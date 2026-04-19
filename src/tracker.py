"""
tracker.py - Object tracking with multi-strategy loss detection and
             automatic VLM re-detection with cooldown.
"""
import cv2
import time
import math
import logging
import threading
from enum import Enum, auto
from typing import Optional, Tuple, Callable

import numpy as np

from .config import get_config
from .vlm_detector import VLMDetector
from .audio_manager import AudioManager

logger = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int]  # (x, y, w, h)


class TrackerState(Enum):
    IDLE = auto()            # No target selected
    DETECTING = auto()       # VLM API call in progress
    TRACKING = auto()        # Actively tracking
    LOST = auto()            # Lost; will retry
    REDETECTING = auto()     # Re-detection API call in progress
    ERROR = auto()           # Max retries exceeded


def _build_cv_tracker(algorithm: str) -> cv2.Tracker:
    algo = algorithm.upper()
    _factories = {
        "CSRT": lambda: cv2.TrackerCSRT.create() if hasattr(cv2, "TrackerCSRT") else None,
        "KCF":  lambda: cv2.TrackerKCF.create()  if hasattr(cv2, "TrackerKCF")  else None,
        "MIL":  lambda: cv2.TrackerMIL.create()  if hasattr(cv2, "TrackerMIL")  else None,
    }
    factory = _factories.get(algo)
    tracker = factory() if factory else None
    if tracker is None:
        # Fallback: try each available tracker in order of preference
        for name in ("CSRT", "MIL", "KCF"):
            f = _factories.get(name)
            if f:
                t = f()
                if t is not None:
                    logger.warning("Tracker '%s' unavailable, using %s", algorithm, name)
                    return t
        raise RuntimeError("No supported OpenCV tracker found (tried CSRT, MIL, KCF)")
    return tracker


class ObjectTracker:
    """
    Wraps an OpenCV tracker and handles:
      - Initialization from VLM bbox
      - Per-frame update with multi-strategy loss detection
      - Automatic re-detection with cooldown
    """

    def __init__(
        self,
        detector: VLMDetector,
        audio: AudioManager,
        on_state_change: Optional[Callable[[TrackerState, str], None]] = None,
    ):
        self._detector = detector
        self._audio = audio
        self._on_state_change = on_state_change

        cfg = get_config()
        tcfg = cfg["tracker"]
        rcfg = cfg["redetection"]

        self._algorithm: str = tcfg["algorithm"]
        self._area_thresh: float = tcfg["area_change_threshold"]
        self._ar_thresh: float = tcfg["aspect_ratio_change_threshold"]
        self._disp_thresh: float = tcfg["displacement_threshold"]
        self._tmpl_thresh: float = tcfg["template_match_threshold"]

        self._cooldown: float = rcfg["cooldown_seconds"]
        self._max_retries: int = rcfg["max_retries"]

        self._loss_confirm_frames: int = tcfg.get("loss_confirmation_frames", 3)

        # Runtime state
        self.state: TrackerState = TrackerState.IDLE
        self.current_bbox: Optional[BBox] = None
        self.description: str = ""

        self._cv_tracker: Optional[cv2.Tracker] = None
        self._initial_area: float = 0.0
        self._initial_ar: float = 1.0
        self._last_center: Optional[Tuple[float, float]] = None
        self._frame_diagonal: float = 1.0

        # Reference bbox dimensions for size-locking
        self._ref_w: int = 0
        self._ref_h: int = 0

        # Consecutive-loss streak counter
        self._loss_streak: int = 0

        # Template for similarity check (taken at init time)
        self._template: Optional[np.ndarray] = None

        # Re-detection bookkeeping
        self._retry_count: int = 0
        self._last_detect_time: float = 0.0
        self._detect_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_detection(self, frame: np.ndarray, description: str):
        """Called when user hits Send. Kicks off async VLM detection."""
        self.description = description
        self._retry_count = 0
        self._set_state(TrackerState.DETECTING, "正在识别目标…")
        self._run_detection_async(frame)

    def update(self, frame: np.ndarray) -> Optional[BBox]:
        """
        Call once per captured frame.
        Returns the current bbox (pixels) if tracking, else None.
        """
        if self.state != TrackerState.TRACKING:
            return self.current_bbox

        assert self._cv_tracker is not None

        h, w = frame.shape[:2]
        ok, raw_bbox = self._cv_tracker.update(frame)

        # Size-stabilized bbox: only lock size for CSRT (MIL center is too noisy for this)
        use_size_lock = ok and self._ref_w > 0 and self._algorithm.upper() == "CSRT"
        if use_size_lock:
            raw_cx = raw_bbox[0] + raw_bbox[2] / 2
            raw_cy = raw_bbox[1] + raw_bbox[3] / 2
            bbox: BBox = (
                int(raw_cx - self._ref_w / 2),
                int(raw_cy - self._ref_h / 2),
                self._ref_w,
                self._ref_h,
            )
        else:
            bbox = tuple(int(v) for v in raw_bbox) if ok else None  # type: ignore

        # Determine loss signals
        lost = (not ok) or (bbox is not None and self._is_lost(bbox, w, h))

        # Template match check: only for CSRT (reliable center), skip for other trackers
        if (not lost and bbox is not None and self._template is not None
                and self._algorithm.upper() == "CSRT"):
            bx, by = bbox[0], bbox[1]
            x1c = max(0, bx)
            y1c = max(0, by)
            x2c = min(w, bx + self._ref_w)
            y2c = min(h, by + self._ref_h)
            vis_w, vis_h = x2c - x1c, y2c - y1c
            if vis_w >= self._ref_w * 0.5 and vis_h >= self._ref_h * 0.5:
                patch = cv2.cvtColor(frame[y1c:y2c, x1c:x2c], cv2.COLOR_BGR2GRAY)
                tmpl = cv2.resize(self._template, (vis_w, vis_h))
                score = float(cv2.matchTemplate(patch, tmpl, cv2.TM_CCOEFF_NORMED)[0][0])
                if score < self._tmpl_thresh:
                    logger.debug("Loss: template match score %.3f < threshold %.3f", score, self._tmpl_thresh)
                    lost = True

        # Require N consecutive loss frames before declaring loss (prevents single-frame noise)
        if lost:
            self._loss_streak += 1
            if self._loss_streak >= self._loss_confirm_frames:
                self._handle_loss(frame, reason=f"loss confirmed over {self._loss_streak} frames")
                return None
            return self.current_bbox  # hold last known position during buildup
        else:
            self._loss_streak = 0
            self.current_bbox = bbox
            self._last_center = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
            return bbox

    def stop(self):
        """Stop tracking and return to IDLE."""
        self._cv_tracker = None
        self.current_bbox = None
        self._set_state(TrackerState.IDLE, "已停止跟踪")

    # ------------------------------------------------------------------
    # Internal: detection flow
    # ------------------------------------------------------------------

    def _run_detection_async(self, frame: np.ndarray):
        self._last_detect_time = time.time()
        t = threading.Thread(
            target=self._detection_worker,
            args=(frame.copy(),),
            daemon=True,
        )
        t.start()

    def _detection_worker(self, frame: np.ndarray):
        bbox = self._detector.detect(frame, self.description)
        if bbox is not None:
            self._on_detected(frame, bbox)
        else:
            self._retry_count += 1
            if self._retry_count >= self._max_retries:
                self._audio.play_error()
                self._set_state(TrackerState.ERROR, f"自动重检失败（{self._max_retries}次），请重新输入描述")
            else:
                self._set_state(
                    TrackerState.LOST,
                    f"未检测到目标，等待重试 ({self._retry_count}/{self._max_retries})…"
                )

    def _on_detected(self, frame: np.ndarray, bbox: BBox):
        """Initialize the OpenCV tracker from a freshly detected bbox."""
        h, w = frame.shape[:2]
        self._frame_diagonal = math.sqrt(w * w + h * h)

        x, y, bw, bh = bbox
        self._initial_area = float(bw * bh)
        self._initial_ar = bw / bh if bh > 0 else 1.0
        self._last_center = (x + bw / 2, y + bh / 2)
        self._ref_w = bw
        self._ref_h = bh
        self._loss_streak = 0

        # Crop template patch for similarity checks
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + bw)
        y2 = min(h, y + bh)
        self._template = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

        self._cv_tracker = _build_cv_tracker(self._algorithm)
        self._cv_tracker.init(frame, bbox)
        self.current_bbox = bbox
        self._retry_count = 0

        self._audio.play_detected()
        self._set_state(TrackerState.TRACKING, f"正在跟踪：{self.description}")

    # ------------------------------------------------------------------
    # Internal: loss detection
    # ------------------------------------------------------------------

    def _is_lost(self, bbox: BBox, frame_w: int, frame_h: int) -> bool:
        x, y, bw, bh = bbox

        # 1. Out of frame — center-based, percentage margin to allow corners outside
        cx = x + bw / 2
        cy = y + bh / 2
        margin_x = frame_w * 0.15
        margin_y = frame_h * 0.15
        if cx < -margin_x or cx > frame_w + margin_x or cy < -margin_y or cy > frame_h + margin_y:
            logger.debug("Loss: center out of frame (cx=%.0f cy=%.0f)", cx, cy)
            return True

        # 2. Single-frame displacement
        if self._last_center is not None:
            dx = cx - self._last_center[0]
            dy = cy - self._last_center[1]
            disp = math.sqrt(dx * dx + dy * dy)
            if disp > self._disp_thresh * self._frame_diagonal:
                logger.debug("Loss: displacement %.1f px too large", disp)
                return True

        return False

    def _handle_loss(self, frame: np.ndarray, reason: str = ""):
        logger.info("Tracking lost: %s", reason)
        self._audio.play_lost()
        self.current_bbox = None
        self._cv_tracker = None
        self._schedule_redetection(frame)

    def _schedule_redetection(self, frame: np.ndarray):
        now = time.time()
        elapsed = now - self._last_detect_time
        wait = max(0.0, self._cooldown - elapsed)

        def _delayed():
            if wait > 0:
                time.sleep(wait)
            self._retry_count += 1
            if self._retry_count > self._max_retries:
                self._audio.play_error()
                self._set_state(TrackerState.ERROR, f"自动重检已达上限（{self._max_retries}次），请重新输入描述")
                return
            self._audio.play_redetecting()
            self._set_state(
                TrackerState.REDETECTING,
                f"目标丢失，重新检测中… ({self._retry_count}/{self._max_retries})"
            )
            self._detection_worker(frame)

        threading.Thread(target=_delayed, daemon=True).start()

    # ------------------------------------------------------------------
    # State machine helper
    # ------------------------------------------------------------------

    def _set_state(self, new_state: TrackerState, message: str = ""):
        self.state = new_state
        logger.info("[State] %s — %s", new_state.name, message)
        if self._on_state_change:
            self._on_state_change(new_state, message)
