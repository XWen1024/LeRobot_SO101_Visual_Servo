"""
yolo_tracker.py - Local YOLO-World based object tracker.

Replaces the VLM + OpenCV tracker pipeline with a fully local model.
YOLO-World is an open-vocabulary detector that accepts arbitrary text
prompts, so no cloud API is needed.

The tracker runs inference in a background worker thread so the camera
display is never blocked even on CPU-only machines.

Public interface is identical to ObjectTracker so ServoController and
the GUI require minimal changes.
"""
import logging
import threading
import time
from enum import Enum, auto
from typing import Optional, Tuple, Callable

import cv2
import numpy as np

from .config import get_config

logger = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int]  # (x, y, w, h)  — same as tracker.py


# ── Reuse TrackerState enum (import to stay compatible with controller) ────────

class TrackerState(Enum):
    IDLE       = auto()
    DETECTING  = auto()   # model loading or first inference in progress
    TRACKING   = auto()   # object found, actively tracking
    LOST       = auto()   # object not found for N frames
    REDETECTING = auto()  # (unused, kept for API compat)
    ERROR      = auto()   # model load failed


# ── Exponential-moving-average smoother for bbox ──────────────────────────────

class _BBoxEMA:
    def __init__(self, alpha: float = 0.5):
        self._alpha = alpha
        self._val: Optional[np.ndarray] = None  # [x, y, w, h] float

    def update(self, bbox: BBox) -> BBox:
        b = np.array(bbox, dtype=float)
        if self._val is None:
            self._val = b
        else:
            self._val = self._alpha * b + (1 - self._alpha) * self._val
        return tuple(int(v) for v in self._val)  # type: ignore

    def reset(self):
        self._val = None


# ── Main tracker class ────────────────────────────────────────────────────────

class YoloTracker:
    """
    YOLO-World based open-vocabulary tracker.

    Runs inference in a background thread; update() is non-blocking and
    always returns the latest available bbox without stalling the camera.
    """

    def __init__(self, on_state_change: Optional[Callable] = None):
        self._on_state_change = on_state_change
        self.state:        TrackerState    = TrackerState.IDLE
        self.current_bbox: Optional[BBox] = None
        self.description:  str            = ""

        cfg  = get_config()
        ycfg = cfg.get("yolo", {})
        self._model_name      = ycfg.get("model", "yolov8s-worldv2.pt")
        self._conf_thresh     = float(ycfg.get("conf_threshold", 0.15))
        self._max_lost_frames = int(ycfg.get("max_lost_frames", 15))
        self._infer_interval  = float(ycfg.get("infer_interval_sec", 0.0))  # 0 = every frame

        # EMA smoother
        self._ema = _BBoxEMA(alpha=float(ycfg.get("ema_alpha", 0.6)))

        # Worker thread state
        self._model            = None
        self._model_error      = None
        self._worker_thread: Optional[threading.Thread] = None
        self._running          = False
        self._lock             = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._pending_result: Optional[Optional[BBox]] = None  # None = no new result yet
        self._last_infer_time  = 0.0
        self._no_detect_streak = 0

        # Load model asynchronously
        self._set_state(TrackerState.DETECTING, "YOLO-World 模型加载中…")
        threading.Thread(target=self._load_model, daemon=True).start()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self):
        try:
            from ultralytics import YOLO
            logger.info("Loading YOLO-World model: %s", self._model_name)
            model = YOLO(self._model_name)
            # Warm-up on a tiny black frame
            model.predict(np.zeros((32, 32, 3), dtype=np.uint8),
                          conf=self._conf_thresh, verbose=False)
            with self._lock:
                self._model = model
            logger.info("YOLO-World model ready")
            self._set_state(TrackerState.IDLE, "模型就绪，请输入目标描述")
        except Exception as exc:
            logger.error("Failed to load YOLO-World: %s", exc)
            with self._lock:
                self._model_error = str(exc)
            self._set_state(TrackerState.ERROR,
                            f"模型加载失败: {exc}\n请运行: pip install ultralytics")

    # ── Public API (same as ObjectTracker) ────────────────────────────────────

    def start_detection(self, frame: np.ndarray, description: str):
        """Begin tracking the described object."""
        self.description = description
        self._ema.reset()
        self._no_detect_streak = 0
        self.current_bbox = None

        with self._lock:
            model = self._model
        if model is None:
            if self._model_error:
                self._set_state(TrackerState.ERROR, f"模型未就绪: {self._model_error}")
            else:
                self._set_state(TrackerState.DETECTING, "模型加载中，请稍候…")
            return

        # Set open-vocabulary class
        try:
            model.set_classes([description])
            logger.info("YOLO-World target: '%s'", description)
        except Exception as exc:
            logger.error("set_classes failed: %s", exc)

        # Submit the first frame immediately and start worker
        self._set_state(TrackerState.DETECTING, f"正在识别 '{description}'…")
        with self._lock:
            self._latest_frame = frame.copy()
        self._ensure_worker()

    def update(self, frame: np.ndarray) -> Optional[BBox]:
        """
        Called once per camera frame.
        Submits the frame to the worker and returns the latest bbox (non-blocking).
        """
        if self.state in (TrackerState.IDLE, TrackerState.ERROR):
            return None

        with self._lock:
            model = self._model
        if model is None:
            return None

        # Respect optional infer interval (0 = every frame)
        now = time.time()
        if now - self._last_infer_time >= self._infer_interval:
            with self._lock:
                self._latest_frame = frame.copy()
            self._ensure_worker()

        # Consume pending result from worker
        with self._lock:
            result = self._pending_result
            self._pending_result = None  # mark consumed

        if result is not None:
            bbox, found = result
            if found:
                smooth = self._ema.update(bbox)
                self.current_bbox = smooth
                self._no_detect_streak = 0
                if self.state != TrackerState.TRACKING:
                    self._set_state(TrackerState.TRACKING,
                                    f"正在跟踪：{self.description}")
            else:
                self._no_detect_streak += 1
                if self._no_detect_streak >= self._max_lost_frames:
                    self.current_bbox = None
                    self._ema.reset()
                    self._set_state(TrackerState.LOST,
                                    f"目标丢失（{self._no_detect_streak} 帧未检测到）")

        return self.current_bbox

    def stop(self):
        """Stop tracking and return to IDLE."""
        self._running = False
        self.current_bbox = None
        self._ema.reset()
        self._no_detect_streak = 0
        self._set_state(TrackerState.IDLE, "已停止跟踪")

    # ── Worker thread ─────────────────────────────────────────────────────────

    def _ensure_worker(self):
        """Start the inference worker if not already running."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="yolo-worker")
        self._worker_thread.start()

    def _worker_loop(self):
        """Background inference loop: grab latest frame, run YOLO, post result."""
        logger.debug("YOLO worker started")
        while self._running:
            with self._lock:
                frame = self._latest_frame
                self._latest_frame = None

            if frame is None:
                time.sleep(0.01)
                continue

            self._last_infer_time = time.time()
            bbox = self._run_inference(frame)
            with self._lock:
                self._pending_result = (bbox, bbox is not None)

        logger.debug("YOLO worker stopped")

    def _run_inference(self, frame: np.ndarray) -> Optional[BBox]:
        """Run one YOLO-World inference. Returns (x, y, w, h) or None."""
        with self._lock:
            model = self._model
        if model is None:
            return None
        try:
            results = model.predict(frame, conf=self._conf_thresh, verbose=False)
            if results and len(results[0].boxes) > 0:
                # Pick the highest-confidence detection
                boxes = results[0].boxes
                best  = int(boxes.conf.argmax())
                x1, y1, x2, y2 = boxes.xyxy[best].cpu().numpy()
                conf = float(boxes.conf[best])
                logger.debug("Detected '%s' conf=%.2f bbox=(%d,%d,%d,%d)",
                             self.description, conf,
                             int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
        except Exception as exc:
            logger.warning("YOLO inference error: %s", exc)
        return None

    # ── State helper ──────────────────────────────────────────────────────────

    def _set_state(self, state: TrackerState, message: str = ""):
        self.state = state
        logger.info("[YoloTracker] %s — %s", state.name, message)
        if self._on_state_change:
            self._on_state_change(state, message)
