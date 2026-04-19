"""
camera_manager.py — Manages camera capture with adaptive resolution.
Runs the capture loop in a background thread and exposes the latest frame
via a thread-safe property.
"""
import cv2
import threading
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _probe_single(i: int) -> Optional[int]:
    """Try to open camera index i and read one frame. Returns i on success."""
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, _ = cap.read()
        cap.release()
        if ret:
            return i
    return None


def probe_cameras(max_index: int = 8) -> List[int]:
    """Return a sorted list of valid camera indices, probed in parallel."""
    with ThreadPoolExecutor(max_workers=max_index) as ex:
        futures = {ex.submit(_probe_single, i): i for i in range(max_index)}
        results = [f.result() for f in as_completed(futures)]
    return sorted(i for i in results if i is not None)


def capture_preview_frame(index: int, width: int = 640, height: int = 480) -> Optional[np.ndarray]:
    """Open a camera very briefly and grab one frame for preview."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # Discard a few frames to let auto-exposure settle
    for _ in range(5):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


class CameraManager:
    """
    Opens a camera and runs a background capture thread.
    The actual resolution is queried from the camera after opening.
    """

    def __init__(self, index: int, preferred_width: int = 1280, preferred_height: int = 720):
        self._index = index
        self._preferred_w = preferred_width
        self._preferred_h = preferred_height
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.width = 0
        self.height = 0

    def start(self) -> bool:
        self._cap = cv2.VideoCapture(self._index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            logger.error("Cannot open camera %d", self._index)
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._preferred_w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._preferred_h)
        # Read actual resolution
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info("Camera %d opened: %dx%d", self._index, self.width, self.height)
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
            self._cap = None

    def switch_camera(self, new_index: int):
        self.stop()
        self._index = new_index
        self.start()

    def read_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def _capture_loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
            else:
                logger.warning("Camera %d: failed to read frame", self._index)
                time.sleep(0.01)
