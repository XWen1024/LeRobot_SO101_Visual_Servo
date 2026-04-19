"""
main_window.py - Main application window.
Uses a QTimer to pull camera frames, runs the tracker,
and renders the annotated feed to a QLabel.
"""
import cv2
import time
import logging
from typing import Optional

import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSizePolicy,
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import QImage, QPixmap, QColor, QPalette, QKeyEvent

from ..config import get_config
from ..camera_manager import CameraManager
from ..vlm_detector import VLMDetector
from ..tracker import ObjectTracker, TrackerState
from ..audio_manager import AudioManager
from .camera_selector import CameraSelector
from .draw_utils import draw_fancy_bbox

logger = logging.getLogger(__name__)

# How often to poll for a new camera frame (ms)
_FRAME_INTERVAL_MS = 30   # ~33 fps


# ── Thread-safe state-change signal bridge ──────────────────────────────────
class _SignalBridge(QObject):
    state_changed = Signal(TrackerState, str)


# ── Overlay drawing helpers ──────────────────────────────────────────────────

_STATE_COLORS = {
    TrackerState.TRACKING:    (105, 240, 174),   # green
    TrackerState.DETECTING:   (255, 241, 118),   # yellow
    TrackerState.REDETECTING: (255, 183, 77),    # amber
    TrackerState.LOST:        (255, 138, 101),   # orange
    TrackerState.ERROR:       (239, 83, 80),     # red
    TrackerState.IDLE:        (79, 195, 247),    # blue
}
_STATE_LABELS_ZH = {
    TrackerState.TRACKING:    "跟踪中",
    TrackerState.DETECTING:   "检测中",
    TrackerState.REDETECTING: "重新检测",
    TrackerState.LOST:        "已丢失",
    TrackerState.ERROR:       "错误",
    TrackerState.IDLE:        "",
}

def _draw_state_badge(frame: np.ndarray, state: TrackerState, message: str) -> np.ndarray:
    """Top-left status badge — delegates to draw_utils."""
    from .draw_utils import draw_state_badge
    color = _STATE_COLORS.get(state, (200, 200, 200))
    badge = _STATE_LABELS_ZH.get(state, "")
    if not badge:
        return frame
    return draw_state_badge(frame, badge, color)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._cfg = get_config()
        self._state: TrackerState = TrackerState.IDLE
        self._status_msg: str = "就绪"
        self._tracking_active: bool = False   # True while user has pressed Start
        self._description: str = ""

        # Components
        self._audio = AudioManager(
            enabled=self._cfg["audio"]["enabled"],
            volume=self._cfg["audio"]["volume"],
        )
        self._detector = VLMDetector()

        bridge = _SignalBridge()
        bridge.state_changed.connect(self._on_state_changed)
        self._bridge = bridge

        self._tracker = ObjectTracker(
            detector=self._detector,
            audio=self._audio,
            on_state_change=lambda s, m: bridge.state_changed.emit(s, m),
        )

        cam_idx = self._cfg["camera"]["default_index"]
        self._cam = CameraManager(
            cam_idx,
            preferred_width=self._cfg["camera"]["preferred_width"],
            preferred_height=self._cfg["camera"]["preferred_height"],
        )

        self._build_ui()
        self._apply_window_style()

        # Start camera (show selector first-time if camera is unavailable)
        if not self._cam.start():
            self._open_camera_selector(force=True)
        else:
            self._start_frame_timer()

    # ── UI Construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle("Visual Grounding Tracker")
        ui_cfg = self._cfg["ui"]
        self.resize(ui_cfg["window_width"], ui_cfg["window_height"])

        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

        # ── Title bar row ────────────────────────────────────────────
        top_row = QHBoxLayout()
        title_lbl = QLabel("🎯  Visual Grounding Tracker")
        title_lbl.setObjectName("titleLabel")
        top_row.addWidget(title_lbl)
        top_row.addStretch()

        self._cam_btn = QPushButton("切换摄像头")
        self._cam_btn.setObjectName("camSwitchButton")
        self._cam_btn.clicked.connect(lambda: self._open_camera_selector())
        top_row.addWidget(self._cam_btn)
        root.addLayout(top_row)

        # ── Camera feed ──────────────────────────────────────────────
        self._cam_label = QLabel()
        self._cam_label.setObjectName("cameraView")
        self._cam_label.setAlignment(Qt.AlignCenter)
        self._cam_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._cam_label.setMinimumHeight(300)
        root.addWidget(self._cam_label, stretch=1)

        # ── Status bar ───────────────────────────────────────────────
        self._status_lbl = QLabel("就绪")
        self._status_lbl.setObjectName("statusLabel")
        self._status_lbl.setProperty("state", "IDLE")
        root.addWidget(self._status_lbl)

        # ── Input row ────────────────────────────────────────────────
        input_row = QHBoxLayout()
        input_row.setSpacing(10)

        self._input = QLineEdit()
        self._input.setPlaceholderText("描述你要跟踪的物体，例如：桌面上的黑色鼠标")
        self._input.returnPressed.connect(self._on_send)
        input_row.addWidget(self._input, stretch=1)

        self._send_btn = QPushButton("发送识别")
        self._send_btn.setObjectName("sendButton")
        self._send_btn.clicked.connect(self._on_send)
        input_row.addWidget(self._send_btn)

        self._track_btn = QPushButton("开始跟踪")
        self._track_btn.setObjectName("trackButton")
        self._track_btn.setProperty("active", "false")
        self._track_btn.setEnabled(False)
        self._track_btn.clicked.connect(self._on_toggle_track)
        input_row.addWidget(self._track_btn)

        root.addLayout(input_row)

    def _apply_window_style(self):
        from .styles import DARK_THEME
        self.setStyleSheet(DARK_THEME)

    # ── Frame timer ──────────────────────────────────────────────────────────

    def _start_frame_timer(self):
        self._frame_timer = QTimer(self)
        self._frame_timer.timeout.connect(self._on_frame_tick)
        self._frame_timer.start(_FRAME_INTERVAL_MS)

    def _on_frame_tick(self):
        frame = self._cam.read_frame()
        if frame is None:
            return

        # Run tracker if active
        if self._tracking_active:
            bbox = self._tracker.update(frame)
            if bbox is not None:
                color = _STATE_COLORS.get(self._state, (79, 195, 247))
                frame = draw_fancy_bbox(frame, bbox, color, self._description)

        frame = _draw_state_badge(frame, self._state, self._status_msg)
        self._display_frame(frame)

    def _display_frame(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        # Scale to fill label while preserving aspect ratio
        lbl_w = self._cam_label.width()
        lbl_h = self._cam_label.height()
        scale = min(lbl_w / w, lbl_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        if new_w > 0 and new_h > 0:
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
        self._cam_label.setPixmap(QPixmap.fromImage(img))

    # ── Slots ────────────────────────────────────────────────────────────────

    def _on_send(self):
        desc = self._input.text().strip()
        if not desc:
            return
        self._description = desc
        self._input.setEnabled(False)
        self._send_btn.setEnabled(False)

        frame = self._cam.read_frame()
        if frame is None:
            self._set_status(TrackerState.ERROR, "无法获取摄像头画面")
            return

        # Stop existing tracking first
        if self._tracking_active:
            self._tracker.stop()
            self._tracking_active = False

        self._tracker.start_detection(frame, desc)

    def _on_toggle_track(self):
        if self._tracking_active:
            self._tracker.stop()
            self._tracking_active = False
            self._track_btn.setText("开始跟踪")
            self._track_btn.setProperty("active", "false")
            self._track_btn.setStyle(self._track_btn.style())  # force repaint
            self._set_status(TrackerState.IDLE, "已停止跟踪")
        else:
            # Re-trigger detection on current frame with last description
            if not self._description:
                return
            frame = self._cam.read_frame()
            if frame is None:
                return
            self._tracker.start_detection(frame, self._description)

    def _on_state_changed(self, state: TrackerState, message: str):
        self._state = state
        self._set_status(state, message)

        if state == TrackerState.TRACKING:
            self._tracking_active = True
            self._track_btn.setText("停止跟踪")
            self._track_btn.setProperty("active", "true")
            self._track_btn.setEnabled(True)
            self._track_btn.setStyle(self._track_btn.style())
            self._input.setEnabled(True)
            self._send_btn.setEnabled(True)
        elif state in (TrackerState.IDLE, TrackerState.ERROR):
            self._tracking_active = False
            self._track_btn.setText("开始跟踪")
            self._track_btn.setProperty("active", "false")
            self._track_btn.setStyle(self._track_btn.style())
            self._track_btn.setEnabled(state == TrackerState.ERROR)
            self._input.setEnabled(True)
            self._send_btn.setEnabled(True)
        elif state in (TrackerState.DETECTING, TrackerState.REDETECTING):
            self._send_btn.setEnabled(False)

    def _set_status(self, state: TrackerState, message: str):
        self._status_msg = message
        self._status_lbl.setText(f"  {message}")
        self._status_lbl.setProperty("state", state.name)
        self._status_lbl.setStyle(self._status_lbl.style())  # force stylesheet recompute

    # ── Camera selector ──────────────────────────────────────────────────────

    def _open_camera_selector(self, force: bool = False):
        dlg = CameraSelector(self, self._cam._index)
        result = dlg.exec()
        if result or force:
            new_idx = dlg.selected_index()
            if new_idx != self._cam._index or force:
                if self._tracking_active:
                    self._tracker.stop()
                    self._tracking_active = False
                self._cam.switch_camera(new_idx)
                if not hasattr(self, "_frame_timer"):
                    self._start_frame_timer()

    # ── Detect disconnected camera ────────────────────────────────────────────

    def _check_camera_health(self):
        if self._cam.read_frame() is None:
            logger.warning("Camera appears disconnected")
            self._audio.play_error()
            self._set_status(TrackerState.ERROR, "摄像头已断开，请重新选择")
            self._open_camera_selector(force=True)

    # ── Clean up ─────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        if hasattr(self, "_frame_timer"):
            self._frame_timer.stop()
        self._cam.stop()
        event.accept()
