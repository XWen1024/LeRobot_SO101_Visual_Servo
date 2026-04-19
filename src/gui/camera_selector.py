"""
camera_selector.py - Camera selection dialog.
Shows a preview thumbnail for each detected camera.
The selected camera index is persisted to config.yaml.
"""
import cv2
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QWidget, QSizePolicy,
    QButtonGroup,
)
from PySide6.QtCore import Qt, QSize, QThread, Signal
from PySide6.QtGui import QImage, QPixmap

from ..camera_manager import probe_cameras, capture_preview_frame
from ..config import save_camera_index, get_config

logger = logging.getLogger(__name__)

_PREVIEW_W = 320
_PREVIEW_H = 180


class _ProbingThread(QThread):
    """Background thread that probes cameras and captures preview frames."""
    finished = Signal(list, list)  # (indices: List[int], frames: List[Optional[np.ndarray]])

    def __init__(self, preferred_w: int, preferred_h: int, parent=None):
        super().__init__(parent)
        self._pw = preferred_w
        self._ph = preferred_h

    def run(self):
        indices = probe_cameras()
        frames = [None] * len(indices)
        if indices:
            with ThreadPoolExecutor(max_workers=len(indices)) as ex:
                future_map = {
                    ex.submit(capture_preview_frame, idx, self._pw, self._ph): pos
                    for pos, idx in enumerate(indices)
                }
                for fut, pos in future_map.items():
                    frames[pos] = fut.result()
        self.finished.emit(indices, frames)


def _frame_to_pixmap(frame, w: int = _PREVIEW_W, h: int = _PREVIEW_H) -> QPixmap:
    if frame is None:
        pm = QPixmap(w, h)
        pm.fill(Qt.black)
        return pm
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
    return QPixmap.fromImage(img).scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class CameraSelector(QDialog):
    """Modal dialog to pick a camera from detected devices."""

    def __init__(self, parent=None, current_index: int = 0):
        super().__init__(parent)
        self.setWindowTitle("选择摄像头")
        self.setMinimumWidth(760)
        self.setModal(True)
        self._selected_index: int = current_index
        self._available: List[int] = []
        self._btn_group = QButtonGroup(self)
        self._btn_group.setExclusive(True)
        self._build_ui()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)

        title = QLabel("请选择摄像头")
        title.setObjectName("selectorTitle")
        title.setAlignment(Qt.AlignCenter)
        root.addWidget(title)

        sub = QLabel("检测到以下摄像头设备，点击选择后确认")
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet("color: #4a6080; font-size: 12px;")
        root.addWidget(sub)

        # Scrollable preview area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        inner = QWidget()
        self._grid = QHBoxLayout(inner)
        self._grid.setSpacing(16)
        self._grid.setAlignment(Qt.AlignCenter)
        scroll.setWidget(inner)
        root.addWidget(scroll, 1)

        self._status_lbl = QLabel("正在探测摄像头…")
        self._status_lbl.setAlignment(Qt.AlignCenter)
        self._status_lbl.setStyleSheet("color: #4a6080; font-size: 11px;")
        root.addWidget(self._status_lbl)

        # Confirm button (disabled until probing finishes)
        row = QHBoxLayout()
        row.addStretch()
        self._confirm_btn = QPushButton("确认选择")
        self._confirm_btn.setObjectName("confirmButton")
        self._confirm_btn.setEnabled(False)
        self._confirm_btn.clicked.connect(self._on_confirm)
        row.addWidget(self._confirm_btn)
        row.addStretch()
        root.addLayout(row)

        # Start background probing
        cfg = get_config()
        pw = cfg["camera"].get("preferred_width", 640)
        ph = cfg["camera"].get("preferred_height", 480)
        self._probe_thread = _ProbingThread(pw, ph, self)
        self._probe_thread.finished.connect(self._on_probe_finished)
        self._probe_thread.start()

    def _on_probe_finished(self, indices: list, frames: list):
        self._available = indices
        if not self._available:
            self._status_lbl.setText("未检测到任何摄像头")
            return

        self._status_lbl.setText(f"检测到 {len(self._available)} 个摄像头设备")
        for idx, frame in zip(self._available, frames):
            pixmap = _frame_to_pixmap(frame)
            self._add_camera_tile(idx, pixmap)
        self._confirm_btn.setEnabled(True)

    def _add_camera_tile(self, index: int, pixmap: QPixmap):
        tile = QWidget()
        tile_layout = QVBoxLayout(tile)
        tile_layout.setAlignment(Qt.AlignCenter)
        tile_layout.setSpacing(6)

        btn = QPushButton()
        btn.setObjectName("camPreviewButton")
        btn.setCheckable(True)
        btn.setFixedSize(QSize(_PREVIEW_W + 8, _PREVIEW_H + 8))
        icon_label = QLabel(btn)
        icon_label.setPixmap(pixmap)
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setGeometry(4, 4, _PREVIEW_W, _PREVIEW_H)

        if index == self._selected_index:
            btn.setChecked(True)

        btn.clicked.connect(lambda checked, i=index: self._on_camera_selected(i))
        self._btn_group.addButton(btn)
        tile_layout.addWidget(btn)

        lbl = QLabel(f"摄像头 {index}")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("color: #78909c; font-size: 12px;")
        tile_layout.addWidget(lbl)

        self._grid.addWidget(tile)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_camera_selected(self, index: int):
        self._selected_index = index

    def _on_confirm(self):
        save_camera_index(self._selected_index)
        self.accept()

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

    def selected_index(self) -> int:
        return self._selected_index
