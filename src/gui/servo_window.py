"""
servo_window.py - Main window for the visual servo grasping application.

Layout
------
Left panel (fixed 280 px): robot connection, position management, controls
Right panel (expanding)  : live camera feed with overlays
Bottom bar               : status message
"""
import csv
import cv2
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QLineEdit, QPushButton, QSizePolicy,
    QFrame, QMessageBox, QDialog, QDialogButtonBox,
    QProgressDialog, QApplication, QScrollArea,
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QThread
from PySide6.QtGui import QImage, QPixmap, QMouseEvent

from ..config import get_config, save_config
from ..camera_manager import CameraManager
from ..yolo_tracker import YoloTracker, TrackerState
from ..audio_manager import AudioManager
from ..robot_manager import RobotManager
from ..servo_controller import ServoController, GraspState, STATE_COLORS, STATE_LABELS
from .draw_utils import draw_fancy_bbox, draw_state_badge, draw_crosshair, draw_approach_bar
from .camera_selector import CameraSelector
from .styles import DARK_THEME

logger = logging.getLogger(__name__)

_FRAME_MS     = 33      # ~30 fps camera poll
_POSITIONS_FILE = Path("positions.json")

# Map tracker states that can bleed into camera feed rendering
_TRACKER_COLORS = {
    TrackerState.TRACKING:    (105, 240, 174),
    TrackerState.DETECTING:   (255, 241, 118),
    TrackerState.REDETECTING: (255, 183, 77),
    TrackerState.LOST:        (255, 138, 101),
    TrackerState.ERROR:       (239, 83, 80),
    TrackerState.IDLE:        (79, 195, 247),
}


# ── Signal bridge (tracker state changes come from background threads) ────────

class _Bridge(QObject):
    grasp_state_changed = Signal(object, str)   # GraspState, message


# ── Robot connection worker thread ────────────────────────────────────────────

class _ConnectWorker(QThread):
    finished = Signal(bool)   # True = success

    def __init__(self, robot, port: str):
        super().__init__()
        self._robot = robot
        self._port  = port

    def run(self):
        ok = self._robot.connect(self._port)
        self.finished.emit(ok)


# ── Recording dialog ──────────────────────────────────────────────────────────

class RecordDialog(QDialog):
    """Instructs the user to move the arm then click Confirm."""

    def __init__(self, pos_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("录制位置")
        self.setModal(True)
        self.setMinimumWidth(360)
        self.setStyleSheet(DARK_THEME)

        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(20, 18, 20, 16)

        title = QLabel(f"录制「{pos_name}」")
        title.setStyleSheet("font-size:15px; font-weight:bold; color:#4fc3f7;")
        layout.addWidget(title)

        info = QLabel(
            "机械臂力矩已关闭，请手动将机械臂移动到目标位置，\n"
            "然后点击【确认录制】。"
        )
        info.setWordWrap(True)
        info.setStyleSheet("color:#c0d0e8; font-size:13px;")
        layout.addWidget(info)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setText("确认录制")
        btns.button(QDialogButtonBox.Cancel).setText("取消")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)


# ── Separator helper ──────────────────────────────────────────────────────────

def _hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet("color: #1e3a5f;")
    return line


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("font-size:11px; color:#4a6080; letter-spacing:1px;")
    return lbl


# ── Main window ───────────────────────────────────────────────────────────────

class ServoGraspWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._cfg = get_config()

        self._audio = AudioManager(
            enabled=self._cfg["audio"]["enabled"],
            volume=self._cfg["audio"]["volume"],
        )
        self._robot = RobotManager()

        # Must be set before YoloTracker is created (its __init__ fires the callback)
        self._grasp_state   = GraspState.DISCONNECTED
        self._tracker_state = TrackerState.IDLE

        self._bridge = _Bridge()
        self._bridge.grasp_state_changed.connect(self._on_grasp_state)

        self._tracker = YoloTracker(on_state_change=self._on_tracker_state)

        self._controller = ServoController(
            robot=self._robot,
            tracker=self._tracker,
            cfg=self._cfg,
            on_state_change=lambda s, m: self._bridge.grasp_state_changed.emit(s, m),
        )

        cam_cfg = self._cfg["camera"]
        self._cam = CameraManager(
            cam_cfg["default_index"],
            preferred_width=cam_cfg["preferred_width"],
            preferred_height=cam_cfg["preferred_height"],
        )

        self._status_msg = "请连接机器人"
        self._description = ""
        self._last_bbox = None
        self._last_area_ratio = 0.0
        self._last_error_x = 0.0
        self._last_error_y = 0.0

        # Center calibration (pixels relative to frame center)
        scfg = self._cfg.get("servo", {})
        self._center_offset_x: int = int(scfg.get("center_offset_x", 0))
        self._center_offset_y: int = int(scfg.get("center_offset_y", 0))
        self._center_calibrating: bool = False
        self._controller.set_center_offset(self._center_offset_x, self._center_offset_y)

        # Worker thread reference (prevent GC)
        self._conn_worker = None

        # Demo recording state
        self._demo_recording  = False
        self._demo_frames: list = []
        self._demo_start_time = 0.0

        # Saved positions (may be loaded from file)
        self._positions: dict = {"standby": None, "pregrasp": None, "placement": None}
        self._load_positions_file()

        self._build_ui()
        self.setStyleSheet(DARK_THEME + _EXTRA_STYLE)
        self.setWindowTitle("视觉伺服抓取")

        if not self._cam.start():
            self._open_camera_selector(force=True)
        else:
            self._start_timer()

        # Auto-connect if a port was previously saved
        saved_port = self._cfg.get("robot", {}).get("port", "").strip()
        if saved_port:
            QTimer.singleShot(600, self._auto_connect)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # Fit the window into the available screen area (90%), with a sane minimum
        screen = QApplication.primaryScreen().availableGeometry()
        w = min(self._cfg["ui"]["window_width"] + 80, int(screen.width()  * 0.95))
        h = min(self._cfg["ui"]["window_height"],      int(screen.height() * 0.92))
        self.resize(w, h)
        self.setMinimumSize(760, 480)
        # Center on screen
        fg = self.frameGeometry()
        fg.moveCenter(screen.center())
        self.move(fg.topLeft())

        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 8)
        root.setSpacing(8)

        # ── Title row ──────────────────────────────────────────────────
        title_row = QHBoxLayout()
        title_lbl = QLabel("🤖  视觉伺服抓取系统")
        title_lbl.setObjectName("titleLabel")
        title_row.addWidget(title_lbl)
        title_row.addStretch()
        cam_btn = QPushButton("切换摄像头")
        cam_btn.setObjectName("camSwitchButton")
        cam_btn.clicked.connect(lambda: self._open_camera_selector())
        title_row.addWidget(cam_btn)
        root.addLayout(title_row)

        # ── Main body (side-panel + camera) ────────────────────────────
        body = QHBoxLayout()
        body.setSpacing(10)
        body.addWidget(self._build_side_panel(), stretch=0)
        body.addWidget(self._build_camera_panel(), stretch=1)
        root.addLayout(body, stretch=1)

        # ── Status bar ─────────────────────────────────────────────────
        self._status_lbl = QLabel(self._status_msg)
        self._status_lbl.setObjectName("statusLabel")
        self._status_lbl.setProperty("gs", "DISCONNECTED")
        root.addWidget(self._status_lbl)

    def _build_side_panel(self) -> QWidget:
        # Wrap side-panel contents in a scroll area so nothing is cut off on small screens
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setFixedWidth(280)
        scroll.setObjectName("sidePanelScroll")
        scroll.setStyleSheet("QScrollArea#sidePanelScroll { border: none; background: transparent; }")

        panel = QWidget()
        panel.setObjectName("sidePanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)

        # ── Robot connection ───────────────────────────────────────────
        layout.addWidget(_section_label("机 器 人 连 接"))
        conn_row = QHBoxLayout()
        self._port_input = QLineEdit()
        self._port_input.setPlaceholderText("COM3")
        self._port_input.setText(self._cfg.get("robot", {}).get("port", "COM3"))
        conn_row.addWidget(self._port_input, stretch=1)
        self._connect_btn = QPushButton("连接")
        self._connect_btn.setObjectName("connectButton")
        self._connect_btn.clicked.connect(self._on_connect_toggle)
        conn_row.addWidget(self._connect_btn)
        layout.addLayout(conn_row)

        self._conn_status = QLabel("● 未连接")
        self._conn_status.setObjectName("connStatus")
        self._conn_status.setProperty("connected", "false")
        layout.addWidget(self._conn_status)

        layout.addWidget(_hline())

        # ── Camera center calibration ──────────────────────────────────
        layout.addWidget(_section_label("摄 像 头 中 心 校 准"))
        self._calib_btn = QPushButton("设置中心点（点击画面）")
        self._calib_btn.setObjectName("posButton")
        self._calib_btn.setCheckable(True)
        self._calib_btn.clicked.connect(self._on_toggle_calibration)
        layout.addWidget(self._calib_btn)

        self._calib_offset_lbl = QLabel(self._fmt_offset())
        self._calib_offset_lbl.setStyleSheet("color:#4a6080; font-size:11px;")
        layout.addWidget(self._calib_offset_lbl)

        reset_calib_btn = QPushButton("重置中心点")
        reset_calib_btn.setObjectName("ioButton")
        reset_calib_btn.clicked.connect(self._on_reset_calibration)
        layout.addWidget(reset_calib_btn)

        layout.addWidget(_hline())

        # ── Position management ────────────────────────────────────────
        layout.addWidget(_section_label("位 置 管 理"))

        self._rec_standby_btn   = self._pos_btn("录制待机位",    lambda: self._record_position("standby"))
        self._rec_pregrasp_btn  = self._pos_btn("录制待抓取位",  lambda: self._record_position("pregrasp"))
        self._rec_place_btn     = self._pos_btn("录制放置位",    lambda: self._record_position("placement"))
        self._ind_standby  = QLabel()
        self._ind_pregrasp = QLabel()
        self._ind_place    = QLabel()

        for btn, ind, key in [
            (self._rec_standby_btn,  self._ind_standby,  "standby"),
            (self._rec_pregrasp_btn, self._ind_pregrasp, "pregrasp"),
            (self._rec_place_btn,    self._ind_place,    "placement"),
        ]:
            row = QHBoxLayout()
            row.addWidget(btn, stretch=1)
            ind.setFixedWidth(60)
            ind.setAlignment(Qt.AlignCenter)
            row.addWidget(ind)
            layout.addLayout(row)

        pos_io_row = QHBoxLayout()
        save_pos_btn = QPushButton("保存")
        save_pos_btn.setObjectName("ioButton")
        save_pos_btn.clicked.connect(self._save_positions_file)
        load_pos_btn = QPushButton("加载")
        load_pos_btn.setObjectName("ioButton")
        load_pos_btn.clicked.connect(self._on_load_positions)
        pos_io_row.addWidget(save_pos_btn)
        pos_io_row.addWidget(load_pos_btn)
        layout.addLayout(pos_io_row)

        self._update_pos_indicators()
        layout.addWidget(_hline())

        # ── Object description ─────────────────────────────────────────
        layout.addWidget(_section_label("识 别 目 标"))
        self._desc_input = QLineEdit()
        self._desc_input.setPlaceholderText("描述抓取目标，如：红色瓶子")
        self._desc_input.returnPressed.connect(self._on_start_tracking)
        layout.addWidget(self._desc_input)

        self._track_btn = QPushButton("开始识别跟踪")
        self._track_btn.setObjectName("trackButton")
        self._track_btn.setProperty("active", "false")
        self._track_btn.setEnabled(False)
        self._track_btn.clicked.connect(self._on_toggle_tracking)
        layout.addWidget(self._track_btn)

        layout.addWidget(_hline())

        # ── Motion controls ────────────────────────────────────────────
        layout.addWidget(_section_label("运 动 控 制"))

        self._goto_standby_btn  = self._ctrl_btn("前往待机位",    self._on_goto_standby,   "navButton")
        self._goto_pregrasp_btn = self._ctrl_btn("前往待抓取位",  self._on_goto_pregrasp,  "navButton")
        self._grasp_btn         = self._ctrl_btn("开始抓取",      self._on_start_grasp,    "graspButton")
        self._place_btn         = self._ctrl_btn("放置物品",      self._on_place,          "placeButton")
        layout.addWidget(self._goto_standby_btn)
        layout.addWidget(self._goto_pregrasp_btn)
        layout.addWidget(self._grasp_btn)
        layout.addWidget(self._place_btn)

        stop_btn = QPushButton("紧急停止")
        stop_btn.setObjectName("stopButton")
        stop_btn.clicked.connect(self._on_stop)
        layout.addWidget(stop_btn)

        layout.addWidget(_hline())

        # ── Gripper test ───────────────────────────────────────────────
        layout.addWidget(_section_label("夹 爪 测 试"))
        gripper_row = QHBoxLayout()
        open_btn  = QPushButton("张开夹爪")
        close_btn = QPushButton("闭合夹爪")
        open_btn.setObjectName("ioButton")
        close_btn.setObjectName("ioButton")
        open_btn.clicked.connect(self._on_gripper_open)
        close_btn.clicked.connect(self._on_gripper_close)
        gripper_row.addWidget(open_btn)
        gripper_row.addWidget(close_btn)
        layout.addLayout(gripper_row)

        self._gripper_val_lbl = QLabel("open=0  close=70")
        self._gripper_val_lbl.setStyleSheet("color:#4a6080; font-size:11px;")
        layout.addWidget(self._gripper_val_lbl)

        layout.addWidget(_hline())

        # ── Demo recording ─────────────────────────────────────────────
        layout.addWidget(_section_label("示 教 录 制"))
        self._demo_btn = QPushButton("开始录制（卸力模式）")
        self._demo_btn.setObjectName("ioButton")
        self._demo_btn.setEnabled(False)
        self._demo_btn.clicked.connect(self._on_demo_toggle)
        layout.addWidget(self._demo_btn)
        self._demo_status_lbl = QLabel("未录制")
        self._demo_status_lbl.setStyleSheet("color:#4a6080; font-size:11px;")
        layout.addWidget(self._demo_status_lbl)

        layout.addStretch()
        self._refresh_button_states()
        scroll.setWidget(panel)
        return scroll

    def _build_camera_panel(self) -> QWidget:
        panel = QWidget()
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self._cam_label = QLabel()
        self._cam_label.setObjectName("cameraView")
        self._cam_label.setAlignment(Qt.AlignCenter)
        self._cam_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._cam_label.mousePressEvent = self._on_cam_click
        v.addWidget(self._cam_label)
        return panel

    @staticmethod
    def _pos_btn(text: str, slot) -> QPushButton:
        btn = QPushButton(text)
        btn.setObjectName("posButton")
        btn.clicked.connect(slot)
        return btn

    @staticmethod
    def _ctrl_btn(text: str, slot, obj_name: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setObjectName(obj_name)
        btn.setEnabled(False)
        btn.clicked.connect(slot)
        return btn

    # ── Timer & frame rendering ───────────────────────────────────────────────

    def _start_timer(self):
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start(_FRAME_MS)

    def _on_tick(self):
        frame = self._cam.read_frame()
        if frame is None:
            return

        # Demo recording mode: bypass the state-machine controller so it
        # never sends joint commands (which would re-engage torque on the servos).
        if self._demo_recording:
            self._on_tick_demo(frame)
            return

        info = self._controller.update(frame)

        # Store latest overlay values
        if info.bbox is not None:
            self._last_bbox      = info.bbox
            self._last_error_x   = info.error_x
            self._last_error_y   = info.error_y
            self._last_area_ratio = info.area_ratio
        else:
            self._last_bbox = None

        self._render_frame(frame, info)

    def _on_tick_demo(self, frame):
        """Camera tick used during demo recording.
        Runs the tracker for visual feedback, reads joints, logs data.
        The state-machine controller is NOT called so no joint commands are sent.
        """
        from ..servo_controller import FrameInfo
        bbox = self._tracker.update(frame)

        info = FrameInfo()
        info.state = self._grasp_state   # keep badge showing current state
        if bbox is not None:
            h, w = frame.shape[:2]
            bx, by, bw, bh = bbox
            info.bbox = bbox
            info.error_x    = (bx + bw / 2 - (w / 2 + self._center_offset_x)) / w
            info.error_y    = (by + bh / 2 - (h / 2 + self._center_offset_y)) / h
            info.area_ratio = (bw * bh) / (w * h)
            self._last_bbox       = bbox
            self._last_error_x    = info.error_x
            self._last_error_y    = info.error_y
            self._last_area_ratio = info.area_ratio
        else:
            self._last_bbox = None

        # Record data
        if self._demo_recording:
            joints = self._robot.get_joints() or {}
            ts = time.time() - self._demo_start_time
            bx2, by2, bw2, bh2 = info.bbox if info.bbox else (0, 0, 0, 0)
            self._demo_frames.append({
                "t":             round(ts, 3),
                "has_bbox":      int(info.bbox is not None),
                "bbox_x":        bx2, "bbox_y": by2,
                "bbox_w":        bw2, "bbox_h": bh2,
                "area_ratio":    round(info.area_ratio, 4),
                "error_x":       round(info.error_x, 4),
                "error_y":       round(info.error_y, 4),
                "shoulder_pan":  round(joints.get("shoulder_pan.pos", 0), 2),
                "shoulder_lift": round(joints.get("shoulder_lift.pos", 0), 2),
                "elbow_flex":    round(joints.get("elbow_flex.pos", 0), 2),
                "wrist_flex":    round(joints.get("wrist_flex.pos", 0), 2),
                "wrist_roll":    round(joints.get("wrist_roll.pos", 0), 2),
                "gripper":       round(joints.get("gripper.pos", 0), 2),
            })
            n = len(self._demo_frames)
            if n % 30 == 0:
                self._demo_status_lbl.setText(f"录制中… {n} 帧")

        self._render_frame(frame, info)

        self._render_frame(frame, info)

    def _render_frame(self, frame: np.ndarray, info):
        state = info.state

        # Draw bounding box
        if info.bbox is not None:
            color = STATE_COLORS.get(state, (79, 195, 247))
            frame = draw_fancy_bbox(frame, info.bbox, color, self._description)

        # In tracking/approaching: crosshair + error arrow (always show offset center)
        if state in (GraspState.TRACKING, GraspState.APPROACHING) or self._center_calibrating:
            frame = draw_crosshair(
                frame, info.error_x, info.error_y,
                offset_x=self._center_offset_x,
                offset_y=self._center_offset_y,
                calibrating=self._center_calibrating,
            )

        # In approaching: progress bar
        if state == GraspState.APPROACHING:
            thresh = self._cfg.get("servo", {}).get("grasp_area_threshold", 0.18)
            frame = draw_approach_bar(frame, info.area_ratio, thresh)

        # State badge
        label = STATE_LABELS.get(state, "")
        color = STATE_COLORS.get(state, (79, 195, 247))
        if label:
            frame = draw_state_badge(frame, label, color)

        self._display(frame)

    def _display(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        lw = self._cam_label.width()
        lh = self._cam_label.height()
        if lw <= 0 or lh <= 0:
            return
        scale  = min(lw / w, lh / h)
        new_w  = max(1, int(w * scale))
        new_h  = max(1, int(h * scale))
        frame  = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img    = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
        self._cam_label.setPixmap(QPixmap.fromImage(img))

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _auto_connect(self):
        """Silently attempt connection using the saved port on startup."""
        if self._robot.is_connected:
            return
        port = self._cfg.get("robot", {}).get("port", "").strip()
        if port:
            self._start_connect(port, silent=True)

    def _on_connect_toggle(self):
        if self._robot.is_connected:
            self._robot.disconnect()
            self._controller.on_robot_disconnected()
            self._conn_status.setText("● 未连接")
            self._conn_status.setProperty("connected", "false")
            self._conn_status.setStyle(self._conn_status.style())
            self._connect_btn.setText("连接")
            self._refresh_button_states()
        else:
            port = self._port_input.text().strip() or "COM3"
            self._start_connect(port, silent=False)

    def _start_connect(self, port: str, silent: bool = False):
        """Spawn worker thread and show progress dialog while connecting."""
        self._connect_btn.setEnabled(False)
        self._connect_btn.setText("连接中…")

        # Progress dialog (non-cancellable)
        self._conn_dlg = QProgressDialog("正在连接机械臂…", None, 0, 0, self)
        self._conn_dlg.setWindowTitle("连接中")
        self._conn_dlg.setWindowModality(Qt.WindowModal)
        self._conn_dlg.setMinimumDuration(0)
        self._conn_dlg.setValue(0)
        self._conn_dlg.setStyleSheet(DARK_THEME)

        self._conn_worker = _ConnectWorker(self._robot, port)
        self._conn_worker.finished.connect(lambda ok: self._on_connect_done(ok, port, silent))
        self._conn_worker.start()

    def _on_connect_done(self, success: bool, port: str, silent: bool):
        self._conn_dlg.close()

        if success:
            # Save port to config
            self._cfg.setdefault("robot", {})["port"] = port
            save_config()

            self._controller.on_robot_connected()
            self._conn_status.setText("● 已连接")
            self._conn_status.setProperty("connected", "true")
            self._conn_status.setStyle(self._conn_status.style())
            self._connect_btn.setText("断开")
            self._connect_btn.setEnabled(True)
        else:
            self._conn_status.setText("● 连接失败")
            self._conn_status.setProperty("connected", "false")
            self._conn_status.setStyle(self._conn_status.style())
            self._connect_btn.setText("连接")
            self._connect_btn.setEnabled(True)
            if not silent:
                QMessageBox.warning(self, "连接失败",
                                    f"无法连接到 {port}\n请检查串口和驱动。")
        self._refresh_button_states()

    def _record_position(self, key: str):
        if not self._robot.is_connected:
            QMessageBox.information(self, "提示", "请先连接机器人。")
            return

        names = {"standby": "待机位", "pregrasp": "待抓取位", "placement": "放置位"}
        dlg = RecordDialog(names.get(key, key), self)

        self._robot.disable_torque()
        result = dlg.exec()
        self._robot.enable_torque()

        if result == QDialog.Accepted:
            joints = self._robot.get_joints()
            if joints:
                self._positions[key] = joints
                self._controller.set_positions(**{key: joints})
                self._update_pos_indicators()
                self._save_positions_file()
                logger.info("Recorded %s: %s", key, joints)
            else:
                QMessageBox.warning(self, "录制失败", "无法读取关节位置，请重试。")

    def _on_start_tracking(self):
        desc = self._desc_input.text().strip()
        if not desc:
            return
        self._description = desc
        frame = self._cam.read_frame()
        if frame is None:
            return
        self._controller.start_tracking(desc, frame)

    def _on_toggle_tracking(self):
        if self._grasp_state in (GraspState.TRACKING, GraspState.DETECTING):
            self._controller.stop()
            self._track_btn.setText("开始识别跟踪")
            self._track_btn.setProperty("active", "false")
            self._track_btn.setStyle(self._track_btn.style())
        else:
            self._on_start_tracking()

    def _on_goto_standby(self):
        self._controller.go_standby()

    def _on_goto_pregrasp(self):
        self._controller.go_pregrasp()

    def _on_start_grasp(self):
        self._controller.start_approach()

    def _on_place(self):
        self._controller.place()

    def _on_stop(self):
        self._controller.stop()

    # ── Demo recording ────────────────────────────────────────────────────────

    def _on_demo_toggle(self):
        if self._demo_recording:
            self._stop_demo()
        else:
            self._start_demo()

    def _start_demo(self):
        desc = self._desc_input.text().strip()
        if not desc:
            QMessageBox.information(self, "提示", "请先在「识别目标」框输入目标描述。")
            return
        # Stop state machine first so it sends no more joint commands
        self._controller.stop()
        self._tracker.stop()
        # Disable torque so arm can be moved freely
        self._robot.disable_torque()
        # Start tracker independently (visual feedback only, no control)
        self._description = desc
        frame = self._cam.read_frame()
        if frame is not None:
            self._tracker.start_detection(frame, desc)
        self._demo_frames = []
        self._demo_start_time = time.time()
        self._demo_recording = True
        self._demo_btn.setText("停止录制并保存")
        self._demo_status_lbl.setText("录制中… 0 帧")
        logger.info("Demo recording started for '%s'", desc)

    def _stop_demo(self):
        self._demo_recording = False
        self._tracker.stop()
        # Re-enable torque
        self._robot.enable_torque()
        self._demo_btn.setText("开始录制（卸力模式）")
        n = len(self._demo_frames)
        if n == 0:
            self._demo_status_lbl.setText("无数据")
            return
        # Save CSV
        out_path = Path(f"demo_{int(self._demo_start_time)}.csv")
        fieldnames = list(self._demo_frames[0].keys())
        try:
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(self._demo_frames)
            msg = f"已保存 {n} 帧 → {out_path.name}"
            logger.info("Demo saved: %s (%d frames)", out_path, n)
        except Exception as exc:
            msg = f"保存失败: {exc}"
            logger.error("Demo save error: %s", exc)
        self._demo_status_lbl.setText(msg)
        self._demo_frames = []

    # ── State change callbacks ────────────────────────────────────────────────

    def _on_grasp_state(self, state: GraspState, message: str):
        self._grasp_state = state
        self._status_msg = message
        self._status_lbl.setText(f"  {message}")
        self._status_lbl.setProperty("gs", state.name)
        self._status_lbl.setStyle(self._status_lbl.style())
        self._refresh_button_states()

        if state == GraspState.TRACKING:
            self._track_btn.setText("停止跟踪")
            self._track_btn.setProperty("active", "true")
            self._track_btn.setStyle(self._track_btn.style())
        elif state in (GraspState.STANDBY, GraspState.PREGRASP,
                       GraspState.ERROR, GraspState.DISCONNECTED):
            self._track_btn.setText("开始识别跟踪")
            self._track_btn.setProperty("active", "false")
            self._track_btn.setStyle(self._track_btn.style())

        # Auto-open gripper and announce after grasping
        if state == GraspState.CLOSING:
            # Brief timer to let gripper close, then mark as GRASPED
            QTimer.singleShot(1200, self._on_grip_done)

    def _on_grip_done(self):
        self._controller.notify_grasped()

    def _on_tracker_state(self, state: TrackerState, message: str):
        self._tracker_state = state
        # Show YOLO model status in the status bar when not yet tracking
        if (state in (TrackerState.DETECTING, TrackerState.ERROR)
                and self._grasp_state == GraspState.DISCONNECTED
                and hasattr(self, "_status_lbl")):
            self._status_lbl.setText(f"  {message}")

    # ── Gripper test ──────────────────────────────────────────────────────────

    def _on_gripper_open(self):
        if not self._robot.is_connected:
            return
        val = self._cfg.get("servo", {}).get("gripper_open", 0)
        ok = self._robot.set_gripper(float(val))
        logger.info("Gripper TEST open → %s  ok=%s", val, ok)
        self._gripper_val_lbl.setText(f"已发送 open={val}")

    def _on_gripper_close(self):
        if not self._robot.is_connected:
            return
        val = self._cfg.get("servo", {}).get("gripper_close", 70)
        ok = self._robot.set_gripper(float(val))
        logger.info("Gripper TEST close → %s  ok=%s", val, ok)
        self._gripper_val_lbl.setText(f"已发送 close={val}")

    # ── Button state management ───────────────────────────────────────────────

    def _refresh_button_states(self):
        if not hasattr(self, "_track_btn"):
            return
        connected = self._robot.is_connected
        state = self._grasp_state
        idle = state in (GraspState.STANDBY, GraspState.PREGRASP,
                         GraspState.GRASPED, GraspState.ERROR)

        self._track_btn.setEnabled(
            connected and state in (GraspState.PREGRASP, GraspState.STANDBY,
                                    GraspState.TRACKING, GraspState.DETECTING,
                                    GraspState.ERROR)
        )
        self._goto_standby_btn.setEnabled(
            connected and idle and self._positions["standby"] is not None
        )
        self._goto_pregrasp_btn.setEnabled(
            connected and idle and self._positions["pregrasp"] is not None
        )
        self._grasp_btn.setEnabled(connected and state == GraspState.TRACKING)
        self._place_btn.setEnabled(
            connected and state == GraspState.GRASPED
            and self._positions["placement"] is not None
        )
        self._demo_btn.setEnabled(connected)

    # ── Position file helpers ─────────────────────────────────────────────────

    def _update_pos_indicators(self):
        _ok  = "✓  已录制"
        _no  = "—  未录制"
        style_ok = "color:#69f0ae; font-size:11px;"
        style_no = "color:#4a6080; font-size:11px;"
        for ind, key in [
            (self._ind_standby,  "standby"),
            (self._ind_pregrasp, "pregrasp"),
            (self._ind_place,    "placement"),
        ]:
            if self._positions.get(key):
                ind.setText(_ok)
                ind.setStyleSheet(style_ok)
            else:
                ind.setText(_no)
                ind.setStyleSheet(style_no)
        self._refresh_button_states()

    def _save_positions_file(self):
        try:
            _POSITIONS_FILE.write_text(
                json.dumps(self._positions, indent=2, ensure_ascii=False)
            )
            logger.info("Positions saved to %s", _POSITIONS_FILE)
        except Exception as exc:
            logger.error("Failed to save positions: %s", exc)

    def _load_positions_file(self):
        if not _POSITIONS_FILE.exists():
            return
        try:
            data = json.loads(_POSITIONS_FILE.read_text(encoding="utf-8"))
            for key in ("standby", "pregrasp", "placement"):
                if data.get(key):
                    self._positions[key] = data[key]
            self._controller.set_positions(
                standby=self._positions.get("standby"),
                pregrasp=self._positions.get("pregrasp"),
                placement=self._positions.get("placement"),
            )
            logger.info("Positions loaded from %s", _POSITIONS_FILE)
        except Exception as exc:
            logger.warning("Could not load positions: %s", exc)

    def _on_load_positions(self):
        self._load_positions_file()
        self._update_pos_indicators()

    # ── Camera center calibration ─────────────────────────────────────────────

    def _fmt_offset(self) -> str:
        return f"偏移: ({self._center_offset_x:+d}, {self._center_offset_y:+d}) px"

    def _on_toggle_calibration(self, checked: bool):
        self._center_calibrating = checked
        if checked:
            self._calib_btn.setText("✓ 点击画面确定中心点")
            self._cam_label.setCursor(Qt.CrossCursor)
        else:
            self._calib_btn.setText("设置中心点（点击画面）")
            self._cam_label.setCursor(Qt.ArrowCursor)

    def _on_reset_calibration(self):
        self._center_offset_x = 0
        self._center_offset_y = 0
        self._controller.set_center_offset(0, 0)
        self._cfg.setdefault("servo", {})["center_offset_x"] = 0
        self._cfg.setdefault("servo", {})["center_offset_y"] = 0
        save_config()
        self._calib_offset_lbl.setText(self._fmt_offset())

    def _on_cam_click(self, event: QMouseEvent):
        if not self._center_calibrating:
            return
        # Map label pixel → frame pixel
        lw = self._cam_label.width()
        lh = self._cam_label.height()
        frame = self._cam.read_frame()
        if frame is None or lw <= 0 or lh <= 0:
            return
        fh, fw = frame.shape[:2]
        scale = min(lw / fw, lh / fh)
        disp_w = int(fw * scale)
        disp_h = int(fh * scale)
        # Image is centered in the label
        pad_x = (lw - disp_w) // 2
        pad_y = (lh - disp_h) // 2
        click_x = event.position().x() - pad_x
        click_y = event.position().y() - pad_y
        if click_x < 0 or click_y < 0 or click_x > disp_w or click_y > disp_h:
            return
        # Convert to frame coordinates, compute offset from center
        frame_x = int(click_x / scale)
        frame_y = int(click_y / scale)
        self._center_offset_x = frame_x - fw // 2
        self._center_offset_y = frame_y - fh // 2
        self._controller.set_center_offset(self._center_offset_x, self._center_offset_y)
        self._cfg.setdefault("servo", {})["center_offset_x"] = self._center_offset_x
        self._cfg.setdefault("servo", {})["center_offset_y"] = self._center_offset_y
        save_config()
        self._calib_offset_lbl.setText(self._fmt_offset())
        logger.info("Center offset set to (%d, %d)", self._center_offset_x, self._center_offset_y)
        # Exit calibration mode
        self._calib_btn.setChecked(False)
        self._on_toggle_calibration(False)

    # ── Camera selector ───────────────────────────────────────────────────────

    def _open_camera_selector(self, force: bool = False):
        dlg = CameraSelector(self, self._cam._index)
        result = dlg.exec()
        if result or force:
            new_idx = dlg.selected_index()
            if new_idx != self._cam._index or force:
                self._cam.switch_camera(new_idx)
                if not hasattr(self, "_timer"):
                    self._start_timer()

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        if hasattr(self, "_timer"):
            self._timer.stop()
        self._tracker.stop()
        self._cam.stop()
        self._robot.disconnect()
        event.accept()


# ── Extra stylesheet for servo-specific widgets ───────────────────────────────

_EXTRA_STYLE = """
QWidget#sidePanel {
    background: #0a0f16;
    border-right: 1px solid #1e3a5f;
    border-radius: 6px;
}

QLabel#connStatus {
    font-size: 12px;
    padding: 2px 6px;
}
QLabel#connStatus[connected="false"] { color: #ef5350; }
QLabel#connStatus[connected="true"]  { color: #69f0ae; }

QPushButton#connectButton {
    background: #1b3a5f;
    color: #90caf9;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
    font-weight: bold;
}
QPushButton#connectButton:hover  { background: #1e4a7a; }
QPushButton#connectButton:disabled { background: #111820; color: #3a5070; }

QPushButton#posButton {
    background: #131c2b;
    color: #b0c4d8;
    border: 1px solid #1e3a5f;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
}
QPushButton#posButton:hover  { background: #1a2840; border-color: #4fc3f7; }
QPushButton#posButton:pressed { background: #0d1620; }

QPushButton#ioButton {
    background: #0f1a28;
    color: #78909c;
    border: 1px solid #1e3a5f;
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 12px;
}
QPushButton#ioButton:hover { background: #162030; }

QPushButton#navButton {
    background: #0d2a4a;
    color: #81d4fa;
    border: 1px solid #1e4a7a;
    border-radius: 6px;
    padding: 7px 12px;
    font-size: 13px;
    font-weight: bold;
}
QPushButton#navButton:hover    { background: #12346a; }
QPushButton#navButton:disabled { background: #0a1520; color: #2a4060; border-color: #142030; }

QPushButton#graspButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #1b5e20, stop:1 #2e7d32);
    color: #c8e6c9;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 13px;
    font-weight: bold;
}
QPushButton#graspButton:hover    { background: #388e3c; }
QPushButton#graspButton:disabled { background: #0f1e14; color: #2a4030; }

QPushButton#placeButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #4a148c, stop:1 #6a1b9a);
    color: #e1bee7;
    border-radius: 6px;
    padding: 7px 12px;
    font-size: 13px;
    font-weight: bold;
}
QPushButton#placeButton:hover    { background: #7b1fa2; }
QPushButton#placeButton:disabled { background: #1a0d28; color: #3a2050; }

QPushButton#stopButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #7f0000, stop:1 #b71c1c);
    color: #ffcdd2;
    border-radius: 6px;
    padding: 9px 12px;
    font-size: 14px;
    font-weight: bold;
    margin-top: 6px;
}
QPushButton#stopButton:hover  { background: #c62828; }
QPushButton#stopButton:pressed { background: #5a0000; }

QLabel#statusLabel[gs="DISCONNECTED"] { color: #9e9e9e; background: rgba(100,100,100,0.08); }
QLabel#statusLabel[gs="STANDBY"]      { color: #4fc3f7; background: rgba(79,195,247,0.08); }
QLabel#statusLabel[gs="MOVING"]       { color: #fff176; background: rgba(255,241,118,0.08); }
QLabel#statusLabel[gs="PREGRASP"]     { color: #81d4fa; background: rgba(129,212,250,0.08); }
QLabel#statusLabel[gs="DETECTING"]    { color: #fff176; background: rgba(255,241,118,0.08); }
QLabel#statusLabel[gs="TRACKING"]     { color: #69f0ae; background: rgba(105,240,174,0.08); }
QLabel#statusLabel[gs="APPROACHING"]  { color: #ffb74d; background: rgba(255,183,77,0.08); }
QLabel#statusLabel[gs="CLOSING"]      { color: #ff8a65; background: rgba(255,138,101,0.10); }
QLabel#statusLabel[gs="GRASPED"]      { color: #69f0ae; background: rgba(105,240,174,0.10); }
QLabel#statusLabel[gs="PLACING"]      { color: #ce93d8; background: rgba(206,147,216,0.10); }
QLabel#statusLabel[gs="ERROR"]        { color: #ef5350; background: rgba(239,83,80,0.10); }
"""
