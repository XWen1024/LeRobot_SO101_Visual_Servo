"""
servo_controller.py - Visual servo grasping state machine.

States
------
DISCONNECTED  Robot not connected
STANDBY       Arm at standby pose, waiting
MOVING        Arm moving to a target pose (transient)
PREGRASP      Arm at pre-grasp pose, ready to detect
DETECTING     VLM detection running (first detection)
TRACKING      Visual servo active — centering object in frame
APPROACHING   Gripper open, arm advancing toward object
CLOSING       Gripper closing
GRASPED       Object in gripper
PLACING       Arm moving to placement pose
ERROR         Unrecoverable error

The controller is driven by `update(frame)` called each camera frame from
the Qt timer.  State transitions are triggered inside `update()` or by
explicit command methods called from the GUI.
"""
import logging
import time
import math
from enum import Enum, auto
from typing import Optional, Callable

import numpy as np

logger = logging.getLogger(__name__)


class GraspState(Enum):
    DISCONNECTED = auto()
    STANDBY      = auto()
    MOVING       = auto()
    PREGRASP     = auto()
    DETECTING    = auto()
    TRACKING     = auto()
    APPROACHING  = auto()
    CLOSING      = auto()
    GRASPED      = auto()
    PLACING      = auto()
    ERROR        = auto()


# Chinese labels shown in the state badge
STATE_LABELS = {
    GraspState.DISCONNECTED: "未连接",
    GraspState.STANDBY:      "待机",
    GraspState.MOVING:       "移动中",
    GraspState.PREGRASP:     "待抓取位",
    GraspState.DETECTING:    "检测中",
    GraspState.TRACKING:     "跟踪中",
    GraspState.APPROACHING:  "逼近中",
    GraspState.CLOSING:      "闭爪",
    GraspState.GRASPED:      "已抓取",
    GraspState.PLACING:      "放置中",
    GraspState.ERROR:        "错误",
}

# RGB colors for each state badge
STATE_COLORS = {
    GraspState.DISCONNECTED: (150, 150, 150),
    GraspState.STANDBY:      (79, 195, 247),
    GraspState.MOVING:       (255, 241, 118),
    GraspState.PREGRASP:     (129, 212, 250),
    GraspState.DETECTING:    (255, 241, 118),
    GraspState.TRACKING:     (105, 240, 174),
    GraspState.APPROACHING:  (255, 183, 77),
    GraspState.CLOSING:      (255, 138, 101),
    GraspState.GRASPED:      (105, 240, 174),
    GraspState.PLACING:      (206, 147, 216),
    GraspState.ERROR:        (239, 83, 80),
}


class FrameInfo:
    """Per-frame data returned by update() for GUI rendering."""
    __slots__ = ("bbox", "error_x", "error_y", "area_ratio", "state", "message")

    def __init__(self):
        self.bbox = None                  # (x, y, w, h) or None
        self.error_x: float = 0.0        # normalized horizontal error
        self.error_y: float = 0.0        # normalized vertical error
        self.area_ratio: float = 0.0     # bbox_area / frame_area
        self.state: GraspState = GraspState.DISCONNECTED
        self.message: str = ""


class ServoController:
    """
    Drives the visual servo grasping state machine.

    Parameters
    ----------
    robot   : RobotManager
    tracker : ObjectTracker  (from src/tracker.py)
    cfg     : dict — the full config dict (get_config() output)
    on_state_change : optional callback(state, message)
    """

    def __init__(self, robot, tracker, cfg: dict,
                 on_state_change: Optional[Callable] = None):
        self._robot = robot
        self._tracker = tracker
        self._cfg = cfg
        self._on_state_change = on_state_change

        scfg = cfg.get("servo", {})
        self._pan_gain      = float(scfg.get("pan_gain", 25.0))
        self._tilt_gain     = float(scfg.get("tilt_gain", 15.0))
        self._approach_step = float(scfg.get("approach_step", 0.3))
        self._center_thresh = float(scfg.get("centering_threshold", 0.04))
        self._grasp_thresh  = float(scfg.get("grasp_area_threshold", 0.18))
        self._gripper_open  = float(scfg.get("gripper_open", 0.0))
        self._gripper_close = float(scfg.get("gripper_close", 70.0))
        self._arrive_thresh   = float(scfg.get("arrive_threshold", 2.0))
        self._move_fps        = int(scfg.get("move_fps", 15))
        self._move_step_limit  = float(scfg.get("move_step_limit", 3.0))   # deg/frame
        self._move_timeout     = float(scfg.get("move_timeout_sec", 15.0)) # seconds
        # Center calibration offset (pixels, set by GUI)
        self._center_offset_x: int = int(scfg.get("center_offset_x", 0))
        self._center_offset_y: int = int(scfg.get("center_offset_y", 0))

        # Recorded positions
        self._pos_standby   = None   # dict of joint positions
        self._pos_pregrasp  = None
        self._pos_placement = None

        # Runtime state
        self._state         = GraspState.DISCONNECTED
        self._description   = ""
        self._current_joints: Optional[dict] = None

        # move_to state
        self._move_target      = None
        self._move_callback    = None
        self._move_start_time  = 0.0
        self._move_interp      = None  # interpolated current command

        # Approach state
        self._approach_pregrasp_joints: Optional[dict] = None  # snapshot before approach
        self._centered_frames = 0   # consecutive frames object is centered
        # Cache last valid (error_x, error_y, area_ratio) so approach continues
        # even when the tracker momentarily has no fresh bbox
        self._last_approach_info: Optional[tuple] = None

    # ── Center calibration ────────────────────────────────────────────────────

    def set_center_offset(self, offset_x: int, offset_y: int):
        """Set the pixel offset of the servo center relative to frame center."""
        self._center_offset_x = offset_x
        self._center_offset_y = offset_y

    # ── Positions ─────────────────────────────────────────────────────────────

    def set_positions(self, standby=None, pregrasp=None, placement=None):
        if standby is not None:
            self._pos_standby = standby
        if pregrasp is not None:
            self._pos_pregrasp = pregrasp
        if placement is not None:
            self._pos_placement = placement

    # ── Commands from GUI ─────────────────────────────────────────────────────

    def on_robot_connected(self):
        self._set_state(GraspState.STANDBY, "已连接，待机中")

    def on_robot_disconnected(self):
        self._set_state(GraspState.DISCONNECTED, "未连接")

    def go_standby(self):
        """Move arm to standby position."""
        if not self._robot.is_connected:
            return
        if self._pos_standby is None:
            self._set_state(GraspState.STANDBY, "待机位未录制")
            return
        self._tracker.stop()
        self._move_to(self._pos_standby, lambda: self._set_state(GraspState.STANDBY, "待机"))

    def go_pregrasp(self):
        """Move arm to pre-grasp position."""
        if not self._robot.is_connected:
            return
        if self._pos_pregrasp is None:
            self._set_state(GraspState.ERROR, "待抓取位未录制")
            return
        self._tracker.stop()
        self._move_to(self._pos_pregrasp, lambda: self._set_state(GraspState.PREGRASP, "就位，等待识别"))

    def start_tracking(self, description: str, frame: np.ndarray):
        """Start VLM detection and tracking from the current frame."""
        if not self._robot.is_connected:
            return
        if self._state not in (GraspState.PREGRASP, GraspState.TRACKING,
                                GraspState.STANDBY, GraspState.DETECTING):
            return
        self._description = description
        self._tracker.stop()
        self._tracker.start_detection(frame, description)
        self._set_state(GraspState.DETECTING, "识别中…")

    def start_approach(self):
        """Begin grasping: open gripper, start approach loop."""
        if self._state != GraspState.TRACKING:
            logger.warning("start_approach ignored: state=%s (need TRACKING)", self._state.name)
            return
        joints = self._robot.get_joints()
        if joints is None:
            logger.warning("start_approach: get_joints() returned None")
            return
        self._approach_pregrasp_joints = dict(joints)
        self._last_approach_info = None
        ok = self._robot.set_gripper(self._gripper_open)
        logger.info("Gripper open → %.1f  (send_ok=%s)", self._gripper_open, ok)
        self._centered_frames = 0
        self._set_state(GraspState.APPROACHING, "逼近中，夹爪已张开")

    def place(self):
        """Move to placement position and open gripper."""
        if not self._robot.is_connected:
            return
        if self._pos_placement is None:
            self._set_state(GraspState.ERROR, "放置位未录制")
            return
        self._tracker.stop()
        self._set_state(GraspState.PLACING, "前往放置位…")
        self._move_to(self._pos_placement, self._on_placed)

    def stop(self):
        """Emergency stop — disable torque."""
        self._tracker.stop()
        self._robot.disable_torque()
        self._move_target = None
        self._set_state(GraspState.STANDBY, "已停止")

    def notify_grasped(self):
        """Call after gripper has closed to transition into GRASPED state."""
        if self._state == GraspState.CLOSING:
            self._set_state(GraspState.GRASPED, "抓取成功！可点击【放置物品】")

    # ── Per-frame update ──────────────────────────────────────────────────────

    def update(self, frame: np.ndarray) -> FrameInfo:
        """
        Call once per camera frame.  Runs the visual servo loop and
        returns a FrameInfo with overlay data for the GUI.
        """
        info = FrameInfo()
        info.state = self._state

        # Nothing to do without a robot
        if self._state == GraspState.DISCONNECTED:
            return info

        # Poll move_to progress
        if self._state == GraspState.MOVING and self._move_target is not None:
            self._tick_move()
            return info

        # States that use the tracker
        if self._state in (GraspState.DETECTING, GraspState.TRACKING,
                           GraspState.APPROACHING):
            # Import compatible TrackerState from whichever tracker is in use
            try:
                from .yolo_tracker import TrackerState
            except ImportError:
                from .tracker import TrackerState

            bbox = self._tracker.update(frame)
            tracker_state = self._tracker.state
            h, w = frame.shape[:2]

            if bbox is not None:
                info.bbox = bbox
                bx, by, bw, bh = bbox
                cx = bx + bw / 2
                cy = by + bh / 2
                info.error_x    = (cx - (w / 2 + self._center_offset_x)) / w
                info.error_y    = (cy - (h / 2 + self._center_offset_y)) / h
                info.area_ratio = (bw * bh) / (w * h)

                # DETECTING → TRACKING as soon as we get a first bbox
                if self._state == GraspState.DETECTING:
                    self._set_state(GraspState.TRACKING, "跟踪中")

                if self._state == GraspState.TRACKING:
                    self._do_centering(info.error_x, info.error_y)

                elif self._state == GraspState.APPROACHING:
                    # Cache latest good reading and approach
                    self._last_approach_info = (info.error_x, info.error_y, info.area_ratio)
                    self._do_approach(info.error_x, info.error_y, info.area_ratio)

            else:
                # No bbox this frame
                if tracker_state == TrackerState.ERROR:
                    if self._state == GraspState.APPROACHING and self._approach_pregrasp_joints:
                        self._move_to(self._approach_pregrasp_joints,
                                      lambda: self._set_state(GraspState.PREGRASP, "跟踪失败，已返回待抓取位"))
                    else:
                        self._set_state(GraspState.PREGRASP, "跟踪失败，请重新识别")

                elif self._state == GraspState.APPROACHING:
                    # Tracker momentarily has no bbox — continue with last known reading
                    if self._last_approach_info is not None:
                        ex, ey, ar = self._last_approach_info
                        info.area_ratio = ar
                        self._do_approach(ex, ey, ar)

                elif self._state == GraspState.TRACKING:
                    self._set_state(GraspState.DETECTING, "重新检测中…")

        info.state = self._state
        return info

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _do_centering(self, error_x: float, error_y: float):
        """Apply proportional pan/tilt correction to center the object.
        All arm joints are sent each frame to prevent gravity drift."""
        joints = self._robot.get_joints()
        if joints is None:
            logger.debug("_do_centering: get_joints() returned None, skipping")
            return

        cur_pan  = joints.get("shoulder_pan.pos", 0.0)
        cur_lift = joints.get("shoulder_lift.pos", 0.0)
        
        # Limit the maximum step per frame to prevent overshoot and PID windup during YOLO latency
        delta_pan  = max(-self._move_step_limit, min(self._move_step_limit, self._pan_gain  * error_x))
        delta_lift = max(-self._move_step_limit, min(self._move_step_limit, self._tilt_gain * error_y))

        new_pan  = cur_pan  + delta_pan
        new_lift = cur_lift + delta_lift

        new_pan  = max(-100.0, min(100.0, new_pan))
        new_lift = max(-100.0, min(100.0, new_lift))

        logger.debug("centering: err=(%.3f,%.3f) pan %.1f→%.1f  lift %.1f→%.1f",
                     error_x, error_y, cur_pan, new_pan, cur_lift, new_lift)

        self._robot.send_joints({
            "shoulder_pan.pos":  new_pan,
            "shoulder_lift.pos": new_lift,
            "elbow_flex.pos":    joints.get("elbow_flex.pos", 0.0),
            "wrist_flex.pos":    joints.get("wrist_flex.pos", 0.0),
            "wrist_roll.pos":    joints.get("wrist_roll.pos", 0.0),
        })

    def _do_approach(self, error_x: float, error_y: float, area_ratio: float):
        """
        While approaching: center the object AND advance the arm forward.
        Closes the gripper when the object occupies enough of the frame.
        """
        logger.debug("_do_approach called: err=(%.3f,%.3f) area=%.3f thresh=%.3f",
                     error_x, error_y, area_ratio, self._grasp_thresh)

        if area_ratio >= self._grasp_thresh:
            ok = self._robot.set_gripper(self._gripper_close)
            logger.info("Gripper close → %.1f  area_ratio=%.3f  (send_ok=%s)",
                        self._gripper_close, area_ratio, ok)
            self._set_state(GraspState.CLOSING, "闭爪中…")
            return

        joints = self._robot.get_joints()
        if joints is None:
            logger.debug("_do_approach: get_joints() returned None, skipping")
            return

        cur_pan   = joints.get("shoulder_pan.pos", 0.0)
        cur_lift  = joints.get("shoulder_lift.pos", 0.0)
        cur_elbow = joints.get("elbow_flex.pos", 0.0)

        # Pan: keep object horizontally centered, with step limit to prevent windup
        delta_pan = max(-self._move_step_limit, min(self._move_step_limit, self._pan_gain * error_x))
        new_pan   = cur_pan + delta_pan
        new_pan   = max(-100.0, min(100.0, new_pan))

        # Approach: Drive forward by increasing shoulder_lift at constant step.
        new_lift  = max(-100.0, min(100.0, cur_lift  + self._approach_step))
        
        # Calculate elbow_flex using the quadratic polynomial derived from user's manual teaching.
        # This polynomial accurately traces a horizontal approach trajectory without dipping down.
        # Polynomial: Elbow = 0.001089 * lift^2 - 1.023 * lift - 5.55
        poly_elbow = 0.001089 * (new_lift ** 2) - 1.023 * new_lift - 5.55
        new_elbow = max(-100.0, min(100.0, poly_elbow))

        logger.debug("approach: pan %.1f→%.1f  lift %.1f→%.1f  elbow %.1f→%.1f",
                     cur_pan, new_pan, cur_lift, new_lift, cur_elbow, new_elbow)

        self._robot.send_joints({
            "shoulder_pan.pos":  new_pan,
            "shoulder_lift.pos": new_lift,
            "elbow_flex.pos":    new_elbow,
            "wrist_flex.pos":    joints.get("wrist_flex.pos", 0.0),
            "wrist_roll.pos":    joints.get("wrist_roll.pos", 0.0),
        })

    def _move_to(self, target: dict, callback: Optional[Callable] = None):
        """Start a non-blocking move to a joint target."""
        ok = self._robot.enable_torque()
        if not ok:
            self._set_state(GraspState.ERROR,
                            "力矩启用失败（舵机过热或通信错误）——请等待冷却后重试")
            return
        self._move_target     = dict(target)
        self._move_callback   = callback
        self._move_start_time = time.time()
        # Seed the interpolator from current joints so motion is smooth
        joints = self._robot.get_joints()
        self._move_interp = dict(joints) if joints else dict(target)
        prev_state = self._state
        self._set_state(GraspState.MOVING, "移动中…")
        self._prev_state_before_move = prev_state

    def _tick_move(self):
        """Called each frame while in MOVING state.
        Steps the interpolated command toward the target at most
        `move_step_limit` degrees per joint per tick.
        Two-phase: tilt/elbow/wrist joints move first; shoulder_pan moves only
        after all other joints have reached their targets (avoids sweeping cables).
        Completes as soon as the *commanded* position reaches the target
        (no dependency on robot joint feedback), with a hard timeout fallback.
        """
        # Timeout guard
        elapsed = time.time() - self._move_start_time
        if elapsed > self._move_timeout:
            logger.warning("move_to timed out after %.1fs, forcing completion", elapsed)
            self._move_target  = None
            self._move_interp  = None
            cb, self._move_callback = self._move_callback, None
            if cb:
                cb()
            return

        step = self._move_step_limit

        # Phase 1: check whether all non-pan joints have reached target
        non_pan_done = True
        for k, v_target in self._move_target.items():
            if k == "shoulder_pan.pos":
                continue
            v_cur = self._move_interp.get(k, v_target)
            if abs(v_target - v_cur) > step:
                non_pan_done = False
                break

        # Advance each joint; hold shoulder_pan until phase 1 is complete
        cmd = {}
        all_reached = True
        for k, v_target in self._move_target.items():
            v_cur = self._move_interp.get(k, v_target)
            if k == "shoulder_pan.pos" and not non_pan_done:
                # Hold pan in place during phase 1
                cmd[k] = v_cur
                all_reached = False
                continue
            diff = v_target - v_cur
            if abs(diff) <= step:
                v_cur = v_target
            else:
                v_cur += math.copysign(step, diff)
                all_reached = False
            cmd[k] = v_cur
        self._move_interp = cmd
        self._robot.send_joints(cmd)

        # Complete as soon as the commanded position has fully reached the target
        if all_reached:
            self._move_target  = None
            self._move_interp  = None
            cb, self._move_callback = self._move_callback, None
            if cb:
                cb()

    def _on_placed(self):
        self._robot.set_gripper(self._gripper_open)
        self._set_state(GraspState.STANDBY, "放置完成，待机")

    def _set_state(self, state: GraspState, message: str = ""):
        self._state = state
        label = STATE_LABELS.get(state, state.name)
        msg = message or label
        logger.debug("GraspState → %s: %s", state.name, msg)
        if self._on_state_change:
            self._on_state_change(state, msg)
