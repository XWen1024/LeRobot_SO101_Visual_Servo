"""
robot_manager.py - Thin wrapper around SO101Follower for connection,
joint reading, and joint writing.  All public methods are safe to call
from the Qt main thread (they are synchronous but fast).
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Joint names in canonical order for SO-101
JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


class RobotManager:
    """Manages the SO-101 robot connection and provides a simple joint API."""

    def __init__(self):
        self._robot = None

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self, port: str, robot_id: str = "so101_slave") -> bool:
        """Connect to the robot.  Returns True on success."""
        try:
            from lerobot.robots.so101_follower.so101_follower import SO101Follower
            from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig

            config = SO101FollowerConfig(port=port, id=robot_id)
            robot = SO101Follower(config)
            robot.connect(calibrate=True)
            self._robot = robot
            logger.info("Connected to SO-101 on %s", port)
            return True
        except Exception as exc:
            logger.error("Failed to connect: %s", exc)
            self._robot = None
            return False

    def disconnect(self):
        if self._robot is not None and self._robot.is_connected:
            try:
                self._robot.bus.disable_torque()
                self._robot.disconnect()
            except Exception as exc:
                logger.warning("Error during disconnect: %s", exc)
            finally:
                self._robot = None

    @property
    def is_connected(self) -> bool:
        return self._robot is not None and self._robot.is_connected

    # ── Torque ────────────────────────────────────────────────────────────────

    # Arm joints only (excludes gripper which may be holding an object)
    _ARM_MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

    def enable_torque(self) -> bool:
        """Enable torque.  If a motor fails (e.g. gripper overload) fall back to
        enabling arm joints individually so movement is never blocked by the gripper."""
        if not self._robot:
            return False
        try:
            self._robot.bus.enable_torque()
            return True
        except Exception as exc:
            logger.warning("enable_torque (all) failed: %s — trying arm motors only", exc)

        # Fallback: enable each arm motor individually, ignore gripper
        ok = True
        for motor in self._ARM_MOTORS:
            try:
                self._robot.bus.enable_torque(motors=motor)
            except Exception as e:
                logger.error("  enable_torque motor=%s: %s", motor, e)
                ok = False
        return ok

    def disable_torque(self):
        if self._robot:
            try:
                self._robot.bus.disable_torque()
            except Exception as exc:
                logger.warning("disable_torque error: %s", exc)

    # ── Joint read / write ────────────────────────────────────────────────────

    def get_joints(self) -> Optional[dict]:
        """Return dict of {joint_name: value} for all 6 joints, or None."""
        if not self.is_connected:
            return None
        try:
            obs = self._robot.get_observation()
            return {k: v for k, v in obs.items() if k.endswith(".pos")}
        except Exception as exc:
            logger.error("get_joints failed: %s", exc)
            return None

    def send_joints(self, joints: dict) -> bool:
        """Send joint position command.  Returns True on success."""
        if not self.is_connected:
            logger.debug("send_joints skipped: not connected")
            return False
        try:
            logger.debug("send_joints → %s",
                         " | ".join(f"{k.split('.')[0]}={v:.1f}" for k, v in sorted(joints.items())))
            self._robot.send_action(joints)
            return True
        except Exception as exc:
            logger.error("send_joints failed: %s", exc)
            return False

    def set_gripper(self, value: float) -> bool:
        """Set gripper position directly (0=open, 100=closed)."""
        joints = self.get_joints()
        if joints is None:
            return self.send_joints({"gripper.pos": float(value)})
        joints["gripper.pos"] = float(value)
        return self.send_joints(joints)
