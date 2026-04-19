"""
servo_grasp_main.py - Entry point for the visual servo grasping application.

Run:
    python servo_grasp_main.py
"""
import sys
import signal
import logging
from pathlib import Path

# Ensure the project root is on sys.path so `src` imports work
sys.path.insert(0, str(Path(__file__).parent))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QTimer

from src.gui.servo_window import ServoGraspWindow

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# Suppress noisy third-party debug logs
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("draccus").setLevel(logging.WARNING)
logging.getLogger("lerobot").setLevel(logging.WARNING)


def main():
    # Allow Ctrl+C to terminate the Qt event loop cleanly
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # Qt blocks Python signal delivery; this timer wakes Python every 200ms
    _sig_timer = QTimer()
    _sig_timer.start(200)
    _sig_timer.timeout.connect(lambda: None)

    win = ServoGraspWindow()
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
