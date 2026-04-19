"""
main.py - Entry point for the Visual Grounding Tracker application.
"""
import sys
import logging
import os

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# Ensure we can find src/
sys.path.insert(0, os.path.dirname(__file__))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from src.gui.main_window import MainWindow
from src.config import load_config


def main():
    # Load config before anything else
    load_config()

    app = QApplication(sys.argv)
    app.setApplicationName("Visual Grounding Tracker")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
