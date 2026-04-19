"""
styles.py - Shared stylesheet constants for the dark high-tech theme.
"""

DARK_THEME = """
/* ── Global ── */
* {
    font-family: "Segoe UI", Arial, sans-serif;
    color: #e0e8f8;
    outline: none;
}

QMainWindow, QDialog, QWidget#centralWidget {
    background-color: #0d1117;
}

/* ── Labels ── */
QLabel {
    color: #c0d0e8;
    background: transparent;
}

QLabel#titleLabel {
    font-size: 20px;
    font-weight: bold;
    color: #4fc3f7;
    letter-spacing: 2px;
}

QLabel#statusLabel {
    font-size: 12px;
    padding: 4px 10px;
    border-radius: 6px;
    background: rgba(79, 195, 247, 0.08);
    color: #90caf9;
}

QLabel#statusLabel[state="TRACKING"] {
    color: #69f0ae;
    background: rgba(105, 240, 174, 0.10);
}

QLabel#statusLabel[state="LOST"] {
    color: #ff8a65;
    background: rgba(255, 138, 101, 0.10);
}

QLabel#statusLabel[state="ERROR"] {
    color: #ef5350;
    background: rgba(239, 83, 80, 0.12);
}

QLabel#statusLabel[state="DETECTING"],
QLabel#statusLabel[state="REDETECTING"] {
    color: #fff176;
    background: rgba(255, 241, 118, 0.10);
}

/* ── Camera view ── */
QLabel#cameraView {
    background-color: #080c12;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
}

/* ── Text input ── */
QLineEdit {
    background: #131c2b;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 14px;
    color: #e0e8f8;
    selection-background-color: #1565c0;
}

QLineEdit:focus {
    border-color: #4fc3f7;
    background: #162030;
}

QLineEdit::placeholder {
    color: #4a6080;
}

/* ── Buttons ── */
QPushButton {
    border-radius: 8px;
    padding: 8px 18px;
    font-size: 13px;
    font-weight: bold;
    border: none;
}

QPushButton#sendButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #0d47a1, stop:1 #1565c0);
    color: #e3f2fd;
}
QPushButton#sendButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #1565c0, stop:1 #1976d2);
}
QPushButton#sendButton:pressed  { background: #0a3068; }
QPushButton#sendButton:disabled { background: #1a2a3a; color: #3a5070; }

QPushButton#trackButton[active="false"] {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #1b5e20, stop:1 #2e7d32);
    color: #c8e6c9;
}
QPushButton#trackButton[active="false"]:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #2e7d32, stop:1 #388e3c);
}
QPushButton#trackButton[active="true"] {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #7f0000, stop:1 #b71c1c);
    color: #ffcdd2;
}
QPushButton#trackButton[active="true"]:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #b71c1c, stop:1 #c62828);
}

QPushButton#camSwitchButton {
    background: #182030;
    color: #78909c;
    border: 1px solid #263545;
    padding: 6px 12px;
    font-size: 12px;
}
QPushButton#camSwitchButton:hover {
    background: #1e2d40;
    color: #90a4ae;
    border-color: #37474f;
}

/* ── Overlay badge pill ── */
QLabel#stateBadge {
    background: rgba(13, 17, 23, 0.82);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 3px 10px;
    font-size: 11px;
    color: #90caf9;
}

/* ── Camera selector dialog ── */
QDialog {
    background: #0d1117;
}

QLabel#selectorTitle {
    font-size: 16px;
    font-weight: bold;
    color: #4fc3f7;
}

QPushButton#camPreviewButton {
    background: #131c2b;
    border: 2px solid #1e3a5f;
    border-radius: 6px;
    padding: 4px;
}
QPushButton#camPreviewButton:hover  { border-color: #4fc3f7; }
QPushButton#camPreviewButton:checked { border-color: #69f0ae; background: #0a1a0e; }

QPushButton#confirmButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #0d47a1, stop:1 #1565c0);
    color: #e3f2fd;
    padding: 8px 28px;
    border-radius: 8px;
    font-weight: bold;
}
QPushButton#confirmButton:hover { background: #1976d2; }

/* ── Scrollbar ── */
QScrollBar:vertical {
    background: #0d1117;
    width: 6px;
    border-radius: 3px;
}
QScrollBar::handle:vertical {
    background: #1e3a5f;
    border-radius: 3px;
}
"""
