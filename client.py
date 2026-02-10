#!/usr/bin/env python3
"""
Voice Chat Client
=================
PyQt5 desktop application with push-to-talk voice chat.

Run:
    python client.py
"""

import sys
import json
import threading
import queue

import ctypes
import ctypes.wintypes

import numpy as np
import sounddevice as sd
import websocket
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFrame, QScrollArea, QMessageBox,
    QSizePolicy,
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QKeySequence

# Windows API for global key-state polling (no hooks / no admin needed)
_user32 = ctypes.windll.user32
_GetAsyncKeyState = _user32.GetAsyncKeyState
_GetAsyncKeyState.argtypes = [ctypes.c_int]
_GetAsyncKeyState.restype  = ctypes.c_short

# Readable names for common virtual-key codes
_VK_NAMES: dict[int, str] = {
    0x01: "MOUSE LEFT", 0x02: "MOUSE RIGHT", 0x04: "MOUSE MIDDLE",
    0x05: "MOUSE 4", 0x06: "MOUSE 5",
    0x08: "BACKSPACE", 0x09: "TAB", 0x0D: "ENTER", 0x10: "SHIFT",
    0x11: "CTRL", 0x12: "ALT", 0x14: "CAPS LOCK", 0x1B: "ESC",
    0x20: "SPACE", 0x21: "PAGE UP", 0x22: "PAGE DOWN", 0x23: "END",
    0x24: "HOME", 0x25: "LEFT", 0x26: "UP", 0x27: "RIGHT", 0x28: "DOWN",
    0x2C: "PRINT SCREEN", 0x2D: "INSERT", 0x2E: "DELETE",
    0x5B: "LEFT WIN", 0x5C: "RIGHT WIN",
    0x60: "NUM 0", 0x61: "NUM 1", 0x62: "NUM 2", 0x63: "NUM 3",
    0x64: "NUM 4", 0x65: "NUM 5", 0x66: "NUM 6", 0x67: "NUM 7",
    0x68: "NUM 8", 0x69: "NUM 9",
    0x6A: "NUM *", 0x6B: "NUM +", 0x6D: "NUM -", 0x6E: "NUM .", 0x6F: "NUM /",
    0x70: "F1", 0x71: "F2", 0x72: "F3", 0x73: "F4", 0x74: "F5",
    0x75: "F6", 0x76: "F7", 0x77: "F8", 0x78: "F9", 0x79: "F10",
    0x7A: "F11", 0x7B: "F12",
    0xA0: "L-SHIFT", 0xA1: "R-SHIFT", 0xA2: "L-CTRL", 0xA3: "R-CTRL",
    0xA4: "L-ALT", 0xA5: "R-ALT",
    0xBB: "=", 0xBC: ",", 0xBD: "-", 0xBE: ".", 0xBF: "/",
    0xC0: "`", 0xDB: "[", 0xDC: "\\", 0xDD: "]", 0xDE: "'",
}

def _vk_name(vk: int) -> str:
    """Return a human-readable name for a Windows virtual-key code."""
    if vk in _VK_NAMES:
        return _VK_NAMES[vk]
    if 0x30 <= vk <= 0x39:          # 0-9
        return chr(vk)
    if 0x41 <= vk <= 0x5A:          # A-Z
        return chr(vk)
    return f"KEY 0x{vk:02X}"

# ═══════════════════════════════════════════════════════════════════════════
# Audio configuration
# ═══════════════════════════════════════════════════════════════════════════
SAMPLE_RATE  = 16000
AUDIO_CHANNELS = 1
BLOCK_SIZE   = 1024        # ~64 ms at 16 kHz
DTYPE        = "int16"

# ═══════════════════════════════════════════════════════════════════════════
# Dark theme (Catppuccin Mocha–inspired)
# ═══════════════════════════════════════════════════════════════════════════
DARK_THEME = """
/* ── base ───────────────────────────────────────────────── */
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: "Segoe UI", "Arial", sans-serif;
    font-size: 13px;
}

/* ── inputs ─────────────────────────────────────────────── */
QLineEdit {
    background-color: #313244;
    border: 2px solid #45475a;
    border-radius: 8px;
    padding: 10px 14px;
    color: #cdd6f4;
    font-size: 14px;
    selection-background-color: #585b70;
}
QLineEdit:focus { border-color: #89b4fa; }

/* ── buttons ────────────────────────────────────────────── */
QPushButton {
    background-color: #89b4fa;
    color: #1e1e2e;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-size: 14px;
    font-weight: bold;
    min-height: 20px;
}
QPushButton:hover   { background-color: #74c7ec; }
QPushButton:pressed { background-color: #94e2d5; }
QPushButton:disabled {
    background-color: #45475a;
    color: #6c7086;
}

QPushButton#danger        { background-color: #f38ba8; color: #1e1e2e; }
QPushButton#danger:hover  { background-color: #eba0ac; }

QPushButton#secondary        { background-color: #45475a; color: #cdd6f4; }
QPushButton#secondary:hover  { background-color: #585b70; }

/* ── scroll area ────────────────────────────────────────── */
QScrollArea { border: none; background-color: transparent; }
QScrollBar:vertical {
    background-color: #181825; width: 8px; border-radius: 4px;
}
QScrollBar::handle:vertical {
    background-color: #45475a; border-radius: 4px; min-height: 30px;
}
QScrollBar::handle:vertical:hover { background-color: #585b70; }
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical     { height: 0; }

/* ── labels (by objectName) ─────────────────────────────── */
QLabel#title    { font-size: 28px; font-weight: bold; color: #89b4fa; }
QLabel#subtitle { font-size: 13px; color: #6c7086; }
QLabel#section  { font-size: 11px; font-weight: bold; color: #a6adc8; letter-spacing: 1px; }
QLabel#status-muted   { color: #f38ba8; font-size: 14px; font-weight: bold; }
QLabel#status-talking  { color: #a6e3a1; font-size: 14px; font-weight: bold; }
QLabel#error    { color: #f38ba8; font-size: 12px; }

/* ── channel frames ─────────────────────────────────────── */
QFrame#channel {
    background-color: #181825;
    border-radius: 8px;
    border: 2px solid transparent;
}
QFrame#channel-active {
    background-color: #313244;
    border-radius: 8px;
    border: 2px solid #89b4fa;
}

/* ── misc ───────────────────────────────────────────────── */
QFrame#separator { background-color: #313244; max-height: 1px; }
"""


# ═══════════════════════════════════════════════════════════════════════════
# AudioManager — capture + playback via sounddevice
# ═══════════════════════════════════════════════════════════════════════════
class AudioManager:
    """Thread-safe microphone capture and speaker playback."""

    def __init__(self):
        self.is_capturing = False
        self.on_audio_data = None          # callback(bytes), called from _send_loop
        self._capture_q  = queue.Queue(maxsize=50)
        self._playback_q = queue.Queue(maxsize=50)
        self._running     = False
        self._in_stream   = None
        self._out_stream  = None
        self._has_input   = True

    # -- lifecycle -----------------------------------------------------------

    def start(self):
        self._running = True

        # Output (speakers) — should almost always succeed
        self._out_stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=AUDIO_CHANNELS,
            dtype=DTYPE,
            blocksize=BLOCK_SIZE,
            callback=self._out_cb,
        )
        self._out_stream.start()

        # Input (microphone) — may fail if no mic is present
        try:
            self._in_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=AUDIO_CHANNELS,
                dtype=DTYPE,
                blocksize=BLOCK_SIZE,
                callback=self._in_cb,
            )
            self._in_stream.start()
            self._has_input = True
        except Exception:
            self._has_input = False

        if self._has_input:
            threading.Thread(target=self._send_loop, daemon=True).start()

    def stop(self):
        self._running = False
        for s in (self._in_stream, self._out_stream):
            if s is not None:
                try:
                    s.stop(); s.close()
                except Exception:
                    pass
        self._in_stream = self._out_stream = None

    # -- sounddevice callbacks (PortAudio thread) ----------------------------

    def _in_cb(self, indata, frames, time_info, status):
        """Always drain the mic buffer; only enqueue when PTT active."""
        if self.is_capturing:
            try:
                self._capture_q.put_nowait(indata.tobytes())
            except queue.Full:
                pass

    def _out_cb(self, outdata, frames, time_info, status):
        try:
            data = self._playback_q.get_nowait()
            arr = np.frombuffer(data, dtype=np.int16)
            needed = frames * AUDIO_CHANNELS
            if arr.size >= needed:
                outdata[:, 0] = arr[:needed]
            else:
                outdata[:arr.size, 0] = arr
                outdata[arr.size:] = 0
        except queue.Empty:
            outdata.fill(0)

    # -- send loop (own thread → network) ------------------------------------

    def _send_loop(self):
        while self._running:
            try:
                data = self._capture_q.get(timeout=0.05)
                cb = self.on_audio_data
                if cb:
                    cb(data)
            except queue.Empty:
                pass

    # -- called from network receive thread ----------------------------------

    def enqueue_playback(self, raw: bytes):
        try:
            self._playback_q.put_nowait(raw)
        except queue.Full:
            try:
                self._playback_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._playback_q.put_nowait(raw)
            except queue.Full:
                pass


# ═══════════════════════════════════════════════════════════════════════════
# NetworkManager — WebSocket client with Qt signals
# ═══════════════════════════════════════════════════════════════════════════
class NetworkManager(QWidget):
    """Thin Qt wrapper around a threaded websocket-client connection."""

    sig_welcome      = pyqtSignal(list)    # channel names
    sig_connected    = pyqtSignal()
    sig_disconnected = pyqtSignal()
    sig_error        = pyqtSignal(str)
    sig_user_list    = pyqtSignal(dict)    # {str(cid): [names]}
    sig_audio        = pyqtSignal(bytes)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.hide()
        self.ws  = None
        self._thread  = None
        self._name    = ""
        self._address = ""

    # -- public API ----------------------------------------------------------

    def connect_to(self, address: str, name: str):
        self._address = address
        self._name    = name
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def send_json(self, obj):
        try:
            w = self.ws
            if w and w.sock and w.sock.connected:
                w.send(json.dumps(obj))
        except Exception:
            pass

    def send_audio(self, raw: bytes):
        try:
            w = self.ws
            if w and w.sock and w.sock.connected:
                w.send(raw, opcode=websocket.ABNF.OPCODE_BINARY)
        except Exception:
            pass

    def disconnect(self):
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass

    # -- background thread ---------------------------------------------------

    def _run(self):
        url = f"ws://{self._address}"
        self.ws = websocket.WebSocketApp(
            url,
            on_open=self._on_open,
            on_data=self._on_data,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self.ws.run_forever(ping_interval=60, ping_timeout=20)

    def _on_open(self, ws):
        self.send_json({"type": "set_name", "name": self._name})
        self.sig_connected.emit()

    def _on_data(self, ws, data, data_type, cont):
        if data_type == websocket.ABNF.OPCODE_TEXT:
            text = data if isinstance(data, str) else data.decode("utf-8", errors="replace")
            msg = json.loads(text)
            t = msg.get("type")
            if t == "welcome":
                self.sig_welcome.emit(msg["channels"])
            elif t == "user_list":
                self.sig_user_list.emit(msg["channels"])
        elif data_type == websocket.ABNF.OPCODE_BINARY:
            self.sig_audio.emit(data)

    def _on_error(self, ws, error):
        self.sig_error.emit(str(error))

    def _on_close(self, ws, code, msg):
        self.sig_disconnected.emit()


# ═══════════════════════════════════════════════════════════════════════════
# ChannelWidget
# ═══════════════════════════════════════════════════════════════════════════
class ChannelWidget(QFrame):
    """A clickable frame representing one voice channel."""

    clicked = pyqtSignal(int)

    def __init__(self, channel_id: int, name: str, parent=None):
        super().__init__(parent)
        self.channel_id = channel_id
        self._name      = name
        self._active    = False
        self.setObjectName("channel")
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(48)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 10, 14, 10)
        root.setSpacing(4)

        # header
        hdr = QHBoxLayout()
        hdr.setSpacing(8)

        icon = QLabel("\U0001f50a")          # speaker icon
        icon.setFixedWidth(22)
        hdr.addWidget(icon)

        self._name_lbl = QLabel(name)
        self._name_lbl.setStyleSheet("font-weight: bold; font-size: 14px;")
        hdr.addWidget(self._name_lbl)

        hdr.addStretch()

        self._count_lbl = QLabel("0")
        self._count_lbl.setStyleSheet(
            "color: #6c7086; font-size: 12px; background: #45475a;"
            "border-radius: 10px; padding: 2px 8px;"
        )
        hdr.addWidget(self._count_lbl)
        root.addLayout(hdr)

        # users list
        self._users_layout = QVBoxLayout()
        self._users_layout.setContentsMargins(30, 0, 0, 0)
        self._users_layout.setSpacing(2)
        root.addLayout(self._users_layout)

    # -- public --------------------------------------------------------------

    def set_users(self, names: list[str]):
        while self._users_layout.count():
            w = self._users_layout.takeAt(0).widget()
            if w:
                w.deleteLater()
        self._count_lbl.setText(str(len(names)))
        for n in names:
            lbl = QLabel(f"  \u2022  {n}")
            lbl.setStyleSheet("color: #a6adc8; font-size: 12px;")
            self._users_layout.addWidget(lbl)

    def set_active(self, active: bool):
        self._active = active
        self.setObjectName("channel-active" if active else "channel")
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    # -- events --------------------------------------------------------------

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.clicked.emit(self.channel_id)
        super().mousePressEvent(ev)

    def enterEvent(self, ev):
        if not self._active:
            self.setStyleSheet("background-color: #252536; border-radius: 8px;")
        super().enterEvent(ev)

    def leaveEvent(self, ev):
        if not self._active:
            self.setStyleSheet("")
        super().leaveEvent(ev)


# ═══════════════════════════════════════════════════════════════════════════
# LoginPage
# ═══════════════════════════════════════════════════════════════════════════
class LoginPage(QWidget):
    sig_connect = pyqtSignal(str, str)   # (name, address)

    def __init__(self, parent=None):
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setAlignment(Qt.AlignCenter)

        box = QVBoxLayout()
        box.setSpacing(12)

        title = QLabel("Voice Chat")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        box.addWidget(title)

        sub = QLabel("Enter your name and server address to connect")
        sub.setObjectName("subtitle")
        sub.setAlignment(Qt.AlignCenter)
        box.addWidget(sub)
        box.addSpacing(20)

        self._name = QLineEdit()
        self._name.setPlaceholderText("Your display name")
        self._name.setMaxLength(24)
        self._name.setFixedWidth(360)
        box.addWidget(self._name, alignment=Qt.AlignCenter)

        self._addr = QLineEdit()
        self._addr.setPlaceholderText("Server address  (e.g. 192.168.1.5:9753)")
        self._addr.setText("16.62.171.242:9753")
        self._addr.setFixedWidth(360)
        box.addWidget(self._addr, alignment=Qt.AlignCenter)

        box.addSpacing(8)

        self._btn = QPushButton("Connect")
        self._btn.setFixedWidth(360)
        self._btn.clicked.connect(self._on_click)
        box.addWidget(self._btn, alignment=Qt.AlignCenter)

        self._err = QLabel("")
        self._err.setObjectName("error")
        self._err.setAlignment(Qt.AlignCenter)
        self._err.setWordWrap(True)
        self._err.setFixedWidth(360)
        box.addWidget(self._err, alignment=Qt.AlignCenter)

        outer.addLayout(box)

        self._name.returnPressed.connect(self._on_click)
        self._addr.returnPressed.connect(self._on_click)

    def _on_click(self):
        name = self._name.text().strip()
        addr = self._addr.text().strip()
        if not name:
            self._err.setText("Please enter a display name.")
            return
        if not addr:
            self._err.setText("Please enter a server address.")
            return
        self._err.setText("")
        self._btn.setEnabled(False)
        self._btn.setText("Connecting\u2026")
        self.sig_connect.emit(name, addr)

    def reset(self, error: str = ""):
        self._btn.setEnabled(True)
        self._btn.setText("Connect")
        self._err.setText(error)


# ═══════════════════════════════════════════════════════════════════════════
# ChatPage — channels + PTT controls
# ═══════════════════════════════════════════════════════════════════════════
class ChatPage(QWidget):
    sig_join_channel   = pyqtSignal(int)
    sig_leave_channel  = pyqtSignal()
    sig_disconnect     = pyqtSignal()
    sig_ptt_changed    = pyqtSignal(int)    # emits the VK code
    sig_open_mic       = pyqtSignal(bool)   # True = open mic on

    def __init__(self, username: str, parent=None):
        super().__init__(parent)
        self._username = username
        self._current  = None
        self._capturing_key = False
        self._open_mic = False
        self._widgets: list[ChannelWidget] = []
        self._build()
        self.setFocusPolicy(Qt.StrongFocus)

    # -- build UI ------------------------------------------------------------

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── left panel (channels) ──────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(340)
        left.setStyleSheet("background-color: #181825;")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(12, 16, 12, 16)
        ll.setSpacing(6)

        hdr = QLabel("VOICE CHANNELS")
        hdr.setObjectName("section")
        ll.addWidget(hdr)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        scroll_w = QWidget()
        scroll_w.setStyleSheet("background-color: transparent;")
        self._ch_layout = QVBoxLayout(scroll_w)
        self._ch_layout.setContentsMargins(0, 0, 0, 0)
        self._ch_layout.setSpacing(4)
        self._ch_layout.addStretch()

        scroll.setWidget(scroll_w)
        ll.addWidget(scroll)
        root.addWidget(left)

        # ── right panel (info + PTT) ───────────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(32, 24, 32, 24)
        rl.setSpacing(16)

        user_lbl = QLabel(f"Connected as  {self._username}")
        user_lbl.setStyleSheet("font-size: 16px; font-weight: bold;")
        rl.addWidget(user_lbl)

        rl.addWidget(self._sep())

        self._ch_status = QLabel("Not in a channel \u2014 click a channel to join")
        self._ch_status.setStyleSheet("font-size: 14px; color: #6c7086;")
        self._ch_status.setWordWrap(True)
        rl.addWidget(self._ch_status)

        rl.addWidget(self._sep())

        mic_hdr = QLabel("MICROPHONE MODE")
        mic_hdr.setObjectName("section")
        rl.addWidget(mic_hdr)

        # Open-mic toggle
        self._open_mic_btn = QPushButton("Open Mic: OFF")
        self._open_mic_btn.setObjectName("secondary")
        self._open_mic_btn.setCheckable(True)
        self._open_mic_btn.clicked.connect(self._toggle_open_mic)
        rl.addWidget(self._open_mic_btn)

        rl.addSpacing(4)

        ptt_hdr = QLabel("PUSH TO TALK")
        ptt_hdr.setObjectName("section")
        rl.addWidget(ptt_hdr)
        self._ptt_hdr = ptt_hdr

        ptt_row = QHBoxLayout()
        ptt_row.setSpacing(12)

        self._ptt_label = QLabel("No key set")
        self._ptt_label.setStyleSheet(
            "background-color: #313244; padding: 10px 18px;"
            "border-radius: 6px; font-size: 14px; font-weight: bold;"
        )
        ptt_row.addWidget(self._ptt_label)

        self._ptt_btn = QPushButton("Set PTT Key")
        self._ptt_btn.setObjectName("secondary")
        self._ptt_btn.clicked.connect(self._start_capture)
        ptt_row.addWidget(self._ptt_btn)
        ptt_row.addStretch()
        self._ptt_row_widget = QWidget()
        self._ptt_row_widget.setLayout(ptt_row)
        rl.addWidget(self._ptt_row_widget)

        self._ptt_status = QLabel("\u25cf  Muted")
        self._ptt_status.setObjectName("status-muted")
        rl.addWidget(self._ptt_status)

        # mic-not-found hint (hidden by default)
        self._mic_hint = QLabel("")
        self._mic_hint.setObjectName("error")
        self._mic_hint.setWordWrap(True)
        self._mic_hint.hide()
        rl.addWidget(self._mic_hint)

        rl.addStretch()

        btn_row = QHBoxLayout()
        self._leave_btn = QPushButton("Leave Channel")
        self._leave_btn.setObjectName("secondary")
        self._leave_btn.setEnabled(False)
        self._leave_btn.clicked.connect(lambda: self.sig_leave_channel.emit())
        btn_row.addWidget(self._leave_btn)

        dc_btn = QPushButton("Disconnect")
        dc_btn.setObjectName("danger")
        dc_btn.clicked.connect(lambda: self.sig_disconnect.emit())
        btn_row.addWidget(dc_btn)
        btn_row.addStretch()
        rl.addLayout(btn_row)

        root.addWidget(right, stretch=1)

    @staticmethod
    def _sep() -> QFrame:
        f = QFrame()
        f.setObjectName("separator")
        f.setFrameShape(QFrame.HLine)
        return f

    # -- channel list --------------------------------------------------------

    def add_channel(self, cid: int, name: str):
        w = ChannelWidget(cid, name)
        w.clicked.connect(self._on_ch_click)
        self._widgets.append(w)
        self._ch_layout.insertWidget(self._ch_layout.count() - 1, w)

    def _on_ch_click(self, cid: int):
        if cid != self._current:
            self.sig_join_channel.emit(cid)

    def set_current_channel(self, cid, name=None):
        self._current = cid
        for w in self._widgets:
            w.set_active(w.channel_id == cid)
        if cid is not None and name:
            self._ch_status.setText(f"In channel:  {name}")
            self._ch_status.setStyleSheet("font-size: 14px; color: #a6e3a1;")
            self._leave_btn.setEnabled(True)
        else:
            self._ch_status.setText("Not in a channel \u2014 click a channel to join")
            self._ch_status.setStyleSheet("font-size: 14px; color: #6c7086;")
            self._leave_btn.setEnabled(False)

    def update_users(self, data: dict):
        for w in self._widgets:
            w.set_users(data.get(str(w.channel_id), []))

    # -- open mic toggle -------------------------------------------------------

    def _toggle_open_mic(self, checked: bool):
        self._open_mic = checked
        if checked:
            self._open_mic_btn.setText("Open Mic: ON")
            self._open_mic_btn.setStyleSheet(
                "background-color: #a6e3a1; color: #1e1e2e; font-weight: bold;"
                "border: none; border-radius: 8px; padding: 10px 24px;"
            )
            # Disable PTT controls
            self._ptt_hdr.setStyleSheet("color: #45475a;")
            self._ptt_row_widget.setEnabled(False)
            self._ptt_status.setText("\u25cf  Open Mic")
            self._ptt_status.setObjectName("status-talking")
            self._ptt_status.style().unpolish(self._ptt_status)
            self._ptt_status.style().polish(self._ptt_status)
        else:
            self._open_mic_btn.setText("Open Mic: OFF")
            self._open_mic_btn.setStyleSheet("")
            # Re-enable PTT controls
            self._ptt_hdr.setStyleSheet("")
            self._ptt_row_widget.setEnabled(True)
            self._ptt_status.setText("\u25cf  Muted")
            self._ptt_status.setObjectName("status-muted")
            self._ptt_status.style().unpolish(self._ptt_status)
            self._ptt_status.style().polish(self._ptt_status)
        self.sig_open_mic.emit(checked)

    # -- PTT key capture (uses Qt keyPressEvent, no external library) ---------

    def _start_capture(self):
        self._ptt_btn.setEnabled(False)
        self._ptt_label.setText("Press any key or mouse button\u2026")
        self._capturing_key = True
        self.setFocus()
        self.grabKeyboard()
        self.grabMouse()

    def _finish_capture(self, vk: int):
        """Common handler once a VK code is captured (keyboard or mouse)."""
        self._capturing_key = False
        self.releaseKeyboard()
        self.releaseMouse()
        self._ptt_label.setText(_vk_name(vk))
        self._ptt_btn.setEnabled(True)
        self.sig_ptt_changed.emit(vk)

    def keyPressEvent(self, event):
        if self._capturing_key:
            vk = event.nativeVirtualKey()
            if vk == 0:
                vk = event.key()
            self._finish_capture(vk)
        else:
            super().keyPressEvent(event)

    # Qt mouse-button → Windows VK mapping
    _MOUSE_VK = {
        Qt.LeftButton:    0x01,   # VK_LBUTTON
        Qt.RightButton:   0x02,   # VK_RBUTTON
        Qt.MiddleButton:  0x04,   # VK_MBUTTON
        Qt.XButton1:      0x05,   # VK_XBUTTON1  (Mouse 4 / Back)
        Qt.XButton2:      0x06,   # VK_XBUTTON2  (Mouse 5 / Forward)
    }

    def mousePressEvent(self, event):
        if self._capturing_key:
            vk = self._MOUSE_VK.get(event.button())
            if vk is not None:
                self._finish_capture(vk)
                return
        super().mousePressEvent(event)

    # -- PTT active indicator ------------------------------------------------

    def set_ptt_active(self, active: bool):
        if active:
            self._ptt_status.setText("\u25cf  Talking")
            self._ptt_status.setObjectName("status-talking")
        else:
            self._ptt_status.setText("\u25cf  Muted")
            self._ptt_status.setObjectName("status-muted")
        self._ptt_status.style().unpolish(self._ptt_status)
        self._ptt_status.style().polish(self._ptt_status)

    def show_mic_warning(self, msg: str):
        self._mic_hint.setText(msg)
        self._mic_hint.show()


# ═══════════════════════════════════════════════════════════════════════════
# MainWindow — orchestrates everything
# ═══════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    _ptt_signal = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice Chat")
        self.resize(820, 540)
        self.setMinimumSize(700, 420)

        self._audio = None
        self._net   = NetworkManager(self)
        self._ptt_vk = None           # Windows virtual-key code
        self._ptt_was_down = False
        self._open_mic = False
        self._username = ""
        self._channel_names: list[str] = []
        self._current_cid = None

        # Timer that polls GetAsyncKeyState every 20 ms (global, works in background)
        self._ptt_timer = QTimer(self)
        self._ptt_timer.setInterval(20)
        self._ptt_timer.timeout.connect(self._poll_ptt)

        # Central stacked area
        self._container = QWidget()
        self._layout    = QVBoxLayout(self._container)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(self._container)

        self._login = LoginPage()
        self._chat  = None

        self._show_login()

        # Signals
        self._login.sig_connect.connect(self._do_connect)
        self._net.sig_connected.connect(lambda: None)     # just acknowledge
        self._net.sig_welcome.connect(self._on_welcome)
        self._net.sig_disconnected.connect(self._on_disconnected)
        self._net.sig_error.connect(self._on_error)
        self._net.sig_user_list.connect(self._on_user_list)
        self._net.sig_audio.connect(self._on_audio)
        self._ptt_signal.connect(self._on_ptt)

    # ── page management ────────────────────────────────────────────────────

    def _clear_layout(self):
        while self._layout.count():
            item = self._layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)

    def _show_login(self):
        self._clear_layout()
        self._layout.addWidget(self._login)
        self._login.show()

    def _show_chat(self, channels: list[str]):
        self._clear_layout()
        self._chat = ChatPage(self._username)
        self._channel_names = list(channels)
        for i, name in enumerate(channels):
            self._chat.add_channel(i, name)
        self._chat.sig_join_channel.connect(self._join_channel)
        self._chat.sig_leave_channel.connect(self._leave_channel)
        self._chat.sig_disconnect.connect(self._disconnect)
        self._chat.sig_ptt_changed.connect(self._set_ptt_key)
        self._chat.sig_open_mic.connect(self._set_open_mic)
        self._layout.addWidget(self._chat)
        self._chat.show()

    # ── connection ─────────────────────────────────────────────────────────

    def _do_connect(self, name: str, address: str):
        self._username = name
        self._net.connect_to(address, name)

    def _on_welcome(self, channels: list[str]):
        self._show_chat(channels)

        # Start audio engine
        self._audio = AudioManager()
        try:
            self._audio.start()
            self._audio.on_audio_data = self._net.send_audio
        except Exception as e:
            QMessageBox.warning(
                self, "Audio Error",
                f"Could not initialise audio devices:\n{e}\n\n"
                "You can still join channels and listen.",
            )

        if self._audio and not self._audio._has_input:
            if self._chat:
                self._chat.show_mic_warning(
                    "No microphone detected — you can listen but not talk."
                )

    def _on_disconnected(self):
        self._cleanup()
        self._login.reset("Disconnected from server.")
        self._show_login()

    def _on_error(self, msg: str):
        self._cleanup()
        self._login.reset(f"Connection error: {msg}")
        self._show_login()

    # ── channels ───────────────────────────────────────────────────────────

    def _join_channel(self, cid: int):
        self._current_cid = cid
        self._net.send_json({"type": "join_channel", "channel_id": cid})
        name = self._channel_names[cid] if cid < len(self._channel_names) else "?"
        if self._chat:
            self._chat.set_current_channel(cid, name)

    def _leave_channel(self):
        self._current_cid = None
        self._net.send_json({"type": "leave_channel"})
        if self._chat:
            self._chat.set_current_channel(None)

    def _on_user_list(self, data: dict):
        if self._chat:
            self._chat.update_users(data)

    # ── audio ──────────────────────────────────────────────────────────────

    def _on_audio(self, data: bytes):
        if self._audio:
            self._audio.enqueue_playback(data)

    # ── PTT ────────────────────────────────────────────────────────────────

    def _set_open_mic(self, enabled: bool):
        self._open_mic = enabled
        if self._audio:
            self._audio.is_capturing = enabled
        if enabled:
            self._ptt_timer.stop()
            self._ptt_was_down = False
        else:
            # Revert to PTT mode; start polling if a key is set
            if self._audio:
                self._audio.is_capturing = False
            if self._ptt_vk is not None:
                self._ptt_timer.start()

    def _set_ptt_key(self, vk: int):
        self._ptt_vk = vk
        self._ptt_was_down = False
        if self._audio:
            self._audio.is_capturing = False
        if not self._open_mic:
            self._ptt_timer.start()

    def _poll_ptt(self):
        """Called every 20 ms — check if the PTT key is physically held."""
        if self._ptt_vk is None or self._open_mic:
            return
        pressed = bool(_GetAsyncKeyState(self._ptt_vk) & 0x8000)
        if pressed and not self._ptt_was_down:
            self._ptt_was_down = True
            if self._audio:
                self._audio.is_capturing = True
            self._ptt_signal.emit(True)
        elif not pressed and self._ptt_was_down:
            self._ptt_was_down = False
            if self._audio:
                self._audio.is_capturing = False
            self._ptt_signal.emit(False)

    def _on_ptt(self, active: bool):
        if self._chat:
            self._chat.set_ptt_active(active)

    # ── cleanup ────────────────────────────────────────────────────────────

    def _cleanup(self):
        self._ptt_timer.stop()
        if self._audio:
            self._audio.is_capturing = False
            self._audio.stop()
            self._audio = None
        self._ptt_vk = None
        self._ptt_was_down = False
        self._current_cid = None

    def _disconnect(self):
        self._net.send_json({"type": "leave_channel"})
        self._net.disconnect()

    def closeEvent(self, event):
        self._cleanup()
        self._net.disconnect()
        super().closeEvent(event)


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_THEME)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
