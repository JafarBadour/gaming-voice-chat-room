#!/usr/bin/env python3
"""
Voice Chat Client (WebRTC)
==========================
PyQt5 desktop application with push-to-talk / open-mic voice chat.
Audio is transported over a WebRTC DataChannel (unreliable + unordered,
UDP-like) for low-latency, non-blocking communication.

Run:
    python client.py
"""

import sys
import json
import logging
import threading
import queue
import asyncio
import traceback
import time

import ctypes
import ctypes.wintypes

import numpy as np
import sounddevice as sd
from aiortc import RTCPeerConnection, RTCSessionDescription
import websockets


try:
    from aec_audio_processing import AudioProcessor as _WebRTCAP
    _HAS_AEC = True
except ImportError:
    _WebRTCAP = None
    _HAS_AEC = False

try:
    from pyrnnoise import RNNoise as _RNNoise
    _HAS_RNNOISE = True
except ImportError:
    _RNNoise = None
    _HAS_RNNOISE = False

# ---------------------------------------------------------------------------
# Client-side logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  [%(levelname)s]  %(name)-12s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("client")

# Quiet down noisy third-party loggers
for _quiet in ("aioice", "aiortc", "aiortc.rtcpeerconnection",
               "aiortc.rtcdtlstransport", "aiortc.rtcicetransport",
               "websockets"):
    logging.getLogger(_quiet).setLevel(logging.WARNING)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFrame, QScrollArea, QMessageBox,
    QSizePolicy,
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont

# ═══════════════════════════════════════════════════════════════════════════
# Windows API for global key-state polling (no hooks / no admin needed)
# ═══════════════════════════════════════════════════════════════════════════
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
SAMPLE_RATE    = 48000
AUDIO_CHANNELS = 1
BLOCK_SIZE     = 960         # 20 ms at 48 kHz (= 2 × 480 RNNoise/AEC frames)
DTYPE          = "int16"

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
# AudioManager — capture + playback with multi-user mixing
# ═══════════════════════════════════════════════════════════════════════════
class _SourceBuffer:
    """Per-source jitter buffer.

    Accumulates PRE_BUFFER chunks before starting playback.
    If the buffer briefly runs dry (network jitter), the last
    known chunk is faded out quickly (over ~2 chunks ≈ 128 ms)
    rather than repeated at full volume or cut to hard silence.
    """

    PRE_BUFFER   = 15           # chunks to buffer before playing (~300 ms)
    FADE_CHUNKS  = 4            # underrun fade-out chunks (~80 ms)

    def __init__(self):
        self.q = queue.Queue(maxsize=150)
        self._started = False
        self._empty_streak = 0
        self._last_chunk: bytes | None = None

    def put(self, data: bytes):
        try:
            self.q.put_nowait(data)
        except queue.Full:
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(data)
            except queue.Full:
                pass

    def get(self) -> bytes | None:
        """Return one chunk, or None if source is truly silent."""
        if not self._started:
            if self.q.qsize() >= self.PRE_BUFFER:
                self._started = True
            else:
                return None

        try:
            data = self.q.get_nowait()
            self._empty_streak = 0
            self._last_chunk = data
            return data
        except queue.Empty:
            self._empty_streak += 1
            if self._empty_streak > self.FADE_CHUNKS or self._last_chunk is None:
                # Sender stopped — reset for next burst
                self._started = False
                self._empty_streak = 0
                self._last_chunk = None
                return None
            # Brief underrun — return a quickly fading version of the
            # last chunk so the tail doesn't repeat at full volume.
            fade = 1.0 / (2 ** self._empty_streak)     # 0.5, 0.25, …
            samples = np.frombuffer(self._last_chunk, dtype=np.int16)
            faded = (samples.astype(np.float32) * fade).astype(np.int16)
            return faded.tobytes()


class _AudioProcessor:
    """WebRTC AEC  +  RNNoise noise suppression.

    Two dedicated libraries, each used for what it does best:

    * **WebRTC AEC** (``aec-audio-processing``) — real adaptive-filter
      echo cancellation.  AGC and NS are disabled (AGC was boosting
      noise; WebRTC NS targets stationary noise, not keyboard clicks).

    * **RNNoise** (``pyrnnoise``) — neural-network denoiser trained on
      non-speech transient sounds (keyboard, mouse, breathing, …).

    Pipeline:  mic → WebRTC AEC → RNNoise → network

    At 48 kHz with BLOCK_SIZE = 960, both WebRTC AEC (480-sample /
    10 ms frames) and RNNoise (480-sample frames) divide evenly into
    each block — no carry buffers, no resampling, no variable output.
    """

    _FRAME = 480               # 10 ms @ 48 kHz

    def __init__(self, block_size: int = BLOCK_SIZE):
        self.B = block_size
        self.enabled = True    # toggled from the UI

        # ── WebRTC AEC (echo cancellation only) ──────────────────────
        self._ap = None
        self._ref_q: queue.Queue = queue.Queue(maxsize=200)

        if _HAS_AEC:
            try:
                self._ap = _WebRTCAP(
                    enable_aec=True,
                    enable_ns=False,     # leave NS to RNNoise
                    enable_agc=False,    # no auto-gain
                )
                self._ap.set_stream_format(SAMPLE_RATE, AUDIO_CHANNELS)
                self._ap.set_reverse_stream_format(SAMPLE_RATE, AUDIO_CHANNELS)
                self._ap.set_stream_delay(40)
            except Exception:
                self._ap = None

        # ── RNNoise (keyboard / mouse / noise suppression) ───────────
        self._denoiser = None
        if _HAS_RNNOISE:
            try:
                # 48 kHz = RNNoise native rate → zero resampling
                self._denoiser = _RNNoise(sample_rate=SAMPLE_RATE)
            except Exception:
                self._denoiser = None

    # -- called from _out_cb (must be fast) --------------------------------

    def feed_reference(self, pcm_int16: np.ndarray):
        """Queue speaker output so the AEC can subtract it later."""
        if self._ap is None or not self.enabled:
            return
        try:
            self._ref_q.put_nowait(pcm_int16.astype(np.int16).copy())
        except queue.Full:
            pass

    # -- called from _send_loop thread -------------------------------------

    def process(self, mic_bytes: bytes) -> bytes:
        if not self.enabled:
            return mic_bytes
        data = self._aec(mic_bytes)
        data = self._rnnoise(data)
        return data

    # ── WebRTC AEC ────────────────────────────────────────────────────

    def _aec(self, mic_bytes: bytes) -> bytes:
        """Remove echo using WebRTC's adaptive-filter AEC."""
        if self._ap is None:
            return mic_bytes

        F = self._FRAME

        # 1. Drain reference queue → feed reverse stream
        while True:
            try:
                ref = self._ref_q.get_nowait()
            except queue.Empty:
                break
            # 960 / 480 = 2 frames exactly — no remainder
            for i in range(0, len(ref), F):
                try:
                    self._ap.process_reverse_stream(
                        ref[i:i + F].tobytes())
                except Exception:
                    pass

        # 2. Process mic in 480-sample (10 ms) frames
        audio = np.frombuffer(mic_bytes, dtype=np.int16)
        parts: list[np.ndarray] = []
        for i in range(0, len(audio), F):
            try:
                out = self._ap.process_stream(audio[i:i + F].tobytes())
                parts.append(np.frombuffer(out, dtype=np.int16))
            except Exception:
                parts.append(audio[i:i + F])

        return np.concatenate(parts).tobytes()

    # ── RNNoise (keyboard / mouse / background noise) ─────────────────

    def _rnnoise(self, pcm_bytes: bytes) -> bytes:
        """Remove keyboard, mouse, and background noise via RNNoise."""
        if self._denoiser is None:
            return pcm_bytes

        audio = np.frombuffer(pcm_bytes, dtype=np.int16)
        # 960 / 480 = 2 frames exactly at native 48 kHz — no resampling
        try:
            parts: list[np.ndarray] = []
            for _prob, frame in self._denoiser.denoise_chunk(
                audio.reshape(1, -1), partial=False,
            ):
                parts.append(frame.flatten())
        except Exception:
            return pcm_bytes

        if not parts:
            return pcm_bytes
        return np.concatenate(parts).astype(np.int16).tobytes()


class AudioManager:
    """Thread-safe microphone capture and speaker playback.

    Incoming audio is tagged with an 8-byte source ID by the server.
    Each source gets its own jitter buffer so consecutive chunks from
    the *same* speaker play back-to-back, while chunks from
    *different* speakers are summed (mixed) in the output callback.

    An ``_AudioProcessor`` chains WebRTC AEC (echo cancellation)
    with RNNoise (keyboard / mouse / noise suppression).
    """

    SOURCE_TAG_LEN = 8          # bytes prepended by the server

    def __init__(self):
        self.is_capturing = False
        self.on_audio_data = None          # callback(bytes)
        self._capture_q    = queue.Queue(maxsize=50)
        self._sources: dict[bytes, _SourceBuffer] = {}
        self._sources_lock = threading.Lock()
        self._running      = False
        self._in_stream    = None
        self._out_stream   = None
        self._has_input    = True
        self._proc         = _AudioProcessor(BLOCK_SIZE)

    # -- lifecycle -----------------------------------------------------------

    def start(self):
        self._running = True

        self._out_stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=AUDIO_CHANNELS,
            dtype=DTYPE,
            blocksize=BLOCK_SIZE,
            callback=self._out_cb,
        )
        self._out_stream.start()

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
        if self.is_capturing:
            try:
                self._capture_q.put_nowait(indata.tobytes())
            except queue.Full:
                pass

    def _out_cb(self, outdata, frames, time_info, status):
        """Take ONE chunk from each source's jitter buffer and mix.

        Each _SourceBuffer holds back audio until PRE_BUFFER chunks
        have accumulated, absorbing network jitter so playback is
        smooth and continuous.
        """
        needed = frames * AUDIO_CHANNELS
        mixed  = np.zeros(needed, dtype=np.float32)
        count  = 0

        with self._sources_lock:
            buffers = list(self._sources.values())

        for buf in buffers:
            data = buf.get()
            if data is not None:
                arr = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                size = min(arr.size, needed)
                mixed[:size] += arr[:size]
                count += 1

        if count:
            out_samples = np.clip(mixed, -32768, 32767).astype(np.int16)
            outdata[:, 0] = out_samples
            # Record speaker output for echo suppression
            self._proc.feed_reference(out_samples)
        else:
            outdata.fill(0)

    # -- send loop (own thread → network) ------------------------------------

    def _send_loop(self):
        while self._running:
            try:
                data = self._capture_q.get(timeout=0.05)
                # WebRTC AEC + NS + AGC → send
                data = self._proc.process(data)
                cb = self.on_audio_data
                if cb:
                    cb(data)
            except queue.Empty:
                pass

    # -- called from network thread ------------------------------------------

    def enqueue_playback(self, raw: bytes):
        """Parse the 8-byte source tag and route to a per-source jitter buffer."""
        if len(raw) <= self.SOURCE_TAG_LEN:
            return
        source_id = raw[:self.SOURCE_TAG_LEN]
        audio     = raw[self.SOURCE_TAG_LEN:]

        with self._sources_lock:
            buf = self._sources.get(source_id)
            if buf is None:
                buf = _SourceBuffer()
                self._sources[source_id] = buf

        buf.put(audio)


# ═══════════════════════════════════════════════════════════════════════════
# NetworkManager — WebSocket signaling + WebRTC DataChannel
# ═══════════════════════════════════════════════════════════════════════════
class NetworkManager(QWidget):
    """Manages signaling (WebSocket) and audio transport (WebRTC DC).

    Runs an asyncio event loop in a background thread.  All heavy I/O
    is non-blocking; the Qt main thread is never stalled.

    If the connection drops unexpectedly, the manager will automatically
    retry up to MAX_RECONNECT_ATTEMPTS times with exponential backoff.
    """

    MAX_RECONNECT_ATTEMPTS = 5
    RECONNECT_BASE_DELAY   = 2      # seconds — doubles each attempt

    sig_welcome      = pyqtSignal(list)    # channel names
    sig_connected    = pyqtSignal()
    sig_reconnecting = pyqtSignal(int)     # attempt number
    sig_disconnected = pyqtSignal()
    sig_error        = pyqtSignal(str)
    sig_user_list    = pyqtSignal(dict)    # {str(cid): [names]}
    sig_speaking     = pyqtSignal(dict)    # {str(cid): [names currently speaking]}
    sig_audio        = pyqtSignal(bytes)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.hide()
        self._ws      = None                # websockets connection
        self._pc      = None                # RTCPeerConnection
        self._dc      = None                # RTCDataChannel
        self._loop    = None                # asyncio event loop
        self._thread  = None
        self._name    = ""
        self._address = ""
        self._closing = False
        self._dc_send_errors = 0
        self._last_channel_id = None        # remember for reconnection

    # -- public API ----------------------------------------------------------

    def connect_to(self, address: str, name: str):
        self._address = address
        self._name    = name
        self._closing = False
        self._thread  = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def send_json(self, obj: dict):
        loop = self._loop
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(self._ws_send(json.dumps(obj)), loop)
        # Track channel for reconnection
        if obj.get("type") == "join_channel":
            self._last_channel_id = obj.get("channel_id")
        elif obj.get("type") == "leave_channel":
            self._last_channel_id = None

    def send_audio(self, raw: bytes):
        """Called from the audio send-loop thread."""
        loop = self._loop
        if self._dc and loop and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(self._dc_send_sync, raw)
            except RuntimeError:
                pass   # loop already closed

    def disconnect(self):
        log.info("Disconnect requested by user")
        self._closing = True
        loop = self._loop
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(self._close(), loop)

    # -- asyncio background thread -------------------------------------------

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main_async())
        except Exception as e:
            if not self._closing:
                log.error("Event loop crashed: %s\n%s", e, traceback.format_exc())
                self.sig_error.emit(str(e))
        finally:
            self.sig_disconnected.emit()
            try:
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            except Exception:
                pass
            self._loop.close()
            self._loop = None

    async def _main_async(self):
        attempt = 0

        while not self._closing:
            disconnect_reason = "unknown"
            try:
                await self._connect_once()
                # If _connect_once returns normally (clean close), don't retry
                if self._closing:
                    return
                disconnect_reason = "clean close"
            except asyncio.CancelledError:
                return
            except ConnectionRefusedError:
                if not self._closing:
                    self.sig_error.emit("Could not connect — is the server running?")
                return
            except websockets.exceptions.ConnectionClosedOK:
                disconnect_reason = "server closed cleanly"
                log.info("Server closed connection cleanly")
                if self._closing:
                    return
            except websockets.exceptions.ConnectionClosedError as e:
                disconnect_reason = f"connection lost (code={e.code} reason={e.reason!r})"
                log.warning("Connection lost: code=%s reason=%s", e.code, e.reason)
            except websockets.exceptions.ConnectionClosed as e:
                disconnect_reason = f"connection closed (code={e.code})"
                log.warning("Connection closed: code=%s", e.code)
            except OSError as e:
                disconnect_reason = f"network error: {e}"
                log.warning("Network error: %s", e)
            except Exception as e:
                disconnect_reason = f"unexpected: {e}"
                log.error("Unexpected connection error: %s\n%s", e, traceback.format_exc())
            finally:
                # Always clean up WebRTC
                if self._pc:
                    try:
                        await self._pc.close()
                    except Exception as e:
                        log.debug("Error closing PC: %s", e)
                    self._pc = None
                    self._dc = None
                self._ws = None

            if self._closing:
                return

            # Reconnection with exponential backoff
            attempt += 1
            if attempt > self.MAX_RECONNECT_ATTEMPTS:
                log.error("Giving up after %d reconnection attempts (last: %s)",
                          attempt - 1, disconnect_reason)
                self.sig_error.emit(
                    f"Lost connection ({disconnect_reason}). "
                    f"Gave up after {attempt - 1} retries.")
                return

            delay = self.RECONNECT_BASE_DELAY * (2 ** (attempt - 1))
            delay = min(delay, 30)  # cap at 30s
            log.info("Reconnecting in %.0fs (attempt %d/%d, reason: %s)",
                     delay, attempt, self.MAX_RECONNECT_ATTEMPTS, disconnect_reason)
            self.sig_reconnecting.emit(attempt)
            await asyncio.sleep(delay)

    async def _connect_once(self):
        """Single connection attempt — run until disconnected or error."""
        url = f"ws://{self._address}"
        log.info("Connecting to %s ...", url)

        async with websockets.connect(
            url,
            max_size=2 ** 20,
            ping_interval=30,
            ping_timeout=30,      # match server — generous for gaming
        ) as ws:
            self._ws = ws
            log.info("WebSocket connected to %s", url)

            await ws.send(json.dumps({
                "type": "set_name", "name": self._name,
            }))

            async for msg in ws:
                if isinstance(msg, str):
                    try:
                        await self._on_message(json.loads(msg))
                    except Exception:
                        log.error("Error handling message: %s\n%s",
                                  msg[:200], traceback.format_exc())

    # -- signaling messages --------------------------------------------------

    async def _on_message(self, data: dict):
        t = data.get("type")
        log.debug("WS recv: %s", t)

        if t == "welcome":
            self.sig_welcome.emit(data["channels"])
            self.sig_connected.emit()
            await self._setup_webrtc()
            # Re-join channel if this is a reconnection
            if self._last_channel_id is not None:
                log.info("Re-joining channel %d after reconnect", self._last_channel_id)
                await self._ws_send(json.dumps({
                    "type": "join_channel",
                    "channel_id": self._last_channel_id,
                }))

        elif t == "user_list":
            self.sig_user_list.emit(data["channels"])

        elif t == "speaking":
            self.sig_speaking.emit(data.get("speakers", {}))

        elif t == "answer":
            if self._pc:
                log.info("SDP answer received, setting remote description")
                ans = RTCSessionDescription(
                    sdp=data["sdp"], type=data["sdp_type"],
                )
                await self._pc.setRemoteDescription(ans)
        else:
            log.debug("Unknown message type: %s", t)

    # -- WebRTC setup --------------------------------------------------------

    async def _setup_webrtc(self):
        log.info("Setting up WebRTC PeerConnection")
        self._pc = RTCPeerConnection()
        self._dc_send_errors = 0

        # Unreliable + unordered DataChannel (UDP-like, ideal for audio)
        self._dc = self._pc.createDataChannel(
            "audio", ordered=False, maxRetransmits=0,
        )

        @self._dc.on("open")
        def on_dc_open():
            log.info("DataChannel OPEN")

        @self._dc.on("close")
        def on_dc_close():
            log.warning("DataChannel CLOSED")

        @self._dc.on("message")
        def on_dc_message(message):
            if isinstance(message, bytes):
                self.sig_audio.emit(message)

        @self._pc.on("connectionstatechange")
        async def _on_rtc_state():
            state = self._pc.connectionState if self._pc else "?"
            log.info("WebRTC state: %s", state)
            if state == "failed":
                log.error("WebRTC connection FAILED — likely network/firewall issue")
            elif state == "disconnected":
                log.warning("WebRTC DISCONNECTED — may recover automatically")

        @self._pc.on("iceconnectionstatechange")
        async def _on_ice_conn():
            state = self._pc.iceConnectionState if self._pc else "?"
            log.debug("ICE connection state: %s", state)

        # Create and send SDP offer
        offer = await self._pc.createOffer()
        await self._pc.setLocalDescription(offer)
        log.info("SDP offer created, gathering ICE candidates...")

        # Wait until all ICE candidates are gathered
        if self._pc.iceGatheringState != "complete":
            done = asyncio.Event()

            @self._pc.on("icegatheringstatechange")
            def _on_ice():
                log.debug("ICE gathering: %s",
                          self._pc.iceGatheringState if self._pc else "?")
                if self._pc and self._pc.iceGatheringState == "complete":
                    done.set()

            try:
                await asyncio.wait_for(done.wait(), timeout=30)
            except asyncio.TimeoutError:
                log.error("ICE gathering timed out after 30s")
                raise

        log.info("ICE gathering complete, sending offer to server")
        await self._ws.send(json.dumps({
            "type": "offer",
            "sdp": self._pc.localDescription.sdp,
            "sdp_type": self._pc.localDescription.type,
        }))

    # -- internal helpers ----------------------------------------------------

    async def _ws_send(self, text: str):
        try:
            if self._ws:
                await self._ws.send(text)
        except Exception as e:
            log.warning("WS send failed: %s", e)

    def _dc_send_sync(self, data: bytes):
        """Synchronous DC send — called via call_soon_threadsafe."""
        try:
            if self._dc and self._dc.readyState == "open":
                self._dc.send(data)
        except Exception as e:
            self._dc_send_errors += 1
            if self._dc_send_errors <= 5 or self._dc_send_errors % 100 == 0:
                log.warning("DC send failed: %s (total errors: %d)",
                            e, self._dc_send_errors)

    async def _close(self):
        try:
            if self._pc:
                await self._pc.close()
                self._pc = None
                self._dc = None
        except Exception as e:
            log.debug("Error closing PC: %s", e)
        try:
            if self._ws:
                await self._ws.close()
        except Exception as e:
            log.debug("Error closing WS: %s", e)


# ═══════════════════════════════════════════════════════════════════════════
# ChannelWidget
# ═══════════════════════════════════════════════════════════════════════════
class ChannelWidget(QFrame):
    """A clickable frame representing one voice channel."""

    clicked = pyqtSignal(int)

    _STYLE_IDLE     = "color: #a6adc8; font-size: 12px;"
    _STYLE_SPEAKING = ("color: #a6e3a1; font-size: 12px; font-weight: bold;")

    def __init__(self, channel_id: int, name: str, parent=None):
        super().__init__(parent)
        self.channel_id = channel_id
        self._name      = name
        self._active    = False
        self._user_labels: dict[str, QLabel] = {}   # name → QLabel
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
        self._user_labels.clear()
        self._count_lbl.setText(str(len(names)))
        for n in names:
            lbl = QLabel(f"  \u2022  {n}")
            lbl.setStyleSheet(self._STYLE_IDLE)
            self._users_layout.addWidget(lbl)
            self._user_labels[n] = lbl

    def set_speakers(self, speaking_names: list[str]):
        """Highlight users who are currently speaking."""
        speaking_set = set(speaking_names)
        for name, lbl in self._user_labels.items():
            if name in speaking_set:
                lbl.setText(f"  \U0001f50a  {name}")
                lbl.setStyleSheet(self._STYLE_SPEAKING)
            else:
                lbl.setText(f"  \u2022  {name}")
                lbl.setStyleSheet(self._STYLE_IDLE)

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
    sig_noise_suppress = pyqtSignal(bool)   # True = echo/noise suppression on

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

        rl.addWidget(self._sep())

        ns_hdr = QLabel("NOISE SUPPRESSION")
        ns_hdr.setObjectName("section")
        rl.addWidget(ns_hdr)

        self._ns_btn = QPushButton("Noise Suppression: ON")
        self._ns_btn.setObjectName("secondary")
        self._ns_btn.setCheckable(True)
        self._ns_btn.setChecked(True)          # enabled by default
        self._ns_btn.setStyleSheet(
            "background-color: #a6e3a1; color: #1e1e2e; font-weight: bold;"
            "border: none; border-radius: 8px; padding: 10px 24px;"
        )
        self._ns_btn.clicked.connect(self._toggle_noise_suppress)
        rl.addWidget(self._ns_btn)

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

    def update_speakers(self, data: dict):
        """Highlight speaking users in each channel widget."""
        for w in self._widgets:
            w.set_speakers(data.get(str(w.channel_id), []))

    # -- open mic toggle -----------------------------------------------------

    def _toggle_open_mic(self, checked: bool):
        self._open_mic = checked
        if checked:
            self._open_mic_btn.setText("Open Mic: ON")
            self._open_mic_btn.setStyleSheet(
                "background-color: #a6e3a1; color: #1e1e2e; font-weight: bold;"
                "border: none; border-radius: 8px; padding: 10px 24px;"
            )
            self._ptt_hdr.setStyleSheet("color: #45475a;")
            self._ptt_row_widget.setEnabled(False)
            self._ptt_status.setText("\u25cf  Open Mic")
            self._ptt_status.setObjectName("status-talking")
            self._ptt_status.style().unpolish(self._ptt_status)
            self._ptt_status.style().polish(self._ptt_status)
        else:
            self._open_mic_btn.setText("Open Mic: OFF")
            self._open_mic_btn.setStyleSheet("")
            self._ptt_hdr.setStyleSheet("")
            self._ptt_row_widget.setEnabled(True)
            self._ptt_status.setText("\u25cf  Muted")
            self._ptt_status.setObjectName("status-muted")
            self._ptt_status.style().unpolish(self._ptt_status)
            self._ptt_status.style().polish(self._ptt_status)
        self.sig_open_mic.emit(checked)

    # -- noise suppression toggle --------------------------------------------

    def _toggle_noise_suppress(self, checked: bool):
        if checked:
            self._ns_btn.setText("Noise Suppression: ON")
            self._ns_btn.setStyleSheet(
                "background-color: #a6e3a1; color: #1e1e2e; font-weight: bold;"
                "border: none; border-radius: 8px; padding: 10px 24px;"
            )
        else:
            self._ns_btn.setText("Noise Suppression: OFF")
            self._ns_btn.setStyleSheet("")
        self.sig_noise_suppress.emit(checked)

    # -- PTT key capture (keyboard + mouse) ----------------------------------

    def _start_capture(self):
        self._ptt_btn.setEnabled(False)
        self._ptt_label.setText("Press any key or mouse button\u2026")
        self._capturing_key = True
        self.setFocus()
        self.grabKeyboard()
        self.grabMouse()

    def _finish_capture(self, vk: int):
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

    _MOUSE_VK = {
        Qt.LeftButton:    0x01,
        Qt.RightButton:   0x02,
        Qt.MiddleButton:  0x04,
        Qt.XButton1:      0x05,   # Mouse 4 / Back
        Qt.XButton2:      0x06,   # Mouse 5 / Forward
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
        if self._open_mic:
            return                     # don't override open-mic label
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
        self._ptt_vk = None
        self._ptt_was_down = False
        self._open_mic = False
        self._username = ""
        self._channel_names: list[str] = []
        self._current_cid = None

        # Timer that polls GetAsyncKeyState every 20 ms
        self._ptt_timer = QTimer(self)
        self._ptt_timer.setInterval(20)
        self._ptt_timer.timeout.connect(self._poll_ptt)

        # Delayed mute on PTT release (500 ms tail so words aren't clipped)
        self._ptt_release_timer = QTimer(self)
        self._ptt_release_timer.setSingleShot(True)
        self._ptt_release_timer.setInterval(500)
        self._ptt_release_timer.timeout.connect(self._ptt_release_fire)

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
        self._net.sig_connected.connect(lambda: log.info("Connected signal received"))
        self._net.sig_welcome.connect(self._on_welcome)
        self._net.sig_reconnecting.connect(self._on_reconnecting)
        self._net.sig_disconnected.connect(self._on_disconnected)
        self._net.sig_error.connect(self._on_error)
        self._net.sig_user_list.connect(self._on_user_list)
        self._net.sig_speaking.connect(self._on_speaking)
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
        self._chat.sig_noise_suppress.connect(self._set_noise_suppress)
        self._layout.addWidget(self._chat)
        self._chat.show()

    # ── connection ─────────────────────────────────────────────────────────

    def _do_connect(self, name: str, address: str):
        log.info("User connecting as %r to %s", name, address)
        self._username = name
        self._net.connect_to(address, name)

    def _on_welcome(self, channels: list[str]):
        log.info("Welcome received, channels: %s", channels)
        self._show_chat(channels)

        self._audio = AudioManager()
        try:
            self._audio.start()
            self._audio.on_audio_data = self._net.send_audio
            log.info("Audio manager started (has_input=%s)", self._audio._has_input)
        except Exception as e:
            log.error("Audio init failed: %s\n%s", e, traceback.format_exc())
            QMessageBox.warning(
                self, "Audio Error",
                f"Could not initialise audio devices:\n{e}\n\n"
                "You can still join channels and listen.",
            )

        if self._audio and not self._audio._has_input:
            if self._chat:
                self._chat.show_mic_warning(
                    "No microphone detected \u2014 you can listen but not talk."
                )

    def _on_reconnecting(self, attempt: int):
        log.info("Reconnecting... attempt %d", attempt)
        if self._chat:
            self._chat._ch_status.setText(
                f"Reconnecting... (attempt {attempt})")
            self._chat._ch_status.setStyleSheet(
                "font-size: 14px; color: #fab387;")  # orange/warning

    def _on_disconnected(self):
        log.info("Disconnected from server")
        self._cleanup()
        self._login.reset("Disconnected from server.")
        self._show_login()

    def _on_error(self, msg: str):
        log.error("Connection error: %s", msg)
        self._cleanup()
        self._login.reset(f"Connection error: {msg}")
        self._show_login()

    # ── channels ───────────────────────────────────────────────────────────

    def _join_channel(self, cid: int):
        self._current_cid = cid
        name = self._channel_names[cid] if cid < len(self._channel_names) else "?"
        log.info("Joining channel %d (%s)", cid, name)
        self._net.send_json({"type": "join_channel", "channel_id": cid})
        if self._chat:
            self._chat.set_current_channel(cid, name)

    def _leave_channel(self):
        log.info("Leaving channel")
        self._current_cid = None
        self._net.send_json({"type": "leave_channel"})
        if self._chat:
            self._chat.set_current_channel(None)

    def _on_user_list(self, data: dict):
        if self._chat:
            self._chat.update_users(data)

    def _on_speaking(self, data: dict):
        if self._chat:
            self._chat.update_speakers(data)

    # ── audio ──────────────────────────────────────────────────────────────

    def _on_audio(self, data: bytes):
        if self._audio:
            self._audio.enqueue_playback(data)

    # ── noise suppression ──────────────────────────────────────────────────

    def _set_noise_suppress(self, enabled: bool):
        if self._audio:
            self._audio._proc.enabled = enabled

    # ── PTT ────────────────────────────────────────────────────────────────

    def _set_open_mic(self, enabled: bool):
        self._open_mic = enabled
        self._ptt_release_timer.stop()
        if self._audio:
            self._audio.is_capturing = enabled
        if enabled:
            self._ptt_timer.stop()
            self._ptt_was_down = False
        else:
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
        if self._ptt_vk is None or self._open_mic:
            return
        pressed = bool(_GetAsyncKeyState(self._ptt_vk) & 0x8000)
        if pressed and not self._ptt_was_down:
            self._ptt_was_down = True
            self._ptt_release_timer.stop()   # cancel pending mute
            if self._audio:
                self._audio.is_capturing = True
            self._ptt_signal.emit(True)
        elif not pressed and self._ptt_was_down:
            self._ptt_was_down = False
            # Keep mic open for 500 ms so the tail of the sentence
            # isn't clipped.  _ptt_release_fire() does the actual mute.
            self._ptt_release_timer.start()

    def _ptt_release_fire(self):
        """Called 500 ms after PTT key was released."""
        if self._ptt_was_down:
            return                             # re-pressed before timeout
        if self._audio:
            self._audio.is_capturing = False
        self._ptt_signal.emit(False)

    def _on_ptt(self, active: bool):
        if self._chat:
            self._chat.set_ptt_active(active)

    # ── cleanup ────────────────────────────────────────────────────────────

    def _cleanup(self):
        self._ptt_timer.stop()
        self._ptt_release_timer.stop()
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
