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
import threading
import queue
import asyncio

import ctypes
import ctypes.wintypes

import numpy as np
import sounddevice as sd
from aiortc import RTCPeerConnection, RTCSessionDescription
import websockets


from pyrnnoise import RNNoise as _RNNoise
_HAS_RNNOISE = True

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
SAMPLE_RATE    = 16000
AUDIO_CHANNELS = 1
BLOCK_SIZE     = 1024        # ~64 ms at 16 kHz
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
    known chunk is *repeated* instead of outputting silence,
    which sounds like a tiny sustain rather than a hard cut.
    """

    PRE_BUFFER  = 5             # chunks to buffer before playing (~320 ms)
    STALE_LIMIT = 50            # consecutive empty reads before resetting

    def __init__(self):
        self.q = queue.Queue(maxsize=60)
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
            if self._empty_streak > self.STALE_LIMIT:
                # Source truly stopped — reset for next burst
                self._started = False
                self._empty_streak = 0
                self._last_chunk = None
                return None
            # Brief underrun — repeat the last chunk to fill the gap
            return self._last_chunk


class _AudioProcessor:
    """Echo suppression  +  RNNoise neural denoiser.

    *Echo suppression* — keeps a ring buffer of recent speaker output
    (the *reference*) and uses power-spectral subtraction to remove
    the speaker bleed that the microphone picks up.

    *Noise suppression* — passes the result through **pyrnnoise**
    (Xiph RNNoise, a recurrent neural network trained for real-time
    speech denoising).  RNNoise operates at 48 kHz so we resample
    16 kHz ↔ 48 kHz around it.

    All processing runs in the send-loop thread, never in PortAudio.
    """

    _REF_BLOCKS = 10            # ~640 ms of reference history

    def __init__(self, block_size: int = BLOCK_SIZE):
        self.B = block_size

        # ── Reference ring buffer (what the speakers played) ───────────
        self._ref_ring = np.zeros(block_size * self._REF_BLOCKS, dtype=np.float64)
        self._ref_wpos = 0
        self._ref_lock = threading.Lock()

        # Spectral-subtraction tuning
        self._alpha = 4.0       # over-subtraction factor
        self._beta  = 0.02      # spectral floor

        # ── RNNoise (48 kHz) ──────────────────────────────────────────
        self._denoiser = None
        if _HAS_RNNOISE:
            try:
                self._denoiser = _RNNoise(sample_rate=48000)
            except Exception:
                pass

    # -- called from _out_cb (must be fast) ------------------------------

    def feed_reference(self, pcm_int16: np.ndarray):
        """Record what the speakers just played."""
        samples = pcm_int16.astype(np.float64)
        n = len(samples)
        with self._ref_lock:
            ring_len = len(self._ref_ring)
            end = self._ref_wpos + n
            if end <= ring_len:
                self._ref_ring[self._ref_wpos:end] = samples
            else:
                first = ring_len - self._ref_wpos
                self._ref_ring[self._ref_wpos:] = samples[:first]
                self._ref_ring[:n - first] = samples[first:]
            self._ref_wpos = end % ring_len

    # -- called from _send_loop thread -----------------------------------

    def process(self, mic_bytes: bytes) -> bytes:
        """Full chain: echo suppress → RNNoise denoise."""
        data = self._echo_suppress(mic_bytes)
        data = self._rnnoise(data)
        return data

    # ── echo suppression (spectral subtraction with reference) ─────────

    def _echo_suppress(self, mic_bytes: bytes) -> bytes:
        mic = np.frombuffer(mic_bytes, dtype=np.int16).astype(np.float64)

        with self._ref_lock:
            ref_all = self._ref_ring.copy()

        ref_energy = np.sqrt(np.mean(ref_all ** 2))
        if ref_energy < 30:
            return mic_bytes                       # speakers are silent

        mic_energy = np.sqrt(np.mean(mic ** 2))
        if mic_energy > ref_energy * 3.0:
            return mic_bytes                       # user is clearly talking

        # Build max power spectrum across recent reference blocks
        n_fft = len(mic)
        R_pow = np.zeros(n_fft // 2 + 1, dtype=np.float64)
        ring_len = len(ref_all)
        for i in range(self._REF_BLOCKS):
            start = ring_len - (i + 1) * self.B
            if start < 0:
                break
            block = ref_all[start:start + self.B]
            if len(block) == n_fft:
                R_pow = np.maximum(R_pow, np.abs(np.fft.rfft(block)) ** 2)

        M       = np.fft.rfft(mic)
        M_mag2  = np.abs(M) ** 2
        M_phase = np.angle(M)

        clean_mag2 = np.maximum(M_mag2 - self._alpha * R_pow,
                                self._beta * M_mag2)

        out = np.fft.irfft(np.sqrt(clean_mag2) * np.exp(1j * M_phase))[:n_fft]
        return np.clip(out, -32768, 32767).astype(np.int16).tobytes()

    # ── RNNoise neural denoiser (48 kHz) ───────────────────────────────

    def _rnnoise(self, pcm_bytes: bytes) -> bytes:
        if self._denoiser is None:
            return pcm_bytes

        audio_16k = np.frombuffer(pcm_bytes, dtype=np.int16)
        n = len(audio_16k)

        # Upsample 16 kHz → 48 kHz (clean 1:3 ratio)
        audio_48k = np.interp(
            np.linspace(0, n - 1, n * 3),
            np.arange(n),
            audio_16k.astype(np.float64),
        ).astype(np.int16)

        # RNNoise expects [channels, samples] int16
        try:
            denoised_parts: list[np.ndarray] = []
            for _prob, frame in self._denoiser.denoise_chunk(
                audio_48k.reshape(1, -1), partial=True,
            ):
                denoised_parts.append(frame.flatten())
        except Exception:
            return pcm_bytes

        if not denoised_parts:
            return pcm_bytes

        denoised_48k = np.concatenate(denoised_parts).astype(np.float64)

        # Downsample 48 kHz → 16 kHz
        denoised_16k = np.interp(
            np.linspace(0, len(denoised_48k) - 1, n),
            np.arange(len(denoised_48k)),
            denoised_48k,
        ).astype(np.int16)

        return denoised_16k.tobytes()


class AudioManager:
    """Thread-safe microphone capture and speaker playback.

    Incoming audio is tagged with an 8-byte source ID by the server.
    Each source gets its own jitter buffer so consecutive chunks from
    the *same* speaker play back-to-back, while chunks from
    *different* speakers are summed (mixed) in the output callback.

    An ``_AudioProcessor`` handles echo suppression (spectral
    subtraction with speaker reference) and noise suppression
    (RNNoise neural network via ``pyrnnoise``).
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
                # Echo suppress → RNNoise denoise → send
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
    """

    sig_welcome      = pyqtSignal(list)    # channel names
    sig_connected    = pyqtSignal()
    sig_disconnected = pyqtSignal()
    sig_error        = pyqtSignal(str)
    sig_user_list    = pyqtSignal(dict)    # {str(cid): [names]}
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

    def send_audio(self, raw: bytes):
        """Called from the audio send-loop thread."""
        loop = self._loop
        if self._dc and loop and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(self._dc_send_sync, raw)
            except RuntimeError:
                pass   # loop already closed

    def disconnect(self):
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
        url = f"ws://{self._address}"
        try:
            async with websockets.connect(
                url, max_size=2 ** 20, ping_interval=30, ping_timeout=10,
            ) as ws:
                self._ws = ws
                await ws.send(json.dumps({
                    "type": "set_name", "name": self._name,
                }))
                async for msg in ws:
                    if isinstance(msg, str):
                        await self._on_message(json.loads(msg))
        except asyncio.CancelledError:
            pass
        except websockets.exceptions.ConnectionClosed:
            pass
        except ConnectionRefusedError:
            if not self._closing:
                self.sig_error.emit("Could not connect — is the server running?")
        except Exception as e:
            if not self._closing:
                self.sig_error.emit(str(e))
        finally:
            if self._pc:
                try:
                    await self._pc.close()
                except Exception:
                    pass
                self._pc = None
                self._dc = None
            self._ws = None

    # -- signaling messages --------------------------------------------------

    async def _on_message(self, data: dict):
        t = data.get("type")

        if t == "welcome":
            self.sig_welcome.emit(data["channels"])
            self.sig_connected.emit()
            await self._setup_webrtc()

        elif t == "user_list":
            self.sig_user_list.emit(data["channels"])

        elif t == "answer":
            if self._pc:
                ans = RTCSessionDescription(
                    sdp=data["sdp"], type=data["sdp_type"],
                )
                await self._pc.setRemoteDescription(ans)

    # -- WebRTC setup --------------------------------------------------------

    async def _setup_webrtc(self):
        self._pc = RTCPeerConnection()

        # Unreliable + unordered DataChannel (UDP-like, ideal for audio)
        self._dc = self._pc.createDataChannel(
            "audio", ordered=False, maxRetransmits=0,
        )

        @self._dc.on("message")
        def on_dc_message(message):
            if isinstance(message, bytes):
                self.sig_audio.emit(message)

        @self._pc.on("connectionstatechange")
        async def _on_rtc_state():
            pass   # could add reconnect logic here

        # Create and send SDP offer
        offer = await self._pc.createOffer()
        await self._pc.setLocalDescription(offer)

        # Wait until all ICE candidates are gathered
        if self._pc.iceGatheringState != "complete":
            done = asyncio.Event()

            @self._pc.on("icegatheringstatechange")
            def _on_ice():
                if self._pc and self._pc.iceGatheringState == "complete":
                    done.set()

            await asyncio.wait_for(done.wait(), timeout=30)

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
        except Exception:
            pass

    def _dc_send_sync(self, data: bytes):
        """Synchronous DC send — called via call_soon_threadsafe."""
        try:
            if self._dc and self._dc.readyState == "open":
                self._dc.send(data)
        except Exception:
            pass

    async def _close(self):
        try:
            if self._pc:
                await self._pc.close()
                self._pc = None
                self._dc = None
        except Exception:
            pass
        try:
            if self._ws:
                await self._ws.close()
        except Exception:
            pass


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
        self._net.sig_connected.connect(lambda: None)
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
                    "No microphone detected \u2014 you can listen but not talk."
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
