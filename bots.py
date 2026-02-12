#!/usr/bin/env python3
"""
Voice Chat Test Bots
====================
Spawns N fake users that connect to the server via WebRTC,
join a channel, and speak with distinct human voices (via Edge TTS).

Usage:
    python bots.py                              # 3 bots on channel 1
    python bots.py --count 5 --channel 0        # 5 bots on General
    python bots.py --addr 192.168.1.5:9753 --count 2 --channel 2
    python bots.py --no-tts                     # fallback: sine tones only
"""

import argparse
import asyncio
import io
import json
import logging
import random

import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription
import websockets

# ---------------------------------------------------------------------------
# Config (must match server / client)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 48000
BLOCK_SIZE  = 960       # samples per chunk (20 ms at 48 kHz)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bots")


# ---------------------------------------------------------------------------
# Voice definitions — each bot gets a unique Edge TTS voice + lines
# ---------------------------------------------------------------------------
VOICES = [
    {
        "voice": "en-US-GuyNeural",
        "label": "Guy (US male)",
        "lines": [
            "Hey everyone, just hopped in. Can you all hear me okay?",
            "So I was thinking we could run some games tonight, what do you guys think?",
            "The audio quality on this thing is actually pretty solid.",
            "Alright, who's ready to go? Let me grab my headset real quick.",
            "That's hilarious dude, I almost spit out my coffee.",
        ],
    },
    {
        "voice": "en-US-JennyNeural",
        "label": "Jenny (US female)",
        "lines": [
            "Hey hey! Yeah I can hear you loud and clear.",
            "Okay wait, let me turn down my speakers so I don't echo.",
            "Has anyone else been having lag today or is it just me?",
            "Oh my god, that play was insane! Did you guys see that?",
            "Alright I'm muting for a sec, someone's at the door.",
        ],
    },
    {
        "voice": "en-GB-RyanNeural",
        "label": "Ryan (British male)",
        "lines": [
            "Evening lads, how's everyone doing tonight?",
            "Right, so the plan is to jump in and see what happens, yeah?",
            "I've been testing this voice chat all day, works a treat.",
            "Hold on, my cat just knocked over my monitor, one second.",
            "Brilliant, absolutely brilliant. Let's keep going.",
        ],
    },
    {
        "voice": "en-AU-NatashaNeural",
        "label": "Natasha (Australian female)",
        "lines": [
            "G'day! Just joined the channel, what are we up to?",
            "No worries, I can hear everyone perfectly fine.",
            "Reckon we should do a quick mic check before we start.",
            "Oh that's classic, you guys are cracking me up.",
            "Alright, I'm ready when you are. Let's get into it!",
        ],
    },
    {
        "voice": "en-IN-NeerjaNeural",
        "label": "Neerja (Indian female)",
        "lines": [
            "Hi everyone! Good to be here. How's the connection?",
            "One moment please, let me adjust my microphone settings.",
            "Testing, testing. Can everyone hear me properly?",
            "That was really impressive, well played!",
            "Okay team, let's focus. We've got this.",
        ],
    },
]


# ---------------------------------------------------------------------------
# TTS → raw mono int16 PCM at SAMPLE_RATE
# ---------------------------------------------------------------------------
async def tts_to_pcm(text: str, voice: str) -> np.ndarray:
    """Convert text to mono int16 PCM at SAMPLE_RATE using Edge TTS + PyAV.

    Note: edge_tts is a client for Microsoft's cloud TTS (Text To Speech)
    service. The audio synthesis is performed on Microsoft's servers,
    not locally, and this function requires an internet connection.
    """
    import edge_tts
    import av

    communicate = edge_tts.Communicate(text, voice)
    mp3_buf = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_buf += chunk["data"]

    # Decode MP3 → PCM using PyAV (bundled with aiortc)
    container = av.open(io.BytesIO(mp3_buf), format="mp3")
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    pcm_parts = []
    for frame in container.decode(audio=0):
        out_frames = resampler.resample(frame)
        for f in out_frames:
            pcm_parts.append(f.to_ndarray().flatten())
    container.close()

    return np.concatenate(pcm_parts).astype(np.int16)


async def generate_voice_clips(voice_def: dict) -> list[np.ndarray]:
    """Generate PCM clips for all lines of one voice definition."""
    clips = []
    for line in voice_def["lines"]:
        log.info("  TTS [%s]: %s", voice_def["label"], line[:50] + "…" if len(line) > 50 else line)
        pcm = await tts_to_pcm(line, voice_def["voice"])
        clips.append(pcm)
    return clips


# ---------------------------------------------------------------------------
# Load audio files from a directory (ogg / wav / mp3 / flac / …)
# ---------------------------------------------------------------------------
def load_audio_file(path: str) -> np.ndarray:
    """Decode any audio file to mono int16 PCM at SAMPLE_RATE via PyAV."""
    import av

    container = av.open(path)
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    pcm_parts = []
    for frame in container.decode(audio=0):
        for f in resampler.resample(frame):
            pcm_parts.append(f.to_ndarray().flatten())
    container.close()
    return np.concatenate(pcm_parts).astype(np.int16)


def load_voices_dir(dirpath: str) -> list[np.ndarray]:
    """Load all audio files from *dirpath*, sorted by name."""
    import os

    exts = {".ogg", ".wav", ".mp3", ".flac", ".opus", ".m4a", ".aac"}
    files = sorted(
        f for f in os.listdir(dirpath)
        if os.path.splitext(f)[1].lower() in exts
    )
    if not files:
        raise FileNotFoundError(f"No audio files found in {dirpath}")

    clips = []
    for f in files:
        full = os.path.join(dirpath, f)
        log.info("  loading %s", f)
        clips.append(load_audio_file(full))
    return clips


# ---------------------------------------------------------------------------
# Fallback: sine tones if --no-tts
# ---------------------------------------------------------------------------
def make_tone_clip(freq: float, duration: float = 2.0, amp: float = 0.25) -> np.ndarray:
    n = int(SAMPLE_RATE * duration)
    t = np.arange(n) / SAMPLE_RATE
    return (amp * 32767 * np.sin(2 * np.pi * freq * t)).astype(np.int16)


# ---------------------------------------------------------------------------
# Stream a PCM clip as BLOCK_SIZE chunks over a DataChannel
# ---------------------------------------------------------------------------
async def stream_clip(dc, clip: np.ndarray):
    """Send one PCM clip through the DataChannel at real-time pace.

    Uses wall-clock scheduling so that cumulative sleep drift on
    Windows (timer granularity ~15 ms) never causes the send rate
    to fall behind the playback rate.
    """
    loop     = asyncio.get_event_loop()
    total    = len(clip)
    offset   = 0
    interval = BLOCK_SIZE / SAMPLE_RATE
    t0       = loop.time()
    chunk_i  = 0

    while offset < total:
        end = min(offset + BLOCK_SIZE, total)
        chunk = clip[offset:end]
        if len(chunk) < BLOCK_SIZE:
            chunk = np.concatenate([chunk, np.zeros(BLOCK_SIZE - len(chunk), dtype=np.int16)])
        if dc.readyState == "open":
            dc.send(chunk.tobytes())
        offset  += BLOCK_SIZE
        chunk_i += 1

        # Sleep until the *absolute* target time for the next chunk
        target = t0 + chunk_i * interval
        delay  = target - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)


# ---------------------------------------------------------------------------
# Single bot coroutine
# ---------------------------------------------------------------------------
async def run_bot(address: str, name: str, channel_id: int, clips: list[np.ndarray]):
    """Connect one bot, join a channel, and loop through voice clips."""

    url = f"ws://{address}"
    log.info("[%s] connecting …", name)

    try:
        async with websockets.connect(url, max_size=2**20, ping_interval=30, ping_timeout=10) as ws:
            await ws.send(json.dumps({"type": "set_name", "name": name}))

            welcome = json.loads(await ws.recv())
            assert welcome["type"] == "welcome"
            channels = welcome["channels"]

            await ws.send(json.dumps({"type": "join_channel", "channel_id": channel_id}))
            log.info("[%s] joined '%s'", name, channels[channel_id])

            # ── WebRTC setup ──────────────────────────────────────────
            pc = RTCPeerConnection()
            dc = pc.createDataChannel("audio", ordered=False, maxRetransmits=0)

            dc_open = asyncio.Event()
            received_chunks = 0

            @dc.on("open")
            def on_open():
                dc_open.set()

            @dc.on("message")
            def on_message(msg):
                nonlocal received_chunks
                if isinstance(msg, bytes):
                    received_chunks += 1

            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)

            if pc.iceGatheringState != "complete":
                done = asyncio.Event()

                @pc.on("icegatheringstatechange")
                def _():
                    if pc.iceGatheringState == "complete":
                        done.set()

                await asyncio.wait_for(done.wait(), timeout=30)

            await ws.send(json.dumps({
                "type": "offer",
                "sdp": pc.localDescription.sdp,
                "sdp_type": pc.localDescription.type,
            }))

            # Read until we get the answer
            while True:
                msg = json.loads(await ws.recv())
                if msg["type"] == "answer":
                    ans = RTCSessionDescription(sdp=msg["sdp"], type=msg["sdp_type"])
                    await pc.setRemoteDescription(ans)
                    break

            await asyncio.wait_for(dc_open.wait(), timeout=15)
            log.info("[%s] DataChannel open — starting voice loop", name)

            # Drain WS messages in background
            async def ws_drain():
                try:
                    async for _ in ws:
                        pass
                except Exception:
                    pass

            drain_task = asyncio.create_task(ws_drain())

            try:
                clip_idx = 0
                while True:
                    clip = clips[clip_idx % len(clips)]
                    log.info("[%s] speaking clip %d/%d  (recv %d from others)",
                             name, clip_idx % len(clips) + 1, len(clips), received_chunks)
                    await stream_clip(dc, clip)

                    # Pause between lines (like a real person)
                    pause = random.uniform(1.5, 4.0)
                    await asyncio.sleep(pause)
                    clip_idx += 1
            except asyncio.CancelledError:
                pass
            finally:
                drain_task.cancel()
                await pc.close()

    except Exception as e:
        log.error("[%s] error: %s", name, e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    parser = argparse.ArgumentParser(description="Voice chat test bots")
    parser.add_argument("--addr",    default="localhost:9753", help="Server address")
    parser.add_argument("--count",   type=int, default=3,     help="Number of bots (1-5)")
    parser.add_argument("--channel", type=int, default=1,     help="Channel ID (0-4)")
    parser.add_argument("--no-tts",  action="store_true",     help="Use sine tones instead of TTS")
    parser.add_argument("--voices-dir", type=str, default=None,
                        help="Path to a folder of audio files (ogg/wav/mp3/…) to use instead of TTS")
    args = parser.parse_args()

    count = min(args.count, len(VOICES))

    # ── Generate / load voice clips ───────────────────────────────────────
    all_clips: list[list[np.ndarray]] = []

    if args.voices_dir:
        # Load real audio files from disk
        log.info("Loading audio files from %s …", args.voices_dir)
        file_clips = load_voices_dir(args.voices_dir)
        log.info("Loaded %d file(s)\n", len(file_clips))
        # Distribute files round-robin across bots
        for i in range(count):
            assigned = [file_clips[j] for j in range(len(file_clips)) if j % count == i]
            if not assigned:
                # More bots than files — give everyone the full set
                assigned = list(file_clips)
            all_clips.append(assigned)

    elif args.no_tts:
        log.info("Using sine-tone fallback (--no-tts)")
        for i in range(count):
            freq = 300 + i * 150
            clips = [make_tone_clip(freq, d) for d in [2.0, 1.5, 2.5, 1.8, 2.2]]
            all_clips.append(clips)

    else:
        log.info("Generating TTS voice clips (this takes a moment) …")
        for i in range(count):
            clips = await generate_voice_clips(VOICES[i])
            all_clips.append(clips)
        log.info("All voice clips ready!\n")

    # ── Launch bots ───────────────────────────────────────────────────────
    bots = []
    for i in range(count):
        label = VOICES[i]["label"] if i < len(VOICES) else f"Voice {i+1}"
        name = f"Bot-{i+1} ({label})"
        bots.append(run_bot(args.addr, name, args.channel, all_clips[i]))

    log.info("Launching %d bots on channel %d → %s\n", count, args.channel, args.addr)
    await asyncio.gather(*bots)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Bots stopped.")
