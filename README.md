# Voice Chat

A simple voice chat application with multiple channels and push-to-talk.

## Features

- **5 voice channels**: General, Voice Channel 1–4
- **Push-to-talk**: configurable hotkey — hold to talk, release to mute
- **Real-time audio**: 16 kHz mono PCM streamed over WebSockets
- **Dark UI**: Catppuccin-inspired theme built with PyQt5

## Setup

Requires **Python 3.9+** on Windows.

```bash
pip install -r requirements.txt
```

> **Note:** `sounddevice` bundles PortAudio, so no extra native libraries are needed.
> The `keyboard` library uses low-level Windows hooks for global push-to-talk —
> some antivirus software may flag this; it is safe.

## Running

### 1. Start the server

```bash
python server.py
```

The server listens on port **9753** by default and prints its LAN IP address on
startup.

### 2. Start the client

```bash
python client.py
```

1. Enter your **display name** and the **server address** (e.g. `192.168.1.5:9753`).  
   Use `localhost:9753` if running on the same machine.
2. Click **Connect**.
3. Click any **voice channel** to join it.
4. Click **Set PTT Key** and press the key you want to use for push-to-talk.
5. **Hold** the PTT key to transmit your microphone — release to mute.

Anyone on the same channel hears you while your key is held.

## Architecture

```
client  ──WebSocket──▶  server  ──WebSocket──▶  other clients
 (audio capture)         (routes audio          (audio playback)
                          by channel)
```

- **server.py** — async WebSocket server (`websockets`). Tracks clients per
  channel and forwards audio binary frames to everyone else in the same channel.
- **client.py** — PyQt5 desktop app. Uses `sounddevice` for audio I/O,
  `keyboard` for global PTT hotkey, and `websocket-client` for networking.
