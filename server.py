#!/usr/bin/env python3
"""
Voice Chat Server
=================
Async WebSocket server that manages voice channels and routes audio
between connected clients.

Run:
    python server.py
"""

import asyncio
import json
import logging
import socket

import websockets

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HOST = "0.0.0.0"
PORT = 9753

CHANNELS = [
    "General",
    "Voice Channel 1",
    "Voice Channel 2",
    "Voice Channel 3",
    "Voice Channel 4",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("server")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_local_ip() -> str:
    """Return the machine's LAN IP (best-effort)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ---------------------------------------------------------------------------
# VoiceServer
# ---------------------------------------------------------------------------
class VoiceServer:
    def __init__(self):
        # ws -> {"name": str, "channel_id": int | None}
        self.clients: dict = {}

    # -- user-list helpers ---------------------------------------------------

    def _channel_users(self) -> dict[int, list[str]]:
        channels: dict[int, list[str]] = {i: [] for i in range(len(CHANNELS))}
        for info in self.clients.values():
            cid = info["channel_id"]
            if cid is not None:
                channels[cid].append(info["name"])
        return channels

    async def _broadcast_user_list(self):
        payload = json.dumps({
            "type": "user_list",
            "channels": {str(k): v for k, v in self._channel_users().items()},
        })
        await asyncio.gather(
            *(self._send_text(ws, payload) for ws in self.clients),
            return_exceptions=True,
        )

    # -- safe senders --------------------------------------------------------

    @staticmethod
    async def _send_text(ws, text: str):
        try:
            await ws.send(text)
        except Exception:
            pass

    @staticmethod
    async def _send_bytes(ws, data: bytes):
        try:
            await ws.send(data)
        except Exception:
            pass

    # -- connection handler --------------------------------------------------

    async def handle(self, ws):
        self.clients[ws] = {"name": "Unknown", "channel_id": None}
        addr = ws.remote_address
        log.info("+ connect  %s", addr)

        try:
            # Send the channel list so the client knows the names.
            await ws.send(json.dumps({
                "type": "welcome",
                "channels": CHANNELS,
            }))

            async for msg in ws:
                if isinstance(msg, str):
                    await self._on_control(ws, json.loads(msg))
                elif isinstance(msg, bytes):
                    await self._on_audio(ws, msg)

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            name = self.clients.pop(ws, {}).get("name", "?")
            log.info("- disconnect  %s  (%s)", name, addr)
            await self._broadcast_user_list()

    # -- control messages ----------------------------------------------------

    async def _on_control(self, ws, data: dict):
        t = data.get("type")

        if t == "set_name":
            self.clients[ws]["name"] = data["name"]
            log.info("  name  %s", data["name"])

        elif t == "join_channel":
            cid = data["channel_id"]
            if 0 <= cid < len(CHANNELS):
                self.clients[ws]["channel_id"] = cid
                log.info("  join  %s -> %s", self.clients[ws]["name"], CHANNELS[cid])
                await self._broadcast_user_list()

        elif t == "leave_channel":
            info = self.clients[ws]
            old = info["channel_id"]
            info["channel_id"] = None
            if old is not None:
                log.info("  leave  %s <- %s", info["name"], CHANNELS[old])
            await self._broadcast_user_list()

    # -- audio forwarding ----------------------------------------------------

    async def _on_audio(self, ws, audio: bytes):
        sender = self.clients.get(ws)
        if not sender or sender["channel_id"] is None:
            return

        cid = sender["channel_id"]
        targets = [
            c for c, info in self.clients.items()
            if c is not ws and info["channel_id"] == cid
        ]
        if targets:
            await asyncio.gather(
                *(self._send_bytes(t, audio) for t in targets),
                return_exceptions=True,
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def main():
    server = VoiceServer()
    local_ip = get_local_ip()

    log.info("=" * 52)
    log.info("  Voice Chat Server")
    log.info("=" * 52)
    log.info("  Listening on     : %s:%d", HOST, PORT)
    log.info("  LAN address      : %s:%d", local_ip, PORT)
    log.info("  Channels         : %s", ", ".join(CHANNELS))
    log.info("=" * 52)

    async with websockets.serve(
        server.handle,
        HOST,
        PORT,
        max_size=2 ** 20,       # 1 MiB max frame
        ping_interval=20,
        ping_timeout=60,
    ):
        await asyncio.Future()  # block forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Server stopped.")
