#!/usr/bin/env python3
"""
Voice Chat Server (WebRTC SFU)
==============================
WebSocket for signaling. WebRTC DataChannels for audio transport.
The server acts as a Selective Forwarding Unit: receives audio from
each client's DataChannel and forwards it to every other client in
the same voice channel.  All I/O is non-blocking (asyncio).

Run:
    python server.py
"""

import asyncio
import json
import logging
import socket
import uuid

from aiortc import RTCPeerConnection, RTCSessionDescription
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
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ---------------------------------------------------------------------------
# Client record
# ---------------------------------------------------------------------------
class Client:
    __slots__ = ("ws", "name", "channel_id", "pc", "dc", "uid")

    def __init__(self, ws):
        self.ws = ws
        self.name: str = "Unknown"
        self.channel_id: int | None = None
        self.pc: RTCPeerConnection | None = None
        self.dc = None          # RTCDataChannel
        self.uid: str = uuid.uuid4().hex[:8]


# ---------------------------------------------------------------------------
# VoiceServer
# ---------------------------------------------------------------------------
class VoiceServer:
    def __init__(self):
        self.clients: dict = {}          # ws → Client

    # -- channel helpers ----------------------------------------------------

    def _channel_users(self) -> dict[int, list[str]]:
        channels: dict[int, list[str]] = {i: [] for i in range(len(CHANNELS))}
        for c in self.clients.values():
            if c.channel_id is not None:
                channels[c.channel_id].append(c.name)
        return channels

    async def _broadcast_user_list(self):
        payload = json.dumps({
            "type": "user_list",
            "channels": {str(k): v for k, v in self._channel_users().items()},
        })
        await asyncio.gather(
            *(self._safe_send(c.ws, payload) for c in self.clients.values()),
            return_exceptions=True,
        )

    @staticmethod
    async def _safe_send(ws, text: str):
        try:
            await ws.send(text)
        except Exception:
            pass

    # -- connection handler -------------------------------------------------

    async def handle(self, ws):
        client = Client(ws)
        self.clients[ws] = client
        log.info("+ connect  %s", ws.remote_address)

        try:
            await ws.send(json.dumps({
                "type": "welcome",
                "channels": CHANNELS,
            }))

            async for msg in ws:
                if isinstance(msg, str):
                    await self._on_control(client, json.loads(msg))

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            name = client.name
            if client.pc:
                try:
                    await client.pc.close()
                except Exception:
                    pass
            self.clients.pop(ws, None)
            log.info("- disconnect  %s  (%s)", name, ws.remote_address)
            await self._broadcast_user_list()

    # -- control messages ---------------------------------------------------

    async def _on_control(self, client: Client, data: dict):
        t = data.get("type")

        if t == "set_name":
            client.name = data["name"]
            log.info("  name  %s", data["name"])

        elif t == "join_channel":
            cid = data["channel_id"]
            if 0 <= cid < len(CHANNELS):
                client.channel_id = cid
                log.info("  join  %s -> %s", client.name, CHANNELS[cid])
                await self._broadcast_user_list()

        elif t == "leave_channel":
            old = client.channel_id
            client.channel_id = None
            if old is not None:
                log.info("  leave  %s <- %s", client.name, CHANNELS[old])
            await self._broadcast_user_list()

        elif t == "offer":
            await self._handle_offer(client, data)

    # -- WebRTC offer / answer ----------------------------------------------

    async def _handle_offer(self, client: Client, data: dict):
        """Create a PeerConnection for *client* and answer their SDP offer."""

        # Tear down any previous connection
        if client.pc:
            try:
                await client.pc.close()
            except Exception:
                pass
            client.pc = None
            client.dc = None

        pc = RTCPeerConnection()
        client.pc = pc

        @pc.on("datachannel")
        def on_datachannel(channel):
            client.dc = channel
            log.info("  datachannel open  %s", client.name)

            @channel.on("message")
            def on_message(message):
                """Forward audio bytes to every other client in the channel.

                Each forwarded frame is prefixed with the sender's 8-byte
                UID so receivers can keep per-source jitter buffers.
                """
                if not isinstance(message, bytes):
                    return
                if client.channel_id is None:
                    return
                cid = client.channel_id
                tagged = client.uid.encode("ascii")[:8].ljust(8, b"\x00") + message
                for other in list(self.clients.values()):
                    if (other is not client
                            and other.channel_id == cid
                            and other.dc is not None
                            and other.dc.readyState == "open"):
                        try:
                            other.dc.send(tagged)
                        except Exception:
                            pass

        @pc.on("connectionstatechange")
        async def on_conn_state():
            state = pc.connectionState
            log.info("  rtc %s  (%s)", state, client.name)
            if state in ("failed", "closed"):
                if client.pc is pc:
                    client.pc = None
                    client.dc = None

        # SDP handshake
        offer = RTCSessionDescription(sdp=data["sdp"], type=data["sdp_type"])
        await pc.setRemoteDescription(offer)

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        # Wait for ICE gathering to complete so the answer carries all candidates
        if pc.iceGatheringState != "complete":
            done = asyncio.Event()

            @pc.on("icegatheringstatechange")
            def _on_ice():
                if pc.iceGatheringState == "complete":
                    done.set()

            await asyncio.wait_for(done.wait(), timeout=30)

        await self._safe_send(client.ws, json.dumps({
            "type": "answer",
            "sdp": pc.localDescription.sdp,
            "sdp_type": pc.localDescription.type,
        }))
        log.info("  answer sent -> %s", client.name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def main():
    server = VoiceServer()
    local_ip = get_local_ip()

    log.info("=" * 52)
    log.info("  Voice Chat Server  (WebRTC)")
    log.info("=" * 52)
    log.info("  Listening on     : %s:%d", HOST, PORT)
    log.info("  LAN address      : %s:%d", local_ip, PORT)
    log.info("  Channels         : %s", ", ".join(CHANNELS))
    log.info("=" * 52)

    async with websockets.serve(
        server.handle,
        HOST,
        PORT,
        max_size=2 ** 20,
        ping_interval=30,
        ping_timeout=10,
    ):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Server stopped.")
    except OSError as e:
        if getattr(e, "errno", None) == 10048 or "already in use" in str(e).lower():
            log.error(
                "Port %d is already in use. "
                "Kill the old server process or pick a different port.", PORT,
            )
        else:
            raise
