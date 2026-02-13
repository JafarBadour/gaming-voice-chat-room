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
import time
import traceback
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

# WebSocket keepalive — generous timeouts so gaming CPU spikes don't
# cause spurious disconnections.
WS_PING_INTERVAL = 30      # send a ping every 30 s
WS_PING_TIMEOUT  = 30      # allow up to 30 s for the pong reply

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  [%(levelname)s]  %(name)-10s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("server")

# Quiet down noisy third-party loggers so our logs stay readable
for _quiet in ("aioice", "aiortc", "aiortc.rtcpeerconnection",
               "aiortc.rtcdtlstransport", "aiortc.rtcicetransport",
               "websockets"):
    logging.getLogger(_quiet).setLevel(logging.WARNING)


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
SPEAK_TIMEOUT = 0.30        # seconds — consider silent after no audio for this long
SPEAK_BROADCAST_INTERVAL = 0.15  # how often we broadcast speaking state changes


class Client:
    __slots__ = ("ws", "name", "channel_id", "pc", "dc", "uid",
                 "connected_at", "last_ws_msg", "dc_send_errors",
                 "last_audio_time", "is_speaking")

    def __init__(self, ws):
        self.ws = ws
        self.name: str = "Unknown"
        self.channel_id: int | None = None
        self.pc: RTCPeerConnection | None = None
        self.dc = None          # RTCDataChannel
        self.uid: str = uuid.uuid4().hex[:8]
        self.connected_at: float = time.time()
        self.last_ws_msg: float = time.time()
        self.dc_send_errors: int = 0
        self.last_audio_time: float = 0.0
        self.is_speaking: bool = False


# ---------------------------------------------------------------------------
# VoiceServer
# ---------------------------------------------------------------------------
class VoiceServer:
    def __init__(self):
        self.clients: dict = {}          # ws → Client
        self._speak_task: asyncio.Task | None = None

    # -- speaking indicator -------------------------------------------------

    async def _speaking_loop(self):
        """Periodically check who is speaking and broadcast changes."""
        try:
            while True:
                await asyncio.sleep(SPEAK_BROADCAST_INTERVAL)
                now = time.time()
                changed = False

                for c in list(self.clients.values()):
                    was = c.is_speaking
                    c.is_speaking = (now - c.last_audio_time) < SPEAK_TIMEOUT
                    if c.is_speaking != was:
                        changed = True

                if changed:
                    await self._broadcast_speaking()
        except asyncio.CancelledError:
            pass

    async def _broadcast_speaking(self):
        """Send the set of currently-speaking names per channel."""
        speakers: dict[int, list[str]] = {}
        for c in self.clients.values():
            if c.is_speaking and c.channel_id is not None:
                speakers.setdefault(c.channel_id, []).append(c.name)

        payload = json.dumps({
            "type": "speaking",
            "speakers": {str(k): v for k, v in speakers.items()},
        })
        await asyncio.gather(
            *(self._safe_send(c.ws, payload) for c in self.clients.values()),
            return_exceptions=True,
        )

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
        except websockets.exceptions.ConnectionClosed as e:
            log.debug("safe_send failed (connection closed: code=%s reason=%s) %s",
                       e.code, e.reason, ws.remote_address)
        except Exception as e:
            log.warning("safe_send failed: %s %s", type(e).__name__, e)

    # -- connection handler -------------------------------------------------

    async def handle(self, ws):
        client = Client(ws)
        self.clients[ws] = client
        log.info("+ CONNECT  %s  (total clients: %d)",
                 ws.remote_address, len(self.clients))

        disconnect_reason = "unknown"
        try:
            await ws.send(json.dumps({
                "type": "welcome",
                "channels": CHANNELS,
            }))

            async for msg in ws:
                client.last_ws_msg = time.time()
                if isinstance(msg, str):
                    try:
                        await self._on_control(client, json.loads(msg))
                    except Exception:
                        log.error("Error handling control message from %s: %s\n%s",
                                  client.name, msg[:200], traceback.format_exc())

        except websockets.exceptions.ConnectionClosedOK as e:
            disconnect_reason = f"clean close (code={e.code})"
        except websockets.exceptions.ConnectionClosedError as e:
            disconnect_reason = f"connection lost (code={e.code} reason={e.reason!r})"
        except websockets.exceptions.ConnectionClosed as e:
            disconnect_reason = f"connection closed (code={e.code} reason={e.reason!r})"
        except asyncio.CancelledError:
            disconnect_reason = "task cancelled"
        except Exception:
            disconnect_reason = f"unexpected error: {traceback.format_exc()}"
        finally:
            name = client.name
            uptime = time.time() - client.connected_at
            if client.pc:
                try:
                    await client.pc.close()
                except Exception as e:
                    log.debug("Error closing PC for %s: %s", name, e)
            self.clients.pop(ws, None)
            log.info("- DISCONNECT  %s  (%s)  reason=%s  "
                     "uptime=%.1fs  dc_send_errors=%d  remaining_clients=%d",
                     name, ws.remote_address, disconnect_reason,
                     uptime, client.dc_send_errors, len(self.clients))
            await self._broadcast_user_list()

    # -- control messages ---------------------------------------------------

    async def _on_control(self, client: Client, data: dict):
        t = data.get("type")

        if t == "set_name":
            client.name = data["name"]
            log.info("  name  %s  (%s)", data["name"], client.ws.remote_address)

        elif t == "join_channel":
            cid = data["channel_id"]
            if 0 <= cid < len(CHANNELS):
                client.channel_id = cid
                log.info("  join  %s -> %s", client.name, CHANNELS[cid])
                await self._broadcast_user_list()
            else:
                log.warning("  invalid channel_id %s from %s", cid, client.name)

        elif t == "leave_channel":
            old = client.channel_id
            client.channel_id = None
            if old is not None:
                log.info("  leave  %s <- %s", client.name, CHANNELS[old])
            await self._broadcast_user_list()

        elif t == "offer":
            await self._handle_offer(client, data)

        else:
            log.warning("  unknown message type %r from %s", t, client.name)

    # -- WebRTC offer / answer ----------------------------------------------

    async def _handle_offer(self, client: Client, data: dict):
        """Create a PeerConnection for *client* and answer their SDP offer."""

        # Tear down any previous connection
        if client.pc:
            log.info("  tearing down old PeerConnection for %s", client.name)
            try:
                await client.pc.close()
            except Exception as e:
                log.debug("  error closing old PC for %s: %s", client.name, e)
            client.pc = None
            client.dc = None

        pc = RTCPeerConnection()
        client.pc = pc
        log.info("  new PeerConnection for %s", client.name)

        @pc.on("datachannel")
        def on_datachannel(channel):
            client.dc = channel
            log.info("  datachannel OPEN  %s  (label=%s)", client.name, channel.label)

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
                client.last_audio_time = time.time()
                cid = client.channel_id
                tagged = client.uid.encode("ascii")[:8].ljust(8, b"\x00") + message
                for other in list(self.clients.values()):
                    if (other is not client
                            and other.channel_id == cid
                            and other.dc is not None
                            and other.dc.readyState == "open"):
                        try:
                            other.dc.send(tagged)
                        except Exception as e:
                            other.dc_send_errors += 1
                            if other.dc_send_errors <= 5 or other.dc_send_errors % 100 == 0:
                                log.warning("  dc.send failed -> %s: %s (total errors: %d)",
                                            other.name, e, other.dc_send_errors)

            @channel.on("close")
            def on_dc_close():
                log.info("  datachannel CLOSED  %s", client.name)

        @pc.on("connectionstatechange")
        async def on_conn_state():
            state = pc.connectionState
            log.info("  RTC state: %-12s  (%s)", state, client.name)
            if state in ("failed", "closed", "disconnected"):
                if state == "failed":
                    log.warning("  WebRTC connection FAILED for %s — "
                                "likely network issue or ICE failure", client.name)
                if client.pc is pc:
                    client.pc = None
                    client.dc = None

        @pc.on("iceconnectionstatechange")
        async def on_ice_state():
            log.debug("  ICE state: %-12s  (%s)", pc.iceConnectionState, client.name)

        # SDP handshake
        try:
            offer = RTCSessionDescription(sdp=data["sdp"], type=data["sdp_type"])
            await pc.setRemoteDescription(offer)

            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
        except Exception:
            log.error("  SDP handshake failed for %s:\n%s",
                      client.name, traceback.format_exc())
            return

        # Wait for ICE gathering to complete so the answer carries all candidates
        if pc.iceGatheringState != "complete":
            done = asyncio.Event()

            @pc.on("icegatheringstatechange")
            def _on_ice():
                log.debug("  ICE gathering: %s  (%s)", pc.iceGatheringState, client.name)
                if pc.iceGatheringState == "complete":
                    done.set()

            try:
                await asyncio.wait_for(done.wait(), timeout=30)
            except asyncio.TimeoutError:
                log.error("  ICE gathering TIMED OUT for %s after 30s", client.name)
                return

        await self._safe_send(client.ws, json.dumps({
            "type": "answer",
            "sdp": pc.localDescription.sdp,
            "sdp_type": pc.localDescription.type,
        }))
        log.info("  SDP answer sent -> %s", client.name)


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

    log.info("  Ping interval    : %ds", WS_PING_INTERVAL)
    log.info("  Ping timeout     : %ds", WS_PING_TIMEOUT)
    log.info("=" * 52)

    async with websockets.serve(
        server.handle,
        HOST,
        PORT,
        max_size=2 ** 20,
        ping_interval=WS_PING_INTERVAL,
        ping_timeout=WS_PING_TIMEOUT,
    ):
        server._speak_task = asyncio.create_task(server._speaking_loop())
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
