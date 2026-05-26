"""
gemini_live.py — Gemini Live voice pipeline with server-side VAD

Flow:
  Browser streams raw PCM16 @ 16 kHz chunks over WebSocket  →
  Server runs WebRTC VAD frame-by-frame                      →
  On speech-end, sends accumulated audio to Gemini Live       →
  Streams Gemini PCM24k audio chunks back to browser

Latency wins vs. hold-to-talk upload:
  • No client-side buffering / upload step
  • VAD fires the moment silence is detected — Gemini call starts instantly
  • Gemini audio chunks forwarded as fast as they arrive (zero buffering)
"""

import asyncio
import os
import struct
from collections import deque

import numpy as np
import webrtcvad
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# ── Gemini config ─────────────────────────────────────────────────────────────
MODEL = "gemini-3.1-flash-live-preview"
GEMINI_TIMEOUT = 20

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

LIVE_CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede")
        )
    ),
    system_instruction=(
        "You are a helpful, friendly realtime voice assistant. "
        "Keep responses short and natural. Never use markdown or lists."
    ),
)

# ── VAD config ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000
FRAME_MS = 30                  # webrtcvad only accepts 10 / 20 / 30 ms frames
FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000) * 2  # int16 → 2 bytes/sample
VAD_MODE = 3                   # 0 = least aggressive, 3 = most

START_SPEECH_FRAMES = 5        # consecutive speech frames to trigger start
END_SILENCE_FRAMES = 35        # consecutive silence frames to trigger end
HANGOVER_FRAMES = 3            # forgive up to N non-speech frames inside speech
ENERGY_THRESHOLD = 0.06        # initial RMS threshold (adapts over time)
MIN_SPEECH_SECONDS = 0.5       # discard segments shorter than this
PRE_ROLL_FRAMES = 10           # keep N frames before speech start (~300ms)


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Gemini streaming ──────────────────────────────────────────────────────────
async def stream_gemini_to_ws(audio_pcm: bytes, ws: WebSocket):
    """Send audio to Gemini Live and forward each PCM chunk to the browser."""
    async with client.aio.live.connect(model=MODEL, config=LIVE_CONFIG) as session:
        await session.send_realtime_input(
            audio=types.Blob(data=audio_pcm, mime_type="audio/pcm;rate=16000")
        )
        await session.send_realtime_input(audio_stream_end=True)

        chunks_sent = 0
        async for response in session.receive():
            content = response.server_content
            if not content:
                continue
            if content.interrupted:
                print("[gemini] interrupted")
                break
            if content.model_turn:
                for part in content.model_turn.parts:
                    data = getattr(part, "inline_data", None)
                    if data and data.mime_type.startswith("audio/pcm"):
                        await ws.send_bytes(data.data)
                        chunks_sent += 1
            if getattr(content, "generation_complete", False):
                print(f"[gemini] generation_complete — {chunks_sent} chunks")
                break
            if content.turn_complete:
                print(f"[gemini] turn_complete — {chunks_sent} chunks")
                break


# ── Server-side VAD processor ─────────────────────────────────────────────────
class VadProcessor:
    """
    Accumulates raw PCM16 bytes from the client, runs webrtcvad frame-by-frame,
    and calls `on_utterance(pcm_bytes)` whenever a complete speech segment ends.
    """

    def __init__(self, on_utterance):
        self._vad        = webrtcvad.Vad(VAD_MODE)
        self._on_utt     = on_utterance
        self._raw_buf    = bytearray()   # partial frame accumulator
        self._pre_roll   = deque(maxlen=PRE_ROLL_FRAMES)
        self._speech_buf = bytearray()   # confirmed speech frames
        self._in_speech  = False
        self._speech_cnt = 0
        self._silence_cnt= 0
        self._hangover   = 0

    def feed(self, chunk: bytes):
        """Feed arbitrary-length raw PCM16 bytes; returns utterance bytes or None."""
        self._raw_buf.extend(chunk)
        result = None

        while len(self._raw_buf) >= FRAME_BYTES:
            frame = bytes(self._raw_buf[:FRAME_BYTES])
            del self._raw_buf[:FRAME_BYTES]
            utterance = self._process_frame(frame)
            if utterance is not None:
                result = utterance   # return the last complete utterance

        return result

    def _process_frame(self, frame: bytes):
        frame_int16 = np.frombuffer(frame, dtype=np.int16)
        frame_float = frame_int16.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(frame_float ** 2)))

        try:
            is_speech = self._vad.is_speech(frame, SAMPLE_RATE)
        except Exception:
            is_speech = False

        speech_detected = is_speech and rms > ENERGY_THRESHOLD

        if speech_detected:
            self._speech_cnt += 1
            self._silence_cnt = 0
            self._hangover = HANGOVER_FRAMES
        else:
            if self._in_speech and self._hangover > 0:
                self._hangover -= 1
                self._speech_cnt += 1
            else:
                self._silence_cnt += 1
                self._speech_cnt = 0

        if not self._in_speech:
            self._pre_roll.append(frame)
            if self._speech_cnt >= START_SPEECH_FRAMES:
                self._in_speech = True
                # prepend pre-roll so we don't clip the onset
                for f in self._pre_roll:
                    self._speech_buf.extend(f)
                self._pre_roll.clear()
                print("[vad] speech START")
        else:
            self._speech_buf.extend(frame)

            if self._silence_cnt >= END_SILENCE_FRAMES:
                print("[vad] speech END")
                utterance = bytes(self._speech_buf)
                self._speech_buf.clear()
                self._in_speech   = False
                self._speech_cnt  = 0
                self._silence_cnt = 0
                self._hangover    = 0

                # discard misfires
                duration = len(utterance) / 2 / SAMPLE_RATE
                if duration < MIN_SPEECH_SECONDS:
                    print(f"[vad] discarded — too short ({duration:.2f}s)")
                    return None

                return utterance

        return None


# ── WebSocket endpoint ────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("[ws] client connected")

    # Queue so VAD callback can safely hand off to the async Gemini task
    utterance_queue: asyncio.Queue[bytes] = asyncio.Queue()

    def on_utterance(pcm: bytes):
        asyncio.get_event_loop().call_soon_threadsafe(
            utterance_queue.put_nowait, pcm
        )

    vad = VadProcessor(on_utterance=on_utterance)

    # Background task: drain the utterance queue and call Gemini
    async def gemini_worker():
        while True:
            pcm = await utterance_queue.get()
            print(f"[gemini] processing utterance {len(pcm)} bytes")
            try:
                await asyncio.wait_for(
                    stream_gemini_to_ws(pcm, ws),
                    timeout=GEMINI_TIMEOUT,
                )
            except asyncio.TimeoutError:
                print("[gemini] timeout")
            except Exception as e:
                print(f"[gemini] error: {e}")
            finally:
                utterance_queue.task_done()
            # Signal browser that Gemini is done so it can show "ready"
            try:
                await ws.send_text("__done__")
            except Exception:
                break

    worker = asyncio.create_task(gemini_worker())

    try:
        while True:
            data = await ws.receive()
            if "bytes" in data and data["bytes"]:
                utterance = vad.feed(data["bytes"])
                if utterance:
                    on_utterance(utterance)
            elif "text" in data and data["text"] == "__stop__":
                # Client can force-flush (e.g. push-to-talk mode)
                pass
    except WebSocketDisconnect:
        print("[ws] client disconnected")
    except Exception as e:
        print(f"[ws] error: {e}")
    finally:
        worker.cancel()