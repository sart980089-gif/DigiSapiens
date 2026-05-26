"""
llm_tts.py — LLM + TTS voice pipeline with server-side VAD over WebSocket

Flow:
  Browser streams raw PCM16 @ 16 kHz chunks over WebSocket →
  Server VAD detects speech segment                         →
  Audio → Gemini (LLM, audio-in / text-out)                →
  Text → TTS server                                         →
  WAV fragments streamed back over WebSocket                →
  Browser plays gaplessly

Latency wins:
  • No upload round-trip — LLM fires the instant VAD closes the gate
  • TTS WAV fragments forwarded as they arrive (no re-buffering)
  • Text-only conversation history (no audio blobs in context window)
"""

import asyncio
import base64
import json
import os
import re
import tempfile
import time
from collections import deque

import requests
import webrtcvad
import wave
import io

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect

from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# ── Config ────────────────────────────────────────────────────────────────────
TTS_SERVER = os.getenv("TTS_SERVER", "http://127.0.0.1:7000")

llm = ChatOpenRouter(model="google/gemini-2.5-flash")

SYSTEM_PROMPT = """
You are Lisa, a friendly voice-first AI assistant.

CRITICAL IDENTITY RULES:
- Your name is Lisa. You are NOT Gemini, NOT Google Assistant, NOT any other AI.
- If asked who you are: say you are Lisa.
- Never reveal the underlying model or technology stack.

You will receive a voice message from the user.

Your task:
1. Transcribe what the user said.
2. Respond conversationally as Lisa.

You MUST respond ONLY with a valid JSON object. No preamble, no markdown, no code fences.

Required JSON schema:
{"transcript":"<exact words the user said>","response":"<your spoken reply>"}

Strict rules for the response field:
- Plain spoken English only
- No markdown, no emojis, no special characters
- No code blocks, no bullet points, no numbered lists, no line breaks
- Keep replies short and conversational (1-3 sentences max)
- Never start your reply with "I" as the very first word
"""

MAX_HISTORY = 12

# ── VAD config ────────────────────────────────────────────────────────────────
SAMPLE_RATE     = 16_000
FRAME_MS        = 30
FRAME_BYTES     = int(SAMPLE_RATE * FRAME_MS / 1000) * 2
VAD_MODE        = 3
START_FRAMES    = 4
END_FRAMES      = 30
PRE_ROLL_FRAMES = 8
MIN_SPEECH_SECS = 0.4

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="voice-pipeline-ws")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── VAD ───────────────────────────────────────────────────────────────────────
class VadProcessor:
    def __init__(self):
        self._vad         = webrtcvad.Vad(VAD_MODE)
        self._raw_buf     = bytearray()
        self._pre_roll    = deque(maxlen=PRE_ROLL_FRAMES)
        self._speech_buf  = bytearray()
        self._in_speech   = False
        self._speech_cnt  = 0
        self._silence_cnt = 0

    def feed(self, chunk: bytes) -> bytes | None:
        self._raw_buf.extend(chunk)
        utterance = None
        while len(self._raw_buf) >= FRAME_BYTES:
            frame = bytes(self._raw_buf[:FRAME_BYTES])
            del self._raw_buf[:FRAME_BYTES]
            u = self._process_frame(frame)
            if u is not None:
                utterance = u
        return utterance

    def _process_frame(self, frame: bytes) -> bytes | None:
        try:
            is_speech = self._vad.is_speech(frame, SAMPLE_RATE)
        except Exception:
            is_speech = False

        if is_speech:
            self._speech_cnt  += 1
            self._silence_cnt  = 0
        else:
            self._silence_cnt += 1
            self._speech_cnt   = 0

        if not self._in_speech:
            self._pre_roll.append(frame)
            if self._speech_cnt >= START_FRAMES:
                self._in_speech = True
                for f in self._pre_roll:
                    self._speech_buf.extend(f)
                self._pre_roll.clear()
                print("[vad] START")
        else:
            self._speech_buf.extend(frame)
            if self._silence_cnt >= END_FRAMES:
                print("[vad] END")
                utterance = bytes(self._speech_buf)
                self._speech_buf.clear()
                self._in_speech   = False
                self._speech_cnt  = 0
                self._silence_cnt = 0
                duration = len(utterance) / 2 / SAMPLE_RATE
                if duration < MIN_SPEECH_SECS:
                    print(f"[vad] discard — too short ({duration:.2f}s)")
                    return None
                return utterance
        return None


# ── Helpers ───────────────────────────────────────────────────────────────────
def pcm_to_webm_file(pcm_bytes: bytes) -> str:
    """Write raw PCM16 to a temp WAV file (LLM accepts audio/wav)."""
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(tf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)
    tf.close()
    return tf.name


def extract_json(raw: str) -> dict | None:
    try:
        return json.loads(raw.strip())
    except Exception:
        pass
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    m = re.search(r"\{.*?\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    t = re.search(r'"transcript"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    r = re.search(r'"response"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    if r:
        return {"transcript": t.group(1) if t else "", "response": r.group(1)}
    return None


def extract_wav_fragments(buf: bytearray) -> list[bytes]:
    fragments = []
    while True:
        if len(buf) < 12:
            break
        pos = buf.find(b"RIFF")
        if pos == -1:
            buf.clear()
            break
        if pos > 0:
            del buf[:pos]
        if len(buf) < 8:
            break
        size = int.from_bytes(buf[4:8], "little") + 8
        if len(buf) < size:
            break
        fragments.append(bytes(buf[:size]))
        del buf[:size]
    return fragments


# ── LLM call ─────────────────────────────────────────────────────────────────
IDENTITY_LEAKS = [
    "large language model", "trained by google",
    "i am gemini", "i'm gemini", "google ai",
]

def invoke_llm(wav_path: str, history: list) -> tuple[str, str]:
    audio_b64 = base64.b64encode(open(wav_path, "rb").read()).decode()

    user_msg = HumanMessage(content=[
        {"type": "text", "text": "Process this voice input and respond as Lisa."},
        {"type": "audio", "base64": audio_b64, "mime_type": "audio/wav"},
    ])

    messages = [SystemMessage(content=SYSTEM_PROMPT)] + history + [user_msg]
    raw = str(llm.invoke(messages).content).strip()
    print(f"[llm] raw: {raw[:120]}")

    parsed = extract_json(raw)
    if parsed:
        transcript = str(parsed.get("transcript", "")).strip()
        response   = str(parsed.get("response", "")).strip()
    else:
        transcript, response = "", ""

    if any(leak in response.lower() for leak in IDENTITY_LEAKS):
        response = "My name is Lisa, your AI assistant. How can I help you?"

    if not response:
        response = "Sorry, something went wrong. Could you try again?"

    return transcript, response


# ── Pipeline ──────────────────────────────────────────────────────────────────
async def run_pipeline(pcm: bytes, ws: WebSocket, history: list) -> tuple[str, str]:
    """LLM → TTS → stream WAV fragments over WebSocket."""
    t0 = time.time()

    wav_path = pcm_to_webm_file(pcm)
    try:
        loop = asyncio.get_event_loop()
        transcript, text = await loop.run_in_executor(
            None, invoke_llm, wav_path, list(history)
        )
        print(f"[llm] {time.time()-t0:.2f}s  user='{transcript}'  lisa='{text}'")

        # Send metadata first so UI can show transcript
        await ws.send_text(json.dumps({
            "type": "meta",
            "transcript": transcript,
            "response": text,
            "llm_ms": int((time.time() - t0) * 1000),
        }))

        tts_resp = await loop.run_in_executor(
            None,
            lambda: requests.post(
                f"{TTS_SERVER}/tts",
                json={"model": "lisa", "reference_id": 6, "text": text},
                stream=True,
                timeout=30,
            )
        )

        if tts_resp.status_code != 200:
            await ws.send_text(json.dumps({"type": "error", "msg": "tts failed"}))
            return transcript, text

        byte_buf = bytearray()
        first = True

        def iter_tts():
            return tts_resp.iter_content(chunk_size=4096)

        for chunk in tts_resp.iter_content(chunk_size=4096):
            if not chunk:
                continue
            byte_buf.extend(chunk)
            for frag in extract_wav_fragments(byte_buf):
                if first:
                    print(f"[tts] first audio {time.time()-t0:.2f}s")
                    first = False
                # send as binary frame
                await ws.send_bytes(frag)

        await ws.send_text(json.dumps({"type": "done"}))
        return transcript, text

    finally:
        try:
            os.remove(wav_path)
        except Exception:
            pass


# ── WebSocket endpoint ────────────────────────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    print("[ws] connected")

    vad = VadProcessor()
    history: list = []
    queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=4)

    async def pipeline_worker():
        while True:
            pcm = await queue.get()
            try:
                transcript, response = await run_pipeline(pcm, ws, history)
                # update text-only history
                if transcript:
                    history.append(HumanMessage(content=transcript))
                history.append(AIMessage(content=response))
                if len(history) > MAX_HISTORY:
                    del history[:-MAX_HISTORY]
            except Exception as e:
                print(f"[pipeline] error: {e}")
            finally:
                queue.task_done()

    worker = asyncio.create_task(pipeline_worker())

    try:
        while True:
            msg = await ws.receive()
            if "bytes" in msg and msg["bytes"]:
                utterance = vad.feed(msg["bytes"])
                if utterance:
                    try:
                        queue.put_nowait(utterance)
                    except asyncio.QueueFull:
                        print("[pipeline] queue full — dropping utterance")
            elif "text" in msg:
                cmd = msg["text"]
                if cmd == "__ping__":
                    await ws.send_text("__pong__")
    except WebSocketDisconnect:
        print("[ws] disconnected")
    except Exception as e:
        print(f"[ws] error: {e}")
    finally:
        worker.cancel()