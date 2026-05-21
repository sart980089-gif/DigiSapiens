import asyncio
import os

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# --- Config ---
# gemini-2.0-flash-live-001 and gemini-live-2.5-flash-native-audio are deprecated.
# Use one of these two current Live API models:
#   "gemini-2.5-flash-native-audio-preview-12-2025"  — native audio output
#   "gemini-3.1-flash-live-preview"                   — latest, lower latency
MODEL = "gemini-3.1-flash-live-preview"

GEMINI_TIMEOUT = 20  # seconds to wait for Gemini before giving up

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

LIVE_CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede")
        )
    ),
    system_instruction="You are a realtime voice assistant. Keep responses short and natural.",
)

# --- App ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def query_gemini(audio_bytes: bytes) -> list[bytes]:
    """Send one audio turn to Gemini and collect all audio response chunks."""
    chunks = []

    async with client.aio.live.connect(model=MODEL, config=LIVE_CONFIG) as session:
        # audio/pcm;rate=16000 must match what the browser actually sends (16 kHz PCM)
        await session.send_realtime_input(
            audio=types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000")
        )
        await session.send_realtime_input(audio_stream_end=True)

        async for response in session.receive():
            content = response.server_content
            if not content or content.interrupted:
                continue

            # turn_complete signals end of this response
            if getattr(content, "turn_complete", False):
                break

            model_turn = content.model_turn
            if not model_turn:
                continue

            for part in model_turn.parts:
                data = getattr(part, "inline_data", None)
                if data and data.mime_type.startswith("audio/pcm"):
                    chunks.append(data.data)

    return chunks


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("Client connected")

    try:
        while True:
            audio_bytes = await ws.receive_bytes()
            print(f"Received audio: {len(audio_bytes)} bytes")

            try:
                # Hard timeout so we never hang if Gemini stalls
                chunks = await asyncio.wait_for(
                    query_gemini(audio_bytes),
                    timeout=GEMINI_TIMEOUT,
                )
            except asyncio.TimeoutError:
                print(f"ERROR: Gemini did not respond within {GEMINI_TIMEOUT}s")
                continue
            except Exception as e:
                print(f"ERROR: Gemini session failed: {e}")
                continue

            print(f"Response complete — {len(chunks)} audio chunk(s)")
            for chunk in chunks:
                await ws.send_bytes(chunk)

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"ERROR: {e}")