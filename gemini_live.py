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
MODEL = "gemini-3.1-flash-live-preview"
GEMINI_TIMEOUT = 20  # seconds before we give up on a silent Gemini session

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


async def stream_gemini_to_ws(audio_bytes: bytes, ws: WebSocket):
    """
    Open a Gemini Live session, send the user's audio, and forward each
    audio chunk to the browser the moment it arrives — no buffering.

    Gemini streams PCM chunks as fast as it generates them (faster than
    real-time). We forward immediately so the browser can start playing
    before generation is even done.

    We stop on:
      - generationComplete  → model finished generating (earliest signal)
      - turn_complete       → fallback if generationComplete never fires
      - interrupted         → user barged in
    """
    async with client.aio.live.connect(model=MODEL, config=LIVE_CONFIG) as session:

        # Send the full user utterance and mark end of input
        await session.send_realtime_input(
            audio=types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000")
        )
        await session.send_realtime_input(audio_stream_end=True)

        chunk_count = 0

        async for response in session.receive():
            content = response.server_content
            if not content:
                continue

            # Interrupted — stop immediately; browser will flush its queue
            if content.interrupted:
                print("Gemini interrupted")
                break

            # Stream each audio part straight to the browser
            if content.model_turn:
                for part in content.model_turn.parts:
                    data = getattr(part, "inline_data", None)
                    if data and data.mime_type.startswith("audio/pcm"):
                        await ws.send_bytes(data.data)
                        chunk_count += 1

            # generationComplete fires as soon as the model stops generating,
            # before the estimated playback delay that precedes turn_complete.
            # Breaking here gives us the lowest latency for the next user turn.
            if getattr(content, "generation_complete", False):
                print(f"Generation complete — {chunk_count} chunk(s) streamed")
                break

            if content.turn_complete:
                print(f"Turn complete — {chunk_count} chunk(s) streamed")
                break


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("Client connected")

    try:
        while True:
            audio_bytes = await ws.receive_bytes()
            print(f"Received audio: {len(audio_bytes)} bytes")

            try:
                await asyncio.wait_for(
                    stream_gemini_to_ws(audio_bytes, ws),
                    timeout=GEMINI_TIMEOUT,
                )
            except asyncio.TimeoutError:
                print(f"ERROR: Gemini did not respond within {GEMINI_TIMEOUT}s")
            except Exception as e:
                print(f"ERROR: Gemini session failed: {e}")

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"ERROR: {e}")