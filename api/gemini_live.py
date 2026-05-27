"""
gemini_live.py — Real-time Gemini Live Voice Pipeline (No Local/Server VAD)

Flow:
  Browser streams raw PCM16 @ 16 kHz chunks continuously over WebSocket →
  Server forwards chunks instantly to a persistent Gemini Live Session →
  Gemini Live handles automatic VAD server-side                       →
  Streams Gemini PCM24k audio chunks back to browser in real-time
"""

import asyncio
import os

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# ── Gemini config ─────────────────────────────────────────────────────────────
MODEL="gemini-3.1-flash-live-preview"

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
        "Keep responses short, natural, and conversational. Never use markdown or lists."
    ),
)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── WebSocket endpoint ────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("[ws] client connected")

    async with client.aio.live.connect(model=MODEL, config=LIVE_CONFIG) as session:

        async def receive_from_gemini():
            try:
                while True:
                    async for response in session.receive():
                        content = response.server_content
                        if not content:
                            continue
                        
                        if content.interrupted:
                            print("[gemini] interrupted")
                            await ws.send_text("__interrupted__")
                            continue
                            
                        if content.model_turn:
                            for part in content.model_turn.parts:
                                inline = getattr(part, "inline_data", None)
                                if inline and inline.mime_type.startswith("audio/pcm"):
                                    await ws.send_bytes(inline.data)
                                    
                        if content.turn_complete:
                            print("[gemini] turn_complete")
                            await ws.send_text("__done__")
                            
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"[gemini task] error: {e}")

        gemini_task = asyncio.create_task(receive_from_gemini())

        try:
            while True:
                data = await ws.receive()
                if data.get("type") == "websocket.disconnect":
                    print("[ws] client disconnected")
                    break
                
                chunk = data.get("bytes")
                if not chunk:
                    continue

                # Stream the incoming mic chunk directly to the Gemini Live session
                await session.send_realtime_input(
                    audio=types.Blob(
                        data=chunk,
                        mime_type="audio/pcm;rate=16000"
                    )
                )

        except WebSocketDisconnect:
            print("[ws] client disconnected")
        except Exception as e:
            print(f"[ws] error: {e}")
        finally:
            gemini_task.cancel()
            await asyncio.gather(gemini_task, return_exceptions=True)
            print("[ws] cleanup completed")