import os,asyncio
from fastapi import FastAPI,WebSocket
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect

from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()

MODEL="gemini-3.1-flash-live-preview"

client=genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

LIVE_CONFIG=types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                voice_name="Aoede"
            )
        )
    ),
    system_instruction=(
        "You are a realtime voice assistant. "
        "Reply naturally and briefly."
    )
)

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.websocket("/ws")
async def ws(ws: WebSocket):

    await ws.accept()

    print("[ws] connected")

    # Establish the stateful session for the lifetime of this connection
    async with client.aio.live.connect(model=MODEL, config=LIVE_CONFIG) as session:

        async def receive_from_gemini():
            try:
                chunks = 0
                while True:
                    async for r in session.receive():
                        c = r.server_content
                        if not c:
                            continue

                        if c.interrupted:
                            print("[gemini] interrupted")
                            await ws.send_text("__interrupted__")
                            continue

                        if c.model_turn:
                            for p in c.model_turn.parts:
                                inline = getattr(p, "inline_data", None)
                                if inline and inline.mime_type.startswith("audio/pcm"):
                                    chunks += 1
                                    await ws.send_bytes(inline.data)

                        if c.turn_complete:
                            print(f"[gemini] done chunks={chunks}")
                            await ws.send_text("__done__")
                            chunks = 0

            except asyncio.CancelledError:
                pass
            except Exception as e:
                print("[gemini task] ERROR:", e)

        gemini_task = asyncio.create_task(receive_from_gemini())

        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    print("[ws] disconnected")
                    break
                data = msg.get("bytes")
                if not data:
                    continue

                print(f"[gemini] forwarding {len(data)} bytes of user speech")
                await ws.send_text("__thinking__")

                await session.send_realtime_input(
                    audio=types.Blob(
                        data=data,
                        mime_type="audio/pcm;rate=16000"
                    )
                )

                await session.send_realtime_input(
                    audio_stream_end=True
                )

        except WebSocketDisconnect:
            print("[ws] disconnected")
        except Exception as e:
            print("[ws] ERROR:", e)
        finally:
            # Clean up the background reader task
            gemini_task.cancel()
            await asyncio.gather(gemini_task, return_exceptions=True)
            print("[ws] cleanup")