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
async def ws(ws:WebSocket):

    await ws.accept()

    print("[ws] connected")

    active_task=None

    async def stop_generation():

        nonlocal active_task

        if active_task and not active_task.done():

            print("[gemini] cancelling old response")

            active_task.cancel()

            try:
                await asyncio.wait_for(
                    active_task,
                    timeout=2
                )
            except:
                pass

        active_task=None

    async def run_gemini(pcm:bytes):

        try:

            dur=len(pcm)/2/16000

            print(
                f"[gemini] request "
                f"{dur:.2f}s "
                f"{len(pcm)} bytes"
            )

            async with client.aio.live.connect(
                model=MODEL,
                config=LIVE_CONFIG
            ) as session:

                await session.send_realtime_input(
                    audio=types.Blob(
                        data=pcm,
                        mime_type="audio/pcm;rate=16000"
                    )
                )

                await session.send_realtime_input(
                    audio_stream_end=True
                )

                chunks=0

                async for r in session.receive():

                    c=r.server_content

                    if not c:
                        continue

                    if c.interrupted:

                        print("[gemini] interrupted")

                        await ws.send_text(
                            "__interrupted__"
                        )

                        continue

                    if c.model_turn:

                        for p in c.model_turn.parts:

                            inline=getattr(
                                p,
                                "inline_data",
                                None
                            )

                            if (
                                inline and
                                inline.mime_type.startswith(
                                    "audio/pcm"
                                )
                            ):

                                chunks+=1

                                await ws.send_bytes(
                                    inline.data
                                )

                    if c.turn_complete:

                        print(
                            f"[gemini] done "
                            f"chunks={chunks}"
                        )

                        await ws.send_text(
                            "__done__"
                        )

                        break

        except asyncio.CancelledError:

            print("[gemini] cancelled")

        except Exception as e:

            print("[gemini] ERROR:",e)

    try:

        while True:

            msg=await ws.receive()

            data=msg.get("bytes")

            if not data:
                continue

            await stop_generation()

            await ws.send_text(
                "__thinking__"
            )

            active_task=asyncio.create_task(
                run_gemini(data)
            )

    except WebSocketDisconnect:

        print("[ws] disconnected")

    except Exception as e:

        print("[ws] ERROR:",e)

    finally:

        await stop_generation()

        print("[ws] cleanup")