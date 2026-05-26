import asyncio,os
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
        "You are Lisa, a realtime voice assistant. "
        "Keep replies short and natural."
    )
)

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

async def event(ws,t):
    try:
        await ws.send_json({"type":t})
    except:
        pass

async def stream(audio,ws):

    async with client.aio.live.connect(
        model=MODEL,
        config=LIVE_CONFIG
    ) as s:

        await event(ws,"assistant_start")

        await s.send_realtime_input(
            audio=types.Blob(
                data=audio,
                mime_type="audio/pcm;rate=16000"
            )
        )

        await s.send_realtime_input(
            audio_stream_end=True
        )

        async for r in s.receive():

            c=r.server_content

            if not c:
                continue

            if c.interrupted:

                print("interrupted")

                await event(ws,"interrupted")

                break

            if c.model_turn:

                for p in c.model_turn.parts:

                    d=getattr(
                        p,
                        "inline_data",
                        None
                    )

                    if not d:
                        continue

                    if not d.mime_type.startswith(
                        "audio/pcm"
                    ):
                        continue

                    await ws.send_bytes(d.data)

            if getattr(
                c,
                "generation_complete",
                False
            ):

                await event(
                    ws,
                    "assistant_end"
                )

                break

            if c.turn_complete:

                await event(
                    ws,
                    "assistant_end"
                )

                break

@app.websocket("/ws")
async def ws_endpoint(ws:WebSocket):

    await ws.accept()

    print("connected")

    try:

        while True:

            m=await ws.receive()

            if "bytes" in m:

                try:

                    await asyncio.wait_for(
                        stream(
                            m["bytes"],
                            ws
                        ),
                        timeout=30
                    )

                except asyncio.TimeoutError:

                    await event(
                        ws,
                        "timeout"
                    )

                except Exception as e:

                    print(e)

                    await event(
                        ws,
                        "error"
                    )

    except WebSocketDisconnect:

        print("disconnected")

    except Exception as e:

        print(e)