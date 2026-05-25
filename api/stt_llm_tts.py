import os
import time
import queue
import wave
import requests
import tempfile
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_openrouter import ChatOpenRouter


app = FastAPI(title="voice-pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TTS_SERVER = os.getenv("TTS_SERVER")
STT_SERVER = os.getenv("STT_SERVER")

# ---------------------------------------------------
llm=ChatOpenRouter(model="google/gemini-3-flash-preview")
# ---------------------------------------------------

SYSTEM_PROMPT = """
You are Aira, a voice-first AI assistant. 
Use only plain English text. Do not use markdown, emojis, bullets, code, parentheses, slashes, brackets, or any special symbols. 
Only comma, period, and apostrophe are allowed as punctuation. 
Punctuate sentences properly, and avoid single long sentences. Split long ideas into shorter sentences using commas and periods.
"""

def invoke_llm(text: str):

    response=llm.invoke(SYSTEM_PROMPT+text)

    return response.content

# ---------------------------------------------------
# WAV FRAGMENT PARSER
# ---------------------------------------------------

def extract_wav_fragments(byte_buffer):

    fragments = []

    while True:

        if len(byte_buffer) < 12:
            break

        riff_pos = byte_buffer.find(b"RIFF")

        if riff_pos == -1:

            byte_buffer.clear()
            break

        if riff_pos > 0:
            del byte_buffer[:riff_pos]

        if len(byte_buffer) < 8:
            break

        wav_size = int.from_bytes(
            byte_buffer[4:8],
            "little"
        ) + 8

        if len(byte_buffer) < wav_size:
            break

        wav_fragment = bytes(
            byte_buffer[:wav_size]
        )

        del byte_buffer[:wav_size]

        fragments.append(wav_fragment)

    return fragments

# ---------------------------------------------------
# PIPELINE
# ---------------------------------------------------

@app.post("/voice")

async def voice_pipeline(
    file: UploadFile = File(...)
):

    pipeline_start = time.time()

    # ----------------------------------------
    # SAVE TEMP AUDIO
    # ----------------------------------------

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".webm"
    ) as tmp:

        tmp.write(await file.read())

        temp_path = tmp.name

    # ----------------------------------------
    # STT
    # ----------------------------------------

    with open(temp_path, "rb") as f:

        stt_response = requests.post(
            f"{STT_SERVER}/transcribe",
            files={
                "file": (
                    "audio.webm",
                    f,
                    "audio/webm"
                )
            }
        )

    if stt_response.status_code != 200:

        return JSONResponse(
            {
                "error": "transcription failed"
            },
            status_code=500
        )

    stt_json = stt_response.json()

    user_text = stt_json.get(
        "text",
        ""
    ).strip()

    if not user_text:

        return JSONResponse(
            {
                "error": "empty transcription"
            },
            status_code=400
        )

    print(f"user: {user_text}")

    # ----------------------------------------
    # LLM
    # ----------------------------------------

    llm_start = time.time()

    assistant_text = invoke_llm(user_text)

    llm_latency = (
        time.time() - llm_start
    ) * 1000

    print(f"assistant: {assistant_text}")

    # ----------------------------------------
    # TTS
    # ----------------------------------------

    tts_response = requests.post(
        f"{TTS_SERVER}/tts",

        json={
            "model": "lisa",
            "reference_id": 6,
            "text": assistant_text
        },

        stream=True
    )

    if tts_response.status_code != 200:

        return JSONResponse(
            {
                "error": "tts failed"
            },
            status_code=500
        )

    # ----------------------------------------
    # STREAM WAV FRAGMENTS
    # ----------------------------------------

    def generate():

        first_chunk = True

        byte_buffer = bytearray()

        for chunk in tts_response.iter_content(
            chunk_size=4096
        ):

            if not chunk:
                continue

            byte_buffer.extend(chunk)

            wav_fragments = extract_wav_fragments(
                byte_buffer
            )

            for fragment in wav_fragments:

                if first_chunk:

                    latency_ms = int(
                        (
                            time.time()
                            - pipeline_start
                        ) * 1000
                    )

                    print(
                        f"voice→audio latency: {latency_ms}ms"
                    )

                    first_chunk = False

                yield fragment

    headers = {

        "X-User-Text": user_text,
        "X-Assistant-Text": assistant_text,
        "X-LLM-Latency": str(
            int(llm_latency)
        )
    }

    return StreamingResponse(
        generate(),
        media_type="application/octet-stream",
        headers=headers
    )