import os
import re
import json
import time
import base64
import tempfile
import requests

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from langchain_openrouter import ChatOpenRouter

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage
)

app = FastAPI(title="voice-pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TTS_SERVER = os.getenv("TTS_SERVER")

llm = ChatOpenRouter(
    model="google/gemini-2.5-flash"
)

SYSTEM_PROMPT = """
You are Lisa, a friendly voice-first AI assistant.

CRITICAL IDENTITY RULES:
- Your name is Lisa. You are NOT Gemini, NOT Google Assistant, NOT any other AI.
- If asked who you are: say you are Lisa.
- Never reveal the underlying model or technology stack.
- Never say you are "a large language model trained by Google" or anything similar.

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

# Text-only conversation history (no audio blobs)
conversation_history = []

MAX_HISTORY = 12


# ---------------------------------------------------
# JSON EXTRACTION — ROBUST
# ---------------------------------------------------

def extract_json(raw: str) -> dict | None:
    """
    Try multiple strategies to extract a JSON object from raw model output.
    Returns a dict on success, None on failure.
    """

    # Strategy 1: direct parse
    try:
        return json.loads(raw.strip())
    except Exception:
        pass

    # Strategy 2: strip markdown fences then parse
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Strategy 3: regex — find first {...} block
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    # Strategy 4: manually extract fields from key:value pattern
    transcript_match = re.search(
        r'"transcript"\s*:\s*"((?:[^"\\]|\\.)*)"', raw
    )
    response_match = re.search(
        r'"response"\s*:\s*"((?:[^"\\]|\\.)*)"', raw
    )
    if response_match:
        return {
            "transcript": transcript_match.group(1) if transcript_match else "",
            "response": response_match.group(1),
        }

    return None


# ---------------------------------------------------
# GEMINI AUDIO INPUT
# ---------------------------------------------------

def invoke_llm_audio(audio_path: str):

    global conversation_history

    audio_data = base64.b64encode(
        open(audio_path, "rb").read()
    ).decode("utf-8")

    # Build the current user message with audio — NOT stored in history
    current_audio_message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Process this voice input and respond as Lisa.",
            },
            {
                "type": "audio",
                "base64": audio_data,
                "mime_type": "audio/webm",
            },
        ]
    )

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.extend(conversation_history)   # text-only history
    messages.append(current_audio_message)

    response = llm.invoke(messages)
    raw = str(response.content).strip()

    print("\nRAW MODEL OUTPUT:")
    print(raw)
    print()

    # ----------------------------------------
    # PARSE
    # ----------------------------------------

    parsed = extract_json(raw)

    if parsed:
        transcript = str(parsed.get("transcript", "")).strip()
        assistant_response = str(parsed.get("response", "")).strip()
    else:
        print("json parse failed — using fallback")
        transcript = ""
        assistant_response = ""

    # ----------------------------------------
    # SAFETY FALLBACKS
    # ----------------------------------------

    # Reject responses that leak model identity
    identity_leaks = [
        "large language model",
        "trained by google",
        "i am gemini",
        "i'm gemini",
        "google ai",
    ]
    if any(leak in assistant_response.lower() for leak in identity_leaks):
        print("Identity leak detected — overriding response")
        assistant_response = (
            "My name is Aira, your AI assistant from DigiSapiens. "
            "How can I help you today?"
        )

    if not assistant_response:
        assistant_response = "Sorry, something went wrong. Could you try again?"

    # ----------------------------------------
    # STORE TEXT-ONLY HISTORY
    # Append only if transcript is non-empty so
    # blank HumanMessages never pollute history.
    # ----------------------------------------

    if transcript:
        conversation_history.append(HumanMessage(content=transcript))

    conversation_history.append(AIMessage(content=assistant_response))

    # Keep history bounded
    if len(conversation_history) > MAX_HISTORY:
        conversation_history = conversation_history[-MAX_HISTORY:]

    return transcript, assistant_response


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

        wav_size = (
            int.from_bytes(byte_buffer[4:8], "little") + 8
        )

        if len(byte_buffer) < wav_size:
            break

        wav_fragment = bytes(byte_buffer[:wav_size])
        del byte_buffer[:wav_size]
        fragments.append(wav_fragment)

    return fragments


# ---------------------------------------------------
# PIPELINE
# ---------------------------------------------------

@app.post("/voice")
async def voice_pipeline(file: UploadFile = File(...)):

    pipeline_start = time.time()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name

    try:

        # -------------------------------------
        # LLM AUDIO
        # -------------------------------------

        llm_start = time.time()
        transcript, assistant_text = invoke_llm_audio(temp_path)
        llm_latency = int((time.time() - llm_start) * 1000)

        print(f"user: {transcript}")
        print(f"assistant: {assistant_text}")

        # -------------------------------------
        # TTS
        # -------------------------------------

        tts_response = requests.post(
            f"{TTS_SERVER}/tts",
            json={
                "model": "lisa",
                "reference_id": 6,
                "text": assistant_text,
            },
            stream=True,
        )

        if tts_response.status_code != 200:
            return JSONResponse({"error": "tts failed"}, status_code=500)

        # -------------------------------------
        # STREAM AUDIO
        # -------------------------------------

        def generate():

            byte_buffer = bytearray()
            first_fragment = True

            for chunk in tts_response.iter_content(chunk_size=4096):

                if not chunk:
                    continue

                byte_buffer.extend(chunk)
                wav_fragments = extract_wav_fragments(byte_buffer)

                for fragment in wav_fragments:

                    if first_fragment:
                        latency = int((time.time() - pipeline_start) * 1000)
                        print(f"voice→audio: {latency}ms")
                        first_fragment = False

                    yield fragment

        headers = {
            "X-User-Text": transcript,
            "X-Assistant-Text": assistant_text,
            "X-LLM-Latency": str(llm_latency),
        }

        return StreamingResponse(
            generate(),
            media_type="application/octet-stream",
            headers=headers,
        )

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)