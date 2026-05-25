# DigiSapiens — A realtime conversation agent

A small experimental repo exploring speech-based conversational agents. The project contains three lightweight FastAPI services (and simple HTML frontends) that demonstrate different pipeline topologies for speech-to-speech interaction.

**Repository structure**

- `gemini_live.py` – A frontend/backed demo integrating a live Gemini audio-capable LLM with `index.html`. Stateless.
- `stt_llm_tts.py` – A classic speech-to-speech pipeline: STT → LLM → TTS. Uses `index2.html` as a simple frontend. Stateless.
- `llm_tts.py` – Sends recorded audio directly to an LLM that accepts audio input, then passes the LLM response to TTS. Uses `index3.html`. Implements a simple text memory/history.

**Features & notes**

- Minimal example apps that show how to wire STT, LLM, and TTS together.
- Each FastAPI app exposes a `POST /voice` endpoint that accepts an audio file upload and returns streaming audio (TTS) in response.
- `llm_tts.py` contains history support (bounded). The other two services are stateless by default.
- Environment variables are supported for external service URLs. `TTS_SERVER` and `STT_SERVER` are read from the environment with sensible fallbacks.

**Run (examples)**

```bash
# run gemini_live (example)
uvicorn gemini_live:app --port 9000
```
`serve the respective index files` using a python http server or something else

The response from tts is streamed as `application/octet-stream` containing WAV fragments; the services also set helpful headers (for example `X-User-Text`, `X-Assistant-Text`, `X-LLM-Latency`).

**Avatar**

not implemented yet
