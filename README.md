# DigiSapiens — A realtime conversation agent

A experimental repo exploring speech-based conversational agents. The project contains three lightweight FastAPI services (and simple HTML frontends) that demonstrate different pipeline topologies for speech-to-speech interaction.

**Repository structure**

- `vad_gemini_live.py` - a live gemini conversation agent demo with in-browser vad and a 2d dynamic avatar. frontend: `index_g_vad.html`
- `gemini_live.py` – a live Gemini audio-capable LLM which uses gemini's native vad, with `index_g_l.html` as its fontend
- `stt_llm_tts.py` – A classic speech-to-speech pipeline: STT → LLM → TTS. Uses `frontend/index2.html` as a simple frontend. Stateless.
- `llm_tts.py` – Sends recorded audio directly to an LLM that accepts audio input, then passes the LLM response to TTS. Uses `frontend/avatar.html`. Implements a simple text memory/history.

**Features & notes**

- Minimal example apps that show how to wire STT, LLM, and TTS together.
- Each FastAPI app exposes a /ws websocket endpoint
- `vad_gemini_live` and `gemini_live.py` both have native conversational history(session) while `llm_tts.py` have a text based history buffer.
- `stt_llm_tts.py` has no history
- Environment variables are supported for external service URLs. `TTS_SERVER` and `STT_SERVER` are read from the environment with sensible fallbacks.

**Run (examples)**

```bash
# run gemini_live (example)
uvicorn api.gemini_live:app --port 9000
```
`serve the respective index files` using a python http server and access then in the browser by entering thier names as endpoints

```bash
python -m http.server 5500
```

**Avatar**

- implemented in all frontend except `index2.html`
