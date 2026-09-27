"""
SignBridge backend. Just one process for the web app, the model endpoints, and the call-signalling relay.

Endpoints:
- GET  /api/health      -> which models/LLM are actually available right now
- POST /api/transcribe  -> 16kHz WAV in, ?lang=en|es|fr|auto, gives back {text, ms, duration_s} (Whisper)
- POST /api/tone        -> {text} in, {tone, score, ms} out (tone tagger)
- POST /api/decode      -> {items} in, {text, repeat, unclear, words} out (dictionary decoder)
- POST /api/smooth      -> {text} in, {sentence, route, ...} out (grammar + local LLM)
- WS   /ws/{room}       -> WebRTC signalling relay for a two-person call, no media ever goes through here
- /                     -> serves the frontend; /artifacts serves the browser's ONNX classifiers

Run with:  uvicorn backend.app:app --port 8000   (from the project root)
"""
import json, os
from typing import Dict, List

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import engines, smoother

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MAX_BYTES = 8 * 1024 * 1024          # roughly 4 min of 16kHz mono 16-bit audio - real utterances are way shorter than this
MAX_SECONDS = 30

app = FastAPI(title="SignBridge", version="1.0")


class Text(BaseModel):
    text: str


@app.get("/api/health")
def health():
    # doubles as a "which pretrained models is this actually running" check - handy when demoing or grading,
    # since it names Whisper/tone/LLM by their real model id rather than just saying the server is up
    return {"status": "ok", "fake_models": engines.FAKE, "whisper_model": engines.WHISPER_MODEL,
            "tone_model": engines.TONE_MODEL, "llm_model": smoother.LLM_MODEL,
            "llm_available": False if engines.FAKE else smoother.ollama_available()}


@app.post("/api/transcribe")
async def transcribe(request: Request, lang: str = "en"):
    data = await request.body()
    if not data or len(data) > MAX_BYTES:
        raise HTTPException(413 if data else 400, "audio missing or too large")
    try:
        audio = engines.load_wav(data)
    except Exception:
        raise HTTPException(400, "body must be a WAV file")
    dur = len(audio) / 16000
    if dur > MAX_SECONDS:
        raise HTTPException(413, f"utterance longer than {MAX_SECONDS}s")
    text, ms = engines.transcribe(audio, lang)
    return {"text": text, "ms": round(ms, 1), "duration_s": round(dur, 2)}


@app.post("/api/tone")
def tone(body: Text):
    text = body.text.strip()[:1000]
    if not text:
        return {"tone": "neutral", "score": 0.0, "ms": 0.0}
    label, score, ms = engines.tone(text)
    return {"tone": label, "score": round(score, 4), "ms": round(ms, 1)}


class Items(BaseModel):
    items: List[dict]


@app.post("/api/decode")
def decode(body: Items):
    if not body.items or len(body.items) > 60:
        raise HTTPException(400, "1 to 60 items expected")
    try:
        return smoother.decode_message(body.items)
    except (ValueError, TypeError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/smooth")
def smooth(body: Text):
    text = body.text.strip()[:500]
    llm = (lambda s: (None, 0.0)) if engines.FAKE else smoother.ollama_sentence
    return smoother.smooth(text, llm=llm)


# ------------------------------------------------------------------ call signalling relay
rooms: Dict[str, List[WebSocket]] = {}


@app.websocket("/ws/{room}")
async def signalling(ws: WebSocket, room: str):
    await ws.accept()
    peers = rooms.setdefault(room, [])
    if len(peers) >= 2:
        await ws.send_text(json.dumps({"type": "room-full"}))
        await ws.close()
        return
    peers.append(ws)
    for other in peers:                                   # whoever was already waiting is the one who makes the offer
        if other is not ws:
            await other.send_text(json.dumps({"type": "peer-joined"}))
    try:
        while True:
            msg = await ws.receive_text()
            for other in list(peers):
                if other is not ws:
                    await other.send_text(msg)
    except WebSocketDisconnect:
        pass
    finally:
        if ws in peers:
            peers.remove(ws)
        for other in peers:
            try:
                await other.send_text(json.dumps({"type": "peer-left"}))
            except Exception:
                pass
        if not peers:
            rooms.pop(room, None)


app.mount("/artifacts", StaticFiles(directory=os.path.join(ROOT, "artifacts")), name="artifacts")
app.mount("/", StaticFiles(directory=os.path.join(ROOT, "frontend"), html=True), name="frontend")
