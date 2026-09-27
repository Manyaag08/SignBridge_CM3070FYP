"""Backend tests. Models get replaced by stubs (SIGNBRIDGE_FAKE_MODELS=1) so nothing needs downloading; the
smoother's LLM is just injected instead. Run with:  python -m pytest tests -v"""
import io, json, os, sys
os.environ["SIGNBRIDGE_FAKE_MODELS"] = "1"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import numpy as np, soundfile as sf
from fastapi.testclient import TestClient
from backend import smoother
from backend.app import app

client = TestClient(app)

def wav_bytes(seconds=1.0, sr=16000):
    buf = io.BytesIO(); sf.write(buf, (0.1 * np.sin(np.arange(int(sr * seconds)) / 20)).astype("float32"), sr, format="WAV"); return buf.getvalue()

# ---------------------------------------------------------------- HTTP endpoints
def test_health_reports_models():
    j = client.get("/api/health").json()
    assert j["status"] == "ok" and j["whisper_model"] and j["llm_model"]

def test_transcribe_accepts_wav_and_reports_duration():
    r = client.post("/api/transcribe?lang=en", content=wav_bytes(1.5), headers={"Content-Type": "audio/wav"})
    assert r.status_code == 200 and r.json()["duration_s"] == 1.5 and r.json()["text"]

def test_transcribe_resamples_other_rates():
    r = client.post("/api/transcribe", content=wav_bytes(1.0, sr=48000))
    assert r.status_code == 200 and r.json()["duration_s"] == 1.0

def test_transcribe_rejects_garbage_empty_and_too_long():
    assert client.post("/api/transcribe", content=b"not audio").status_code == 400
    assert client.post("/api/transcribe", content=b"").status_code == 400
    assert client.post("/api/transcribe", content=wav_bytes(31)).status_code == 413

def test_tone_endpoint():
    assert client.post("/api/tone", json={"text": "my chest hurts"}).json()["tone"] == "negative"
    assert client.post("/api/tone", json={"text": ""}).json()["tone"] == "neutral"

def test_frontend_and_models_are_served():
    assert "SignBridge" in client.get("/").text
    assert client.get("/js/call.js").status_code == 200
    assert client.get("/artifacts/asl_landmark_mlp_multisigner_web.onnx").status_code in (200, 404)

# ---------------------------------------------------------------- sentence smoother (grammar -> dictionary -> LLM -> guard)
def no_llm(signs): raise AssertionError("LLM must not be called")

def test_smoother_rules_handle_real_words_without_llm():
    r = smoother.smooth("hello how are you", llm=no_llm)
    assert r["route"] == "rules" and r["sentence"].startswith("Hello")
    assert smoother.smooth("how are you?", llm=no_llm)["sentence"].endswith("?")

def test_smoother_uses_llm_for_a_non_word_and_accepts_faithful_output():
    r = smoother.smooth("hello i am hungy", llm=lambda s: ("Hello, I am hungry.", 5.0))
    assert r["route"] == "llm" and r["sentence"] == "Hello, I am hungry."

def test_smoother_guard_rejects_invented_content():
    r = smoother.smooth("hello i am hungy", llm=lambda s: ("Hello, I would like to book a flight to Paris.", 5.0))
    assert r["route"] == "llm_rejected_by_guard" and "Paris" not in r["sentence"]

def test_smoother_falls_back_to_rules_when_llm_is_down():
    r = smoother.smooth("hello i am hungy", llm=lambda s: (None, 0.0))
    assert r["route"] == "rules_llm_unavailable" and r["sentence"]

def test_smoother_empty():
    assert smoother.smooth("   ", llm=no_llm)["route"] == "empty"

# ---------------------------------------------------------------- call signalling relay
def test_signalling_relays_between_two_peers_and_rejects_a_third():
    with client.websocket_connect("/ws/r1") as a:
        with client.websocket_connect("/ws/r1") as b:
            assert a.receive_json() == {"type": "peer-joined"}          # the waiting peer is told to make the offer
            a.send_text(json.dumps({"type": "offer", "sdp": "x"}))
            assert b.receive_json()["type"] == "offer"
            b.send_text(json.dumps({"type": "answer", "sdp": "y"}))
            assert a.receive_json()["type"] == "answer"
            with client.websocket_connect("/ws/r1") as c:
                assert c.receive_json() == {"type": "room-full"}
        assert a.receive_json() == {"type": "peer-left"}

def test_rooms_are_isolated():
    with client.websocket_connect("/ws/x") as a, client.websocket_connect("/ws/y") as b:
        a.send_text(json.dumps({"type": "offer"}))
        b.send_text(json.dumps({"type": "offer"}))   # would deadlock/relay if rooms leaked; reaching here is the check

# ---------------------------------------------------------------- dictionary decoder endpoint
def _probs(word, conf=0.9, wrong=None):
    """probability rows that spell `word`; `wrong` maps position -> letter that is (wrongly) ranked first instead"""
    rows = []
    for i, c in enumerate(word):
        top = (wrong or {}).get(i, c)
        r = np.full(26, (1 - conf) / 25); r[ord(top) - 65] = conf
        if top != c: r[ord(c) - 65] = 0.35; r[ord(top) - 65] = 0.5
        rows.append((r / r.sum()).tolist())
    return rows

def test_decode_keeps_clean_words_and_phrases():
    r = client.post("/api/decode", json={"items": [{"text": "I"}, {"probs": _probs("HELP")}]}).json()
    assert not r["repeat"] and r["text"] == "I help"

def test_decode_corrects_a_near_miss_with_the_dictionary():
    r = client.post("/api/decode", json={"items": [{"probs": _probs("WATER", wrong={0: "V"})}]}).json()
    assert r["words"][0]["raw"] == "VATER" and r["text"] == "water" and not r["repeat"]

def test_decode_asks_to_repeat_when_a_word_is_unclear():
    flat = np.full((4, 26), 1 / 26).tolist()
    r = client.post("/api/decode", json={"items": [{"text": "I"}, {"probs": flat}]}).json()
    assert r["repeat"] and r["unclear"] == [1]

def test_decode_rejects_bad_input():
    assert client.post("/api/decode", json={"items": []}).status_code == 400
    assert client.post("/api/decode", json={"items": [{"probs": [[0.5, 0.5]]}]}).status_code == 400
