"""
Model engines for the backend. Models load lazily on first use (Whisper and the tone model are large), and every
engine can be replaced by a stub with SIGNBRIDGE_FAKE_MODELS=1, which the automated tests use so they need no downloads.
"""
import io, os, threading, time
import numpy as np

FAKE = os.getenv("SIGNBRIDGE_FAKE_MODELS") == "1"
# OpenAI's own whisper package, run unmodified. "base" is the deliberate choice here: it fits comfortably on a
# laptop CPU and transcribes fast enough for a live call, at some accuracy cost against "small"/"medium" - fine
# for the short utterances this pipeline actually sees. Override with SIGNBRIDGE_WHISPER_MODEL for a bigger one.
WHISPER_MODEL = os.getenv("SIGNBRIDGE_WHISPER_MODEL", "base")
# Pretrained sentiment classifier pulled straight from the HuggingFace Hub, no fine-tuning of my own. Picked this
# one over the more common English-only sentiment models because a Deaf/hearing pair using this app may not be
# speaking English, and this one covers the transcript regardless of language.
TONE_MODEL = "tabularisai/multilingual-sentiment-analysis"
# The model's own labels are 5-way (very negative..very positive); I only need enough signal to colour a caption,
# so I collapse that down to 3 classes here rather than exposing 5 the UI has no real use for.
THREE = {"very negative": "negative", "negative": "negative", "neutral": "neutral", "positive": "positive", "very positive": "positive"}

_lock = threading.Lock()          # neither Whisper nor the HF pipeline is safe to call from several threads at once
_whisper = None
_tone = None


def load_wav(data: bytes, target=16000):
    import soundfile as sf
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    audio = audio.mean(1)
    if sr != target:
        n = int(len(audio) * target / sr)
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)
    return audio


def transcribe(audio: np.ndarray, lang="en"):
    """Whisper transcription of 16 kHz mono float32 audio -> (text, ms). Segments Whisper itself flags as
    non-speech are dropped, which stops the well-known 'Thank you.' hallucination on silence and noise."""
    t0 = time.perf_counter()
    if FAKE:
        return ("hello how are you" if len(audio) else ""), (time.perf_counter() - t0) * 1000
    global _whisper
    with _lock:
        if _whisper is None:
            import whisper  # OpenAI's pretrained speech model, loaded as-is - lazy so a run that never records
                             # audio (e.g. the sign-only pipeline) never pays for it
            _whisper = whisper.load_model(WHISPER_MODEL)
        res = _whisper.transcribe(audio, language=None if lang == "auto" else lang, fp16=False, temperature=0.0,
                                  condition_on_previous_text=False)
    segs = [s for s in res.get("segments", []) if s.get("no_speech_prob", 0) < 0.6]
    text = " ".join(s["text"].strip() for s in segs).strip() if segs else ""
    return text, (time.perf_counter() - t0) * 1000


def tone(text: str):
    """Sentiment / tone of a transcript -> (label in negative|neutral|positive, score, ms)."""
    t0 = time.perf_counter()
    if FAKE:
        neg = any(w in text.lower() for w in ("hurt", "pain", "bad", "sorry"))
        return ("negative" if neg else "neutral"), 0.9, (time.perf_counter() - t0) * 1000
    global _tone
    with _lock:
        if _tone is None:
            from transformers import pipeline
            # first call downloads TONE_MODEL from the HF Hub and caches it locally; device=-1 keeps this on CPU
            # since it doesn't need to fight Whisper for a GPU that most examiners' laptops won't have anyway
            _tone = pipeline("text-classification", model=TONE_MODEL, top_k=1, device=-1)
        r = _tone(text, truncation=True)
    r = r[0][0] if isinstance(r[0], list) else r[0]
    return THREE.get(r["label"].lower(), "neutral"), float(r["score"]), (time.perf_counter() - t0) * 1000
