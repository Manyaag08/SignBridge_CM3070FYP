"""
Pipeline B evaluation: Whisper (base) speech-to-text across many speakers and conditions.

Test cases
  S1  LibriSpeech test-clean  - read speech, many real speakers (per-speaker WER)
  S2  LibriSpeech test-other  - harder / accented / less clear speakers
  S3  Noise robustness        - S1 subset with white noise at SNR 20/10/5/0 dB
  S4  Call phrases            - short conversational phrases spoken by several synthetic voices (macOS `say`)
Metrics: corpus WER (Whisper English normaliser), per-speaker WER, latency p50/p95, real-time factor.
"""
import io, json, os, subprocess, sys, tempfile, time, urllib.request
import numpy as np, soundfile as sf, torch, whisper
from jiwer import wer
from whisper.normalizers import EnglishTextNormalizer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "artifacts", "eval")
os.makedirs(OUT, exist_ok=True)
API = "https://datasets-server.huggingface.co/rows?dataset=openslr/librispeech_asr&config={cfg}&split=test&offset={off}&length={n}"
RNG = np.random.RandomState(7)
norm = EnglishTextNormalizer()
torch.set_num_threads(4)

def http_get(url, tries=8):
    """GET with exponential backoff (the public datasets API rate-limits bursts)."""
    import urllib.error
    for a in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=90).read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503) or a == tries - 1:
                raise
            time.sleep(min(60, 3 * 2 ** a))

def fetch_rows(cfg, n_total, n_blocks, per_block=5):
    """Evenly spaced blocks of consecutive utterances: the corpus is ordered by speaker, so this spans many speakers."""
    rows, step = [], n_total // n_blocks
    for off in range(0, step * n_blocks, step):
        d = json.loads(http_get(API.format(cfg=cfg, off=off, n=per_block)))
        for rr in d["rows"]:
            r = rr["row"]
            audio, sr = sf.read(io.BytesIO(http_get(r["audio"][0]["src"])), dtype="float32")
            rows.append({"id": r["id"], "speaker": r["speaker_id"], "ref": r["text"], "audio": audio, "sr": sr})
        time.sleep(1)
    return rows

model = whisper.load_model("base")
def transcribe(audio):
    t0 = time.perf_counter()
    text = model.transcribe(audio, language="en", fp16=False, temperature=0.0)["text"]
    return text.strip(), (time.perf_counter() - t0) * 1000

def run_set(name, items):
    refs, hyps, lat, rtf, per = [], [], [], [], []
    for it in items:
        hyp, ms = transcribe(it["audio"])
        r, h = norm(it["ref"]), norm(hyp)
        refs.append(r); hyps.append(h); lat.append(ms)
        dur = len(it["audio"]) / 16000; rtf.append(ms / 1000 / dur)
        per.append({"id": it["id"], "speaker": it.get("speaker"), "ref": it["ref"], "hyp": hyp,
                    "wer": round(wer(r, h), 4) if r else None, "latency_ms": round(ms, 1), "dur_s": round(dur, 2)})
    spk = {}
    for p, r, h in zip(per, refs, hyps):
        spk.setdefault(str(p["speaker"]), ([], []))
        spk[str(p["speaker"])][0].append(r); spk[str(p["speaker"])][1].append(h)
    spk_wer = {k: round(wer(v[0], v[1]), 4) for k, v in spk.items()}
    res = {"n": len(items), "n_speakers": len(spk), "wer": round(wer(refs, hyps), 4),
           "per_speaker_wer": spk_wer,
           "worst_speaker_wer": max(spk_wer.values()), "best_speaker_wer": min(spk_wer.values()),
           "latency_p50_ms": round(float(np.percentile(lat, 50)), 1),
           "latency_p95_ms": round(float(np.percentile(lat, 95)), 1),
           "rtf_mean": round(float(np.mean(rtf)), 3)}
    print(name, {k: v for k, v in res.items() if k != "per_speaker_wer"}, flush=True)
    return res, per

results, detail = {"model": "openai-whisper base (CPU, fp16=False, 4 threads)"}, {}

clean = fetch_rows("clean", 2620, 20)
results["S1_librispeech_test_clean"], detail["S1"] = run_set("S1", clean)
other = fetch_rows("other", 2939, 12)
results["S2_librispeech_test_other"], detail["S2"] = run_set("S2", other)

noise = {}
sub = clean[:30]
for snr in [20, 10, 5, 0]:
    items = []
    for it in sub:
        a = it["audio"]; p = np.mean(a ** 2)
        n = RNG.normal(0, np.sqrt(p / (10 ** (snr / 10))), a.shape).astype(np.float32)
        items.append({**it, "audio": a + n})
    noise[f"snr_{snr}db"], _ = run_set(f"S3 snr{snr}", items)
base_sub, _ = run_set("S3 clean-subset", sub)
noise["clean"] = base_sub
results["S3_noise_robustness"] = {k: {"wer": v["wer"], "latency_p50_ms": v["latency_p50_ms"]} for k, v in noise.items()}

PHRASES = ["Hello, how are you?", "I am good, thank you.", "Love you.", "Nice to meet you.",
           "What is your name?", "My name is Manya.", "Can you help me please?", "I need a doctor.",
           "Where does it hurt?", "Please take a seat.", "I am sorry, I did not understand.",
           "Can you repeat that more slowly?", "The appointment is at three o'clock tomorrow.",
           "Do you have any allergies?", "I feel dizzy and my chest hurts.", "See you later.",
           "Thank you for waiting.", "Is this seat free?", "Call an ambulance now.", "Yes, that is right."]
avail = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
VOICES = [v for v in ["Samantha", "Daniel", "Karen", "Rishi", "Moira", "Tessa", "Fred", "Albert"] if v + " " in avail]
items = []
with tempfile.TemporaryDirectory() as td:
    for v in VOICES:
        for i, ph in enumerate(PHRASES):
            f = os.path.join(td, f"{v}_{i}.wav")
            subprocess.run(["say", "-v", v, "-o", f, "--data-format=LEI16@16000", ph], check=True)
            a, _ = sf.read(f, dtype="float32")
            items.append({"id": f"{v}_{i}", "speaker": v, "ref": ph, "audio": a})
results["S4_call_phrases_synthetic_voices"], detail["S4"] = run_set("S4", items)
results["S4_call_phrases_synthetic_voices"]["voices"] = VOICES

json.dump(results, open(os.path.join(OUT, "speech_metrics.json"), "w"), indent=2)
json.dump(detail, open(os.path.join(OUT, "speech_detail.json"), "w"), indent=1)
print("saved speech_metrics.json")
