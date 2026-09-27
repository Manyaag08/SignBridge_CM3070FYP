"""
End-to-end evaluation of both SignBridge pipelines, run as one orchestrated system on one machine.

Pipeline A  sign -> speech
  landmark frames of a signer held out of training (Surrey signer A, and the original-corpus signer) at 30 fps,
  1 s per letter + 0.33 s transition -> letter classifier (leave-one-source-out model, and for Surrey A also
  the original single-signer model) ->
  hold-to-type commit logic -> sentence smoother (gloss grammar, then local LLM) -> TTS (macOS `say`)
  -> Whisper listens to the synthesised audio (closed loop: did the hearing caller receive the message?)
Pipeline B  speech -> captions + tone + signing avatar
  spoken call phrases (2 voices) -> Whisper -> tone tagger -> avatar sign plan -> avatar fingerspelling
  frames read back by the letter classifier (can a recogniser read the avatar?)
Latency: per stage, measured sequentially on an otherwise idle machine (compute latency; the deliberate
1 s letter hold is reported separately as interaction time).
"""
import json, os, subprocess, sys, tempfile, time
import numpy as np, pandas as pd, soundfile as sf, torch, whisper
from jiwer import wer, cer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from whisper.normalizers import EnglishTextNormalizer

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from signbridge_core import LETTERS, augment, build_message, commit_letters_segment, commit_letters_vote, make_clf, mirror, orchestrate_smoother, personalise, plan_avatar
ROOT = os.path.join(HERE, ".."); OUT = os.path.join(ROOT, "artifacts", "eval")
RNG = np.random.RandomState(3); torch.set_num_threads(4); norm = EnglishTextNormalizer()

D = pd.read_csv(os.path.join(OUT, "landmarks_all_sources.csv"))
FEAT = [f"f{i}" for i in range(42)]
X = D[FEAT].values.astype(np.float32); y = np.array([LETTERS.index(l) for l in D.letter]); src = D.source.values
ms = json.load(open(os.path.join(OUT, "multisigner_metrics.json")))
REVISED_TEMP = json.load(open(os.path.join(OUT, "calibration_temperature.json")))["selected_temperature"]
DEC = json.load(open(os.path.join(OUT, "message_decoding_params.json")))   # chosen on validation messages
rng = np.random.RandomState(42)

# ---------------------------------------------------------------- models
print("loading models", flush=True)
wmodel = whisper.load_model("base")
from transformers import pipeline as hf_pipeline, AutoModelForCausalLM, AutoTokenizer
tone = hf_pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis", top_k=1, device=-1)
LLM = "Qwen/Qwen2.5-0.5B-Instruct"; tok = AutoTokenizer.from_pretrained(LLM); lm = AutoModelForCausalLM.from_pretrained(LLM, dtype=torch.float32)
SYSTEM_PROMPT = open(os.path.join(ROOT, "evaluation", "eval_smoother.py")).read().split('SYSTEM_PROMPT = """')[1].split('"""')[0]

def timed(fn, *a):
    t0 = time.perf_counter(); r = fn(*a); return r, (time.perf_counter() - t0) * 1000
def smooth(signs):
    if not signs:
        return ""
    g = orchestrate_smoother(signs)
    if g: return g
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "Signs: " + " ".join(signs)}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    with torch.no_grad(): out = lm.generate(**enc, max_new_tokens=40, do_sample=False)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip().split("\n")[0]
def tts(text):
    f = tempfile.mktemp(suffix=".wav"); subprocess.run(["say", "-v", "Samantha", "-o", f, "--data-format=LEI16@16000", text], check=True)
    a, _ = sf.read(f, dtype="float32"); os.remove(f); return a
def stt(audio): return wmodel.transcribe(audio, language="en", fp16=False, temperature=0.0)["text"].strip()

# ---------------------------------------------------------------- Pipeline A
MESSAGES = ["HI", "HELLO", "YES", "NO", "HELP", "THANKS", "GOOD", "SORRY", "PLEASE WAIT", "CALL A DOCTOR",
            "MY NAME IS ANNA", "SEE YOU LATER", "I FEEL SICK", "WATER PLEASE", "THANK YOU"]
rows, improved_rows = [], []
for s in ["SA", "O"]:
    tr = src != s
    Xa, ya = augment(X[tr], y[tr], rng)
    models = {"multi_signer": make_clf().fit(Xa, ya)}
    if s == "SA":   # the original model has never seen Surrey A (it was trained on O only)
        models["single_signer"] = make_clf().fit(X[src == "O"], y[src == "O"])
    single_mirror = ms["E1_single_signer_model"]["SA"]["best_convention"] == "mirrored"
    te = D[src == s]
    # calibration frames: the first 5 of each letter; message frames: later in the recording (Surrey >= 15 frames on, O its last 30%)
    pools, ses0 = {}, {}
    for L, g in te.groupby("letter"):
        g = g.sort_values("seq")
        ses0[L] = g[FEAT].values[:5].astype(np.float32)
        rest = g.iloc[int(0.7 * len(g)):] if s == "O" else g.iloc[15:]
        pools[L] = rest[FEAT].values.astype(np.float32)
    for msg in MESSAGES:
        frames, truth = [], []
        for w_i, word in enumerate(msg.split()):
            if w_i: frames += [None] * 20; truth.append("_")
            for L in word:
                pool = pools[L]; st = RNG.randint(0, len(pool))   # a held letter = consecutive frames of the recording
                frames += [pool[(st + i) % len(pool)] for i in range(30)] + [None] * 10; truth.append(L)
        for mname, model in models.items():
            for calib in (["none", "revised", "shipped"] if mname == "multi_signer" else ["none", "revised"]):
                feats = np.array([f if f is not None else np.zeros(42, np.float32) for f in frames])
                Xin = mirror(feats) if (mname == "single_signer" and single_mirror) else feats
                t0 = time.perf_counter(); proba = model.predict_proba(Xin); cls_ms = (time.perf_counter() - t0) * 1000 / len(frames)
                full = np.zeros((len(frames), 26)); full[:, model.classes_] = proba
                if calib != "none":
                    cal = {LETTERS.index(L): (mirror(v) if (mname == "single_signer" and single_mirror) else v) for L, v in ses0.items()}
                    temp = REVISED_TEMP if calib == "revised" else 0.8
                    full = np.array([personalise(full[i], Xin[i], cal, 0.6, temp=temp) for i in range(len(frames))])
                preds = [(None, 0.0) if f is None else (LETTERS[int(p.argmax())], float(p.max())) for f, p in zip(frames, full)]
                hand = [i for i, f in enumerate(frames) if f is not None]
                gold = []
                for w_i, word in enumerate(msg.split()):
                    if w_i: gold += [None] * 20
                    for L in word: gold += [L] * 30 + [None] * 10
                frame_acc = float(np.mean([preds[i][0] == gold[i] for i in hand]))
                confident = float(np.mean([preds[i][1] >= 0.55 for i in hand]))
                # hold-to-type (20 frames) with >= 20 empty frames typed as a word space, as a signer pausing between words
                typed, run, last, since, gaprun = [], 0, None, 20, 0
                for L, p in preds:
                    since += 1
                    if L is None or p < 0.55:
                        gaprun += 1; run, last = 0, None
                        if gaprun == 20 and typed and typed[-1] != "_": typed.append("_")
                        continue
                    gaprun = 0; run = run + 1 if L == last else 1; last = L
                    if run >= 20 and since >= 20: typed.append(L); since = 0; run = 0
                while typed and typed[-1] == "_": typed.pop()
                sentence, llm_ms = timed(smooth, typed)
                if sentence:
                    audio, tts_ms = timed(tts, sentence)
                    heard, stt_ms = timed(stt, audio)
                else:
                    heard, tts_ms, stt_ms = "", np.nan, np.nan
                if mname == "multi_signer" and calib == "revised":
                    hand_mask = [f is not None for f in frames]
                    for cname, typing, dec in [("improved_vote", lambda P, h: commit_letters_vote(P, h, agree=DEC["chosen_vote"]["agree"]), DEC["chosen_vote"]["decoder"]),
                                               ("improved_segment", commit_letters_segment, DEC["chosen"]["decoder"] if DEC["chosen"]["typing"] == "segment" else DEC["table"]["segment"]["best_decoder"]["params"])]:
                        words = typing(full, hand_mask)
                        typed_i = " ".join("".join(LETTERS[i] for i, _ in w) for w in words)
                        (sent_i, detail), dec_ms = timed(lambda w: build_message(w, **dec), words)
                        if sent_i:
                            audio_i, tts_i = timed(tts, sent_i); heard_i, stt_i = timed(stt, audio_i)
                        else:
                            heard_i, tts_i, stt_i = "", np.nan, np.nan
                        said = sent_i.rstrip(".?").upper() if sent_i else None
                        improved_rows.append({"signer": s, "config": cname, "message": msg, "typed": typed_i,
                                              "decoded": " ".join(f"{w}({st})" for w, st in detail), "sentence": sent_i or "[please repeat]",
                                              "heard": heard_i, "exact": said == msg, "wrong_spoken": said is not None and said != msg,
                                              "repeat_request": said is None, "letter_accuracy": round(1 - min(1.0, cer(msg, typed_i)), 4),
                                              "heard_wer": round(min(1.0, wer(norm(msg), norm(heard_i))), 4) if sent_i and norm(heard_i) else None,
                                              "decode_ms": dec_ms, "tts_ms": tts_i})
                rows.append({"signer": s, "model": mname, "calibrated": calib, "message": msg, "typed": "".join(typed).replace("_", " "),
                             "sentence": sentence, "heard": heard,
                             "letter_cer": round(cer(msg, "".join(typed).replace("_", " ")), 4),
                             "message_exact": "".join(typed).replace("_", " ") == msg,
                             "heard_wer": round(min(1.0, wer(norm(msg), norm(heard))) if norm(heard) else 1.0, 4),
                             "frame_accuracy": round(frame_acc, 4), "frames_above_threshold": round(confident, 4),
                             "cls_ms_per_frame": cls_ms, "smoother_ms": llm_ms, "tts_ms": tts_ms, "stt_ms": stt_ms,
                             "interaction_s": round(len(frames) / 30, 2)})
    print("A held-out signer", s, "done", flush=True)
A = pd.DataFrame(rows)
summaryA = {}
for (m, c), g in A.groupby(["model", "calibrated"]):
    summaryA[f"{m}" + ("" if c == "none" else f"_calibrated_k5_{c}_temp")] = {
        "n_messages": int(len(g)), "letter_accuracy": round(1 - float(g.letter_cer.mean()), 4),
        "message_exact_rate": round(float(g.message_exact.mean()), 4),
        "closed_loop_heard_wer_capped": round(float(g.heard_wer.mean()), 4),
        "per_frame_accuracy": round(float(g.frame_accuracy.mean()), 4), "frames_above_0.55_confidence": round(float(g.frames_above_threshold.mean()), 4),
        "per_signer_message_exact": {s: round(float(gg.message_exact.mean()), 3) for s, gg in g.groupby("signer")}}
I = pd.DataFrame(improved_rows)
for cname, g in I.groupby("config"):
    summaryA[cname] = {"n_messages": int(len(g)), "letter_accuracy": round(float(g.letter_accuracy.mean()), 4),
                       "message_exact_rate": round(float(g.exact.mean()), 4), "wrong_spoken_rate": round(float(g.wrong_spoken.mean()), 4),
                       "repeat_request_rate": round(float(g.repeat_request.mean()), 4),
                       "closed_loop_heard_wer_spoken_only": round(float(g.heard_wer.dropna().mean()), 4) if g.heard_wer.notna().any() else None,
                       "decode_ms_p50": round(float(g.decode_ms.median()), 2),
                       "per_signer_message_exact": {sg: round(float(gg.exact.mean()), 3) for sg, gg in g.groupby("signer")}}
for (m, c), g in A[A.model == "multi_signer"].groupby(["model", "calibrated"]):
    if c == "revised":        # the old pipeline on the same frames, with the same safety metrics for comparison
        spoken = g.sentence.fillna("").str.len() > 0
        exact_old = g.message_exact
        summaryA["baseline_wrong_spoken_rate"] = round(float((spoken & ~exact_old).mean()), 4)
I.to_csv(os.path.join(OUT, "e2e_pipelineA_improved_cases.csv"), index=False)
lat = A[(A.model == "multi_signer") & (A.calibrated == "none")]
summaryA["latency_ms_p50"] = {"mediapipe_per_frame": ms["E0_mediapipe_ms_p50"],
                              "classifier_per_frame": round(float(lat.cls_ms_per_frame.median()), 4),
                              "smoother": round(float(lat.smoother_ms.median()), 1), "smoother_p95": round(float(np.percentile(lat.smoother_ms, 95)), 1), "tts_synthesis": round(float(np.nanmedian(lat.tts_ms)), 1),
                              "whisper_closed_loop_check": round(float(np.nanmedian(lat.stt_ms)), 1)}
summaryA["latency_ms_p50"]["compute_sign_to_speech_excl_hold"] = round(summaryA["latency_ms_p50"]["mediapipe_per_frame"] + summaryA["latency_ms_p50"]["classifier_per_frame"] + summaryA["latency_ms_p50"]["smoother"] + summaryA["latency_ms_p50"]["tts_synthesis"], 1)
summaryA["interaction_time_s_mean"] = round(float(lat.interaction_s.mean()), 2)
print(json.dumps(summaryA, indent=1), flush=True)

# ---------------------------------------------------------------- Pipeline B
PHRASES = [("Hello, how are you?", "neutral"), ("I am good, thank you.", "positive"), ("Love you.", "positive"), ("Nice to meet you.", "positive"),
           ("What is your name?", "neutral"), ("My name is Manya.", "neutral"), ("Can you help me please?", "neutral"), ("I need a doctor.", "negative"),
           ("Where does it hurt?", "neutral"), ("Please take a seat.", "neutral"), ("I am sorry, I did not understand.", "negative"),
           ("The appointment is at three o'clock tomorrow.", "neutral"), ("Do you have any allergies?", "neutral"), ("I feel dizzy and my chest hurts.", "negative"),
           ("See you later.", "neutral"), ("Thank you for waiting.", "positive"), ("Call an ambulance now.", "negative"), ("Yes, that is right.", "positive"),
           ("I am worried something is wrong.", "negative"), ("That's wonderful news.", "positive")]
REF = json.load(open(os.path.join(ROOT, "artifacts", "ref_skeletons.json")))
Xall, yall = augment(X, y, rng)
reader_ms = make_clf().fit(Xall, yall)                    # the multi-signer model that ships in the browser
reader_single = make_clf().fit(X[src == "O"], y[src == "O"])
to3 = lambda l: {"very negative": "negative", "very positive": "positive"}.get(l, l)
rowsB = []
for voice in ["Samantha", "Daniel"]:
    for text, tone_ref in PHRASES:
        f = tempfile.mktemp(suffix=".wav"); subprocess.run(["say", "-v", voice, "-o", f, "--data-format=LEI16@16000", text], check=True)
        audio, _ = sf.read(f, dtype="float32"); os.remove(f)
        hyp, stt_ms = timed(stt, audio)
        tone_out, tone_ms = timed(lambda t: to3(tone(t)[0][0]["label"].lower()), hyp)
        (plan, dur), plan_ms = timed(plan_avatar, hyp)
        ref_plan, _ = plan_avatar(text)
        spelled = "".join(p["value"] for p in plan if p["type"] == "spell")
        letters = [c for c in spelled if c in REF]
        feats = np.array([np.array(REF[c], np.float32).reshape(-1) for c in letters]) if letters else np.zeros((0, 42))
        rt_ms = "".join(LETTERS[i] for i in reader_ms.predict(feats)) if len(feats) else ""
        rt_single = "".join(LETTERS[i] for i in reader_single.predict(feats)) if len(feats) else ""
        rowsB.append({"voice": voice, "text": text, "heard": hyp, "stt_wer": round(wer(norm(text), norm(hyp)), 4), "stt_ms": stt_ms,
                      "tone": tone_out, "tone_ref": tone_ref, "tone_ms": tone_ms, "plan": " ".join(p["value"] if p["type"] == "sign" else "#" + p["value"] for p in plan),
                      "plan_matches_reference": plan == ref_plan, "signs": sum(p["type"] == "sign" for p in plan), "spelled_words": sum(p["type"] == "spell" for p in plan),
                      "avatar_duration_s": dur, "speech_duration_s": round(len(audio) / 16000, 2), "plan_ms": plan_ms,
                      "fingerspelled_letters": "".join(letters), "roundtrip_multisigner_reader": rt_ms, "roundtrip_single_signer_reader": rt_single})
B = pd.DataFrame(rowsB)
tot_letters = B.fingerspelled_letters.str.len().sum()
def rt_acc(col): return round(float(sum(sum(a == b for a, b in zip(x, y)) for x, y in zip(B.fingerspelled_letters, B[col])) / max(1, tot_letters)), 4)
summaryB = {"n_utterances": int(len(B)), "stt_wer": round(float(B.stt_wer.mean()), 4),
            "tone_accuracy_vs_author_label": round(float((B.tone == B.tone_ref).mean()), 4),
            "avatar_plan_identical_to_plan_from_reference_text": round(float(B.plan_matches_reference.mean()), 4),
            "avatar_lexical_sign_share": round(float(B.signs.sum() / (B.signs.sum() + B.spelled_words.sum())), 4),
            "avatar_fingerspelled_letters": int(tot_letters),
            "avatar_roundtrip_letter_accuracy_multisigner_reader": rt_acc("roundtrip_multisigner_reader"),
            "avatar_roundtrip_letter_accuracy_single_signer_reader": rt_acc("roundtrip_single_signer_reader"),
            "avatar_to_speech_duration_ratio_mean": round(float((B.avatar_duration_s / B.speech_duration_s).mean()), 2),
            "latency_ms_p50": {"whisper": round(float(B.stt_ms.median()), 1), "tone": round(float(B.tone_ms.median()), 1),
                               "avatar_plan": round(float(B.plan_ms.median()), 3)}}
summaryB["latency_ms_p50"]["compute_speech_to_caption_tone_plan"] = round(sum(summaryB["latency_ms_p50"].values()), 1)
print(json.dumps(summaryB, indent=1), flush=True)
json.dump({"pipeline_A_sign_to_speech": summaryA, "pipeline_B_speech_to_sign": summaryB}, open(os.path.join(OUT, "e2e_metrics.json"), "w"), indent=2)
A.to_csv(os.path.join(OUT, "e2e_pipelineA_cases.csv"), index=False); B.to_csv(os.path.join(OUT, "e2e_pipelineB_cases.csv"), index=False)
print("saved e2e_metrics.json")
