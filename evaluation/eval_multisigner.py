"""
Multi-signer evaluation of the Pipeline A letter recogniser.

Sources of hands - each is a person, or group of people, the model can be tested on without having seen them:
  O   the single-signer corpus used so far (MohdDilshad-nitk/ASL_Recognition): 1 signer, 26 letters
  SA  Surrey ASL Finger Spelling dataset (Pugeault & Bowden, 2011), signer A: the 24 static letters
  SB  Surrey signer B: letters A-F only (the public server throttled the download after the first two signers)
  AN  ASLNow! (sid220/asl-now-fingerspelling): many participants signing into browser webcams, landmarks from
      MediaPipe's web hand landmarker, 26 letters. Participant identities are not recorded, so AN is a pooled
      set of unseen people, and the camera aspect ratio is unknown (4:3 is used; 16:9 is reported as a check)

Surrey frames go through MediaPipe Hands here, with the same wrist-relative / max-abs preprocessing as the browser.

Experiments (accuracy, and macro-F1 over the letters present in the test source)
  E0  MediaPipe detection rate on the Surrey frames
  E1  the deployed single-signer model tested on each unseen source (both left/right conventions, because the
      demo asks the user to pick their signing hand once)
  E2  leave-one-source-out: train on the other sources, test on the held-out one - no augmentation
  E3  leave-one-source-out with training-time augmentation (rotation, scale, jitter, mirror)
  E4  random 5-fold over all pooled frames (optimistic contrast: every test person also appears in training)
  E5  few-shot calibration of a new signer with the live demo's personalise() blend, k = 1, 3, 5, 10 samples per
      letter; test frames come from later in the same recording (>= 10 kept frames later; for O its last 30%).
      Reported for the shipped blend temperature (0.8) and the revised one chosen on validation signer SB,
      which is therefore excluded from E5.
"""
import csv, glob, json, os, re, sys, time
from collections import Counter
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from signbridge_core import preprocess, personalise, LETTERS, make_clf, mirror, augment
ROOT = os.path.join(HERE, ".."); OUT = os.path.join(ROOT, "artifacts", "eval"); os.makedirs(OUT, exist_ok=True)
SURREY = os.environ.get("SURREY_DIR"); ASLNOW = os.environ.get("ASLNOW_DIR")
SURREY_CSV = os.path.join(OUT, "surrey_landmarks.csv"); ALL_CSV = os.path.join(OUT, "landmarks_all_sources.csv")
FEAT = [f"f{i}" for i in range(42)]
rng = np.random.RandomState(42)

# ------------------------------------------------------------------ E0: Surrey landmark extraction
if not os.path.exists(SURREY_CSV):
    if not SURREY or not os.path.isdir(SURREY):
        sys.exit("No cached Surrey landmarks (artifacts/eval/surrey_landmarks.csv) and SURREY_DIR is not set.\n"
                 "Download https://www.cvssp.org/FingerSpellingKinect2011/fingerspelling5.tar.bz2, extract it and set\n"
                 "SURREY_DIR to its dataset5 folder, or restore the cached CSV.")
    import cv2, mediapipe as mp
    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)
    files = sorted(glob.glob(os.path.join(SURREY, "**", "color_*.png"), recursive=True))
    print("extracting landmarks from", len(files), "Surrey frames", flush=True)
    unreadable = 0
    with open(SURREY_CSV, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["signer", "letter", "seq", "detected", "mp_ms"] + FEAT)
        for n, f in enumerate(files):
            m = re.search(r"/([A-E])/([a-z])/color_\d+_(\d+)\.png$", f)
            img = cv2.imread(f)
            if img is None:        # truncated / corrupt file: skip it and report the count
                unreadable += 1; continue
            pad = int(max(img.shape[:2]) * 0.5)
            img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            t0 = time.perf_counter(); r = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ms = (time.perf_counter() - t0) * 1000
            row = [m.group(1), m.group(2).upper(), int(m.group(3))]
            if r.multi_hand_landmarks:
                lm = [(p.x, p.y) for p in r.multi_hand_landmarks[0].landmark]
                feat = preprocess(lm, img.shape[1] * 4, img.shape[0] * 4)   # finer pixel grid, same aspect as the frame
                w.writerow(row + [1, round(ms, 2)] + [round(float(v), 5) for v in feat])
            else:
                w.writerow(row + [0, round(ms, 2)] + [""] * 42)
            if n % 500 == 0: print(" ", n, flush=True)
    print("unreadable image files skipped:", unreadable, flush=True)
sur = pd.read_csv(SURREY_CSV)

# ------------------------------------------------------------------ assemble all sources
orig = pd.read_csv(os.path.join(ROOT, "ASL_Recognition", "ACM_Hand_gesture", "keypoint_classifier", "keypoint.csv"), header=None)
O = pd.DataFrame(orig.iloc[:, 1:43].values.astype(np.float32), columns=FEAT)
O["letter"] = [LETTERS[i] for i in orig.iloc[:, 0]]; O["source"] = "O"
O["seq"] = O.groupby("letter").cumcount()
S = sur[sur.detected == 1].copy(); S["source"] = "S" + S.signer
an_rows, an_169 = [], []
if not ASLNOW or not glob.glob(os.path.join(ASLNOW, "*", "*.json")):
    sys.exit("ASLNOW_DIR is not set or empty - run: python3 evaluation/fetch_aslnow.py evaluation/data/aslnow")
for f in sorted(glob.glob(os.path.join(ASLNOW, "*", "*.json"))):
    L = os.path.basename(os.path.dirname(f))
    if L not in LETTERS: continue
    lm = json.load(open(f))
    if len(lm) != 21: continue
    xy = [(p["x"], p["y"]) for p in lm]
    an_rows.append([L] + list(preprocess(xy, 640, 480))); an_169.append(preprocess(xy, 1280, 720))
AN = pd.DataFrame([r[1:] for r in an_rows], columns=FEAT); AN["letter"] = [r[0] for r in an_rows]; AN["source"] = "AN"
AN["seq"] = AN.groupby("letter").cumcount()
AN169 = np.array(an_169, np.float32)
D = pd.concat([O[["source", "letter", "seq"] + FEAT], S[["source", "letter", "seq"] + FEAT], AN[["source", "letter", "seq"] + FEAT]], ignore_index=True)
D.to_csv(ALL_CSV, index=False)
X = D[FEAT].values.astype(np.float32); y = np.array([LETTERS.index(l) for l in D.letter]); src = D.source.values
SOURCES = ["O", "SA", "SB", "AN"]
DESC = {"O": "single-signer corpus (1 person)", "SA": "Surrey signer A (1 person)", "SB": "Surrey signer B (1 person, letters A-F)",
        "AN": "ASLNow! (multiple participants, pooled)"}
res = {"sources": {s: {"description": DESC[s], "frames": int((src == s).sum()), "letters": int(len(set(y[src == s])))} for s in SOURCES}}
res["E0_detection_rate"] = {s: round(float(sur[sur.signer == s[1]].detected.mean()), 4) for s in ["SA", "SB"]}
res["E0_mediapipe_ms_p50"] = round(float(sur.mp_ms.median()), 2)
print("sources", res["sources"], "detection", res["E0_detection_rate"], flush=True)

def metrics(yt, p):
    labs = sorted(set(yt.tolist()))
    return {"accuracy": round(float(accuracy_score(yt, p)), 4), "macro_f1": round(float(f1_score(yt, p, labels=labs, average="macro", zero_division=0)), 4), "n": int(len(yt))}
def mean_of(per, key="accuracy"):
    return round(float(np.mean([v[key] for v in per.values()])), 4)

# ------------------------------------------------------------------ E1: deployed single-signer model on unseen sources
clf_orig = make_clf().fit(X[src == "O"], y[src == "O"])
E1 = {}
for s in ["SA", "SB", "AN"]:
    m = src == s
    a = metrics(y[m], clf_orig.predict(X[m])); b = metrics(y[m], clf_orig.predict(mirror(X[m])))
    best = "as_captured" if a["accuracy"] >= b["accuracy"] else "mirrored"
    E1[s] = {"as_captured": a, "mirrored": b, "best_convention": best, "best": a if best == "as_captured" else b}
m = src == "AN"
E1["AN_16x9_check"] = {"as_captured": metrics(y[m], clf_orig.predict(AN169)), "mirrored": metrics(y[m], clf_orig.predict(mirror(AN169)))}
res["E1_single_signer_model"] = E1
res["E1_mean_best_convention"] = round(float(np.mean([E1[s]["best"]["accuracy"] for s in ["SA", "SB", "AN"]])), 4)
res["E1_mean_as_captured"] = round(float(np.mean([E1[s]["as_captured"]["accuracy"] for s in ["SA", "SB", "AN"]])), 4)
res["E1_note"] = "best_convention = the left/right setting a user picks once in the demo; both are reported"
print("E1", {s: (E1[s]["as_captured"]["accuracy"], E1[s]["mirrored"]["accuracy"]) for s in ["SA", "SB", "AN"]}, flush=True)

# ------------------------------------------------------------------ E2 / E3: leave-one-source-out
E2, E3, models3 = {}, {}, {}
cm = np.zeros((26, 26), int)
for s in SOURCES:
    tr, te = src != s, src == s
    E2[s] = metrics(y[te], make_clf().fit(X[tr], y[tr]).predict(X[te]))
    Xa, ya = augment(X[tr], y[tr], rng)
    m3 = make_clf().fit(Xa, ya); p3 = m3.predict(X[te]); models3[s] = m3
    E3[s] = metrics(y[te], p3); cm += confusion_matrix(y[te], p3, labels=list(range(26)))
    print("LOSO", s, "no-aug", E2[s], "aug", E3[s], flush=True)
res["E2_leave_one_source_out"] = {"per_source": E2, "mean_accuracy": mean_of(E2), "mean_macro_f1": mean_of(E2, "macro_f1")}
res["E3_leave_one_source_out_augmented"] = {"per_source": E3, "mean_accuracy": mean_of(E3), "mean_macro_f1": mean_of(E3, "macro_f1"),
    "min_source": min(E3, key=lambda k: E3[k]["accuracy"]), "min_accuracy": min(v["accuracy"] for v in E3.values()),
    "max_accuracy": max(v["accuracy"] for v in E3.values())}
res["E3_AN_16x9_check"] = metrics(y[src == "AN"], models3["AN"].predict(AN169))
pairs = sorted([(LETTERS[i], LETTERS[j], int(cm[i, j])) for i in range(26) for j in range(26) if i != j and cm[i, j]], key=lambda t: -t[2])
res["E3_top_confusions"] = [{"true": a, "pred": b, "count": c} for a, b, c in pairs[:10]]
res["E3_per_letter_recall"] = {LETTERS[i]: round(float(cm[i, i] / cm[i].sum()), 3) for i in range(26) if cm[i].sum()}

# ------------------------------------------------------------------ E4: optimistic pooled contrast
pred = np.zeros_like(y)
for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
    pred[te] = make_clf().fit(X[tr], y[tr]).predict(X[te])
res["E4_pooled_random_5fold"] = metrics(y, pred)
print("E4", res["E4_pooled_random_5fold"], flush=True)

# ------------------------------------------------------------------ E5: few-shot calibration of a new signer
def calib_split(s, k):
    cal, test = {}, []
    for L, g in D[src == s].groupby("letter"):
        g = g.sort_values("seq")
        cal[LETTERS.index(L)] = g[FEAT].values[:k].astype(np.float32)
        test += list(g.index[int(0.7 * len(g)):] if s == "O" else g.index[k + 10:])
    return cal, np.array(test, dtype=int)
TEMP_FILE = os.path.join(OUT, "calibration_temperature.json")
TEMPS = {"shipped_temp_0.8": 0.8}
if os.path.exists(TEMP_FILE):     # chosen on validation signer SB by select_calibration_temperature.py
    TEMPS["revised_temp"] = json.load(open(TEMP_FILE))["selected_temperature"]
def fewshot(model, s, k, orient, temp):
    cal, test = calib_split(s, max(k, 1))
    if len(test) == 0: return None
    Xt = orient(X[test]); yt = y[test]
    P = np.zeros((len(Xt), 26)); P[:, model.classes_] = model.predict_proba(Xt)
    if k:
        calX = {c: orient(v) for c, v in cal.items()}
        P = np.array([personalise(P[i], Xt[i], calX, weight=0.6, temp=temp) for i in range(len(Xt))])
    am, mx = P.argmax(1), P.max(1)
    r = metrics(yt, am)
    r["confident_correct"] = round(float(np.mean((am == yt) & (mx >= 0.55))), 4)   # frames that would type the right letter
    return r
ident = lambda A: A
REPORT = {"single_signer_model": ["SA"], "leave_one_source_out_model": ["O", "SA"]}   # SB = validation signer, excluded
E5 = {"validation_signer_excluded": "SB", "temperatures": TEMPS}
for tname, temp in TEMPS.items():
    E5[tname] = {"single_signer_model": {}, "leave_one_source_out_model": {}}
    for s in REPORT["single_signer_model"]:
        orient = ident if E1[s]["best_convention"] == "as_captured" else mirror
        E5[tname]["single_signer_model"][s] = {f"k={k}": fewshot(clf_orig, s, k, orient, temp) for k in [0, 1, 3, 5, 10]}
    for s in REPORT["leave_one_source_out_model"]:
        E5[tname]["leave_one_source_out_model"][s] = {f"k={k}": fewshot(models3[s], s, k, ident, temp) for k in [0, 1, 3, 5, 10]}
    for name in ["single_signer_model", "leave_one_source_out_model"]:
        per = E5[tname][name]
        per["mean"] = {f"k={k}": {"accuracy": round(float(np.mean([v[f"k={k}"]["accuracy"] for kk, v in per.items() if kk != "mean"])), 4),
                                  "confident_correct": round(float(np.mean([v[f"k={k}"]["confident_correct"] for kk, v in per.items() if kk != "mean"])), 4)}
                       for k in [0, 1, 3, 5, 10]}
res["E5_fewshot_calibration"] = E5
print("E5", {t: {n: E5[t][n]["mean"]["k=5"] for n in REPORT} for t in TEMPS}, flush=True)

# ------------------------------------------------------------------ export the multi-signer browser model (all sources + augmentation)
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
Xall, yall = augment(X, y, rng)
final = make_clf().fit(Xall, yall)
onx = convert_sklearn(final, initial_types=[("input", FloatTensorType([None, 42]))], target_opset=17, options={id(final): {"zipmap": False}})
path = os.path.join(ROOT, "artifacts", "asl_landmark_mlp_multisigner_web.onnx")
open(path, "wb").write(onx.SerializeToString())
sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
chk = Xall[::7][:2000]
res["multisigner_export"] = {"file": os.path.basename(path), "size_kb": round(os.path.getsize(path) / 1024, 1), "train_rows_after_augmentation": int(len(Xall)),
                             "onnx_parity": round(float(np.mean(sess.run(None, {"input": chk})[0].reshape(-1) == final.predict(chk))), 4)}

# ------------------------------------------------------------------ figures
lab = {"O": "Original\ncorpus", "SA": "Surrey\nsigner A", "SB": "Surrey\nsigner B", "AN": "ASLNow!\n(pooled)"}
fig, ax = plt.subplots(figsize=(8, 3.8)); xs = np.arange(len(SOURCES)); wd = 0.26
e1v = [np.nan] + [E1[s]["best"]["accuracy"] for s in ["SA", "SB", "AN"]]
for i, (name, vals) in enumerate([("Single-signer model (E1)", e1v), ("Leave-one-source-out (E2)", [E2[s]["accuracy"] for s in SOURCES]),
                                  ("… + augmentation (E3)", [E3[s]["accuracy"] for s in SOURCES])]):
    ax.bar(xs + (i - 1) * wd, vals, wd, label=name)
ax.axhline(0.85, ls="--", c="grey", lw=1)
ax.set_xticks(xs); ax.set_xticklabels([lab[s] for s in SOURCES]); ax.set_ylim(0, 1); ax.set_ylabel("Accuracy on the unseen source")
ax.set_title("Letter recognition on people the model never saw in training"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_cross_signer.png"), dpi=150); plt.close()
fig, ax = plt.subplots(figsize=(6.4, 3.6)); ks = [0, 1, 3, 5, 10]
for tname in TEMPS:
    per = E5[tname]["leave_one_source_out_model"]["mean"]
    ax.plot(ks, [per[f"k={k}"]["confident_correct"] for k in ks], "o-", label=f"frames typing the right letter, {tname.replace('_', ' ')}")
ax.axhline(0.85, ls="--", c="grey", lw=1); ax.set_xlabel("Calibration samples per letter from the new signer (k)")
ax.set_ylabel("Share of frames (original signer, Surrey A)"); ax.set_ylim(0, 1); ax.grid(alpha=.3); ax.legend(fontsize=7)
ax.set_title("Few-shot calibration: shipped vs revised blend temperature"); plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_fewshot.png"), dpi=150); plt.close()

json.dump(res, open(os.path.join(OUT, "multisigner_metrics.json"), "w"), indent=2)
print(json.dumps({k: res[k] for k in ["E1_mean_best_convention", "E1_mean_as_captured", "E4_pooled_random_5fold", "multisigner_export"]}, indent=1))
print("E2", res["E2_leave_one_source_out"]["mean_accuracy"], "E3", res["E3_leave_one_source_out_augmented"]["mean_accuracy"])
