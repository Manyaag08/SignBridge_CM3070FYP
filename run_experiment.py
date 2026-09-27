"""
SignBridge feature prototype: landmark-based ASL fingerspelling classifier.

Validates the core custom-trained component of the SignBridge pipeline:
  MediaPipe 21 hand landmarks (42 coords)  ->  ONNX MLP  ->  ASL letter label.

Produces: metrics.json + four figures, and exports the trained model to ONNX.
Also builds a representative pixel-CNN as an ONNX graph and benchmarks both
classifiers under the SAME runtime (onnxruntime, single CPU thread) to test the
design claim that landmark classification is far cheaper than a raw-pixel CNN.
"""
import json, time, os
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

RNG = np.random.RandomState(42)

# All paths are resolved relative to THIS script's folder, so the prototype runs
# from wherever you unzip it (no hard-coded machine paths).
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "artifacts")
os.makedirs(OUT, exist_ok=True)
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
results = {}

# ---------------------------------------------------------------- load data
# The landmark CSV comes from the public repo MohdDilshad-nitk/ASL_Recognition.
# Clone it INSIDE this prototype folder, next to run_experiment.py:
#     git clone https://github.com/MohdDilshad-nitk/ASL_Recognition
# The script looks for the clone here first, then one level up, so it works
# whether you cloned it inside or beside the prototype folder.
REL = os.path.join("ASL_Recognition", "ACM_Hand_gesture",
                   "keypoint_classifier", "keypoint.csv")
candidates = [os.path.join(HERE, REL), os.path.join(HERE, "..", REL)]
CSV = next((c for c in candidates if os.path.exists(c)), None)
if CSV is None:
    raise SystemExit(
        "Could not find the dataset CSV.\n"
        "From inside this folder, run:\n"
        "    git clone https://github.com/MohdDilshad-nitk/ASL_Recognition\n"
        "then re-run:  python run_experiment.py\n"
        "Looked in:\n  " + "\n  ".join(os.path.abspath(c) for c in candidates))
df = pd.read_csv(CSV, header=None)
X = df.iloc[:, 1:43].values.astype(np.float32)   # 21 landmarks x (x,y) = 42 coords
y = df.iloc[:, 0].values.astype(np.int64)
n, d = X.shape
results["dataset"] = {
    "source": "MohdDilshad-nitk/ASL_Recognition (MediaPipe Hands keypoint.csv)",
    "n_samples": int(n), "n_features": int(d), "n_classes": int(len(np.unique(y))),
    "per_class_counts": {LABELS[k]: int((y == k).sum()) for k in np.unique(y)},
    "note": "single contiguous block per letter (one recording session per class)"
}
print(f"Loaded {n} samples, {d} features, {len(np.unique(y))} classes")

def make_clf():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                              alpha=1e-3, max_iter=600, random_state=42))])

# ------------------------------------------- Eval 1: stratified 5-fold (optimistic)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred_cv = cross_val_predict(make_clf(), X, y, cv=skf)
acc_cv = accuracy_score(y, y_pred_cv)
f1_cv = f1_score(y, y_pred_cv, average="macro")
results["eval_stratified_kfold"] = {
    "accuracy": round(float(acc_cv), 4), "macro_f1": round(float(f1_cv), 4),
    "caveat": "frames within a class are temporally correlated; this over-estimates "
              "real performance because correlated frames leak across folds"}
print(f"[Eval 1] stratified 5-fold: acc={acc_cv:.4f}  macroF1={f1_cv:.4f}")

# ------------------------------------------- Eval 2: temporal hold-out (honest)
# train on first 70% of each class's contiguous frames, test on last 30%
tr_idx, te_idx = [], []
for k in np.unique(y):
    idx = np.where(y == k)[0]            # already contiguous & in capture order
    cut = int(0.7 * len(idx))
    tr_idx += list(idx[:cut]); te_idx += list(idx[cut:])
tr_idx, te_idx = np.array(tr_idx), np.array(te_idx)
clf = make_clf().fit(X[tr_idx], y[tr_idx])
y_pred_temp = clf.predict(X[te_idx])
acc_temp = accuracy_score(y[te_idx], y_pred_temp)
f1_temp = f1_score(y[te_idx], y_pred_temp, average="macro")
results["eval_temporal_split"] = {
    "accuracy": round(float(acc_temp), 4), "macro_f1": round(float(f1_temp), 4),
    "train_n": int(len(tr_idx)), "test_n": int(len(te_idx)),
    "note": "later frames of each recording held out -> reduces frame leakage"}
print(f"[Eval 2] temporal hold-out: acc={acc_temp:.4f}  macroF1={f1_temp:.4f}")

# ------------------------------------------- per-class accuracy + confusion matrix
cm = confusion_matrix(y, y_pred_cv, labels=list(range(26)))
per_class_acc = cm.diagonal() / cm.sum(axis=1)
results["per_class_accuracy"] = {LABELS[k]: round(float(per_class_acc[k]), 3)
                                 for k in range(26)}
# most-confused off-diagonal pairs
conf_pairs = []
for i in range(26):
    for j in range(26):
        if i != j and cm[i, j] > 0:
            conf_pairs.append((LABELS[i], LABELS[j], int(cm[i, j])))
conf_pairs.sort(key=lambda t: -t[2])
results["top_confusions"] = [{"true": a, "pred": b, "count": c}
                             for a, b, c in conf_pairs[:8]]
print("[Confusions] top:", conf_pairs[:6])

# Figure: confusion matrix (row-normalised)
cmn = cm / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(7.2, 6.2))
im = ax.imshow(cmn, cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(26)); ax.set_xticklabels(LABELS, fontsize=7)
ax.set_yticks(range(26)); ax.set_yticklabels(LABELS, fontsize=7)
ax.set_xlabel("Predicted letter"); ax.set_ylabel("True letter")
ax.set_title("Row-normalised confusion matrix (stratified 5-fold)")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="proportion")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_confusion.png", dpi=150); plt.close()

# Figure: per-class accuracy
order = np.argsort(per_class_acc)
fig, ax = plt.subplots(figsize=(8, 3.4))
colors = ["#d9534f" if per_class_acc[k] < 0.97 else "#3aa17e" for k in order]
ax.bar([LABELS[k] for k in order], per_class_acc[order], color=colors)
ax.set_ylim(0.8, 1.0); ax.set_ylabel("Accuracy")
ax.set_title("Per-letter accuracy (sorted) - red = below 97%")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_perclass.png", dpi=150); plt.close()

# ------------------------------------------- Eval 3: robustness to landmark noise
# Simulate signer/camera variation by perturbing the 21 (x,y) landmarks with
# isotropic jitter, small in-plane rotation, and scale changes, then re-test.
clf_full = make_clf().fit(X, y)   # fit once; test on perturbed copies of all data

def perturb(Xb, jitter, rot_deg, scale_sd):
    pts = Xb.reshape(-1, 21, 2).copy()
    # rotation about centroid
    th = np.deg2rad(RNG.uniform(-rot_deg, rot_deg, size=len(pts)))
    c, s = np.cos(th), np.sin(th)
    ctr = pts.mean(axis=1, keepdims=True)
    pts -= ctr
    xr = pts[..., 0] * c[:, None] - pts[..., 1] * s[:, None]
    yr = pts[..., 0] * s[:, None] + pts[..., 1] * c[:, None]
    pts = np.stack([xr, yr], axis=-1)
    # per-sample isotropic scale
    sc = 1.0 + RNG.normal(0, scale_sd, size=(len(pts), 1, 1))
    pts *= sc
    pts += ctr
    # gaussian jitter
    pts += RNG.normal(0, jitter, size=pts.shape)
    return pts.reshape(len(Xb), 42).astype(np.float32)

levels = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10]
rob_acc = []
for lv in levels:
    Xp = perturb(X, jitter=lv, rot_deg=lv * 150, scale_sd=lv * 1.2)
    rob_acc.append(float(accuracy_score(y, clf_full.predict(Xp))))
results["robustness_curve"] = {"perturbation_level": levels,
                               "accuracy": [round(a, 4) for a in rob_acc]}
print("[Eval 3] robustness:", [round(a, 3) for a in rob_acc])

fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot([l * 100 for l in levels], rob_acc, "o-", color="#2f6f8f", lw=2)
ax.set_xlabel("Perturbation level (% of normalised hand size)")
ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.02)
ax.set_title("Robustness to landmark jitter + rotation + scale")
ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_robustness.png", dpi=150); plt.close()

# ------------------------------------------- Export landmark MLP to ONNX + parity
onx = convert_sklearn(clf_full, initial_types=[("input", FloatTensorType([None, 42]))],
                      target_opset=17)
mlp_path = f"{OUT}/asl_landmark_mlp.onnx"
with open(mlp_path, "wb") as f:
    f.write(onx.SerializeToString())
mlp_size = os.path.getsize(mlp_path)
sess_mlp = ort.InferenceSession(mlp_path, providers=["CPUExecutionProvider"])
in_name = sess_mlp.get_inputs()[0].name
onnx_pred = np.array([int(r) if np.ndim(r) == 0 else int(r[0])
                      for r in sess_mlp.run(None, {in_name: X})[0]])
parity = float(np.mean(onnx_pred == clf_full.predict(X)))
results["onnx_export"] = {"model_size_bytes": int(mlp_size),
                          "model_size_kb": round(mlp_size / 1024, 1),
                          "sklearn_vs_onnx_agreement": round(parity, 4)}
print(f"[ONNX] MLP size={mlp_size/1024:.1f} KB  parity={parity:.4f}")

# ------------------------------------------- Build representative pixel-CNN (ONNX)
def init(name, arr):
    return numpy_helper.from_array(arr.astype(np.float32), name)

def conv_w(o, i, k=3): return RNG.randn(o, i, k, k).astype(np.float32) * 0.05
inits, nodes = [], []
def conv_block(x_in, w_name, b_name, oc, ic, out_name, pool_name):
    inits.append(init(w_name, conv_w(oc, ic)))
    inits.append(init(b_name, np.zeros(oc, np.float32)))
    nodes.append(helper.make_node("Conv", [x_in, w_name, b_name], [out_name + "_c"],
                                   kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]))
    nodes.append(helper.make_node("Relu", [out_name + "_c"], [out_name + "_r"]))
    nodes.append(helper.make_node("MaxPool", [out_name + "_r"], [pool_name],
                                   kernel_shape=[2, 2], strides=[2, 2]))

conv_block("image", "w1", "b1", 32, 3, "c1", "p1")     # 200->100
conv_block("p1", "w2", "b2", 64, 32, "c2", "p2")        # 100->50
conv_block("p2", "w3", "b3", 128, 64, "c3", "p3")       # 50->25
nodes.append(helper.make_node("GlobalAveragePool", ["p3"], ["gap"]))   # [1,128,1,1]
nodes.append(helper.make_node("Flatten", ["gap"], ["flat"], axis=1))   # [1,128]
inits.append(init("fw1", (RNG.randn(128, 128) * 0.05)))
inits.append(init("fb1", np.zeros(128, np.float32)))
nodes.append(helper.make_node("Gemm", ["flat", "fw1", "fb1"], ["d1"]))
nodes.append(helper.make_node("Relu", ["d1"], ["d1r"]))
inits.append(init("fw2", (RNG.randn(128, 26) * 0.05)))
inits.append(init("fb2", np.zeros(26, np.float32)))
nodes.append(helper.make_node("Gemm", ["d1r", "fw2", "fb2"], ["logits"]))
graph = helper.make_graph(
    nodes, "pixel_cnn",
    [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 200, 200])],
    [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 26])],
    inits)
cnn_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
cnn_model.ir_version = 9
onnx.checker.check_model(cnn_model)
cnn_path = f"{OUT}/pixel_cnn_baseline.onnx"
onnx.save(cnn_model, cnn_path)
cnn_size = os.path.getsize(cnn_path)

# analytical MACs
mlp_macs = 42*128 + 128*64 + 64*26
cnn_macs = (200*200*32*(3*3*3) + 100*100*64*(3*3*32) + 50*50*128*(3*3*64)
            + 128*128 + 128*26)

# ------------------------------------------- Benchmark both under onnxruntime (1 thread)
so = ort.SessionOptions()
so.intra_op_num_threads = 1
so.inter_op_num_threads = 1
sess_mlp1 = ort.InferenceSession(mlp_path, sess_options=so, providers=["CPUExecutionProvider"])
sess_cnn1 = ort.InferenceSession(cnn_path, sess_options=so, providers=["CPUExecutionProvider"])
cnn_in = sess_cnn1.get_inputs()[0].name

def bench(sess, name, feed, runs=200, warm=20):
    for _ in range(warm): sess.run(None, feed)
    t = []
    for _ in range(runs):
        s = time.perf_counter(); sess.run(None, feed); t.append((time.perf_counter()-s)*1000)
    t = np.array(t)
    return {"p50_ms": round(float(np.percentile(t, 50)), 3),
            "p95_ms": round(float(np.percentile(t, 95)), 3),
            "mean_ms": round(float(t.mean()), 3)}

one_lm = X[:1].astype(np.float32)
one_img = (RNG.rand(1, 3, 200, 200).astype(np.float32))
mlp_bench = bench(sess_mlp1, "mlp", {in_name: one_lm})
cnn_bench = bench(sess_cnn1, "cnn", {cnn_in: one_img})
ratio = cnn_bench["p50_ms"] / mlp_bench["p50_ms"]

results["benchmark_cpu_1thread"] = {
    "landmark_mlp": {**mlp_bench, "size_kb": round(mlp_size/1024, 1),
                     "params_macs": int(mlp_macs),
                     "max_fps_single_thread": round(1000/mlp_bench["p50_ms"], 0)},
    "pixel_cnn": {**cnn_bench, "size_kb": round(cnn_size/1024, 1),
                  "params_macs": int(cnn_macs),
                  "max_fps_single_thread": round(1000/cnn_bench["p50_ms"], 1)},
    "cnn_slower_than_mlp_x": round(ratio, 1),
    "cnn_more_macs_x": round(cnn_macs/mlp_macs, 0)}
print(f"[Bench] MLP p50={mlp_bench['p50_ms']}ms  CNN p50={cnn_bench['p50_ms']}ms  "
      f"ratio={ratio:.1f}x  MAC ratio={cnn_macs/mlp_macs:.0f}x")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.6))
a1.bar(["Landmark\nMLP", "Pixel\nCNN"], [mlp_bench["p50_ms"], cnn_bench["p50_ms"]],
       color=["#3aa17e", "#d9534f"])
a1.set_ylabel("Inference latency p50 (ms, log)"); a1.set_yscale("log")
a1.set_title(f"CPU latency - CNN {ratio:.0f}x slower")
for i, v in enumerate([mlp_bench["p50_ms"], cnn_bench["p50_ms"]]):
    a1.text(i, v, f" {v:.2f}ms", ha="center", va="bottom", fontsize=9)
a2.bar(["Landmark\nMLP", "Pixel\nCNN"], [mlp_macs, cnn_macs], color=["#3aa17e", "#d9534f"])
a2.set_ylabel("Multiply-accumulates (log)"); a2.set_yscale("log")
a2.set_title(f"Compute - CNN {cnn_macs/mlp_macs:.0f}x more MACs")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_latency.png", dpi=150); plt.close()

# end-to-end budget context (design target < 500 ms sign->speech)
results["latency_budget_context"] = {
    "design_target_end_to_end_ms": 500,
    "classifier_p50_ms": mlp_bench["p50_ms"],
    "classifier_share_of_budget_pct": round(100*mlp_bench["p50_ms"]/500, 2)}

with open(f"{OUT}/metrics.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved metrics.json and 4 figures to", OUT)
print(json.dumps({k: results[k] for k in
      ["eval_stratified_kfold", "eval_temporal_split", "benchmark_cpu_1thread"]}, indent=2))
