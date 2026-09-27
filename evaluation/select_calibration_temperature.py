"""
Choose the temperature of the few-shot calibration blend (personalise) on a VALIDATION signer only.

The end-to-end test showed that the live demo's temperature (0.8) flattens the blended probabilities so much
that most frames fall below the 0.55 confidence needed to type a letter. The replacement value is selected
here on Surrey signer B, which is then excluded from every reported calibration result, so the choice is not
tuned on the numbers the report quotes. Criterion: frames that are both correct and >= 0.55 confident (i.e.
frames that would actually type the right letter), with k = 5 calibration samples per letter.
"""
import json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from signbridge_core import LETTERS, augment, make_clf, personalise
OUT = os.path.join(HERE, "..", "artifacts", "eval")
D = pd.read_csv(os.path.join(OUT, "landmarks_all_sources.csv")); FEAT = [f"f{i}" for i in range(42)]
X = D[FEAT].values.astype(np.float32); y = np.array([LETTERS.index(l) for l in D.letter]); src = D.source.values
VAL = "SB"; rng = np.random.RandomState(42)
Xa, ya = augment(X[src != VAL], y[src != VAL], rng); model = make_clf().fit(Xa, ya)
cal, test = {}, []
for L, g in D[src == VAL].groupby("letter"):
    g = g.sort_values("seq"); cal[LETTERS.index(L)] = g[FEAT].values[:5].astype(np.float32); test += list(g.index[15:])
Xt, yt = X[test], y[test]
P0 = np.zeros((len(Xt), 26)); P0[:, model.classes_] = model.predict_proba(Xt)
grid = {}
for t in [0.8, 0.5, 0.3, 0.2, 0.15, 0.1, 0.05]:
    P = np.array([personalise(P0[i], Xt[i], cal, weight=0.6, temp=t) for i in range(len(Xt))])
    am, mx = P.argmax(1), P.max(1)
    grid[str(t)] = {"argmax_accuracy": round(float(np.mean(am == yt)), 4), "confident_correct": round(float(np.mean((am == yt) & (mx >= 0.55))), 4),
                    "median_confidence": round(float(np.median(mx)), 3)}
am0, mx0 = P0.argmax(1), P0.max(1)
base = {"argmax_accuracy": round(float(np.mean(am0 == yt)), 4), "confident_correct": round(float(np.mean((am0 == yt) & (mx0 >= 0.55))), 4)}
best = max(grid, key=lambda t: (grid[t]["confident_correct"], -abs(float(t) - 0.8)))
res = {"validation_source": VAL, "shipped_temperature": 0.8, "selected_temperature": float(best), "no_calibration": base, "grid": grid}
json.dump(res, open(os.path.join(OUT, "calibration_temperature.json"), "w"), indent=2)
print(json.dumps(res, indent=1))
