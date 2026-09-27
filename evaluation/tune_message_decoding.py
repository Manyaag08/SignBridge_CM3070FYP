"""
Choose the typing rule and dictionary-decoder settings on VALIDATION messages only.

Validation messages share no words with the 15 end-to-end test messages, and are signed by the ASLNow!
participants (model trained leave-ASLNow!-out, calibrated with 5 samples per letter at the revised temperature,
exactly like the test). Each message is played twice with different frame draws.
Objective: exact messages - 2 x wrong messages spoken (saying the wrong thing is worse than asking to repeat).
"""
import itertools, json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from signbridge_core import LETTERS, augment, build_message, commit_letters_segment, commit_letters_vote, make_clf, personalise
OUT = os.path.join(HERE, "..", "artifacts", "eval")
VAL_MESSAGES = ["COME HOME", "OPEN DOOR", "WHAT TIME", "BUS STOP", "NURSE", "BACK PAIN", "TURN RIGHT", "TURN LEFT", "WE CAN GO NOW",
                "BOOK TABLE", "LUCY", "RAHUL", "TIRED", "COLD", "HUNGRY", "BATHROOM", "MORNING", "EVENING", "PHONE", "FAMILY"]
TEST_WORDS = set("HI HELLO YES NO HELP THANKS GOOD SORRY PLEASE WAIT CALL A DOCTOR MY NAME IS ANNA SEE YOU LATER I FEEL SICK WATER THANK".split())
assert not TEST_WORDS & {w for m in VAL_MESSAGES for w in m.split()}, "validation must not reuse test words"

D = pd.read_csv(os.path.join(OUT, "landmarks_all_sources.csv")); FEAT = [f"f{i}" for i in range(42)]
X = D[FEAT].values.astype(np.float32); y = np.array([LETTERS.index(l) for l in D.letter]); src = D.source.values
TEMP = json.load(open(os.path.join(OUT, "calibration_temperature.json")))["selected_temperature"]
VAL = "AN"; rng = np.random.RandomState(42); RNG = np.random.RandomState(11)
Xa, ya = augment(X[src != VAL], y[src != VAL], rng); model = make_clf().fit(Xa, ya)
te = D[src == VAL]; pools, cal = {}, {}
for L, g in te.groupby("letter"):
    g = g.sort_values("seq"); cal[LETTERS.index(L)] = g[FEAT].values[:5].astype(np.float32); pools[L] = g[FEAT].values[5:].astype(np.float32)

plays = []                                                  # (message, probs, hand)
for msg in VAL_MESSAGES:
    for _ in range(2):
        frames = []
        for wi, word in enumerate(msg.split()):
            if wi: frames += [None] * 20
            for L in word:
                st = RNG.randint(0, len(pools[L])); frames += [pools[L][(st + i) % len(pools[L])] for i in range(30)] + [None] * 10
        feats = np.array([f if f is not None else np.zeros(42, np.float32) for f in frames])
        P = np.zeros((len(frames), 26)); P[:, model.classes_] = model.predict_proba(feats)
        P = np.array([personalise(P[i], feats[i], cal, 0.6, temp=TEMP) for i in range(len(P))])
        plays.append((msg, P, [f is not None for f in frames]))

def score(typing, dec):
    exact = wrong = repeat = 0
    for msg, P, hand in plays:
        words = typing(P, hand)
        if dec is None:                                     # no decoder: say exactly what was typed
            sent = " ".join("".join(LETTERS[i] for i, _ in w) for w in words) or None
            norm = sent
        else:
            sent, _ = build_message(words, **dec)
            norm = sent.rstrip(".?").upper() if sent else None
        if norm is None: repeat += 1
        elif norm == msg: exact += 1
        else: wrong += 1
    n = len(plays)
    return {"exact": round(exact / n, 3), "wrong_spoken": round(wrong / n, 3), "repeat_request": round(repeat / n, 3), "objective": round((exact - 2 * wrong) / n, 3)}

TYPING = {f"vote_{a}": (lambda a: lambda P, h: commit_letters_vote(P, h, agree=a))(a) for a in [0.5, 0.6, 0.7]}
TYPING["segment"] = lambda P, h: commit_letters_segment(P, h)
GRID = [dict(freq_weight=f, indel_penalty=i, keep_raw_margin=k, min_letter_logp=float(np.log(m)))
        for f, i, k, m in itertools.product([0.25, 0.5, 1.0], [1.5, 3.0], [2.0, 4.0], [0.15, 0.3])]
table = {}
for tname, tf in TYPING.items():
    table[tname] = {"no_decoder": score(tf, None)}
    best = None
    for dec in GRID:
        r = score(tf, dec)
        if best is None or r["objective"] > best[0]["objective"]:
            best = (r, dec)
    table[tname]["best_decoder"] = {**best[0], "params": {**best[1], "min_letter_logp": round(best[1]["min_letter_logp"], 4)}}
    print(tname, table[tname], flush=True)
best_typing = max(TYPING, key=lambda t: table[t]["best_decoder"]["objective"])
best_vote = max([t for t in TYPING if t.startswith("vote")], key=lambda t: table[t]["best_decoder"]["objective"])
res = {"validation": f"{len(VAL_MESSAGES)} messages x 2 plays, ASLNow! participants (held out), calibrated k=5, temp {TEMP}",
       "objective": "exact - 2 x wrong spoken", "table": table,
       "chosen": {"typing": best_typing, "decoder": table[best_typing]["best_decoder"]["params"]},
       "chosen_vote": {"agree": float(best_vote.split("_")[1]), "decoder": table[best_vote]["best_decoder"]["params"]}}
json.dump(res, open(os.path.join(OUT, "message_decoding_params.json"), "w"), indent=2)
print(json.dumps(res["chosen"]), json.dumps(res["chosen_vote"]))
