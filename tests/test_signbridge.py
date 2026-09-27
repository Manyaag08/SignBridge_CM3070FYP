"""
Unit and integration tests for SignBridge.  Run:  python -m pytest tests -v
Covers preprocessing, the hold-to-type logic, the sign grammar, the sentence-smoother routing, the avatar
planner, few-shot personalisation, both browser ONNX models, and that the Python logic used for evaluation
matches the JavaScript that actually runs in the frontend.
"""
import json, os, re, sys
import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from signbridge_core import (LETTERS, PHRASE_RULES, AVATAR_LEXICON, commit_letters, gloss_to_english,
                             normalise_sentence, orchestrate_smoother, personalise, plan_avatar, preprocess)
import glob
# the browser app is split into frontend/index.html + frontend/js/*.js; the tests read the JavaScript that actually runs
HTML = "\n".join(open(f, encoding="utf-8").read()
                 for f in [os.path.join(ROOT, "frontend", "index.html")] + sorted(glob.glob(os.path.join(ROOT, "frontend", "js", "*.js"))))

# ---------------------------------------------------------------- preprocessing
def test_preprocess_is_wrist_relative_and_scaled():
    rng = np.random.RandomState(0); lm = rng.uniform(0.2, 0.8, (21, 2))
    f = preprocess(lm, 640, 480)
    assert f.shape == (42,) and f[0] == 0 and f[1] == 0
    assert np.isclose(np.abs(f).max(), 1.0)

def test_preprocess_mirror_flips_x_only():
    rng = np.random.RandomState(1); lm = rng.uniform(0.2, 0.8, (21, 2))
    a, b = preprocess(lm, 640, 480).reshape(21, 2), preprocess(lm, 640, 480, mirror=True).reshape(21, 2)
    assert np.allclose(a[:, 1], b[:, 1], atol=0.02) and np.allclose(a[:, 0], -b[:, 0], atol=0.02)

def test_preprocess_matches_training_csv_scale():
    import pandas as pd
    csv = os.path.join(ROOT, "ASL_Recognition", "ACM_Hand_gesture", "keypoint_classifier", "keypoint.csv")
    if not os.path.exists(csv): pytest.skip("dataset not cloned")
    X = pd.read_csv(csv, header=None).iloc[:, 1:43].values
    assert np.allclose(X[:, :2], 0) and np.allclose(np.abs(X).max(1), 1.0)

# ---------------------------------------------------------------- hold-to-type
def test_commit_letters_types_each_held_letter_once():
    frames = [("A", 0.9)] * 45 + [("B", 0.9)] * 45
    assert commit_letters(frames, hold=30, gap=30) == ["A", "B"]

def test_commit_letters_ignores_low_confidence_and_flicker():
    assert commit_letters([("A", 0.4)] * 90) == []
    assert commit_letters([("A", 0.9), ("B", 0.9)] * 60) == []

# ---------------------------------------------------------------- grammar + smoother
@pytest.mark.parametrize("signs,expected", [
    (["ME", "GOOD"], "I am good."), (["HOW", "YOU"], "How are you?"), (["LOVE", "YOU"], "Love you."),
    (["ILY"], "I love you."), (["HELLO", "HOW", "YOU"], "Hello, how are you?"),
    (["H", "E", "L", "L", "O"], "Hello."), (["M", "Y", "_", "N", "A", "M", "E"], "My name."),
    (["WHERE", "B", "U", "S"], "Where bus?")])
def test_gloss_to_english(signs, expected):
    assert gloss_to_english(signs) == expected

def test_python_grammar_matches_live_demo():
    js = re.findall(r'\{text:"([^"]+)",\s*g:\[([^\]]+)\]\}', HTML)
    js_rules = {tuple(x.strip().strip('"') for x in g.split(",")): t for t, g in js}
    assert js_rules == {g: t for g, t in PHRASE_RULES}

def test_smoother_routes_unknown_words_to_llm():
    assert orchestrate_smoother(["ME", "GOOD"]) == "I am good."
    assert orchestrate_smoother(["H", "E", "L", "L", "O"]) == "Hello."
    assert orchestrate_smoother(["H", "E", "L", "L", "L", "O"]) is None     # misrecognised: needs repair

def test_normalise_sentence():
    assert normalise_sentence("I'm GOOD!") == normalise_sentence("i am good")

# ---------------------------------------------------------------- avatar
def test_avatar_plan_asl_order_and_fingerspelling():
    plan, dur = plan_avatar("Hello, how are you? My name is Manya.")
    assert [p["value"] for p in plan] == ["HELLO", "HOW", "YOU", "ME", "NAME", "MANYA"]
    assert [p["type"] for p in plan][-1] == "spell" and dur > 0

def test_avatar_thank_you_is_one_sign():
    assert [p["value"] for p in plan_avatar("Thank you so much")[0]] == ["THANK-YOU", "SO", "MUCH"]

def test_avatar_lexicon_matches_live_demo():
    m = re.search(r"const AVATAR_LEXICON=\{(.*?)\};", HTML, re.S)
    js = dict(re.findall(r'([a-z]+):"([A-Z-]+)"', m.group(1)))
    assert js == AVATAR_LEXICON

def test_every_avatar_sign_has_keyframes_and_handshapes():
    signs = re.search(r"const SIGNS=\{(.*?)\n\};", HTML, re.S).group(1)
    for gloss in set(AVATAR_LEXICON.values()):
        assert f'"{gloss}"' in signs, gloss
    ref = json.load(open(os.path.join(ROOT, "artifacts", "ref_skeletons.json")))
    for shape in set(re.findall(r'shape:"([A-Z])"', signs)):
        assert shape in ref

# ---------------------------------------------------------------- personalisation
def test_personalise_pulls_towards_calibrated_letter():
    probs = np.full(26, 1 / 26); feat = np.zeros(42, np.float32)
    out = personalise(probs, feat, {LETTERS.index("M"): np.zeros((3, 42))}, weight=0.6)
    assert out.argmax() == LETTERS.index("M") and np.isclose(out.sum(), 1.0)

def test_personalise_is_identity_without_exemplars():
    probs = np.random.RandomState(2).dirichlet(np.ones(26))
    assert np.allclose(personalise(probs, np.zeros(42), {}), probs)

# ---------------------------------------------------------------- deployed models
@pytest.mark.parametrize("name", ["asl_landmark_mlp_web.onnx", "asl_landmark_mlp_multisigner_web.onnx"])
def test_browser_onnx_models(name):
    import onnxruntime as ort
    path = os.path.join(ROOT, "artifacts", name)
    if not os.path.exists(path): pytest.skip(f"{name} not built yet (run evaluation/eval_multisigner.py)")
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    ref = json.load(open(os.path.join(ROOT, "artifacts", "ref_skeletons.json")))
    X = np.array([np.array(ref[L], np.float32).reshape(-1) for L in LETTERS])
    label, probs = sess.run(["label", "probabilities"], {"input": X})
    assert probs.shape == (26, 26) and np.allclose(probs.sum(1), 1, atol=1e-4)
    assert (probs.argmax(1) == label).all()
    assert (label == np.arange(26)).mean() >= 0.8   # the reference handshapes are read back correctly

def test_live_demo_references_existing_models():
    for path in re.findall(r'"\./(artifacts/[^"]+\.onnx)"', HTML):
        assert os.path.exists(os.path.join(ROOT, path)) or "multisigner" in path, path

def test_calibration_temperature_consistent_across_demo_python_and_selection():
    import inspect
    js = float(re.search(r"const PERS_TEMP=([0-9.]+);", HTML).group(1))
    py = inspect.signature(personalise).parameters["temp"].default
    assert js == py
    sel = os.path.join(ROOT, "artifacts", "eval", "calibration_temperature.json")
    if os.path.exists(sel):
        assert json.load(open(sel))["selected_temperature"] == js

def test_calibrated_letter_clears_typing_threshold():
    """Realistic case from the end-to-end test: the model is confident in the right letter, the new signer's own
    sample for it is 0.3 away and the samples for the other 25 letters 0.6 away. The shipped temperature (0.8)
    flattened that into a top probability below the 0.55 typing threshold, so nothing was typed; 0.1 must not."""
    probs = np.zeros(26); probs[LETTERS.index("M")] = 1.0
    feat = np.zeros(42, np.float32)
    ex = {}
    for k in range(26):
        e = np.zeros((1, 42), np.float32); e[0, k] = 0.3 if k == LETTERS.index("M") else 0.6
        ex[k] = e
    assert personalise(probs, feat, ex, weight=0.6).argmax() == LETTERS.index("M")
    assert personalise(probs, feat, ex, weight=0.6).max() >= 0.55
    assert personalise(probs, feat, ex, weight=0.6, temp=0.8).max() < 0.55

# ---------------------------------------------------------------- robust message typing and decoding
from signbridge_core import build_message, commit_letters_segment, commit_letters_vote, decode_word, faithful

def _letters(word, p=0.8, swap=None):
    out = []
    for i, c in enumerate(word):
        v = np.full(26, (1 - p) / 25); v[LETTERS.index(c)] = p
        if swap and i in swap:
            v = np.full(26, 0.05 / 24); v[LETTERS.index(swap[i])] = 0.5; v[LETTERS.index(c)] = 0.45
        out.append((int(v.argmax()), v))
    return out

def _frames(word, misread=0.45, seed=0):
    rng = np.random.RandomState(seed); P, hand = [], []
    for c in word:
        for _ in range(30):
            v = np.full(26, 0.01); v[LETTERS.index(c) if rng.rand() > misread else rng.randint(26)] = 0.75
            P.append(v / v.sum()); hand.append(True)
        P += [np.zeros(26)] * 10; hand += [False] * 10
    return np.array(P), hand

def test_segment_typing_does_not_drop_letters_on_noisy_frames():
    P, hand = _frames("SORRY")
    assert ["".join(LETTERS[i] for i, _ in w) for w in commit_letters_segment(P, hand)] == ["SORRY"]

def test_vote_typing_types_each_clean_letter_once_and_splits_words():
    P1, h1 = _frames("NO", misread=0.0); P2, h2 = _frames("GO", misread=0.0)
    P = np.concatenate([P1, np.zeros((20, 26)), P2]); hand = h1 + [False] * 20 + h2
    assert ["".join(LETTERS[i] for i, _ in w) for w in commit_letters_vote(P, hand)] == ["NO", "GO"]

def test_decoder_fixes_a_misread_letter_using_probabilities():
    assert decode_word(_letters("HELLO", swap={1: "A"})) == ("HELLO", "word")

def test_decoder_keeps_a_confident_name_and_flags_garbage():
    assert decode_word(_letters("MANYA", p=0.9)) == ("MANYA", "raw")
    assert decode_word(_letters("XQZJ", p=0.2))[1] == "unclear"

def test_unclear_word_means_please_repeat_not_a_guess():
    sentence, _ = build_message([_letters("CALL"), _letters("XQZJ", p=0.2)])
    assert sentence is None

def test_faithfulness_guard_rejects_invented_content():
    assert faithful("Please call a doctor.", ["CALL", "A", "DOCTOR"])
    assert not faithful("The letter H stands for Hello.", ["H", "L"])


def test_demo_vote_threshold_matches_validation_choice():
    js = float(re.search(r"const VOTE_AGREE=([0-9.]+);", HTML).group(1))
    sel = os.path.join(ROOT, "artifacts", "eval", "message_decoding_params.json")
    if not os.path.exists(sel):
        pytest.skip("run evaluation/tune_message_decoding.py first")
    assert json.load(open(sel))["chosen_vote"]["agree"] == js
