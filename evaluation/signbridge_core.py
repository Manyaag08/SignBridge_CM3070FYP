"""
Shared SignBridge logic used by the evaluation harness and the tests.
Each function mirrors the behaviour of the running system so the numbers describe what ships:
  - preprocess()        == handFeatures() in live_demo.html and the training CSV format
  - personalise()       == personalise() in live_demo.html (few-shot signer calibration)
  - commit_letters()    == the hold-to-type logic of live_demo.html
  - gloss_to_english()  == the sign grammar (PHRASE_RULES) in live_demo.html, plus a word lexicon
  - plan_avatar()       == the text -> sign plan that drives the signing avatar
"""
import os
import re
import numpy as np

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# ---------------------------------------------------------------- landmarks
def preprocess(lm_xy, w, h, mirror=False):
    """21 normalised (x, y) landmarks -> 42-d wrist-relative, max-abs scaled vector (pixel grid, like training)."""
    pts = np.array([[min(int((1 - x if mirror else x) * w), w - 1), min(int(y * h), h - 1)] for x, y in lm_xy], np.float32)
    rel = (pts - pts[0]).reshape(-1)
    m = np.abs(rel).max() or 1.0
    return rel / m

def personalise(probs, feat, exemplars, weight=0.6, temp=0.1, floor=2.0):
    """Blend model probabilities with a nearest-exemplar softmax over this signer's calibrated letters."""
    if not exemplars or weight <= 0:
        return probs
    pp = np.zeros_like(probs)
    for k, ex in exemplars.items():
        d = np.sqrt(((np.asarray(ex) - feat) ** 2).sum(1)).min()
        pp[k] = np.exp(-d / temp)
    fl = np.exp(-floor / temp); total = pp.sum() + fl
    if pp.sum() <= 0:
        return probs
    return (1 - weight) * probs + weight * (pp / total + (fl / total) * probs)

def commit_letters(frames, conf=0.55, hold=30, gap=30):
    """frames: [(letter, p)] at 30 fps. A letter is typed after `hold` consecutive confident frames; a new
    letter needs another `gap` frames (mirrors COMMIT_MS = 1000 ms in the demo)."""
    out, last, run, since_commit = [], None, 0, gap
    for letter, p in frames:
        since_commit += 1
        if letter is None or p < conf:
            last, run = None, 0; continue
        run = run + 1 if letter == last else 1
        last = letter
        if run >= hold and since_commit >= gap:
            out.append(letter); since_commit = 0; run = 0
    return out

# ---------------------------------------------------------------- sign grammar
PHRASE_RULES = [
    (("ME", "GOOD"), "I am good."), (("HOW", "YOU"), "How are you?"), (("HELLO", "HOW", "YOU"), "Hello, how are you?"),
    (("LOVE", "YOU"), "Love you."), (("ME", "LOVE", "YOU"), "I love you."), (("ILY",), "I love you."), (("YOU", "GOOD"), "You good?"),
]
LEXICON = {"ME": "I", "YOU": "you", "MY": "my", "NAME": "name", "GOOD": "good", "LOVE": "love", "HOW": "how",
           "HELLO": "hello", "THANK-YOU": "thank you", "NEED": "need", "HELP": "help", "WHERE": "where", "WHAT": "what",
           "FEEL": "feel", "YES": "yes", "NO": "no", "ILY": "I love you", "PLEASE": "please", "SORRY": "sorry"}
QUESTION = {"how", "what", "where", "who", "why", "when"}

def _tokens(signs):
    """Group single letters into fingerspelled words; '_' (or SPACE) marks a word boundary."""
    toks, word = [], ""
    for s in signs:
        if s in ("_", "SPACE"):
            if word: toks.append(("word", word)); word = ""
        elif len(s) == 1 and s.isalpha():
            word += s
        else:
            if word: toks.append(("word", word)); word = ""
            toks.append(("gloss", s))
    if word: toks.append(("word", word))
    return toks

def gloss_to_english(signs, strict=False):
    toks = _tokens(signs)
    gl = tuple(v for t, v in toks if t == "gloss")
    if len(gl) == len(toks):
        for g, text in PHRASE_RULES:
            if g == gl:
                return text
    if strict:
        return None
    words = [LEXICON.get(v, v.lower()) if t == "gloss" else v.lower() for t, v in toks]
    if not words:
        return ""
    s = " ".join(words)
    s = s[0].upper() + s[1:]
    return s + ("?" if words[0].split()[0] in QUESTION else ".")

_DICT = None
def _dictionary():
    global _DICT
    if _DICT is None:
        path = "/usr/share/dict/words"
        _DICT = {w.strip().lower() for w in open(path)} if os.path.exists(path) else set()
        _DICT |= {v.lower() for v in LEXICON.values()} | {"hi", "thanks", "okay", "ok"}
    return _DICT

def orchestrate_smoother(signs):
    """Deterministic path first: the sign grammar, then lexicon + fingerspelled words that are real
    dictionary words. Returns None when the LLM is needed (a spelled word is not a dictionary word,
    which usually means a recognition error the language model may be able to repair)."""
    g = gloss_to_english(signs, strict=True)
    if g:
        return g
    words = [v.lower() for t, v in _tokens(signs) if t == "word"]
    if all(w in _dictionary() for w in words):
        return gloss_to_english(signs)
    return None

def normalise_sentence(s):
    s = s.lower().replace("’", "'").strip()
    for a, b in [("i'm", "i am"), ("what's", "what is"), ("name's", "name is"), ("it's", "it is"), ("you're", "you are")]:
        s = s.replace(a, b)
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

# ---------------------------------------------------------------- signing avatar planner
# Lexical signs the avatar can animate (keyframed); everything else is fingerspelled.
AVATAR_LEXICON = {
    "hello": "HELLO", "hi": "HELLO", "how": "HOW", "you": "YOU", "your": "YOU", "i": "ME", "me": "ME", "my": "ME",
    "good": "GOOD", "fine": "GOOD", "love": "LOVE", "yes": "YES", "thank": "THANK-YOU", "thanks": "THANK-YOU",
    "please": "PLEASE", "sorry": "SORRY", "help": "HELP", "name": "NAME", "no": "NO", "what": "WHAT",
    "where": "WHERE", "need": "NEED", "feel": "FEEL",
}
DROP = {"am", "is", "are", "a", "an", "the", "to", "do", "does", "be"}   # ASL drops copulas and articles
SIGN_S, LETTER_S, PAUSE_S = 0.9, 0.4, 0.25

def plan_avatar(text):
    """English text -> ordered sign plan [{'type': 'sign'|'spell', 'value': ...}] plus a duration estimate."""
    words = re.findall(r"[a-z']+", text.lower())
    plan, skip_next = [], False
    for i, w in enumerate(words):
        if skip_next:
            skip_next = False; continue
        if w == "thank" and i + 1 < len(words) and words[i + 1] == "you":
            plan.append({"type": "sign", "value": "THANK-YOU"}); skip_next = True; continue
        if w in DROP:
            continue
        w = w.replace("'", "")
        if w in AVATAR_LEXICON:
            plan.append({"type": "sign", "value": AVATAR_LEXICON[w]})
        else:
            plan.append({"type": "spell", "value": re.sub(r"[^a-z]", "", w).upper()})
    dur = sum(SIGN_S if p["type"] == "sign" else LETTER_S * len(p["value"]) for p in plan) + PAUSE_S * max(0, len(plan) - 1)
    return plan, round(dur, 2)

# ---------------------------------------------------------------- shared training helpers (evaluation scripts)
def make_clf():
    """The letter classifier exactly as run_experiment.py defines it."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([("scaler", StandardScaler()),
                     ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", alpha=1e-3, max_iter=600, random_state=42))])

def mirror(X):
    """Flip 42-d landmark vectors left-right (a left hand becomes a right hand)."""
    P = np.asarray(X, np.float32).reshape(-1, 21, 2).copy()
    P[..., 0] *= -1
    return P.reshape(len(P), 42)

def augment(X, y, rng, copies=2):
    """Training-time augmentation: rotation (+-15 deg), scale (+-10%), jitter, re-normalised; plus a mirrored copy."""
    X = np.asarray(X, np.float32)
    outX, outy = [X], [y]
    for _ in range(copies):
        P = X.reshape(-1, 21, 2)
        th = np.deg2rad(rng.uniform(-15, 15, len(P))); c, s = np.cos(th), np.sin(th)
        R = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)
        Q = np.einsum("nij,nkj->nki", R, P) * rng.uniform(0.9, 1.1, (len(P), 1, 1)) + rng.normal(0, 0.02, P.shape)
        Q = Q / np.abs(Q).reshape(len(Q), -1).max(1)[:, None, None]
        outX.append(Q.reshape(len(X), 42).astype(np.float32)); outy.append(y)
    outX.append(mirror(X)); outy.append(y)
    return np.concatenate(outX), np.concatenate(outy)

# ---------------------------------------------------------------- robust typing + faithful sentence building
def commit_letters_vote(probs, hand, win=20, agree=0.7, conf=0.55, gap=20):
    """Majority-vote typing. probs: (T, 26) per-frame probabilities; hand: per-frame bool (a hand is visible).
    A letter is typed when at least `agree` of the last `win` hand frames confidently vote for it; one misread
    frame no longer restarts the count. The typed letter is locked until the hand drops or another letter wins.
    `gap` hand-free frames mark a word boundary. Returns a list of words; each word is a list of
    (letter_index, mean probability vector over the voting window)."""
    words, cur, buf, locked, empty = [], [], [], None, 0
    for p, h in zip(probs, hand):
        if not h:
            empty += 1; buf, locked = [], None
            if empty == gap and cur:
                words.append(cur); cur = []
            continue
        empty = 0
        buf.append(p); buf = buf[-win:]
        if len(buf) < win:
            continue
        votes = np.bincount([int(q.argmax()) for q in buf if q.max() >= conf], minlength=26)
        top = int(votes.argmax())
        if votes[top] >= agree * win and top != locked:
            cur.append((top, np.mean(buf, 0))); locked = top
    if cur:
        words.append(cur)
    return words

_LEX = None
def word_lexicon(n=30000):
    """The n most frequent English words (wordfreq) with their Zipf frequency, plus the sign-grammar words."""
    global _LEX
    if _LEX is None:
        from wordfreq import top_n_list, zipf_frequency
        words = {w.upper() for w in top_n_list("en", n) if w.isalpha() and w.isascii()}
        words |= {v.upper() for v in LEXICON.values() if " " not in v}
        _LEX = {}
        for w in words:
            _LEX.setdefault(len(w), []).append((w, zipf_frequency(w.lower(), "en")))
        for L, items in _LEX.items():
            _LEX[L] = ([w for w, _ in items], np.array([[LETTERS.index(c) for c in w] for w, _ in items]), np.array([z for _, z in items]))
    return _LEX

def commit_letters_segment(probs, hand, min_frames=10, gap=20):
    """Segment-then-classify typing: every run of hand frames between pauses is one held letter, typed as the
    letter with the highest mean probability over that run, so letters are not dropped when frame-by-frame
    agreement is low. Assumes letters are separated by a brief pause or transition (as in the test playback);
    a live camera needs motion-based segmentation for this. `gap` hand-free frames mark a word boundary."""
    words, cur, seg, empty = [], [], [], 0
    def close():
        if len(seg) >= min_frames:
            m = np.mean(seg, 0); cur.append((int(m.argmax()), m))
    for p, h in zip(probs, hand):
        if h:
            if empty and seg:
                close(); seg = []
            if empty >= gap and cur:
                words.append(cur); cur = []
            empty = 0; seg.append(p)
        else:
            empty += 1
    if seg:
        close()
    if cur:
        words.append(cur)
    return words

def decode_word(letters, freq_weight=0.5, indel_penalty=4.0, keep_raw_margin=3.0, min_letter_logp=float(np.log(0.3)), raw_conf=float(np.log(0.6))):
    """Correct one fingerspelled word with the recogniser's probability for every letter, not just its top guess.
    Candidates: dictionary words of the same length, or with one letter dropped / one extra letter typed.
    letter score = sum of log-probabilities of the candidate's letters (minus an indel penalty);
    total = letter score + freq_weight * Zipf word frequency.
    Returns (word, status): "word" = dictionary word; "raw" = a confidently typed non-word such as a name, kept;
    "unclear" = even the best word fits the letters badly, so the signer should repeat."""
    lex = word_lexicon()
    P = np.log(np.clip(np.array([v for _, v in letters]), 1e-6, 1.0)); n = len(P)
    raw = "".join(LETTERS[int(r.argmax())] for r in P)
    raw_letter = float(P.max(1).sum())
    cands = []                                   # (total, letter_score, word)
    if n in lex:
        ws, idx, z = lex[n]; ls = P[np.arange(n), idx].sum(1); tot = ls + freq_weight * z
        i = int(tot.argmax()); cands.append((tot[i], ls[i], ws[i]))
    if n + 1 in lex:                             # a letter was dropped while typing
        ws, idx, z = lex[n + 1]; ls = np.full(len(ws), -np.inf)
        for k in range(n + 1):
            keep = [j for j in range(n + 1) if j != k]
            ls = np.maximum(ls, P[np.arange(n), idx[:, keep]].sum(1))
        ls = ls - indel_penalty; tot = ls + freq_weight * z
        i = int(tot.argmax()); cands.append((tot[i], ls[i], ws[i]))
    if n >= 2 and n - 1 in lex:                  # an extra letter was typed
        ws, idx, z = lex[n - 1]; ls = np.full(len(ws), -np.inf)
        for k in range(n):
            keep = [j for j in range(n) if j != k]
            ls = np.maximum(ls, P[keep][np.arange(n - 1), idx].sum(1))
        ls = ls - indel_penalty; tot = ls + freq_weight * z
        i = int(tot.argmax()); cands.append((tot[i], ls[i], ws[i]))
    best = max(cands) if cands else None
    if _is_nonword(raw, lex) and raw_letter / n >= raw_conf and (best is None or best[1] < raw_letter - keep_raw_margin):
        return raw, "raw"                        # confident letters that no dictionary word explains well: keep them
    if best is None or best[1] / max(len(best[2]), 1) < min_letter_logp:
        return raw, "unclear"
    return best[2], "word"

def _is_nonword(w, lex):
    return len(w) not in lex or w not in set(lex[len(w)][0])

FUNCTION_WORDS = {"a", "an", "the", "is", "am", "are", "was", "be", "i", "me", "my", "you", "your", "to", "of", "for", "and",
                  "it", "do", "can", "will", "at", "in", "on", "with", "that", "this", "please", "im", "its", "have", "has"}

def faithful(sentence, words, max_edit=1):
    """Accept a language-model sentence only if every word in it was signed (allowing small inflections) or is a
    function word; otherwise the model has added content the signer did not sign."""
    import difflib
    signed = [w.lower() for w in words]
    for tok in normalise_sentence(sentence).replace("'", "").split():
        if tok in FUNCTION_WORDS or tok in signed:
            continue
        if any(difflib.SequenceMatcher(None, tok, s).ratio() >= 0.8 and abs(len(tok) - len(s)) <= max_edit + 1 for s in signed):
            continue
        return False
    return True

def build_message(words, **kw):
    """Decoded words -> (sentence or None, detail). None means "please repeat": at least one word was unclear,
    so nothing is spoken rather than risking a wrong message."""
    decoded = [decode_word(w, **kw) for w in words if w]
    if not decoded:
        return None, decoded
    if any(status == "unclear" for _, status in decoded):
        return None, decoded
    return gloss_to_english([c for w, _ in decoded for c in (list(w) + ["_"])][:-1]), decoded
