"""
Sentence smoother for Pipeline A (sign -> text -> speech).

This is the v3 orchestration, the one that got evaluated in evaluation/eval_smoother.py:
  1. try the sign grammar / lexicon rules first
  2. dictionary check - if every fingerspelled word turns out to be a real word, the rules alone give the sentence
  3. otherwise (some spelled word isn't a real word, usually just a recognition error) ask the local LLM
  4. only accept what the LLM gives back if it passes the faithfulness guard (i.e. it hasn't added words the
     signer never signed) - if it fails the guard, fall back to the rule-based sentence instead

Ollama serves the LLM on this machine. If Ollama isn't running we just use the rule sentence, so the whole
thing still works, it just can't repair misspelled words.
"""
import json, os, re, sys, time, urllib.request
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from signbridge_core import decode_word, LETTERS, _tokens, faithful, gloss_to_english, orchestrate_smoother   # noqa: E402

SYSTEM_PROMPT = """You are a sign language interpreter assistant.
You receive a sequence of ASL signs (some may be fingerspelled letters, some whole words).
Your job is to convert the sequence into a single clear, natural English sentence.
Rules:
- Output ONLY the final sentence. No explanation, no preamble.
- If the input is unclear, produce the most likely intended sentence.
- Preserve the user's intent. Do not add information not implied by the signs.
- Keep it concise — one sentence maximum."""

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
# Llama 3.2, the 3B checkpoint, served locally through Ollama - pretrained, nothing fine-tuned here. Went with 3B
# over a bigger Llama checkpoint because this only ever has to turn a handful of signed words into one sentence
# (see SYSTEM_PROMPT below), and 3B answers in well under a second on a laptop CPU where a call can't afford to
# wait several seconds per message. Override with SIGNBRIDGE_LLM_MODEL if a bigger model is available.
LLM_MODEL = os.getenv("SIGNBRIDGE_LLM_MODEL", "llama3.2:3b")


def ollama_available(timeout=0.6):
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=timeout) as r:
            tags = json.loads(r.read())
        names = {m.get("name", "") for m in tags.get("models", [])}
        return LLM_MODEL in names or (":" not in LLM_MODEL and any(n.startswith(LLM_MODEL + ":") for n in names))
    except Exception:
        return False


def ollama_sentence(signs, timeout=40):
    """Asks the local LLM for one sentence back. Returns (sentence, ms), or (None, ms) if anything goes wrong."""
    body = json.dumps({
        "model": LLM_MODEL, "stream": False, "options": {"temperature": 0, "num_predict": 48},
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": "Signs: " + " ".join(signs)}]}).encode()
    req = urllib.request.Request(OLLAMA_URL + "/api/chat", data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())["message"]["content"].strip().split("\n")[0]
        return out, (time.perf_counter() - t0) * 1000
    except Exception:
        return None, (time.perf_counter() - t0) * 1000


def _prime_dictionary():
    """signbridge_core checks fingerspelled words against /usr/share/dict/words, which exists on macOS and most
    Linux boxes. If that file's missing, fall back to the top 100k English words so real words don't get sent to the LLM."""
    import signbridge_core as core
    d = core._dictionary()
    if len(d) < 1000:
        try:
            from wordfreq import top_n_list
            d |= {w for w in top_n_list("en", 100000) if w.isalpha() and w.isascii()}
        except Exception:
            pass


_prime_dictionary()


def text_to_signs(text):
    """Turns the typed message (letters + phrases) into the sign sequence the orchestrator wants - letters with
    an underscore between each word."""
    words = re.findall(r"[A-Za-z']+", text)
    signs = []
    for w in words:
        signs += list(w.upper().replace("'", "")) + ["_"]
    return signs[:-1]


def signed_words(signs):
    return [v for t, v in _tokens(signs)] + [w for t, v in _tokens(signs) if t == "gloss" for w in gloss_to_english([v]).rstrip(".?").split()]


def smooth(text, llm=ollama_sentence):
    """Returns {sentence, route, llm_ms, llm_sentence}. `llm` can be swapped out so tests don't need a real model."""
    signs = text_to_signs(text)
    if not signs:
        return {"sentence": "", "route": "empty", "llm_ms": None, "llm_sentence": None}
    question = text.strip().endswith("?")

    def fix(s):
        return (s[:-1] + "?") if question and s.endswith(".") else s

    rules = orchestrate_smoother(signs)
    if rules:
        return {"sentence": fix(rules), "route": "rules", "llm_ms": None, "llm_sentence": None}
    fallback = fix(gloss_to_english(signs))
    out, ms = llm(signs)
    if out is None:
        return {"sentence": fallback, "route": "rules_llm_unavailable", "llm_ms": None, "llm_sentence": None}
    if faithful(out, signed_words(signs)):
        return {"sentence": out, "route": "llm", "llm_ms": ms, "llm_sentence": out}
    return {"sentence": fallback, "route": "llm_rejected_by_guard", "llm_ms": ms, "llm_sentence": out}


# these decoder settings were picked on validation messages by evaluation/tune_message_decoding.py (vote typing, 0.5 agreement)
DECODER_PARAMS = dict(freq_weight=1.0, indel_penalty=1.5, keep_raw_margin=2.0, min_letter_logp=-1.204)

def decode_message(items):
    """items look like: [{"text": "I am good."} | {"probs": [[26 floats], ...]}]. Each spelled word gets corrected
    against a dictionary using the recogniser's per-letter probabilities. Returns {text, repeat, unclear:[indexes],
    words:[{raw, word, status}]}. If any spelled word comes back unclear the caller should ask the signer to repeat
    it rather than speak - a wrong message is worse than having to ask again."""
    words, out, unclear = [], [], []
    for k, it in enumerate(items):
        if "probs" in it:
            P = np.array(it["probs"], dtype=float)
            if P.ndim != 2 or P.shape[1] != len(LETTERS) or not len(P):
                raise ValueError("probs must be N x 26")
            raw = "".join(LETTERS[int(r.argmax())] for r in P)
            word, status = decode_word([(int(r.argmax()), r) for r in P], **DECODER_PARAMS)
            words.append({"raw": raw, "word": word, "status": status})
            if status == "unclear":
                unclear.append(k)
            out.append(word.lower())
        else:
            t = str(it.get("text", "")).strip()
            words.append({"raw": t, "word": t, "status": "text"})
            out.append(t)
    return {"text": " ".join(x for x in out if x), "repeat": bool(unclear), "unclear": unclear, "words": words}
