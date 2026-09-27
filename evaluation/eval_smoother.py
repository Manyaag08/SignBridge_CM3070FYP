"""
Sentence-smoother evaluation (Pipeline A, text modality).

The designed model is Llama 3:8b via Ollama. It could not be installed on the evaluation machine
(4.7 GB model, <4 GB free disk), so a small local instruction model of the same family of
decoder-only LLMs (Qwen2.5-0.5B-Instruct) is evaluated with the identical system prompt as a
lower-bound stand-in, against a deterministic rule/gloss-grammar baseline.

30 test cases in four categories, as planned in the report (Table 5):
  C1 clean glosses (ASL word order)  C2 fingerspelled words with boundaries
  C3 noisy recogniser output (confusable-letter substitutions, stutters)  C4 mixed glosses + fingerspelling
Metrics: exact match, acceptable match, word error rate vs the expected sentence, latency p50.
"""
import json, os, re, sys, time
import numpy as np, torch
from jiwer import wer
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from signbridge_core import faithful, gloss_to_english, normalise_sentence, orchestrate_smoother, _tokens
OUT = os.path.join(HERE, "..", "artifacts", "eval")
SYSTEM_PROMPT = """You are a sign language interpreter assistant.
You receive a sequence of ASL signs (some may be fingerspelled letters, some whole words).
Your job is to convert the sequence into a single clear, natural English sentence.
Rules:
- Output ONLY the final sentence. No explanation, no preamble.
- If the input is unclear, produce the most likely intended sentence.
- Preserve the user's intent. Do not add information not implied by the signs.
- Keep it concise — one sentence maximum."""

CASES = json.load(open(os.path.join(HERE, "smoother_cases.json")))
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
torch.set_num_threads(4)

def llm(signs):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "Signs: " + " ".join(signs)}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = lm.generate(**enc, max_new_tokens=40, do_sample=False)
    ms = (time.perf_counter() - t0) * 1000
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip().split("\n")[0], ms

def score(name, fn):
    rows, lat = [], []
    for c in CASES:
        out, ms = fn(c["signs"]); lat.append(ms)
        o = normalise_sentence(out)
        exp = normalise_sentence(c["expected"]); acc = [normalise_sentence(a) for a in c.get("acceptable", [])]
        rows.append({"id": c["id"], "cat": c["category"], "signs": " ".join(c["signs"]), "expected": c["expected"], "output": out,
                     "exact": o == exp, "acceptable": o == exp or o in acc, "wer": round(wer(exp, o), 3)})
    by = {}
    for r in rows:
        b = by.setdefault(r["cat"], {"n": 0, "exact": 0, "acceptable": 0})
        b["n"] += 1; b["exact"] += r["exact"]; b["acceptable"] += r["acceptable"]
    return {"exact_match_rate": round(np.mean([r["exact"] for r in rows]), 4),
            "acceptable_match_rate": round(np.mean([r["acceptable"] for r in rows]), 4),
            "mean_wer": round(float(np.mean([r["wer"] for r in rows])), 4),
            "latency_p50_ms": round(float(np.percentile(lat, 50)), 1), "by_category": by, "rows": rows}

res = {"n_cases": len(CASES), "designed_model": "llama3:8b (Ollama) - not runnable on eval machine (disk)",
       "evaluated_llm": MODEL + " (greedy decoding, same system prompt)"}
res["rule_gloss_grammar"] = score("rules", lambda s: (lambda t0: (gloss_to_english(s), (time.perf_counter() - t0) * 1000))(time.perf_counter()))
print("rules", {k: v for k, v in res["rule_gloss_grammar"].items() if k != "rows"}, flush=True)
res["llm"] = score("llm", llm)
print("llm", {k: v for k, v in res["llm"].items() if k != "rows"}, flush=True)
# orchestrated: deterministic grammar first, LLM only when the grammar has no full-sequence rule
def hybrid(s):
    t0 = time.perf_counter(); g = gloss_to_english(s, strict=True)
    if g: return g, (time.perf_counter() - t0) * 1000
    return llm(s)
res["hybrid_v1_grammar_then_llm"] = score("hybrid", hybrid)
print("hybrid v1", {k: v for k, v in res["hybrid_v1_grammar_then_llm"].items() if k != "rows"}, flush=True)
# v2 (design iteration after v1): the LLM is only consulted when the rules cannot produce real English words,
# i.e. a fingerspelled word is not in the system dictionary (likely a recognition error)
def hybrid_v2(s):
    t0 = time.perf_counter(); g = orchestrate_smoother(s)
    if g: return g, (time.perf_counter() - t0) * 1000
    return llm(s)
res["hybrid_v2_grammar_dictionary_then_llm"] = score("hybrid2", hybrid_v2)
print("hybrid v2", {k: v for k, v in res["hybrid_v2_grammar_dictionary_then_llm"].items() if k != "rows"}, flush=True)
res["llm_share_of_cases_v2"] = round(sum(orchestrate_smoother(c["signs"]) is None for c in CASES) / len(CASES), 3)
# v3: the same v2 routing, but an LLM sentence is only used if it passes the faithfulness check (no words the signer
# did not sign); otherwise the deterministic rule sentence is used instead of the invented one
def signed_words(signs):
    return [v for t, v in _tokens(signs)] + [w for t, v in _tokens(signs) if t == "gloss" for w in gloss_to_english([v]).rstrip(".?").split()]
guard_log = []
def hybrid_v3(s):
    t0 = time.perf_counter(); g = orchestrate_smoother(s)
    if g: return g, (time.perf_counter() - t0) * 1000
    out, ms = llm(s)
    ok = faithful(out, signed_words(s)); guard_log.append({"signs": " ".join(s), "llm": out, "accepted": ok})
    return (out if ok else gloss_to_english(s)), ms
res["hybrid_v3_v2_plus_faithfulness_guard"] = score("hybrid3", hybrid_v3)
res["faithfulness_guard"] = {"llm_calls": len(guard_log), "rejected": sum(not g["accepted"] for g in guard_log), "log": guard_log}
v3rows = res["hybrid_v3_v2_plus_faithfulness_guard"]["rows"]; v2rows = res["hybrid_v2_grammar_dictionary_then_llm"]["rows"]
res["invented_content_rate"] = {name: round(sum(1 for r in rows if r["wer"] > 0.5 and not r["acceptable"]) / len(rows), 3)
                                for name, rows in [("llm", res["llm"]["rows"]), ("v2", v2rows), ("v3_guarded", v3rows)]}
print("hybrid v3", {k: v for k, v in res["hybrid_v3_v2_plus_faithfulness_guard"].items() if k != "rows"}, "guard", {k: v for k, v in res["faithfulness_guard"].items() if k != "log"}, flush=True)
json.dump(res, open(os.path.join(OUT, "smoother_metrics.json"), "w"), indent=2)
print("saved smoother_metrics.json")
