"""
Tone tagger evaluation (Pipeline B, text modality).

Model shipped in backend/pipeline/speech_pipeline.py: tabularisai/multilingual-sentiment-analysis
(a 5-class multilingual DistilBERT fine-tune).  Baseline: VADER (the rejected rule-based alternative).

Test cases
  T1  SST-5 test set (2,210 public human-labelled sentences, 5 classes) - also collapsed to 3 classes
  T2  Call-dialogue set: 60 conversational/medical utterances, 20 per class (author-labelled)
  T3  Multilingual: 20 of the T2 utterances in Spanish and French (author translations)
Metrics: accuracy and macro-F1 (5-class and 3-class), per-class F1, VADER comparison.
"""
import json, os, time, urllib.request
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "artifacts", "eval")
FIVE = ["very negative", "negative", "neutral", "positive", "very positive"]
to3 = lambda l: {"very negative": "negative", "very positive": "positive"}.get(l, l)


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

clf = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis", top_k=1, device=-1)
vader = SentimentIntensityAnalyzer()
def tag(texts, bs=32):
    out = []
    for i in range(0, len(texts), bs):
        out += [r[0]["label"].lower() for r in clf(texts[i:i + bs], truncation=True)]
    return out
def vader3(t):
    c = vader.polarity_scores(t)["compound"]
    return "positive" if c >= 0.05 else "negative" if c <= -0.05 else "neutral"
def scores(y, p, labels):
    return {"accuracy": round(accuracy_score(y, p), 4), "macro_f1": round(f1_score(y, p, labels=labels, average="macro", zero_division=0), 4),
            "per_class_f1": dict(zip(labels, [round(x, 3) for x in f1_score(y, p, labels=labels, average=None, zero_division=0)])),
            "confusion": confusion_matrix(y, p, labels=labels).tolist()}

res = {"model": "tabularisai/multilingual-sentiment-analysis (DistilBERT-multilingual, 5-class)", "baseline": "VADER (rule-based)"}

# T1 SST-5
rows = []
for off in range(0, 2210, 100):
    d = json.loads(http_get(f"https://datasets-server.huggingface.co/rows?dataset=SetFit/sst5&config=default&split=test&offset={off}&length=100")); time.sleep(1)
    rows += [r["row"] for r in d["rows"]]
texts = [r["text"] for r in rows]; y5 = [r["label_text"] for r in rows]
t0 = time.perf_counter(); p5 = tag(texts); ms = (time.perf_counter() - t0) * 1000 / len(texts)
res["T1_sst5"] = {"n": len(texts), "five_class": scores(y5, p5, FIVE),
                  "three_class": scores([to3(v) for v in y5], [to3(v) for v in p5], ["negative", "neutral", "positive"]),
                  "vader_three_class": scores([to3(v) for v in y5], [vader3(t) for t in texts], ["negative", "neutral", "positive"]),
                  "mean_ms_per_sentence_batched": round(ms, 2)}
print("T1", res["T1_sst5"]["five_class"]["macro_f1"], res["T1_sst5"]["three_class"]["macro_f1"], "vader", res["T1_sst5"]["vader_three_class"]["macro_f1"], flush=True)

# T2 call-dialogue set (author-labelled, 20 per class)
POS = ["I am good, thank you so much.", "That's wonderful news, I'm really happy.", "Love you, see you soon!",
       "Thanks for being so patient with me.", "The results came back clear, what a relief.", "It was lovely to meet you.",
       "You explained that really well.", "I feel much better today.", "Great, that works perfectly for me.",
       "Congratulations on the new baby!", "I'm excited to start the new job.", "The staff here have been so kind.",
       "That's a brilliant idea.", "I really enjoyed our chat.", "Yes! I passed my exam.",
       "Thank you, you've been a huge help.", "Everything went smoothly.", "I'm glad you could make it.",
       "The pain has completely gone.", "What a beautiful day."]
NEU = ["How are you?", "My name is Manya.", "The appointment is at three o'clock tomorrow.", "Please take a seat.",
       "Can you repeat that more slowly?", "What is your date of birth?", "The pharmacy is on the second floor.",
       "I will send you the form by email.", "Do you have any allergies?", "The bus leaves at half past nine.",
       "Please write your address here.", "Which hand do you write with?", "I usually take the train to work.",
       "The meeting has moved to Thursday.", "Can I have a glass of water?", "We need to fill in this form first.",
       "Is this seat free?", "Where is the nearest exit?", "I am calling about my prescription.", "Let me check the schedule."]
NEG = ["I feel dizzy and my chest hurts.", "I'm really scared about the operation.", "This is the third time you've cancelled.",
       "I don't understand and it's frustrating.", "My father passed away last week.", "The pain is getting worse.",
       "Nobody is listening to me.", "I'm sorry, that's terrible news.", "I've been waiting for two hours, this is unacceptable.",
       "I can't breathe properly.", "I lost my job yesterday.", "That was really rude.", "I'm so tired of this.",
       "The medicine isn't working at all.", "Call an ambulance now, he collapsed.", "I feel very lonely.",
       "I am worried something is wrong.", "The interpreter didn't turn up again.", "I hate waiting like this.",
       "My child is very ill."]
t2 = POS + NEU + NEG; y2 = ["positive"] * 20 + ["neutral"] * 20 + ["negative"] * 20
p2 = [to3(v) for v in tag(t2)]
res["T2_call_dialogue"] = {"n": 60, "labels": "author-labelled", "three_class": scores(y2, p2, ["negative", "neutral", "positive"]),
                           "vader_three_class": scores(y2, [vader3(t) for t in t2], ["negative", "neutral", "positive"]),
                           "errors": [(t, y, p) for t, y, p in zip(t2, y2, p2) if y != p]}
print("T2", res["T2_call_dialogue"]["three_class"]["macro_f1"], "vader", res["T2_call_dialogue"]["vader_three_class"]["macro_f1"], flush=True)

# T3 multilingual (author translations of 20 T2 utterances: 7 pos, 7 neu, 6 neg)
ES = [("Estoy bien, muchas gracias.", "positive"), ("Es una noticia maravillosa, estoy muy feliz.", "positive"), ("Te quiero, ¡nos vemos pronto!", "positive"),
      ("Gracias por tener tanta paciencia conmigo.", "positive"), ("Hoy me siento mucho mejor.", "positive"), ("Fue un placer conocerte.", "positive"), ("El personal ha sido muy amable.", "positive"),
      ("¿Cómo estás?", "neutral"), ("Me llamo Manya.", "neutral"), ("La cita es mañana a las tres.", "neutral"), ("Por favor, siéntese.", "neutral"),
      ("¿Tiene alguna alergia?", "neutral"), ("La farmacia está en el segundo piso.", "neutral"), ("¿Está libre este asiento?", "neutral"),
      ("Me siento mareado y me duele el pecho.", "negative"), ("Tengo mucho miedo de la operación.", "negative"), ("El dolor está empeorando.", "negative"),
      ("Nadie me escucha.", "negative"), ("No puedo respirar bien.", "negative"), ("Mi hijo está muy enfermo.", "negative")]
FR = [("Je vais bien, merci beaucoup.", "positive"), ("C'est une merveilleuse nouvelle, je suis très heureuse.", "positive"), ("Je t'aime, à bientôt !", "positive"),
      ("Merci d'être si patient avec moi.", "positive"), ("Je me sens beaucoup mieux aujourd'hui.", "positive"), ("C'était un plaisir de vous rencontrer.", "positive"), ("Le personnel a été très gentil.", "positive"),
      ("Comment ça va ?", "neutral"), ("Je m'appelle Manya.", "neutral"), ("Le rendez-vous est demain à trois heures.", "neutral"), ("Veuillez vous asseoir.", "neutral"),
      ("Avez-vous des allergies ?", "neutral"), ("La pharmacie est au deuxième étage.", "neutral"), ("Cette place est-elle libre ?", "neutral"),
      ("J'ai des vertiges et mal à la poitrine.", "negative"), ("J'ai très peur de l'opération.", "negative"), ("La douleur empire.", "negative"),
      ("Personne ne m'écoute.", "negative"), ("Je n'arrive pas à bien respirer.", "negative"), ("Mon enfant est très malade.", "negative")]
res["T3_multilingual"] = {}
for name, data in [("spanish", ES), ("french", FR)]:
    tx = [t for t, _ in data]; yy = [l for _, l in data]; pp = [to3(v) for v in tag(tx)]
    res["T3_multilingual"][name] = {"n": len(data), "three_class": scores(yy, pp, ["negative", "neutral", "positive"]),
                                    "vader_three_class": scores(yy, [vader3(t) for t in tx], ["negative", "neutral", "positive"])}
    print("T3", name, res["T3_multilingual"][name]["three_class"]["macro_f1"], "vader", res["T3_multilingual"][name]["vader_three_class"]["macro_f1"], flush=True)

json.dump(res, open(os.path.join(OUT, "tone_metrics.json"), "w"), indent=2)
print("saved tone_metrics.json")
