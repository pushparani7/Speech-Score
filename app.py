"""
app.py — SpeakScore v2 Upgraded Backend
========================================
Upgrades:
  1. WER (Word Error Rate) — before and after preprocessing
  2. Model comparison endpoint — whisper-base vs whisper-small
  3. Semantic similarity — sentence-transformers embeddings
  4. Audio preprocessing — noise reduction + silence trimming
  5. Honest language — baseline framing throughout

Install: pip install -r requirements.txt
Run:     python app.py
Open:    http://localhost:5000
"""

import os, re, time, difflib, tempfile, warnings
import numpy as np
import soundfile as sf
import librosa
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import whisper

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__, static_folder=".", template_folder=".")
CORS(app)

print("[SpeakScore] Loading Whisper base model...")
MODEL_BASE  = whisper.load_model("base")
MODEL_SMALL = None   # loaded lazily on first /compare call
EMBEDDER    = None   # loaded lazily on first semantic request

print("[SpeakScore] Ready → http://localhost:5000")


# ═══════════════════════════════════════
# UPGRADE 4 — Preprocessing
# ═══════════════════════════════════════

def preprocess_audio(path):
    """Noise reduction + silence trim. Returns path to cleaned wav."""
    try:
        import noisereduce as nr
        y, sr = librosa.load(path, sr=16000, mono=True)
        y, _ = librosa.effects.trim(y, top_db=25)
        frames = int(0.5 * sr)
        noise  = y[:frames] if len(y) > frames else y
        y = nr.reduce_noise(y=y, y_noise=noise, sr=sr, stationary=True)
    except ImportError:
        y, sr = librosa.load(path, sr=16000, mono=True)
        y, _ = librosa.effects.trim(y, top_db=25)

    out = tempfile.NamedTemporaryFile(suffix="_pre.wav", delete=False)
    sf.write(out.name, y, sr)
    return out.name


# ═══════════════════════════════════════
# UPGRADE 1 — WER
# ═══════════════════════════════════════

def normalize(text):
    return re.sub(r"[^\w\s']", "", text.lower()).split()

def wer_breakdown(ref_text, hyp_text):
    ref = normalize(ref_text)
    hyp = normalize(hyp_text)
    n   = len(ref)
    if n == 0:
        return {"wer": 0.0, "substitutions": 0, "deletions": 0, "insertions": 0, "ref_words": 0}

    m = len(hyp)
    d = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): d[i][0] = i
    for j in range(m+1): d[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            d[i][j] = d[i-1][j-1] if ref[i-1]==hyp[j-1] else 1+min(d[i-1][j], d[i][j-1], d[i-1][j-1])

    i, j = n, m
    subs = dels = ins = 0
    while i > 0 or j > 0:
        if i>0 and j>0 and ref[i-1]==hyp[j-1]:
            i-=1; j-=1
        elif i>0 and j>0 and d[i][j]==d[i-1][j-1]+1:
            subs+=1; i-=1; j-=1
        elif j>0 and d[i][j]==d[i][j-1]+1:
            ins+=1; j-=1
        else:
            dels+=1; i-=1

    return {"wer": round((subs+dels+ins)/n, 4),
            "substitutions": subs, "deletions": dels,
            "insertions": ins, "ref_words": n}


# ═══════════════════════════════════════
# UPGRADE 3 — Semantic Similarity
# ═══════════════════════════════════════

def get_embedder():
    global EMBEDDER
    if EMBEDDER is None:
        try:
            from sentence_transformers import SentenceTransformer
            print("[SpeakScore] Loading sentence-transformers...")
            EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            EMBEDDER = "unavailable"
    return EMBEDDER

def compute_semantic(ref, hyp):
    emb = get_embedder()
    if emb == "unavailable":
        return {"score": None, "label": "install sentence-transformers", "cosine": None}
    from numpy.linalg import norm
    vecs   = emb.encode([ref, hyp])
    cosine = float(np.dot(vecs[0], vecs[1]) / (norm(vecs[0])*norm(vecs[1])))
    cosine = max(0.0, min(1.0, cosine))
    score  = int(round(cosine * 100))
    label  = ("High semantic match" if score>=85 else
              "Partial semantic overlap" if score>=65 else
              "Low semantic similarity" if score>=40 else
              "Significant meaning difference")
    return {"score": score, "label": label, "cosine": round(cosine, 4)}


# ═══════════════════════════════════════
# Grammar helpers
# ═══════════════════════════════════════

def compute_difflib_score(ref, hyp):
    r, h = normalize(ref), normalize(hyp)
    if not r: return 0
    return max(0, min(100, int(round(difflib.SequenceMatcher(None,r,h).ratio()*100))))

def word_diff(ref, hyp):
    r, h = normalize(ref), normalize(hyp)
    missing, extra = [], []
    for tag,i1,i2,j1,j2 in difflib.SequenceMatcher(None,r,h).get_opcodes():
        if tag=="delete":  missing.extend(r[i1:i2])
        elif tag=="insert": extra.extend(h[j1:j2])
        elif tag=="replace": missing.extend(r[i1:i2]); extra.extend(h[j1:j2])
    return missing, extra

GRAMMAR_RULES = [
    (r"\b(he|she|it)\s+don't\b",            "Subject-verb disagreement: use 'doesn't' with he/she/it"),
    (r"\b(they|we|i)\s+was\b",              "Subject-verb disagreement: use 'were' with they/we/I"),
    (r"\bdon't\s+know\s+nothing\b",          "Double negative: use 'doesn't know anything'"),
    (r"\b\d+\s+year\b(?!s)",               "Plural error: use 'years' after a number"),
    (r"\b(i|he|she|they|we)\s+working\b",   "Missing auxiliary: e.g. 'is working' or 'has been working'"),
    (r"\b(i|he|she|they|we)\s+(go|goes)\b", "Tense error — check verb form"),
    (r"\b(i|he|she|they|we)\s+tell\b",      "Tense error: use 'told' (past tense)"),
    (r"\bchildrens\b",                       "Incorrect plural: 'children' is already plural"),
]
AUXILIARIES  = {"has","have","had","is","are","was","were","be","been","being",
                "do","does","did","will","would","shall","should","may","might","must","can","could"}
ARTICLES     = {"a","an","the"}
PREPOSITIONS = {"in","on","at","for","to","from","with","by","of","about"}

def detect_errors(transcription, missing):
    errors, low = [], transcription.lower()
    for pat, msg in GRAMMAR_RULES:
        if re.search(pat, low): errors.append(msg)
    aux  = [w for w in missing if w in AUXILIARIES]
    art  = [w for w in missing if w in ARTICLES]
    prep = [w for w in missing if w in PREPOSITIONS]
    if aux:  errors.append(f"Missing auxiliary verb(s): {', '.join(aux)}")
    if art:  errors.append(f"Missing article(s): {', '.join(art)}")
    if prep: errors.append(f"Missing preposition(s): {', '.join(prep)}")
    return list(dict.fromkeys(errors))

def make_feedback(ref, missing, extra, errors, semantic):
    parts = []
    sem = (semantic or {}).get("score")
    if sem is not None:
        if sem < 50:
            parts.append(f"Semantic similarity is low ({sem}%) — overall meaning differs significantly.")
        elif sem < 75:
            parts.append(f"Semantic similarity is moderate ({sem}%) — some meaning preserved but key ideas differ.")
    if missing: parts.append(f"Words omitted: {', '.join(repr(w) for w in missing[:6])}.")
    if extra:   parts.append(f"Unexpected words spoken: {', '.join(repr(w) for w in extra[:4])}.")
    for e in errors[:3]: parts.append(e+".")
    if not parts:
        return ("Transcription is close to the reference. "
                "Note: this is a baseline system — subtle errors may not be detected.")
    return " ".join(parts) + f'  ▸ Reference: "{ref}"'


# ═══════════════════════════════════════
# Core analysis pipeline
# ═══════════════════════════════════════

def run_analysis(audio_path, reference, model, apply_preprocessing):
    result = {}
    pre_path = None

    t0 = time.time()
    raw = model.transcribe(audio_path, language="en", task="transcribe")
    raw_text = raw["text"].strip()
    result["transcription_raw"]    = raw_text
    result["inference_time_raw"]   = round(time.time()-t0, 2)

    if apply_preprocessing:
        try:
            pre_path = preprocess_audio(audio_path)
            t1 = time.time()
            pre = model.transcribe(pre_path, language="en", task="transcribe")
            result["transcription"]              = pre["text"].strip()
            result["preprocessing_applied"]      = True
            result["inference_time_preprocessed"]= round(time.time()-t1, 2)
        except Exception as e:
            result["transcription"]         = raw_text
            result["preprocessing_applied"] = False
            result["preprocessing_error"]   = str(e)
    else:
        result["transcription"]         = raw_text
        result["preprocessing_applied"] = False

    final = result["transcription"]

    if reference:
        result["wer_before"]    = wer_breakdown(reference, raw_text)
        if apply_preprocessing and result["preprocessing_applied"]:
            result["wer_after"] = wer_breakdown(reference, final)
            result["wer_improved"] = result["wer_before"]["wer"] > result["wer_after"]["wer"]
        else:
            result["wer_after"]    = result["wer_before"]
            result["wer_improved"] = False

        result["difflib_score"] = compute_difflib_score(reference, final)
        result["semantic"]      = compute_semantic(reference, final)
        missing, extra          = word_diff(reference, final)
        result["missing_words"] = missing
        result["extra_words"]   = extra
        errors                  = detect_errors(final, missing)
        result["errors"]        = errors
        result["feedback"]      = make_feedback(reference, missing, extra, errors, result["semantic"])

        sem_s = (result["semantic"] or {}).get("score")
        result["score"] = (int(round(0.6*sem_s + 0.4*result["difflib_score"]))
                           if sem_s is not None else result["difflib_score"])
    else:
        result["score"]         = None
        result["errors"]        = []
        result["feedback"]      = ("No reference provided — transcription only. "
                                   "This is a baseline system; accuracy varies with accent and audio quality.")
        result["missing_words"] = []
        result["extra_words"]   = []
        result["wer_before"]    = None
        result["wer_after"]     = None
        result["semantic"]      = None

    if pre_path and os.path.exists(pre_path):
        os.unlink(pre_path)
    return result


# ═══════════════════════════════════════
# Routes
# ═══════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    reference  = request.form.get("reference", "").strip()
    audio_file = request.files.get("audio")
    preprocess = request.form.get("preprocess", "true").lower() == "true"

    if not audio_file:
        return jsonify({"error": "No audio file provided"}), 400

    suffix = os.path.splitext(audio_file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        audio_file.save(tmp.name); tmp_path = tmp.name

    try:
        result = run_analysis(tmp_path, reference, MODEL_BASE, preprocess)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path): os.unlink(tmp_path)

    return jsonify(result)


@app.route("/compare", methods=["POST"])
def compare():
    """Upgrade 2 — run both base and small, return side-by-side."""
    global MODEL_SMALL
    reference  = request.form.get("reference", "").strip()
    audio_file = request.files.get("audio")

    if not audio_file:
        return jsonify({"error": "No audio file provided"}), 400

    suffix = os.path.splitext(audio_file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        audio_file.save(tmp.name); tmp_path = tmp.name

    if MODEL_SMALL is None:
        print("[SpeakScore] Loading whisper-small for comparison...")
        MODEL_SMALL = whisper.load_model("small")

    try:
        t0 = time.time()
        base_r = run_analysis(tmp_path, reference, MODEL_BASE,  apply_preprocessing=False)
        base_t = round(time.time()-t0, 2)

        t1 = time.time()
        small_r= run_analysis(tmp_path, reference, MODEL_SMALL, apply_preprocessing=False)
        small_t= round(time.time()-t1, 2)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path): os.unlink(tmp_path)

    def _card(r, t):
        return {"transcription":  r["transcription"],
                "score":          r.get("score"),
                "wer":            (r.get("wer_after") or {}).get("wer"),
                "semantic_score": (r.get("semantic") or {}).get("score"),
                "inference_sec":  t,
                "errors":         r.get("errors",[])}

    b_wer = (base_r.get("wer_after") or {}).get("wer", 1.0)
    s_wer = (small_r.get("wer_after") or {}).get("wer", 1.0)
    speed  = round(small_t/base_t, 1) if base_t > 0 else 1.0

    if s_wer < b_wer:
        diff = round((b_wer-s_wer)*100, 1)
        verdict = (f"whisper-small reduced WER by {diff}pp but was {speed}x slower. "
                   f"Recommended for heavily accented learner speech despite the speed cost.")
    elif b_wer < s_wer:
        diff = round((s_wer-b_wer)*100, 1)
        verdict = (f"whisper-base achieved lower WER ({diff}pp better) and was {speed}x faster. "
                   f"This audio may not benefit from the larger model.")
    else:
        verdict = (f"Both models produced identical output on this audio. "
                   f"whisper-base is {speed}x faster — prefer it for real-time use.")

    return jsonify({"base": _card(base_r, base_t),
                    "small": _card(small_r, small_t),
                    "verdict": verdict})


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)