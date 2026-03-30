"""
Non-Native Speech Transcription System with Error Analysis
GSoC Prototype — speech_analyzer.py


Usage:
    python speech_analyzer.py --audio input.wav --reference "She has been working here for five years"
    python speech_analyzer.py --audio input.wav  # (no reference = transcription only)
"""

import argparse
import json
import difflib
import whisper
import re
from dataclasses import dataclass, asdict
from typing import Optional


# ─────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────

@dataclass
class AnalysisResult:
    transcription: str
    score: int
    errors: list[str]
    feedback: str
    missing_words: list[str]
    extra_words: list[str]


# ─────────────────────────────────────────────
# 1. Speech-to-Text (Whisper)
# ─────────────────────────────────────────────

def transcribe(audio_path: str, model_size: str = "base") -> str:
    """
    Transcribe audio to text using OpenAI Whisper.
    Model sizes: tiny, base, small, medium, large
    Larger = more accurate, slower. For non-native speech: 'small' or 'medium' recommended.
    """
    print(f"[Whisper] Loading model '{model_size}'...")
    model = whisper.load_model(model_size)

    print(f"[Whisper] Transcribing: {audio_path}")
    result = model.transcribe(audio_path, language="en", task="transcribe")

    transcription = result["text"].strip()
    print(f"[Whisper] Done: '{transcription}'")
    return transcription


# ─────────────────────────────────────────────
# 2. Word-level diff
# ─────────────────────────────────────────────

def normalize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into words."""
    text = text.lower()
    text = re.sub(r"[^\w\s']", "", text)
    return text.split()


def word_diff(reference: str, hypothesis: str) -> dict:
    """
    Compare reference vs hypothesis word-by-word.
    Returns missing words, extra words, and matched words.
    """
    ref_words = normalize(reference)
    hyp_words = normalize(hypothesis)

    matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)
    missing, extra, matched = [], [], []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            matched.extend(ref_words[i1:i2])
        elif tag == "delete":
            missing.extend(ref_words[i1:i2])
        elif tag == "insert":
            extra.extend(hyp_words[j1:j2])
        elif tag == "replace":
            missing.extend(ref_words[i1:i2])
            extra.extend(hyp_words[j1:j2])

    return {"missing": missing, "extra": extra, "matched": matched}


# ─────────────────────────────────────────────
# 3. Scoring
# ─────────────────────────────────────────────

def compute_score(reference: str, hypothesis: str) -> int:
    """
    Compute a similarity score (0–100) using SequenceMatcher on words.
    This is a word-level F1-inspired approach.
    """
    ref_words = normalize(reference)
    hyp_words = normalize(hypothesis)

    if not ref_words:
        return 0

    # SequenceMatcher ratio on word lists
    ratio = difflib.SequenceMatcher(None, ref_words, hyp_words).ratio()
    score = int(round(ratio * 100))
    return max(0, min(100, score))


# ─────────────────────────────────────────────
# 4. Error detection
# ─────────────────────────────────────────────

GRAMMAR_PATTERNS = [
    # Auxiliary verb patterns
    (r"\b(i|he|she|they|we)\s+(go|went|goes)\b",
     "Possible tense error — check verb agreement"),
    (r"\b(he|she|it)\s+don't\b",
     "Subject-verb agreement error: use 'doesn't' with he/she/it"),
    (r"\b(they|we|i)\s+was\b",
     "Subject-verb agreement error: use 'were' with they/we/I"),
    (r"\bdon't\s+know\s+nothing\b",
     "Double negative: use 'doesn't know anything'"),
    # Plural errors (simplified)
    (r"\b(\d+)\s+year\b(?!s)",
     "Plural error: use 'years' after a number"),
    (r"\b(\d+)\s+month\b(?!s)",
     "Plural error: use 'months' after a number"),
    # Missing progressive auxiliary
    (r"\b(i|he|she|they|we)\s+working\b",
     "Missing auxiliary verb: e.g. 'has been working' or 'is working'"),
]


def detect_grammar_errors(text: str) -> list[str]:
    """Rule-based grammar error detection."""
    errors = []
    lower = text.lower()
    for pattern, message in GRAMMAR_PATTERNS:
        if re.search(pattern, lower):
            errors.append(message)
    return errors


def classify_missing_words(missing: list[str]) -> list[str]:
    """Classify missing words into error types."""
    errors = []
    auxiliaries = {"has", "have", "had", "is", "are", "was", "were", "be", "been", "being",
                   "do", "does", "did", "will", "would", "shall", "should", "may", "might",
                   "must", "can", "could"}
    articles = {"a", "an", "the"}
    prepositions = {"in", "on", "at", "for", "to", "from", "with", "by", "of", "about"}

    missing_aux = [w for w in missing if w in auxiliaries]
    missing_art = [w for w in missing if w in articles]
    missing_prep = [w for w in missing if w in prepositions]

    if missing_aux:
        errors.append(f"Missing auxiliary verb(s): {', '.join(missing_aux)}")
    if missing_art:
        errors.append(f"Missing article(s): {', '.join(missing_art)}")
    if missing_prep:
        errors.append(f"Missing preposition(s): {', '.join(missing_prep)}")
    return errors


# ─────────────────────────────────────────────
# 5. Feedback generator
# ─────────────────────────────────────────────

def generate_feedback(reference: str, hypothesis: str, diff: dict, errors: list[str]) -> str:
    """Generate human-readable, constructive feedback."""
    parts = []

    if diff["missing"]:
        parts.append(f"Missing words: {', '.join(repr(w) for w in diff['missing'][:5])}.")

    if diff["extra"]:
        parts.append(f"Unexpected words: {', '.join(repr(w) for w in diff['extra'][:3])}.")

    for e in errors[:3]:
        parts.append(e + ".")

    if not parts:
        return "Good job! Minor differences from the reference, but overall very close."

    feedback = " ".join(parts)
    feedback += f" Reference: \"{reference}\""
    return feedback


# ─────────────────────────────────────────────
# 6. Full pipeline
# ─────────────────────────────────────────────

def analyze(audio_path: str, reference: Optional[str] = None,
            model_size: str = "base") -> AnalysisResult:
    """
    Full pipeline:
      1. Transcribe audio with Whisper
      2. If reference provided: compute score, diff, errors, feedback
      3. Return AnalysisResult
    """
    transcription = transcribe(audio_path, model_size)

    if not reference:
        return AnalysisResult(
            transcription=transcription,
            score=-1,
            errors=[],
            feedback="No reference sentence provided — transcription only.",
            missing_words=[],
            extra_words=[],
        )

    score = compute_score(reference, transcription)
    diff = word_diff(reference, transcription)

    grammar_errors = detect_grammar_errors(transcription)
    word_errors = classify_missing_words(diff["missing"])
    all_errors = list(dict.fromkeys(grammar_errors + word_errors))  # deduplicate, preserve order

    feedback = generate_feedback(reference, transcription, diff, all_errors)

    return AnalysisResult(
        transcription=transcription,
        score=score,
        errors=all_errors,
        feedback=feedback,
        missing_words=diff["missing"],
        extra_words=diff["extra"],
    )


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Non-Native Speech Error Analyzer")
    parser.add_argument("--audio", required=True, help="Path to .wav audio file")
    parser.add_argument("--reference", default=None, help="Expected/reference sentence")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    result = analyze(args.audio, args.reference, args.model)

    if args.json:
        print(json.dumps(asdict(result), indent=2))
    else:
        print("\n" + "=" * 50)
        print("SPEECH ANALYSIS RESULT")
        print("=" * 50)
        print(f"Transcription : {result.transcription}")
        if result.reference_provided if hasattr(result, 'reference_provided') else result.score >= 0:
            print(f"Score         : {result.score}%")
            print(f"Missing words : {result.missing_words or 'none'}")
            print(f"Extra words   : {result.extra_words or 'none'}")
            print(f"Errors        : {result.errors or ['none detected']}")
            print(f"Feedback      : {result.feedback}")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    main()