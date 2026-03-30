# SpeakScore — Non-Native Speech Transcription & Error Analysis

> A baseline system that transcribes non-native English speech, detects grammar errors, scores semantic similarity, and delivers structured feedback end to end from raw audio to JSON output.

---

## Demo

| Input Audio | Reference | Transcription | Score |
|---|---|---|---|
| `tc01_broken.wav` | She has been working here for five years | She working here five year | 77% |
| `tc03_broken.wav` | He doesn't know anything about it | He don't know nothing about it | 67% |
| `tc04_broken.wav` | They were very happy to see us | They was very happy to see us | 86% |
| `tc07_broken.wav` | The children are playing in the garden | Childrens is playing in garden | 50% |

**Batch evaluation — 8 test cases, whisper-base:**
- Average similarity score: **70.9%**
- Average Word Error Rate: **0.050**
- Errors detected: **8/8**

---

## What it does

1. **Transcribes audio** using OpenAI Whisper — preserves grammatical errors as spoken, does not auto-correct
2. **Preprocesses audio** — noise reduction + silence trimming before transcription
3. **Computes WER** — Word Error Rate before and after preprocessing to measure improvement
4. **Scores similarity** — combines difflib (word-overlap) and sentence-transformers (semantic meaning) into a composite score
5. **Detects grammar errors** — rule-based classifier catches subject-verb disagreement, missing auxiliaries, double negatives, tense errors, plural errors
6. **Generates feedback** — semantic-aware, structured feedback explaining what was wrong and how to fix it
7. **Compares models** — runs whisper-base vs whisper-small side by side and reports accuracy vs speed tradeoff
8. **Outputs JSON** — structured result ready for any frontend or downstream API

---

## Architecture

```
Audio Input (.wav / .mp3 / .m4a)
        │
        ▼
Preprocessing (noisereduce + librosa silence trim)
        │
        ├──► Raw transcription (WER before)
        │
        ▼
Whisper Transcription (WER after)
        │
        ├──► Difflib score (word-overlap)
        ├──► Semantic score (sentence-transformers cosine similarity)
        ├──► Composite score (60% semantic + 40% difflib)
        ├──► Grammar error detection (rule-based regex)
        ├──► Word diff (LCS algorithm — missing + substituted words)
        └──► Structured JSON feedback
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Speech-to-text | OpenAI Whisper (base / small) |
| Semantic similarity | sentence-transformers — all-MiniLM-L6-v2 |
| Audio preprocessing | librosa, noisereduce, soundfile |
| Word diff | difflib (SequenceMatcher + LCS) |
| Backend API | Flask, flask-cors |
| Frontend | Vanilla HTML / CSS / JavaScript |
| Test audio generation | gTTS, pydub |
| Evaluation metric | Word Error Rate (WER) |

---

## Project Structure

```
speakscore/
├── app.py                  # Flask backend — /analyze and /compare endpoints
├── index.html              # Frontend UI — upload, record, visualize results
├── speech_analyzer.py      # Standalone CLI analyzer
├── batch_eval.py           # Batch evaluation runner with WER reporting
├── generate_test_audio.py  # Generates broken-English test audio via gTTS
├── requirements.txt        # All dependencies
└── test_audio/

```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/speakscore.git
cd speakscore
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

> Note: `torch` installation depends on your OS and CUDA setup.
> Visit https://pytorch.org/get-started/locally/ if you hit issues.

**3. Run the server**
```bash
python app.py
```

**4. Open the UI**
```
http://localhost:5000
```

---

## Usage — Web UI

1. Upload a `.wav` / `.mp3` / `.m4a` file, or click **Record mic** to speak directly
2. Type the **reference sentence** (what the speaker should have said)
3. Toggle **audio preprocessing** on or off
4. Click **Analyze Speech** — results appear below:
   - Composite score, WER before/after, semantic score
   - Word diff with missing and substituted word tags
   - Grammar errors list
   - Feedback message
   - Full JSON output
5. Click **Compare base vs small** to see model accuracy vs speed tradeoff

---

## Usage — CLI

```bash
# Analyze a single audio file
python speech_analyzer.py --audio input.wav --reference "She has been working here for five years"

# JSON output
python speech_analyzer.py --audio input.wav --reference "..." --json

# Generate test audio pairs
python generate_test_audio.py

# Run batch evaluation on all test cases
python batch_eval.py
python batch_eval.py --model small
```

---

## Sample JSON Output

```json
{
  "transcription": "She working here five year",
  "transcription_raw": "She working here five year",
  "score": 72,
  "difflib_score": 62,
  "semantic": {
    "score": 78,
    "label": "Partial semantic overlap",
    "cosine": 0.7841
  },
  "wer_before": {
    "wer": 0.375,
    "substitutions": 0,
    "deletions": 3,
    "insertions": 0,
    "ref_words": 8
  },
  "wer_after": {
    "wer": 0.375,
    "substitutions": 0,
    "deletions": 3,
    "insertions": 0,
    "ref_words": 8
  },
  "missing_words": ["has", "been", "for"],
  "extra_words": [],
  "errors": [
    "Missing auxiliary verb: e.g. 'is working' or 'has been working'",
    "Missing auxiliary verb(s): has, been",
    "Missing preposition(s): for"
  ],
  "feedback": "Words omitted: 'has', 'been', 'for'. Missing auxiliary verb: e.g. 'is working' or 'has been working'. ▸ Reference: \"She has been working here for five years\"",
  "preprocessing_applied": true
}
```

---

## Batch Evaluation Results

```
Test cases run       : 8
Avg similarity score : 70.9%
Avg WER              : 0.050
Cases with errors    : 8/8

ID      Score    WER   Note
──────  ──────  ─────  ─────────────────────────────────────────
tc01     77%    0.20   Missing auxiliary, plural error          ✓
tc02     73%    0.00   Wrong tense, missing article             ✓
tc03     67%    0.00   Subject-verb agreement, double negative  ✗
tc04     86%    0.00   Subject-verb agreement (was vs were)     ✓
tc05     80%    0.00   Missing modal verb, missing article      ✓
tc06     67%    0.00   Missing auxiliary, wrong tense           ✗
tc07     50%    0.20   Incorrect plural, SV agreement, article  ✗
tc08     67%    0.00   Wrong tense in reported speech           ✗
```

---

## Known Limitations

- This is a **baseline system** — accuracy varies with accent, background noise, and audio quality
- The rule-based grammar detector covers common error patterns but will miss errors outside its current rule set
- Whisper may normalize uncommon word forms (e.g. "Childrens" → "Children's"), causing edge-case scoring issues
- Semantic similarity does not reliably distinguish subtle tense or grammatical nuance — best used alongside WER and rule-based errors
- WER measures transcription accuracy only, not grammatical correctness
- whisper-small loads on first model comparison request (~30 seconds)

---

## License

MIT

---

*Built as a GSoC 2026 prototype for HumanAI — Non-Native Speech Transcription System with Error Analysis.*
