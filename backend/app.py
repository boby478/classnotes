import os
import json
import re
import urllib.request
import urllib.error

from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder="../frontend", static_url_path="")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

GEMINI_MODEL = "gemini-2.5-flash"


def call_gemini(prompt):
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/{GEMINI_MODEL}:generateContent"
        f"?key={GEMINI_API_KEY}"
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json"
        }
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(req, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result["candidates"][0]["content"]["parts"][0]["text"]


def parse_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("Gemini returned invalid JSON.")
        return json.loads(match.group(0))


def split_transcript(text, chunk_words=1500):
    words = text.split()

    return [
        " ".join(words[i:i + chunk_words])
        for i in range(0, len(words), chunk_words)
    ]


def summarize_chunk(chunk):
    prompt = f"""
You are ClassNotes, an AI classroom study assistant.

Analyze this section of a classroom transcript.

Return ONLY valid JSON:

{{
  "summary": "",
  "key_points": [],
  "important_terms": [],
  "questions": [],
  "things_to_remember": [],
  "exam_points": []
}}

Rules:
- Use ONLY information in the transcript.
- Do not invent facts.
- Remove repetition and filler.
- Correct obvious speech-recognition errors when the meaning is clear.
- Keep the information useful for studying.
- Important terms should include short definitions.

TRANSCRIPT SECTION:

{chunk}
"""

    return parse_json(call_gemini(prompt))


def merge_summaries(chunk_summaries):
    combined = json.dumps(chunk_summaries, ensure_ascii=False)

    prompt = f"""
You are ClassNotes.

Combine the following summaries from different sections of ONE classroom lesson.

Create one complete study pack.

Return ONLY valid JSON in exactly this structure:

{{
  "title": "",
  "summary": "",
  "key_points": [],
  "important_terms": [
    {{
      "term": "",
      "definition": ""
    }}
  ],
  "questions": [],
  "things_to_remember": [],
  "exam_points": [],
  "q_cards": [
    {{
      "question": "",
      "answer": ""
    }}
  ]
}}

Rules:
- Combine duplicate information.
- Do not invent information.
- Preserve important details from every section.
- Make the summary concise.
- Make key points easy to study.
- Give important terms short, clear definitions.
- Make questions useful for revision.
- Make exam points focused on information likely to matter in an assessment.
- Create useful Q Cards covering the important material.
- Keep answers short but accurate.
- Return JSON only.

CHUNK SUMMARIES:

{combined}
"""

    return parse_json(call_gemini(prompt))


def summarize_long_transcript(text, chunk_words=1500):
    chunks = split_transcript(text, chunk_words)

    chunk_summaries = []

    for chunk in chunks:
        chunk_summaries.append(summarize_chunk(chunk))

    return merge_summaries(chunk_summaries)


@app.route("/")
def home():
    return send_from_directory("../frontend", "index.html")


@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "service": "ClassNotes",
        "ai": "online" if GEMINI_API_KEY else "not_configured"
    })


@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "No transcript supplied"}), 400

    if not GEMINI_API_KEY:
        return jsonify({
            "error": "AI is not configured yet."
        }), 503

    try:
        # Short transcript: process normally.
        if len(text.split()) <= 1500:
            result = summarize_chunk(text)

            # Turn the single chunk into the full study-pack format.
            result = merge_summaries([result])

        # Long transcript: automatically split it.
        else:
            result = summarize_long_transcript(text)

        return jsonify(result)

    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")

        return jsonify({
            "error": "Gemini request failed",
            "details": error_body[:1500]
        }), 502

    except Exception as e:
        return jsonify({
            "error": "AI processing failed",
            "details": str(e)
        }), 500


@app.route("/study-pack", methods=["POST"])
def study_pack():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "No transcript supplied"}), 400

    if not GEMINI_API_KEY:
        return jsonify({
            "error": "AI is not configured yet."
        }), 503

    try:
        result = summarize_long_transcript(text)

        return jsonify({
            "success": True,
            "chunks_processed": len(split_transcript(text)),
            "study_pack": result
        })

    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")

        return jsonify({
            "error": "Gemini request failed",
            "details": error_body[:1500]
        }), 502

    except Exception as e:
        return jsonify({
            "error": "Study pack generation failed",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
