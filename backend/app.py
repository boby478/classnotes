import os
import json
import re
import urllib.request
import urllib.error

from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder="../frontend", static_url_path="")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

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
            "error": "AI is not configured yet. Add GEMINI_API_KEY on the hosting service."
        }), 503

    prompt = f"""
You are ClassNotes, an AI classroom study assistant.

Turn the following classroom transcript into accurate, concise study notes.

Return ONLY valid JSON in exactly this structure:

{{
  "summary": "A clear short summary of the lesson.",
  "key_points": [],
  "important_terms": [],
  "questions": [],
  "things_to_remember": [],
  "exam_points": []
}}

Rules:
- Use ONLY information contained in the transcript.
- Do not invent facts.
- Remove repetition and filler.
- Correct obvious speech-recognition mistakes when the intended meaning is clear.
- Keep the notes concise but useful for studying.
- Explain difficult ideas simply.
- Make questions useful for revision.
- Include important definitions and concepts.
- Include likely exam-worthy information.

TRANSCRIPT:
{text}
"""

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-2.5-flash:generateContent"
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

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))

        output = result["candidates"][0]["content"]["parts"][0]["text"]

        try:
            notes = json.loads(output)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", output, re.DOTALL)
            if not match:
                raise ValueError("AI returned invalid JSON")
            notes = json.loads(match.group(0))

        return jsonify(notes)

    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        return jsonify({
            "error": "AI request failed",
            "details": error_body[:1000]
        }), 502

    except Exception as e:
        return jsonify({
            "error": "AI processing failed",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
