from flask import Flask, request, jsonify, send_file
import subprocess
import json
import re
import os

app = Flask(__name__)

MODEL = os.path.expanduser("~/models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
LLAMA = os.path.expanduser("~/llama.cpp/build/bin/llama-cli")

@app.route("/")
def home():
    return send_file("index.html")

@app.route("/health")
def health():
    return jsonify({
        "service": "ClassNotes",
        "status": "online",
        "version": "1.0"
    })

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No transcript supplied"}), 400

    prompt = f"""You are ClassNotes, an assistant for an online student.

Analyze the following class transcript.

Return ONLY valid JSON in exactly this structure:

{{
  "summary": "A short clear summary of what the teacher explained.",
  "key_points": [
    "Important point 1",
    "Important point 2",
    "Important point 3"
  ],
  "important_terms": [
    "Term 1",
    "Term 2"
  ],
  "questions": [
    "Question the student should understand or study"
  ]
}}

Rules:
- Only use information actually present in the transcript.
- Do not invent facts.
- Make the notes useful for studying.
- Keep the summary concise.
- Include the most important concepts.
- Return JSON only.

TRANSCRIPT:
{text}
"""

    try:
        result = subprocess.run(
            [
                LLAMA,
                "-m", MODEL,
                "-c", "2048",
                "-n", "700",
                "-t", "2",
                "--temp", "0.2",
                "-p", prompt
            ],
            capture_output=True,
            text=True,
            timeout=180
        )

        output = result.stdout.strip()

        # Find JSON object inside llama output
        match = re.search(r'\{.*\}', output, re.DOTALL)

        if not match:
            return jsonify({
                "error": "Model did not return valid JSON",
                "raw": output
            }), 500

        json_text = match.group(0)
        notes = json.loads(json_text)

        return jsonify(notes)

    except subprocess.TimeoutExpired:
        return jsonify({"error": "Summarization timed out"}), 504

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("===================================")
    print("        CLASSNOTES BACKEND")
    print("===================================")
    print("Status: ONLINE")
    print("Model:", MODEL)
    print("===================================")

    app.run(host="0.0.0.0", port=8000, debug=False)
