import os
import json
import tempfile
import urllib.request
import urllib.error
import uuid

from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder="../frontend", static_url_path="")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

TRANSCRIBE_MODEL = "gemini-3.5-transcribe"
STUDY_MODEL = "gemini-3.8-flash"


def gemini_request(model, payload):
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured on Render.")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/{model}:generateContent"
    )

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Gemini HTTP {e.code}: {body[:3000]}"
        )


def extract_text(result):
    try:
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        raise RuntimeError(
            "Gemini returned an unexpected response: "
            + json.dumps(result)[:3000]
        )


def parse_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1:
            raise RuntimeError("AI returned invalid JSON.")

        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            raise RuntimeError("AI returned invalid JSON.")


def upload_audio(file_bytes, mime_type, filename):
    boundary = "----ClassNotes" + uuid.uuid4().hex

    metadata = json.dumps({
        "file": {
            "display_name": filename
        }
    })

    body = bytearray()

    body.extend(
        (
            f"--{boundary}\r\n"
            "Content-Disposition: form-data; name=\"metadata\"\r\n"
            "Content-Type: application/json; charset=UTF-8\r\n\r\n"
            f"{metadata}\r\n"
        ).encode()
    )

    body.extend(
        (
            f"--{boundary}\r\n"
            "Content-Disposition: form-data; name=\"file\"; "
            f"filename=\"{filename}\"\r\n"
            f"Content-Type: {mime_type}\r\n\r\n"
        ).encode()
    )

    body.extend(file_bytes)
    body.extend(f"\r\n--{boundary}--\r\n".encode())

    url = (
        "https://generativelanguage.googleapis.com/upload/v1beta/files"
        "?uploadType=multipart"
    )

    req = urllib.request.Request(
        url,
        data=bytes(body),
        headers={
            "Content-Type": f"multipart/related; boundary={boundary}",
            "x-goog-api-key": GEMINI_API_KEY,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Gemini upload HTTP {e.code}: {body[:3000]}"
        )

    file_info = result.get("file", result)

    uri = file_info.get("uri")
    returned_mime = file_info.get("mimeType", mime_type)

    if not uri:
        raise RuntimeError(
            "Gemini uploaded the audio but returned no file URI."
        )

    return uri, returned_mime


def transcribe_audio(file_bytes, mime_type, filename):
    uri, uploaded_mime = upload_audio(
        file_bytes,
        mime_type,
        filename
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "fileData": {
                            "fileUri": uri,
                            "mimeType": uploaded_mime
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1
        }
    }

    result = gemini_request(
        TRANSCRIBE_MODEL,
        payload
    )

    transcript = extract_text(result).strip()

    if not transcript:
        raise RuntimeError("No speech was detected in the recording.")

    return transcript


def study_pack(text):
    prompt = f"""
You are ClassNotes, an AI classroom study assistant.

Turn the following classroom transcript into a useful study pack.

Use ONLY information contained in the transcript.
Do not invent facts.
Remove repetition and filler.
Correct obvious speech-recognition mistakes when the intended meaning is clear.

Return ONLY valid JSON with EXACTLY this structure:

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

Requirements:

- title: short lesson title
- summary: concise but useful explanation
- key_points: important facts and ideas
- important_terms: important vocabulary with short definitions
- questions: useful revision questions
- things_to_remember: memorable facts or concepts
- exam_points: material likely to matter in an assessment
- q_cards: useful flashcards covering the important material
- Keep answers accurate and reasonably short.

TRANSCRIPT:

{text}
"""

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json"
        }
    }

    result = gemini_request(
        STUDY_MODEL,
        payload
    )

    data = parse_json(extract_text(result))

    data.setdefault("title", "Class notes")
    data.setdefault("summary", "")
    data.setdefault("key_points", [])
    data.setdefault("important_terms", [])
    data.setdefault("questions", [])
    data.setdefault("things_to_remember", [])
    data.setdefault("exam_points", [])
    data.setdefault("q_cards", [])

    return data


@app.route("/")
def home():
    return send_from_directory("../frontend", "index.html")


@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "service": "ClassNotes",
        "ai": "configured" if GEMINI_API_KEY else "not_configured",
        "transcription_model": TRANSCRIBE_MODEL,
        "study_model": STUDY_MODEL
    })


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if not GEMINI_API_KEY:
        return jsonify({
            "error": "GEMINI_API_KEY is not configured on Render."
        }), 503

    if "audio" not in request.files:
        return jsonify({
            "error": "No audio file was received."
        }), 400

    audio = request.files["audio"]

    try:
        data = audio.read()

        if not data:
            return jsonify({
                "error": "The recording was empty."
            }), 400

        mime_type = (
            audio.mimetype
            or "audio/webm"
        )

        transcript = transcribe_audio(
            data,
            mime_type,
            audio.filename or "class-recording.webm"
        )

        return jsonify({
            "transcript": transcript
        })

    except Exception as e:
        print("TRANSCRIPTION ERROR:", repr(e))

        return jsonify({
            "error": "Transcription failed.",
            "details": str(e)
        }), 502


@app.route("/study-pack", methods=["POST"])
def create_study_pack():
    if not GEMINI_API_KEY:
        return jsonify({
            "error": "GEMINI_API_KEY is not configured on Render."
        }), 503

    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({
            "error": "No transcript supplied."
        }), 400

    try:
        result = study_pack(text)

        return jsonify(result)

    except Exception as e:
        print("STUDY PACK ERROR:", repr(e))

        return jsonify({
            "error": "Study-pack generation failed.",
            "details": str(e)
        }), 502


@app.route("/summarize", methods=["POST"])
def summarize_compatibility():
    return create_study_pack()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )
