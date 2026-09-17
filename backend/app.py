import os
import json
import tempfile
import urllib.request
import urllib.error
import uuid
import io
from xml.sax.saxutils import escape as xml_escape

from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder="../frontend", static_url_path="")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
LOGFARE_API_KEY = os.environ.get("LOGFARE_API_KEY", "")

TRANSCRIBE_MODEL = "gemini-3.5-transcribe"
STUDY_MODEL = "gemma-4-26b"


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
                        "text": """
Transcribe the audio recording VERBATIM.

This is a classroom recording.

IMPORTANT:
- Transcribe ALL audible spoken words.
- Do NOT summarize.
- Do NOT shorten the recording.
- Do NOT omit sentences or words.
- Preserve the speaker's actual wording.
- Preserve repetitions when they are spoken.
- Preserve natural filler words such as "um", "uh", "okay", etc. when audible.
- Do not turn the recording into notes.
- Do not explain anything.
- Do not add information that was not spoken.
- Do not invent missing words.
- If the speaker is speaking isiZulu/Zulu, transcribe the isiZulu words exactly as spoken.
- If the speaker switches between English and isiZulu, preserve both languages.
- Use punctuation where the speech clearly indicates it.
- Return ONLY the transcript.
"""
                    },
                    {
                        "fileData": {
                            "fileUri": uri,
                            "mimeType": uploaded_mime
                        }
                    }
                ]
            }
        ],
    }

    result = gemini_request(
        TRANSCRIBE_MODEL,
        payload
    )

    transcript = extract_text(result).strip()

    if not transcript:
        raise RuntimeError(
            "No speech was detected in the recording."
        )

    return transcript


def study_pack(text):
    if not LOGFARE_API_KEY:
        raise RuntimeError("LOGFARE_API_KEY is not configured on Render.")

    prompt = f"""
You are ClassNotes, an AI classroom study assistant.

Turn the classroom transcript below into a high-quality study pack.

IMPORTANT:
- Use ONLY information contained in the transcript.
- Do not invent facts.
- Remove repetition, filler and irrelevant conversation.
- Correct obvious speech-recognition mistakes when the intended meaning is clear.
- Keep the notes suitable for a school student.
- Prioritize information that would actually help the student study and prepare for tests.
- Keep the output concise but useful.
- Return ONLY valid JSON.
- Do NOT return markdown.
- Do NOT return code fences.
- Do NOT return explanations outside the JSON.
- Important terms MUST contain both a term and a definition.

Return EXACTLY this structure:

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
- summary: concise explanation of the whole lesson
- key_points: the most important facts and ideas
- important_terms: important vocabulary WITH short accurate definitions
- questions: useful revision questions based only on the lesson
- things_to_remember: especially memorable facts, rules or concepts
- exam_points: material especially likely to matter in an assessment
- q_cards: useful question-and-answer flashcards
- Avoid duplicates.
- Do not add information that was not taught.

TRANSCRIPT:
{text}
"""

    payload = {
        "model": STUDY_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are ClassNotes. "
                    "Return ONLY the requested final JSON. "
                    "Do not include reasoning, analysis, commentary, "
                    "markdown fences, or explanations outside the JSON."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.2,
        "max_tokens": 4000
    }

    req = urllib.request.Request(
        "https://logfare.ai/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LOGFARE_API_KEY}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Logfare HTTP {e.code}: {body[:3000]}"
        )

    try:
        content = result["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(
            "Logfare returned an unexpected response: "
            + json.dumps(result)[:3000]
        )

    data = parse_json(content)

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
        "ai": "configured" if (GEMINI_API_KEY or LOGFARE_API_KEY) else "not_configured",
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
    if not LOGFARE_API_KEY:
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



# ============================================================
# EXPORT FEATURES
# ============================================================

def normalise_lesson(data):
    data = data or {}

    return {
        "title": data.get("title") or "Class Notes",
        "date": data.get("date") or "",
        "summary": data.get("summary") or "",
        "transcript": data.get("transcript") or "",
        "key_points": data.get("key_points") or [],
        "important_terms": data.get("important_terms") or [],
        "questions": data.get("questions") or [],
        "things_to_remember": data.get("things_to_remember") or [],
        "exam_points": data.get("exam_points") or [],
        "q_cards": data.get("q_cards") or data.get("flashcards") or [],
    }


def lesson_text_sections(data):
    lesson = normalise_lesson(data)

    sections = []

    sections.append(("Class Notes", lesson["title"]))
    sections.append(("Date", lesson["date"]))
    sections.append(("Summary", lesson["summary"]))

    sections.append(
        ("Key Points", lesson["key_points"])
    )

    terms = []
    for item in lesson["important_terms"]:
        if isinstance(item, dict):
            terms.append(
                f'{item.get("term", "")}: {item.get("definition", "")}'
            )
        else:
            terms.append(str(item))

    sections.append(("Important Terms", terms))
    sections.append(("Study Questions", lesson["questions"]))
    sections.append(("Things to Remember", lesson["things_to_remember"]))
    sections.append(("Exam Points", lesson["exam_points"]))

    cards = []
    for card in lesson["q_cards"]:
        if isinstance(card, dict):
            q = card.get("question", card.get("q", ""))
            a = card.get("answer", card.get("a", ""))
            cards.append(f"Q: {q}\nA: {a}")
        else:
            cards.append(str(card))

    sections.append(("Q Cards", cards))
    sections.append(("Transcript", lesson["transcript"]))

    return sections


def create_docx(data):
    from docx import Document
    from docx.shared import Pt

    lesson = normalise_lesson(data)

    document = Document()

    title = document.add_heading(
        lesson["title"],
        level=0
    )

    document.add_paragraph(
        f'Date: {lesson["date"]}'
    )

    for heading, content in lesson_text_sections(data)[2:]:
        document.add_heading(heading, level=1)

        if isinstance(content, list):
            for item in content:
                paragraph = document.add_paragraph(
                    style="List Bullet"
                )
                paragraph.add_run(str(item))
        else:
            document.add_paragraph(str(content))

    for paragraph in document.paragraphs:
        for run in paragraph.runs:
            run.font.name = "Arial"
            run.font.size = Pt(10.5)

    output = io.BytesIO()
    document.save(output)
    output.seek(0)

    return output


def create_pdf(data):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        ListFlowable,
        ListItem,
        PageBreak,
    )

    lesson = normalise_lesson(data)

    output = io.BytesIO()

    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )

    styles = getSampleStyleSheet()

    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    body_style = styles["BodyText"]

    body_style.leading = 15

    story = []

    story.append(
        Paragraph(
            xml_escape(lesson["title"]),
            title_style
        )
    )

    story.append(
        Paragraph(
            "Date: " + xml_escape(lesson["date"]),
            body_style
        )
    )

    story.append(Spacer(1, 10))

    for heading, content in lesson_text_sections(data)[2:]:
        story.append(
            Paragraph(
                xml_escape(heading),
                heading_style
            )
        )

        if isinstance(content, list):
            items = []

            for item in content:
                safe = xml_escape(str(item))
                items.append(
                    ListItem(
                        Paragraph(safe, body_style)
                    )
                )

            if items:
                story.append(
                    ListFlowable(
                        items,
                        bulletType="bullet",
                        leftIndent=18,
                    )
                )

        else:
            story.append(
                Paragraph(
                    xml_escape(str(content)),
                    body_style
                )
            )

        story.append(Spacer(1, 8))

    document.build(story)

    output.seek(0)

    return output


def create_xlsx(data):
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment

    lesson = normalise_lesson(data)

    workbook = Workbook()

    sheet = workbook.active
    sheet.title = "Class Notes"

    sheet.append(["ClassNotes"])
    sheet["A1"].font = Font(
        bold=True,
        size=18
    )

    sheet.append(["Title", lesson["title"]])
    sheet.append(["Date", lesson["date"]])
    sheet.append([])

    rows = [
        ("Summary", lesson["summary"]),
    ]

    for item in lesson["key_points"]:
        rows.append(("Key Point", item))

    for item in lesson["important_terms"]:
        if isinstance(item, dict):
            rows.append(
                (
                    "Important Term",
                    f'{item.get("term", "")}: {item.get("definition", "")}'
                )
            )
        else:
            rows.append(("Important Term", str(item)))

    for item in lesson["questions"]:
        rows.append(("Study Question", item))

    for item in lesson["things_to_remember"]:
        rows.append(("Remember", item))

    for item in lesson["exam_points"]:
        rows.append(("Exam Point", item))

    for card in lesson["q_cards"]:
        if isinstance(card, dict):
            rows.append(
                (
                    "Q Card",
                    f'Q: {card.get("question", card.get("q", ""))}\n'
                    f'A: {card.get("answer", card.get("a", ""))}'
                )
            )

    rows.append(("Transcript", lesson["transcript"]))

    for row in rows:
        sheet.append(list(row))

    sheet.column_dimensions["A"].width = 24
    sheet.column_dimensions["B"].width = 100

    for row in sheet.iter_rows():
        for cell in row:
            cell.alignment = Alignment(
                vertical="top",
                wrap_text=True
            )

    output = io.BytesIO()

    workbook.save(output)
    output.seek(0)

    return output


@app.route("/export/<file_format>", methods=["POST"])
def export_lesson(file_format):
    data = request.get_json(silent=True) or {}

    if not data:
        return jsonify({
            "error": "No lesson data supplied."
        }), 400

    file_format = file_format.lower()

    try:
        if file_format == "docx":
            output = create_docx(data)
            mimetype = (
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            )
            extension = "docx"

        elif file_format == "pdf":
            output = create_pdf(data)
            mimetype = "application/pdf"
            extension = "pdf"

        elif file_format == "xlsx":
            output = create_xlsx(data)
            mimetype = (
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            )
            extension = "xlsx"

        else:
            return jsonify({
                "error": "Unsupported export format."
            }), 400

        safe_title = "".join(
            c if c.isalnum() or c in " _-" else "_"
            for c in normalise_lesson(data)["title"]
        ).strip()

        if not safe_title:
            safe_title = "ClassNotes"

        from flask import send_file

        return send_file(
            output,
            mimetype=mimetype,
            as_attachment=True,
            download_name=f"{safe_title}.{extension}",
        )

    except Exception as error:
        print("EXPORT ERROR:", repr(error))

        return jsonify({
            "error": "Export failed.",
            "details": str(error)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )
