import os
import uuid
import fitz  # PyMuPDF
import pytesseract
from flask import Flask, request, jsonify, session, send_from_directory, render_template
from werkzeug.utils import secure_filename
import openai
from dotenv import load_dotenv
from PIL import Image

# ---- Load environment variables ----
load_dotenv()

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24).hex())

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {"pdf"}

openai.api_key = os.environ.get("OPENAI_API_KEY")

# ---- Dynamic Tesseract Path ----
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# ---- Utility Functions ----
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(filepath):
    text = []
    page_count = 0
    try:
        with fitz.open(filepath) as doc:
            page_count = doc.page_count
            for page in doc:
                page_text = page.get_text()
                if not page_text.strip():
                    print(f"[OCR] No text found on page {page.number}, running OCR...")
                    pix = page.get_pixmap()
                    img_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}_page.png")
                    pix.save(img_path)
                    ocr_text = pytesseract.image_to_string(Image.open(img_path))
                    os.remove(img_path)
                    text.append(ocr_text)
                else:
                    print(f"[PDF] Page {page.number} text: {page_text[:100]}")  # Print first 100 chars
                    text.append(page_text)
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return "", 0
    all_text = "\n".join(text)
    print(f"[FINAL EXTRACTED TEXT] (first 400 chars):\n{all_text[:400]}")
    return all_text, page_count


def get_session_pdfs():
    return session.setdefault("pdfs", {})

def save_session_pdfs(pdfs):
    session["pdfs"] = pdfs

def get_chat_history(pdf_name):
    history = session.setdefault("history", {})
    return history.setdefault(pdf_name, [])

def add_chat_history(pdf_name, role, content):
    history = session.setdefault("history", {})
    history.setdefault(pdf_name, []).append({"role": role, "content": content})

def clear_chat_history():
    session["history"] = {}

# ---- Routes ----
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    pdfs = get_session_pdfs()
    files = request.files.getlist("pdfs")
    summaries = {}
    for f in files:
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            save_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex[:8]}_{filename}")
            f.save(save_path)
            text, page_count = extract_text_from_pdf(save_path)
            file_size = round(os.path.getsize(save_path) / 1024.0, 2)  # in KB
            pdfs[filename] = {
                "path": save_path,
                "text": text,
                "pages": page_count,
                "size": file_size
            }
            summaries[filename] = {
                "pages": page_count,
                "size": file_size
            }
    save_session_pdfs(pdfs)
    return jsonify({"status": "ok", "pdf_names": list(pdfs.keys()), "summaries": summaries})

@app.route("/get_pdfs")
def get_pdfs():
    pdfs = get_session_pdfs()
    summaries = {name: {"pages": pdfs[name]["pages"], "size": pdfs[name]["size"]} for name in pdfs}
    return jsonify({"pdf_names": list(pdfs.keys()), "summaries": summaries})

@app.route("/get_history")
def get_history():
    pdf_name = request.args.get("pdf_name", "")
    history = session.get("history", {})
    if pdf_name == "__ALL__":
        all_history = []
        for h in history.values():
            all_history.extend(h)
        return jsonify({"history": all_history})
    return jsonify({"history": history.get(pdf_name, [])})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_msg = data.get("message", "").strip()
    pdf_name = data.get("pdf_name", "")

    pdfs = get_session_pdfs()
    if not pdfs:
        return jsonify({"reply": "No PDF uploaded. Please upload a PDF first."})

    # Context: all PDFs or one
    if pdf_name == "__ALL__":
        context = "\n\n".join(
            [pdfs[name]["text"] for name in pdfs if pdfs[name]["text"].strip()]
        )
        context_names = ", ".join([name for name in pdfs])
        system_prompt = (
            f"You are an AI assistant for PDF chat. The user uploaded these PDFs: {context_names}.\n"
            f"Answer based on the content. Use markdown if useful."
        )
    else:
        if pdf_name not in pdfs:
            return jsonify({"reply": "PDF not found or has been removed. Please upload or select a valid PDF."})
        context = pdfs[pdf_name]["text"]
        system_prompt = (
            f"You are an AI assistant for PDF chat. The user uploaded this PDF: {pdf_name}.\n"
            f"Answer based on the content. Use markdown if useful."
        )

    add_chat_history(pdf_name, "user", user_msg)
    chat_history = get_chat_history(pdf_name)

    messages = [{"role": "system", "content": system_prompt}]
    for h in chat_history[-6:]:
        messages.append({"role": h["role"], "content": h["content"]})
    # Append user's latest with context chunk (limit to 3500 chars)
    messages.append({"role": "user", "content": f"{user_msg}\n\nPDF content:\n{context[:3500]}"})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.2,
            max_tokens=800
        )
        reply = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI error: {e}")
        reply = "Sorry, there was an error connecting to the AI. Please try again later."

    add_chat_history(pdf_name, "bot", reply)
    return jsonify({"reply": reply})

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/remove_pdf", methods=["POST"])
def remove_pdf():
    data = request.get_json()
    filename = data.get("filename")
    pdfs = get_session_pdfs()
    if filename in pdfs:
        file_path = pdfs[filename]["path"]
        if os.path.exists(file_path):
            os.remove(file_path)
        pdfs.pop(filename)
        save_session_pdfs(pdfs)
        if "history" in session and filename in session["history"]:
            session["history"].pop(filename)
        return jsonify({"status": "ok", "pdf_names": list(pdfs.keys())})
    return jsonify({"status": "error", "message": "PDF not found"})

@app.route("/remove_all_pdfs", methods=["POST"])
def remove_all_pdfs():
    pdfs = get_session_pdfs()
    for pdf in pdfs.values():
        if os.path.exists(pdf["path"]):
            os.remove(pdf["path"])
    session["pdfs"] = {}
    session["history"] = {}
    return jsonify({"status": "ok", "pdf_names": []})

@app.route("/reset", methods=["POST"])
def reset():
    session.clear()
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    # Bind to 0.0.0.0 and port 5000 for dev; production uses gunicorn (see render.yaml)
    app.run(host="0.0.0.0", port=5000)
