import os, io, uuid
from collections import OrderedDict
from threading import Lock
import chromadb
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from google import genai
from google.genai import errors, types
from pypdf import PdfReader
from ingest import embed, chunk_text

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 3 * 1024 * 1024

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
TOP_K = 4
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.6"))  # cosine distance cut-off
MAX_HISTORY = 6          # keep the last 6 messages
MAX_INPUT = 500
MAX_FILE_BYTES = 2 * 1024 * 1024
MIN_TEXT_CHARS = 200
MAX_TEXT_CHARS = 150_000
MAX_UPLOADS = 20
ALLOWED = {".txt", ".md", ".pdf"}

conversations = {}            # conversation_id -> list of messages
uploads = OrderedDict()       # doc_id -> {"name", "collection", "chunks"}
lock = Lock()

SYSTEM_PROMPT = """You are 'Notes Assistant', a careful assistant that answers ONLY from the
document excerpts provided in each message.

Rules:
1. Use only the provided excerpts. Never use outside knowledge for facts.
2. Cite the excerpt(s) you used like [1], [2] at the end of the relevant sentence.
3. If the excerpts do not contain the answer, reply exactly:
   "I couldn't find that in my documents." and suggest what the documents do cover.
4. Keep answers short (2-5 sentences) and quote numbers exactly as written.

Examples:
Excerpts: [1] The model's MAE was 1.03 qtl/ha.
Question: What was the MAE?
Answer: The MAE was 1.03 qtl/ha [1].

Excerpts: [1] The model's MAE was 1.03 qtl/ha.
Question: Who won the 2011 cricket World Cup?
Answer: I couldn't find that in my documents. I can answer questions about the forecasting notes and the prompt engineering report."""

# One persistent Chroma client. Sample docs live in "docs" (built by ingest.py);
# uploaded files live in temporary collections named "u_<id>".
DB = chromadb.PersistentClient(path="chroma_db")
for _c in DB.list_collections():
    _name = _c if isinstance(_c, str) else _c.name
    if _name.startswith("u_"):
        try:
            DB.delete_collection(_name)
        except Exception:
            pass


def extract_text(filename, data):
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED:
        raise ValueError("Unsupported file type. Please upload a .txt, .md or .pdf file.")
    if ext == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                raise ValueError("This PDF is password-protected.")
            return "\n\n".join((p.extract_text() or "") for p in reader.pages)
        except ValueError:
            raise
        except Exception:
            raise ValueError("Could not read this PDF. Try a text-based PDF or a .txt file.")
    return data.decode("utf-8", errors="ignore")


def retrieve(col, question):
    qv = embed([question], "RETRIEVAL_QUERY")[0]
    res = col.query(query_embeddings=[qv], n_results=TOP_K)
    return list(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]))


@app.errorhandler(413)
def too_large(e):
    return jsonify(error="File is too large. The limit is 2 MB."), 413


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify(ok=True)


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(error="Please choose a file to upload."), 400
    name = os.path.basename(f.filename)[:80]
    data = f.read()
    if not data:
        return jsonify(error="The file is empty."), 400
    if len(data) > MAX_FILE_BYTES:
        return jsonify(error="File is too large. The limit is 2 MB."), 413

    try:
        text = extract_text(name, data).strip()
    except ValueError as e:
        return jsonify(error=str(e)), 400
    if len(text) < MIN_TEXT_CHARS:
        return jsonify(error="Not enough readable text in this file (a scanned PDF has no text layer)."), 400

    truncated = len(text) > MAX_TEXT_CHARS
    text = text[:MAX_TEXT_CHARS]
    chunks = chunk_text(text)

    try:
        vectors = embed(chunks, "RETRIEVAL_DOCUMENT")
    except errors.APIError as e:
        print("EMBED ERROR:", e.code, e)
        msg = ("Too many requests right now. Please wait a minute and upload again."
               if e.code == 429 else "The AI service had a problem while reading your file. Please try again.")
        return jsonify(error=msg), 502
    except Exception as e:
        print("UPLOAD ERROR:", repr(e))
        return jsonify(error="Could not process the file. Please try again."), 500

    doc_id = uuid.uuid4().hex[:16]
    cname = "u_" + doc_id
    col = DB.create_collection(cname, metadata={"hnsw:space": "cosine"})
    col.add(
        ids=[f"{doc_id}-{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=vectors,
        metadatas=[{"source": name, "chunk": i} for i in range(len(chunks))],
    )

    with lock:
        uploads[doc_id] = {"name": name, "collection": cname, "chunks": len(chunks)}
        while len(uploads) > MAX_UPLOADS:
            _, old = uploads.popitem(last=False)
            try:
                DB.delete_collection(old["collection"])
            except Exception:
                pass

    return jsonify(doc_id=doc_id, name=name, chunks=len(chunks), truncated=truncated)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    msg = (data.get("message") or "").strip()
    cid = data.get("conversation_id") or str(uuid.uuid4())
    doc_id = data.get("doc_id")
    if not msg:
        return jsonify(error="Please type a question."), 400
    if len(msg) > MAX_INPUT:
        return jsonify(error=f"Question is too long (max {MAX_INPUT} characters)."), 400

    upload_name = None
    try:
        if doc_id:
            entry = uploads.get(doc_id)
            if not entry:
                return jsonify(error="Your uploaded document is no longer available. Please upload it again."), 404
            col = DB.get_collection(entry["collection"])
            upload_name = entry["name"]
        else:
            col = DB.get_collection("docs")

        history = conversations.setdefault(cid, [])
        hits = retrieve(col, msg)
        relevant = [h for h in hits if h[2] <= MAX_DISTANCE]
        sources = [{"n": i + 1, "source": m["source"], "chunk": m["chunk"],
                    "distance": round(d, 3), "snippet": c[:300]}
                   for i, (c, m, d) in enumerate(relevant)]

        if not relevant:
            if upload_name:
                reply = (f"I couldn't find that in your uploaded document ({upload_name}). "
                         "Try rephrasing, or ask about something the document covers.")
            else:
                reply = ("I couldn't find that in my documents. I can answer questions about "
                         "the agribusiness forecasting notes and the prompt engineering report.")
        else:
            context = "\n\n".join(f"[{i+1}] ({m['source']}) {c}" for i, (c, m, d) in enumerate(relevant))
            user_turn = f"Excerpts:\n{context}\n\nQuestion: {msg}\nAnswer:"
            contents = [types.Content(role=h["role"], parts=[types.Part(text=h["text"])])
                        for h in history[-MAX_HISTORY:]]
            contents.append(types.Content(role="user", parts=[types.Part(text=user_turn)]))
            r = client.models.generate_content(
                model=MODEL, contents=contents,
                config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.2),
            )
            reply = r.text or "Sorry, I couldn't generate an answer. Please try again."

        history.append({"role": "user", "text": msg})
        history.append({"role": "model", "text": reply})
        del history[:-MAX_HISTORY]
        return jsonify(reply=reply, sources=sources, conversation_id=cid)

    except errors.APIError as e:
        print("API ERROR:", e.code, e)
        err = ("Too many requests right now. Please wait a moment and try again."
               if e.code == 429 else "The AI service had a problem. Please try again.")
        return jsonify(error=err), 502
    except Exception as e:
        print("ERROR:", repr(e))
        return jsonify(error="Something went wrong on the server. Please try again."), 500


@app.route("/reset", methods=["POST"])
def reset():
    cid = (request.get_json(silent=True) or {}).get("conversation_id")
    conversations.pop(cid, None)
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(debug=True)