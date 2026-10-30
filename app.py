import os, io, uuid, time, threading
from collections import OrderedDict
import chromadb
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from google import genai
from google.genai import errors, types
from pypdf import PdfReader
from ingest import chunk_text, EMBED_MODEL

load_dotenv()
# A request timeout (milliseconds) so a slow Gemini call fails fast instead of hanging.
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(timeout=45_000),
)
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 3 * 1024 * 1024

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
TOP_K = 4
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.6"))  # cosine distance cut-off
MAX_HISTORY = 6          # keep the last 6 messages
MAX_INPUT = 500
MAX_FILE_BYTES = 2 * 1024 * 1024
MIN_TEXT_CHARS = 200

MAX_TEXT_CHARS = 16_000
MAX_CHUNKS = 8
EMBED_BATCH = 4
INDEX_DEADLINE = 180
MAX_RUNNING_JOBS = 1
MAX_UPLOADS = 5
MAX_UPLOADS = 20
ALLOWED = {".txt", ".md", ".pdf"}

conversations = {}            # conversation_id -> list of messages
uploads = OrderedDict()       # doc_id -> {"name", "collection", "chunks"}
jobs = {}                     # job_id -> progress of a background indexing job
lock = threading.Lock()       # protects uploads / jobs
db_lock = threading.Lock()    # protects Chroma writes

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

# Sample docs live in the "docs" collection (built by ingest.py at deploy time);
# uploaded files live in temporary collections named "u_<id>".
DB = chromadb.PersistentClient(path="chroma_db")
for _c in DB.list_collections():
    _name = _c if isinstance(_c, str) else _c.name
    if _name.startswith("u_"):
        try:
            DB.delete_collection(_name)
        except Exception:
            pass


def embed_batch(texts, task, retries=3):
    """Embed a small batch, waiting and retrying on rate limits or temporary errors."""
    delay = 4
    for attempt in range(retries + 1):
        try:
            r = client.models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(task_type=task),
            )
            return [e.values for e in r.embeddings]
        except errors.APIError as e:
            if e.code in (429, 500, 503) and attempt < retries:
                time.sleep(delay)
                delay *= 2
                continue
            raise


def extract_text(filename, data):
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED:
        raise ValueError("Unsupported file type. Please upload a .txt, .md or .pdf file.")
    if ext == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                raise ValueError("This PDF is password-protected.")
            parts, total = [], 0
            for page in reader.pages:          # stop early once we have enough text
                t = page.extract_text() or ""
                parts.append(t)
                total += len(t)
                if total >= MAX_TEXT_CHARS:
                    break
            return "\n\n".join(parts)
        except ValueError:
            raise
        except Exception:
            raise ValueError("Could not read this PDF. Try a text-based PDF or a .txt file.")
    return data.decode("utf-8", errors="ignore")


def retrieve(col, question):
    qv = embed_batch([question], "RETRIEVAL_QUERY")[0]
    res = col.query(query_embeddings=[qv], n_results=TOP_K)
    return list(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]))


def prune_jobs():
    cutoff = time.time() - 900
    with lock:
        for jid in [j for j, v in jobs.items() if v["created"] < cutoff and v["status"] != "running"]:
            jobs.pop(jid, None)


def run_index_job(job_id, name, data):
    job = jobs[job_id]
    started = time.time()
    try:
        job["stage"] = "reading"
        text = extract_text(name, data).strip()
        if len(text) < MIN_TEXT_CHARS:
            raise ValueError("Not enough readable text in this file (a scanned PDF has no text layer).")

        truncated = len(text) > MAX_TEXT_CHARS
        chunks = chunk_text(text[:MAX_TEXT_CHARS])
        if len(chunks) > MAX_CHUNKS:
            chunks = chunks[:MAX_CHUNKS]
            truncated = True

        job["stage"] = "indexing"
        job["total"] = len(chunks)
        vectors = []
        for i in range(0, len(chunks), EMBED_BATCH):
            if time.time() - started > INDEX_DEADLINE:
                raise ValueError("Indexing took too long. Please try a shorter document.")
            vectors += embed_batch(chunks[i:i + EMBED_BATCH], "RETRIEVAL_DOCUMENT")
            job["done"] = min(i + EMBED_BATCH, len(chunks))

        doc_id = uuid.uuid4().hex[:16]
        cname = "u_" + doc_id
        with db_lock:
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
                    with db_lock:
                        DB.delete_collection(old["collection"])
                except Exception:
                    pass
        job.update(status="done", doc_id=doc_id, name=name, chunks=len(chunks), truncated=truncated)

    except ValueError as e:
        job.update(status="error", error=str(e))
    except errors.APIError as e:
        print("EMBED ERROR:", e.code, e)
        msg = ("Too many requests right now. Please wait a minute and upload again."
               if e.code == 429 else "The AI service had a problem while reading your file. Please try again.")
        job.update(status="error", error=msg)
    except Exception as e:
        print("UPLOAD ERROR:", repr(e))
        job.update(status="error", error="Could not process the file. Please try again.")


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
    ext = os.path.splitext(name)[1].lower()
    if ext not in ALLOWED:
        return jsonify(error="Unsupported file type. Please upload a .txt, .md or .pdf file."), 400
    data = f.read()
    if not data:
        return jsonify(error="The file is empty."), 400
    if len(data) > MAX_FILE_BYTES:
        return jsonify(error="File is too large. The limit is 2 MB."), 413

    prune_jobs()
    with lock:
        running = sum(1 for v in jobs.values() if v["status"] == "running")
        if running >= MAX_RUNNING_JOBS:
            return jsonify(error="The server is busy indexing other uploads. Please try again in a minute."), 429
        job_id = uuid.uuid4().hex[:16]
        jobs[job_id] = {"status": "running", "stage": "queued", "done": 0, "total": 0,
                        "created": time.time()}
    threading.Thread(target=run_index_job, args=(job_id, name, data), daemon=True).start()
    return jsonify(job_id=job_id), 202


@app.route("/upload/<job_id>")
def upload_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify(error="This upload was not found. Please upload the file again."), 404
    out = {k: v for k, v in job.items() if k != "created"}
    return jsonify(out)


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
