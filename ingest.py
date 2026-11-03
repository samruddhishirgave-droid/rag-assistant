import os, io, re, sqlite3, uuid, time, threading
from contextlib import contextmanager

try:
    import chromadb
except ModuleNotFoundError:  # pragma: no cover - optional in local/test environments
    chromadb = None

from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template

try:
    from google import genai
    from google.genai import errors, types
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional in local/test environments
    genai = None
    errors = None
    types = None

try:
    from pypdf import PdfReader
except ModuleNotFoundError:  # pragma: no cover - optional in local/test environments
    PdfReader = None

from ingest import chunk_text, EMBED_MODEL

load_dotenv()


def get_client():
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not set.")
    if genai is None or types is None:
        raise RuntimeError("google-genai is not installed.")
    global client
    if client is None:
        client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY"),
            http_options=types.HttpOptions(timeout=45_000),
        )
    return client


client = None
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 3 * 1024 * 1024

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
TOP_K = 4
try:
    MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.6"))  # cosine distance cut-off
except ValueError:
    MAX_DISTANCE = 0.6
MAX_HISTORY = 6          # keep the last 6 messages
MAX_INPUT = 500
MAX_FILE_BYTES = 2 * 1024 * 1024
MIN_TEXT_CHARS = 200
MAX_TEXT_CHARS = 60_000  # about 35 chunks: small enough to index in well under a minute
MAX_CHUNKS = 40
EMBED_BATCH = 10
INDEX_DEADLINE = 100     # seconds before an upload is given up with a clear message
MAX_RUNNING_JOBS = 2
MAX_UPLOADS = 20
ALLOWED = {".txt", ".md", ".pdf"}
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
JOB_DB_PATH = os.getenv("UPLOAD_JOBS_DB", os.path.join(CHROMA_PATH, "upload_jobs.sqlite3"))

conversations = {}            # conversation_id -> list of messages
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
DB = None


def get_db():
    global DB
    if chromadb is None:
        raise RuntimeError("chromadb is not installed.")
    if DB is None:
        os.makedirs(CHROMA_PATH, exist_ok=True)
        DB = chromadb.PersistentClient(path=CHROMA_PATH)
    return DB


@contextmanager
def connect_job_db():
    os.makedirs(os.path.dirname(os.path.abspath(JOB_DB_PATH)), exist_ok=True)
    conn = sqlite3.connect(JOB_DB_PATH, timeout=30)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS upload_jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                stage TEXT NOT NULL,
                done INTEGER NOT NULL,
                total INTEGER NOT NULL,
                created REAL NOT NULL,
                doc_id TEXT,
                name TEXT,
                chunks INTEGER,
                truncated INTEGER,
                error TEXT
            )"""
        )
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def create_job(job_id):
    now = time.time()
    with connect_job_db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "DELETE FROM upload_jobs WHERE status != 'running' AND created < ?",
            (now - 900,),
        )
        conn.execute(
            """UPDATE upload_jobs
               SET status = 'error', stage = 'error',
                   error = 'Upload processing was interrupted. Please upload the file again.'
               WHERE status = 'running' AND created < ?""",
            (now - 900,),
        )
        running = conn.execute(
            "SELECT COUNT(*) FROM upload_jobs WHERE status = 'running'"
        ).fetchone()[0]
        if running >= MAX_RUNNING_JOBS:
            return False
        conn.execute(
            """INSERT INTO upload_jobs (job_id, status, stage, done, total, created)
               VALUES (?, 'running', 'queued', 0, 0, ?)""",
            (job_id, now),
        )
    return True


def update_job(job_id, **values):
    allowed = {"status", "stage", "done", "total", "doc_id", "name", "chunks", "truncated", "error"}
    if not values or not values.keys() <= allowed:
        raise ValueError("Invalid upload job update.")
    assignments = ", ".join(f"{key} = ?" for key in values)
    with connect_job_db() as conn:
        result = conn.execute(
            f"UPDATE upload_jobs SET {assignments} WHERE job_id = ?",
            (*values.values(), job_id),
        )
        if result.rowcount != 1:
            raise RuntimeError(f"Upload job {job_id} no longer exists.")


def get_job(job_id):
    with connect_job_db() as conn:
        row = conn.execute(
            "SELECT * FROM upload_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
    return dict(row) if row else None


def prune_jobs():
    cutoff = time.time() - 900
    with connect_job_db() as conn:
        conn.execute(
            "DELETE FROM upload_jobs WHERE status != 'running' AND created < ?",
            (cutoff,),
        )
        conn.execute(
            """UPDATE upload_jobs
               SET status = 'error', stage = 'error',
                   error = 'Upload processing was interrupted. Please upload the file again.'
               WHERE status = 'running' AND created < ?""",
            (cutoff,),
        )


def prune_uploaded_collections(db):
    uploaded = []
    for collection in db.list_collections():
        name = collection if isinstance(collection, str) else collection.name
        if not name.startswith("u_"):
            continue
        metadata = db.get_collection(name).metadata or {}
        if metadata.get("kind") == "upload":
            uploaded.append((metadata.get("created", 0), name))
    uploaded.sort()
    for _, name in uploaded[:-MAX_UPLOADS]:
        try:
            db.delete_collection(name)
        except Exception as e:
            print("UPLOAD CLEANUP ERROR:", repr(e))


def api_error_code(error):
    if errors is not None and isinstance(error, errors.APIError):
        return getattr(error, "code", None)
    return None


def embed_batch(texts, task, retries=3):
    """Embed a small batch, waiting and retrying on rate limits or temporary errors."""
    if errors is None or types is None:
        raise RuntimeError("google-genai is not installed.")
    c = get_client()
    delay = 4
    for attempt in range(retries + 1):
        try:
            r = c.models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(task_type=task),
            )
            return [e.values for e in r.embeddings]
        except errors.APIError as e:
            if getattr(e, "code", None) in (429, 500, 503) and attempt < retries:
                time.sleep(delay)
                delay *= 2
                continue
            raise


def extract_text(filename, data):
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED:
        raise ValueError("Unsupported file type. Please upload a .txt, .md or .pdf file.")
    if ext == ".pdf":
        if PdfReader is None:
            raise ValueError("PDF reading support is unavailable. Install pypdf to upload PDFs.")
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


def run_index_job(job_id, name, data):
    started = time.time()
    try:
        update_job(job_id, stage="reading")
        text = extract_text(name, data).strip()
        if len(text) < MIN_TEXT_CHARS:
            raise ValueError("Not enough readable text in this file (a scanned PDF has no text layer).")

        truncated = len(text) > MAX_TEXT_CHARS
        chunks = chunk_text(text[:MAX_TEXT_CHARS])
        if len(chunks) > MAX_CHUNKS:
            chunks = chunks[:MAX_CHUNKS]
            truncated = True

        update_job(job_id, stage="indexing", total=len(chunks))
        vectors = []
        for i in range(0, len(chunks), EMBED_BATCH):
            if time.time() - started > INDEX_DEADLINE:
                raise ValueError("Indexing took too long. Please try a shorter document.")
            vectors += embed_batch(chunks[i:i + EMBED_BATCH], "RETRIEVAL_DOCUMENT")
            update_job(job_id, done=min(i + EMBED_BATCH, len(chunks)))

        doc_id = uuid.uuid4().hex[:16]
        cname = "u_" + doc_id
        db = get_db()
        with db_lock:
            col = db.create_collection(
                cname,
                metadata={
                    "hnsw:space": "cosine",
                    "kind": "upload",
                    "source": name,
                    "created": time.time(),
                    "chunks": len(chunks),
                },
            )
            col.add(
                ids=[f"{doc_id}-{i}" for i in range(len(chunks))],
                documents=chunks,
                embeddings=vectors,
                metadatas=[{"source": name, "chunk": i} for i in range(len(chunks))],
            )
        prune_uploaded_collections(db)
        update_job(
            job_id,
            status="done",
            stage="done",
            doc_id=doc_id,
            name=name,
            chunks=len(chunks),
            truncated=int(truncated),
        )

    except ValueError as e:
        update_job(job_id, status="error", stage="error", error=str(e))
    except Exception as e:
        code = api_error_code(e)
        if code is not None:
            print("EMBED ERROR:", code, e)
            msg = ("Too many requests right now. Please wait a minute and upload again."
                   if code == 429 else "The AI service had a problem while reading your file. Please try again.")
        else:
            print("UPLOAD ERROR:", repr(e))
            msg = "Could not process the file. Please try again."
        update_job(job_id, status="error", stage="error", error=msg)


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
    try:
        get_client()
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 503

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
    job_id = uuid.uuid4().hex[:16]
    if not create_job(job_id):
        return jsonify(error="The server is busy indexing other uploads. Please try again in a minute."), 429
    threading.Thread(target=run_index_job, args=(job_id, name, data), daemon=True).start()
    return jsonify(job_id=job_id), 202


@app.route("/upload/<job_id>")
def upload_status(job_id):
    job = get_job(job_id)
    if not job:
        return jsonify(error="This upload was not found. Please upload the file again."), 404
    out = {k: v for k, v in job.items() if k != "created"}
    if out.get("truncated") is not None:
        out["truncated"] = bool(out["truncated"])
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
        try:
            get_client()
        except RuntimeError as exc:
            return jsonify(error=str(exc)), 503

        db = get_db()
        if doc_id:
            if not isinstance(doc_id, str) or not re.fullmatch(r"[a-f0-9]{16}", doc_id):
                return jsonify(error="Your uploaded document is no longer available. Please upload it again."), 404
            collection_name = "u_" + doc_id
            available = {
                collection if isinstance(collection, str) else collection.name
                for collection in db.list_collections()
            }
            if collection_name not in available:
                return jsonify(error="Your uploaded document is no longer available. Please upload it again."), 404
            col = db.get_collection(collection_name)
            upload_name = (col.metadata or {}).get("source")
        else:
            col = db.get_collection("docs")

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
            r = get_client().models.generate_content(
                model=MODEL, contents=contents,
                config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.2),
            )
            reply = r.text or "Sorry, I couldn't generate an answer. Please try again."

        history.append({"role": "user", "text": msg})
        history.append({"role": "model", "text": reply})
        del history[:-MAX_HISTORY]
        return jsonify(reply=reply, sources=sources, conversation_id=cid)

    except Exception as e:
        code = api_error_code(e)
        if code is not None:
            print("API ERROR:", code, e)
            err = ("Too many requests right now. Please wait a moment and try again."
                   if code == 429 else "The AI service had a problem. Please try again.")
            return jsonify(error=err), 502
        print("ERROR:", repr(e))
        return jsonify(error="Something went wrong on the server. Please try again."), 500


@app.route("/reset", methods=["POST"])
def reset():
    cid = (request.get_json(silent=True) or {}).get("conversation_id")
    conversations.pop(cid, None)
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(debug=True)
