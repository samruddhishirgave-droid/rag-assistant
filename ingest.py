import os, glob, time

try:
    import chromadb
except ModuleNotFoundError:  # pragma: no cover - optional in local/test environments
    chromadb = None

from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional in local/test environments
    genai = None
    types = None

load_dotenv()


client = None


def get_client():
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not set.")
    if genai is None or types is None:
        raise RuntimeError("google-genai is not installed.")
    global client
    if client is None:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return client

EMBED_MODEL = "gemini-embedding-001"
CHUNK_SIZE = 2000   # about 500 tokens
OVERLAP = 300       # characters shared between neighbouring chunks
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")

def chunk_text(text):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            # prefer to cut at a paragraph or sentence boundary
            cut = text.rfind("\n\n", start, end)
            if cut == -1:
                cut = text.rfind(". ", start, end)
            if cut > start + CHUNK_SIZE // 2:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - OVERLAP
    return [c for c in chunks if c]

def embed(texts, task):
    if types is None:
        raise RuntimeError("google-genai is not installed.")
    c = get_client()
    vectors = []
    for i in range(0, len(texts), 50):
        batch = texts[i:i + 50]
        r = c.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task),
        )
        vectors += [e.values for e in r.embeddings]
        time.sleep(1)
    return vectors

if __name__ == "__main__":
    if chromadb is None:
        raise RuntimeError("chromadb is not installed.")
    try:
        get_client()
    except RuntimeError as exc:
        raise SystemExit(str(exc))

    ids, docs, metas = [], [], []
    for path in glob.glob("docs/*.md") + glob.glob("docs/*.txt"):
        text = open(path, encoding="utf-8").read()
        for n, chunk in enumerate(chunk_text(text)):
            ids.append(f"{os.path.basename(path)}-{n}")
            docs.append(chunk)
            metas.append({"source": os.path.basename(path), "chunk": n})

    print(f"{len(docs)} chunks created. Embedding...")
    vectors = embed(docs, "RETRIEVAL_DOCUMENT")

    os.makedirs(CHROMA_PATH, exist_ok=True)
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        db.delete_collection("docs")
    except Exception:
        pass
    col = db.create_collection("docs", metadata={"hnsw:space": "cosine"})
    col.add(ids=ids, documents=docs, embeddings=vectors, metadatas=metas)
    print("Stored", col.count(), "chunks in Chroma.")

    # inspect a few chunks manually (a task requirement)
    for d in docs[:3]:
        print("\n--- SAMPLE CHUNK ---\n", d[:500])
