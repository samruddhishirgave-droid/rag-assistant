import os, glob, time
import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

EMBED_MODEL = "gemini-embedding-001"
CHUNK_SIZE = 2000   # about 500 tokens
OVERLAP = 300       # characters shared between neighbouring chunks

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
    vectors = []
    for i in range(0, len(texts), 50):
        batch = texts[i:i + 50]
        r = client.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task),
        )
        vectors += [e.values for e in r.embeddings]
        time.sleep(1)
    return vectors

if __name__ == "__main__":
    ids, docs, metas = [], [], []
    for path in glob.glob("docs/*.md") + glob.glob("docs/*.txt"):
        text = open(path, encoding="utf-8").read()
        for n, chunk in enumerate(chunk_text(text)):
            ids.append(f"{os.path.basename(path)}-{n}")
            docs.append(chunk)
            metas.append({"source": os.path.basename(path), "chunk": n})

    print(f"{len(docs)} chunks created. Embedding...")
    vectors = embed(docs, "RETRIEVAL_DOCUMENT")

    db = chromadb.PersistentClient(path="chroma_db")
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