import os, uuid
import chromadb
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from google import genai
from google.genai import errors, types
from ingest import embed

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
app = Flask(__name__)

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
TOP_K = 4
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.6"))  # cosine distance cut-off
MAX_HISTORY = 6          # context management: keep the last 6 messages
MAX_INPUT = 500
conversations = {}       # conversation_id -> list of {"role", "text"}

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

_col = None
def collection():
    global _col
    if _col is None:
        _col = chromadb.PersistentClient(path="chroma_db").get_collection("docs")
    return _col

def retrieve(question):
    qv = embed([question], "RETRIEVAL_QUERY")[0]
    res = collection().query(query_embeddings=[qv], n_results=TOP_K)
    return list(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify(ok=True)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    msg = (data.get("message") or "").strip()
    cid = data.get("conversation_id") or str(uuid.uuid4())
    if not msg:
        return jsonify(error="Please type a question."), 400
    if len(msg) > MAX_INPUT:
        return jsonify(error=f"Question is too long (max {MAX_INPUT} characters)."), 400

    history = conversations.setdefault(cid, [])
    try:
        hits = retrieve(msg)
        relevant = [h for h in hits if h[2] <= MAX_DISTANCE]
        sources = [{"n": i + 1, "source": m["source"], "chunk": m["chunk"],
                    "distance": round(d, 3), "snippet": c[:300]}
                   for i, (c, m, d) in enumerate(relevant)]

        if not relevant:
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