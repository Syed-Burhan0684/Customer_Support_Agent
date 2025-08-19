# src/app.py
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os, csv, time
from dotenv import load_dotenv

# Create the FastAPI app at import time (uvicorn expects this)
app = FastAPI(title="Customer Support Agent (Retrieval-first)")

# Basic CORS so the UI can call the API during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Minimal Pydantic models for requests/responses
class ChatRequest(BaseModel):
    user_id: Optional[str] = "anon"
    message: str

class ChatResponse(BaseModel):
    reply: str
    confident: bool
    similarity: float
    candidates: List[Dict[str, Any]] = []

class EscalateRequest(BaseModel):
    user_id: Optional[str] = "anon"
    message: str

class EscalateResponse(BaseModel):
    ticket_id: str

# mount static folder (if exists)
if not os.path.exists("static"):
    os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Globals that will be set at startup
embed_model = None
faq_items = []
faq_embs = None
chroma_client = None
chroma_collection = None

# Config (read .env at startup)
load_dotenv()
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
FAQ_CSV = os.getenv("FAQ_CSV", "data/faq.csv")
LOCAL_EMB_MODEL = os.getenv("LOCAL_EMB_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = os.getenv("FAQ_COLLECTION_NAME", "faq_collection")
FAQ_CONFIDENCE_THRESHOLD = float(os.getenv("FAQ_CONFIDENCE_THRESHOLD", "0.70"))
TICKETS_FILE = os.path.join("data", "tickets.csv")

# Helper functions (safe to import)
def ensure_tickets_file():
    os.makedirs(os.path.dirname(TICKETS_FILE), exist_ok=True)
    if not os.path.exists(TICKETS_FILE):
        with open(TICKETS_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ticket_id", "timestamp", "user_id", "message", "status"])

def create_ticket(user_id: str, message: str) -> str:
    ensure_tickets_file()
    ticket_id = str(int(time.time() * 1000))
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(TICKETS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([ticket_id, timestamp, user_id, message, "open"])
    return ticket_id

# Startup: heavy work here so import stays lightweight
@app.on_event("startup")
def startup_event():
    global embed_model, faq_items, faq_embs, chroma_client, chroma_collection
    try:
        # load embedding model lazily
        from sentence_transformers import SentenceTransformer
        import numpy as np
        import chromadb
        from chromadb.config import Settings

        print("Startup: loading embedding model:", LOCAL_EMB_MODEL)
        embed_model = SentenceTransformer(LOCAL_EMB_MODEL)

        # load FAQ CSV into memory
        faqs = []
        if os.path.exists(FAQ_CSV):
            with open(FAQ_CSV, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    q = (row.get("question") or "").strip()
                    a = (row.get("answer") or "").strip()
                    _id = str(row.get("id") or len(faqs))
                    cat = row.get("category", "") or ""
                    if not q or not a:
                        continue
                    faqs.append({"id": _id, "question": q, "answer": a, "category": cat})
        faq_items = faqs
        print(f"Startup: loaded {len(faq_items)} FAQ rows from {FAQ_CSV}")

        # precompute embeddings (if any faqs)
        if faq_items:
            texts = [item["question"] for item in faq_items]
            arr = embed_model.encode(texts, show_progress_bar=False)
            if hasattr(arr, "tolist"):
                faq_embs = np.asarray(arr)
            else:
                import numpy as _np
                faq_embs = _np.asarray([list(map(float, v)) for v in arr])
        else:
            faq_embs = None

        # init persistent chroma client (for fallback candidates)
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(allow_reset=True))
        try:
            chroma_collection = chroma_client.get_collection(COLLECTION_NAME)
        except Exception:
            chroma_collection = chroma_client.create_collection(name=COLLECTION_NAME)

        print("Startup complete.")
    except Exception as e:
        # make startup errors visible in logs (so uvicorn will show them), but keep app defined
        print("ERROR during startup:", e)
        raise

# Utility used by endpoints (safe once startup finished)
def cosine_similarity(a, b):
    import numpy as np
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def find_best_local(query: str):
    global embed_model, faq_items, faq_embs
    if not faq_items or faq_embs is None:
        return 0.0, {}
    q_emb = embed_model.encode(query, show_progress_bar=False).tolist()
    best_sim = -1.0
    best_idx = -1
    for i, emb in enumerate(faq_embs):
        sim = cosine_similarity(q_emb, emb)
        if sim > best_sim:
            best_sim = sim
            best_idx = i
    return max(best_sim, 0.0), (faq_items[best_idx] if best_idx >= 0 else {})

def query_chroma_candidates(query: str, k: int = 5):
    try:
        res = chroma_collection.query(query_texts=[query], n_results=k)
        # build candidate dict list
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        ids = res.get("ids", [[]])[0]
        dists = res.get("distances", [[]])[0] if "distances" in res else [None]*len(docs)
        candidates = []
        seen = set()
        for i, doc in enumerate(docs):
            _id = ids[i] if i < len(ids) else str(i)
            if _id in seen:
                continue
            seen.add(_id)
            meta_q = metas[i].get("question", "") if i < len(metas) else ""
            candidates.append({"id": _id, "question": meta_q, "answer": doc, "distance": dists[i] if i < len(dists) else None})
        return candidates
    except Exception:
        return []

@app.get("/health")
def health():
    return {"status": "ok", "faqs_loaded": len(faq_items)}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if embed_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet (startup in progress or failed). Check server logs.")
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Empty message")

    best_sim, best_faq = find_best_local(message)
    confident = best_sim >= FAQ_CONFIDENCE_THRESHOLD and bool(best_faq)
    if confident:
        return ChatResponse(reply=best_faq["answer"], confident=True, similarity=float(best_sim), candidates=[])
    candidates = query_chroma_candidates(message, k=5)
    return ChatResponse(reply=(candidates[0]["answer"] if candidates else "No match found"), confident=False, similarity=float(best_sim), candidates=candidates)

@app.post("/escalate", response_model=EscalateResponse)
def escalate(req: EscalateRequest):
    ticket_id = create_ticket(req.user_id or "anon", req.message or "")
    return EscalateResponse(ticket_id=ticket_id)
