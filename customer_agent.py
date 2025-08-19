# src/customer_agent_agno.py
"""
Agno-based Customer Support Agent (retrieval-first, local-only).
- Uses sentence-transformers for local embeddings & cosine similarity
- Uses Chroma PersistentClient for candidate retrieval (fallback)
Behavior:
 - If local cosine similarity for best FAQ >= CONFIDENCE_THRESHOLD -> return that single FAQ only
 - Otherwise -> show top Chroma candidates (deduped) and allow escalation
This keeps Agno in the stack (tool/Agent), but uses local embeddings to pick the single best FAQ.
"""

import os
import csv
import time
from dotenv import load_dotenv
from typing import Dict, Any, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Agno imports (be tolerant)
from agno.agent import Agent
try:
    from agno.tools import Tool  # type: ignore
except Exception:
    Tool = None

load_dotenv()

# Config
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("FAQ_COLLECTION_NAME", "faq_collection")
FAQ_CSV = os.path.join("data", "faq.csv")
LOCAL_EMB_MODEL = os.getenv("LOCAL_EMB_MODEL", "all-MiniLM-L6-v2")
CONFIDENCE_THRESHOLD = float(os.getenv("FAQ_CONFIDENCE_THRESHOLD", "0.70"))  # cosine 0..1

# Load local embedding model (used both for ingestion earlier and for local similarity checks)
print("Loading SentenceTransformer model:", LOCAL_EMB_MODEL)
embed_model = SentenceTransformer(LOCAL_EMB_MODEL)

# Load FAQ CSV into memory and precompute embeddings for questions (or question+answer)
def load_faqs_and_embeddings(csv_path: str) -> Tuple[List[Dict[str, str]], List[List[float]]]:
    faqs = []
    if not os.path.exists(csv_path):
        return faqs, []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = (row.get("question") or "").strip()
            a = (row.get("answer") or "").strip()
            _id = str(row.get("id") or len(faqs))
            if not q or not a:
                continue
            faqs.append({"id": _id, "question": q, "answer": a})

    if not faqs:
        return faqs, []

    # Choose embedding text: question alone or question + answer
    texts = [item["question"] for item in faqs]  # change to question + " " + answer if you prefer
    np_embs = embed_model.encode(texts, show_progress_bar=False)
    emb_list = []
    for v in np_embs:
        if hasattr(v, "tolist"):
            emb_list.append(v.tolist())
        else:
            emb_list.append(list(map(float, v)))
    return faqs, emb_list

faqs_in_memory, faq_embeddings = load_faqs_and_embeddings(FAQ_CSV)
print(f"Loaded {len(faqs_in_memory)} FAQ entries into memory for similarity checks.")

# Connect to persistent Chroma created by ingest script (for candidate list fallback)
client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(allow_reset=True))
try:
    faq_collection = client.get_collection(COLLECTION_NAME)
except Exception as e:
    print("Error getting collection:", e)
    raise SystemExit("Run ingestion first: python src/ingest_faq.py")

# Cosine similarity helpers
def cosine_similarity(a: List[float], b: List[float]) -> float:
    a_np = np.asarray(a, dtype=float)
    b_np = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a_np) * np.linalg.norm(b_np)
    if denom == 0:
        return 0.0
    return float(np.dot(a_np, b_np) / denom)

def find_best_local_match(query: str) -> Tuple[float, Dict[str, str]]:
    """
    Return (best_similarity (0..1), best_faq_dict)
    If no faqs loaded, returns (0.0, {}).
    """
    if not faq_embeddings or not faqs_in_memory:
        return 0.0, {}
    q_emb = embed_model.encode(query, show_progress_bar=False)
    # ensure python list
    if hasattr(q_emb, "tolist"):
        q_emb = q_emb.tolist()
    best_sim = -1.0
    best_idx = -1
    for i, emb in enumerate(faq_embeddings):
        sim = cosine_similarity(q_emb, emb)
        if sim > best_sim:
            best_sim = sim
            best_idx = i
    best_faq = faqs_in_memory[best_idx] if best_idx >= 0 else {}
    best_sim_clamped = max(best_sim, 0.0)
    return best_sim_clamped, best_faq

# Query Chroma for candidate fallback (deduplicated formatting)
def query_chroma_candidates(query: str, k: int = 5) -> Dict[str, Any]:
    try:
        res = faq_collection.query(query_texts=[query], n_results=k)
        return res
    except Exception:
        # If query_texts unsupported, raise a clear message
        raise RuntimeError("Chroma query_texts not supported in this chroma build. Ensure you ingested with compatible chroma.")

def format_chroma_results(res: Dict[str, Any]) -> str:
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    ids = res.get("ids", [[]])[0]
    dists = res.get("distances", [[]])[0] if "distances" in res else [None] * len(docs)
    lines = []
    seen = set()
    for i, doc in enumerate(docs):
        idx = ids[i] if i < len(ids) else f"idx{i}"
        if idx in seen:
            continue
        seen.add(idx)
        q = metas[i].get("question", "") if i < len(metas) else ""
        score = dists[i] if i < len(dists) else None
        lines.append(f"[FAQ#{idx}] Q: {q}\nA: {doc}\n(distance={score})")
    return "\n\n".join(lines)

# Optional Agno tool wrapper (use decorator if available)
if Tool is not None:
    @Tool
    def faq_tool(query: str) -> str:
        # For compatibility, return formatted chroma results (fallback)
        res = query_chroma_candidates(query, k=5)
        return format_chroma_results(res)
else:
    def faq_tool(query: str) -> str:
        res = query_chroma_candidates(query, k=5)
        return format_chroma_results(res)
    faq_tool.name = "FAQTool"
    faq_tool.description = "Returns top FAQ matches from local Chroma"

# Build Agno Agent (best-effort; model=None may be acceptable)
agent = None
try:
    agent = Agent(
        name="AgnoCustomerSupport",
        role="Customer support agent using local FAQ.",
        model=None,
        instructions=[
            "Use the FAQ database first. If confident, return the FAQ answer. Otherwise suggest escalation."
        ],
        tools=[faq_tool],
    )
    print("Agno Agent created successfully.")
except Exception as e:
    agent = None
    print("Warning: could not create Agent with current Agno installation. Continuing with retrieval-only tool.")
    print("Agent creation error:", e)

# Interactive loop
def interactive():
    print("Agno-based Customer Support (type 'exit' to quit)")
    print(f"Local confidence threshold (cosine) = {CONFIDENCE_THRESHOLD}\n")
    while True:
        q = input("User: ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break

        # 1) Local best match check
        best_sim, best_faq = find_best_local_match(q)
        print(f"[debug] best local cosine similarity = {best_sim:.4f}")
        if best_sim >= CONFIDENCE_THRESHOLD and best_faq:
            print("\nAgent (CONFIDENT - single FAQ):")
            print(f"[FAQ#{best_faq.get('id')}] Q: {best_faq.get('question')}\nA: {best_faq.get('answer')}\n(similarity={best_sim:.3f})\n")
            continue

        # 2) Not confident -> show chroma top candidates (deduped)
        try:
            print("\nAgent: I couldn't find a confident FAQ match. Here are top candidates:")
            chroma_res = query_chroma_candidates(q, k=5)
            print(format_chroma_results(chroma_res))
        except Exception as e:
            print("\nAgent: Could not query Chroma candidates:", e)

        # Offer escalation
        choice = input("\nType 'escalate' to create a ticket for human support, or press Enter to continue: ").strip().lower()
        if choice == "escalate":
            ticket_id = str(int(time.time() * 1000))
            # simple file logging (append)
            os.makedirs("data", exist_ok=True)
            with open(os.path.join("data", "tickets.csv"), "a", encoding="utf-8") as f:
                f.write(f"{ticket_id},{time.strftime('%Y-%m-%d %H:%M:%S')},{q},open\n")
            print(f"\nAgent: A ticket has been created (ID: {ticket_id}). Our team will follow up.")
        else:
            print("\nAgent: Okay — you can escalate if needed.\n")

if __name__ == "__main__":
    interactive()
