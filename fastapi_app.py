# fastapi_app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import json, os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import logging

# Try to reuse your logger_setup (if present)
try:
    from logger_setup import get_logger as user_get_logger
    logger = user_get_logger("chatapi", logs_dir="logs")
except Exception:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("chatapi")

app = FastAPI(title="Chat Index Retriever")

INDEX_PREFIX = os.getenv("INDEX_PREFIX", "chat_index")
MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# request/response models
class QueryRequest(BaseModel):
    q: str
    topk: Optional[int] = 3
    generate: Optional[bool] = False

class DocResult(BaseModel):
    score: float
    text: str
    file: Optional[str] = None

class QueryResponse(BaseModel):
    query: str
    results: List[DocResult]
    generated: Optional[str] = None

# global objects
_embedder = None
_index = None
_docs_map = None
_generator = None

def normalize_docs(raw):
    out = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, dict):
                content = v.get("content") or v.get("text") or json.dumps(v)
                file = v.get("file")
            else:
                content = str(v); file = None
            out[str(k)] = {"content": content, "file": file}
        return out
    if isinstance(raw, list):
        for i, item in enumerate(raw):
            if isinstance(item, dict):
                content = item.get("content") or item.get("text") or item.get("doc") or json.dumps(item)
                file = item.get("file") or item.get("filename") or item.get("path")
            else:
                content = str(item); file = None
            out[str(i)] = {"content": content, "file": file}
        return out
    raise ValueError("Unsupported chat_index_docs.json format")

@app.on_event("startup")
def load_models():
    global _embedder, _index, _docs_map, _generator
    prefix = INDEX_PREFIX
    idx_path = prefix + ".faiss"
    docs_path = prefix + "_docs.json"
    if not os.path.exists(idx_path) or not os.path.exists(docs_path):
        logger.error("Index/docs missing: %s or %s", idx_path, docs_path)
        raise RuntimeError("Index/docs missing; run build script first.")
    logger.info("Loading docs from %s", docs_path)
    with open(docs_path, "r", encoding="utf8") as f:
        raw = json.load(f)
    _docs_map = normalize_docs(raw)
    logger.info("Docs loaded: %d", len(_docs_map))

    logger.info("Loading embedder model: %s", MODEL_NAME)
    _embedder = SentenceTransformer(MODEL_NAME)

    logger.info("Loading FAISS index: %s", idx_path)
    _index = faiss.read_index(idx_path)
    logger.info("Loaded FAISS ntotal=%d", int(_index.ntotal))

    # optional generator (if transformers installed)
    try:
        from transformers import pipeline
        _generator = pipeline("text2text-generation", model=os.getenv("GEN_MODEL", "google/flan-t5-small"))
        logger.info("Generator loaded.")
    except Exception as e:
        _generator = None
        logger.info("Generator not available: %s", e)


def encode_query(q: str):
    emb = _embedder.encode([q], convert_to_numpy=True, normalize_embeddings=True)
    emb = np.array(emb).astype("float32")
    try: faiss.normalize_L2(emb)
    except Exception: pass
    return emb


def search_index(q: str, topk: int = 3):
    q_emb = encode_query(q)
    distances, indices = _index.search(q_emb, topk)
    results = []
    for score, idx in zip(distances[0], indices[0]):
        if int(idx) < 0:
            continue
        key = str(int(idx))
        meta = _docs_map.get(key, {})
        results.append({"score": float(score), "text": meta.get("content", "<missing>"), "file": meta.get("file")})
    # sort descending by score
    results.sort(key=lambda r: r["score"], reverse=True)
    return results

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    logger.info("Query: %s (topk=%d generate=%s)", req.q, req.topk, req.generate)
    if not req.q.strip():
        raise HTTPException(status_code=400, detail="Empty query")
    results = search_index(req.q, topk=req.topk)
    generated = None
    if req.generate:
        if _generator is None:
            raise HTTPException(status_code=503, detail="Generator not available on server.")
        # simple prompt: concatenate top results as context
        context = "\n\n".join([r["text"][:1500] for r in results])
        prompt = f"Context:\n{context}\n\nAnswer concisely: {req.q}\nAnswer:"
        try:
            gen = _generator(prompt, max_new_tokens=256, do_sample=False)[0]["generated_text"]
            generated = str(gen).strip()
        except Exception as e:
            logger.exception("Generator error: %s", e)
            raise HTTPException(status_code=500, detail="Generator failed.")
    return {"query": req.q, "results": results, "generated": generated}

@app.get("/health")
def health():
    return {"status": "ok", "docs": len(_docs_map) if _docs_map is not None else 0}
