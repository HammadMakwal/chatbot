#!/usr/bin/env python3
"""
run_query_reload.py

Interactive retriever (+ optional generator) with:
 - confidence gating (--conf-thresh)
 - optional cross-encoder reranking (--rerank)
 - robust logging (text or JSON) with per-session log file + rotation

Usage examples:
  # retrieval only, text logs (daily rotation)
  python run_query_reload.py --no-generator --topk 4

  # retrieval only, JSON logs, size-based rotation
  python run_query_reload.py --no-generator --json --logs-dir logs_json --rotation size --topk 4

  # retrieval + rerank (slower, higher precision)
  python run_query_reload.py --no-generator --rerank --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2

  # if index missing, run builder automatically
  python run_query_reload.py --auto-build
"""
import os
import sys
import json
import argparse
import subprocess
import datetime
import inspect
import logging
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler
import numpy as np

from sentence_transformers import SentenceTransformer
import faiss

# Optional: transformers generator
try:
    from transformers import pipeline
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# Optional: user-supplied logger factory (logger_setup.get_logger)
try:
    from logger_setup import get_logger as user_get_logger
except Exception:
    user_get_logger = None

DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_GENERATOR = "google/flan-t5-small"
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# --------------------------
# Utilities
# --------------------------
def normalize_docs(raw):
    out = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, dict):
                content = v.get("content") or v.get("text") or json.dumps(v)
                file = v.get("file")
            else:
                content = str(v)
                file = None
            out[str(k)] = {"content": content, "file": file}
        return out
    if isinstance(raw, list):
        for i, item in enumerate(raw):
            if isinstance(item, dict):
                content = item.get("content") or item.get("text") or item.get("doc") or json.dumps(item)
                file = item.get("file") or item.get("filename") or item.get("path")
            else:
                content = str(item)
                file = None
            out[str(i)] = {"content": content, "file": file}
        return out
    raise ValueError("Unsupported chat_index_docs.json format")


def ensure_index_exists(prefix, auto_build, logger):
    idx_path = prefix + ".faiss"
    docs_path = prefix + "_docs.json"
    if os.path.exists(idx_path) and os.path.exists(docs_path):
        return idx_path, docs_path
    logger.warning("Index or docs missing: %s or %s", idx_path, docs_path)
    if auto_build:
        logger.info("Auto-build requested: running run_query.py --rebuild")
        try:
            subprocess.run([sys.executable, "run_query.py", "--rebuild"], check=True)
            logger.info("Auto-build completed.")
        except subprocess.CalledProcessError as e:
            logger.exception("Auto-build failed: %s", e)
            raise
        if os.path.exists(idx_path) and os.path.exists(docs_path):
            return idx_path, docs_path
        else:
            raise FileNotFoundError("Auto-build finished but index/docs still missing.")
    else:
        raise FileNotFoundError(f"Missing index/docs: {idx_path}, {docs_path}. Run run_query.py --rebuild or pass --auto-build.")


# --------------------------
# Embedding / Search helpers
# --------------------------
def encode_query(embedder, q):
    emb = embedder.encode([q], convert_to_numpy=True, normalize_embeddings=True)
    emb = np.array(emb).astype("float32")
    try:
        faiss.normalize_L2(emb)
    except Exception:
        pass
    return emb


def convert_distances_to_similarity(distances, index):
    """
    Convert FAISS 'distances' to a similarity-like score where higher == better.
    For METRIC_L2: use sim = 1 / (1 + distance)  (range (0,1], higher when closer)
    For METRIC_INNER_PRODUCT: the returned 'distances' are actually scores (higher better).
    For other metrics: return as-is.
    """
    try:
        metric = getattr(index, "metric_type", None)
    except Exception:
        metric = None

    # faiss constant names exist, but compare by value to be safe
    try:
        if metric == faiss.METRIC_L2:
            # distances array (nq x k)
            return 1.0 / (1.0 + distances)
        else:
            # assume higher is better (inner product)
            return distances
    except Exception:
        # fallback: return distances (user can tune conf threshold accordingly)
        return distances


def search_index(query, embedder, index, docs_map, top_k=3):
    q_emb = encode_query(embedder, query)
    distances, indices = index.search(q_emb, top_k)
    sims = convert_distances_to_similarity(distances, index)
    results = []
    for sim_row, idx_row in zip(sims, indices):
        for s, idx in zip(sim_row, idx_row):
            if int(idx) < 0:
                continue
            key = str(int(idx))
            doc_meta = docs_map.get(key, {})
            results.append({
                "score": float(s),
                "text": doc_meta.get("content", "<missing doc content>"),
                "file": doc_meta.get("file")
            })
    # results currently contains up to top_k items for the single query
    # ensure sorted descending by score
    results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
    return results


# --------------------------
# Logger initializer (robust)
# --------------------------
def init_logger(name: str, logs_dir: str, json_mode: bool, rotation: str):
    """
    Try to call user_get_logger with only the parameters it accepts.
    If user_get_logger is not available or fails, fall back to stdlib logging.
    Attach a 'logfile' attribute.
    """
    if user_get_logger is not None:
        # try to call user_get_logger with names it accepts
        sig = inspect.signature(user_get_logger)
        kwargs = {}
        for p in sig.parameters:
            if p in ("name", "log_name", "logger_name", "app_name"):
                kwargs[p] = name
            elif p in ("logs_dir", "log_dir", "logsdir", "logdir"):
                kwargs[p] = logs_dir
            elif p in ("json_mode", "json", "json_enabled", "json_logs"):
                kwargs[p] = json_mode
            elif p in ("rotation", "rot", "rotation_policy"):
                kwargs[p] = rotation
        try:
            logger = user_get_logger(**kwargs)
        except Exception:
            try:
                logger = user_get_logger(name, logs_dir)
            except Exception:
                logger = None
        if logger is not None:
            if not hasattr(logger, "logfile"):
                try:
                    os.makedirs(logs_dir, exist_ok=True)
                    logfile = os.path.join(logs_dir, f"{name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
                    logger.logfile = logfile
                except Exception:
                    logger.logfile = None
            return logger

    # fallback to stdlib logging
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logfile = os.path.join(logs_dir, f"session_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # avoid adding duplicate handlers if reinitializing in same process
    if not logger.handlers:
        if rotation == "daily":
            fh = TimedRotatingFileHandler(logfile, when="midnight", backupCount=7, encoding="utf8")
        elif rotation == "hourly":
            fh = TimedRotatingFileHandler(logfile, when="H", interval=1, backupCount=48, encoding="utf8")
        else:  # size
            fh = RotatingFileHandler(logfile, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf8")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    logger.logfile = logfile
    return logger


# --------------------------
# Main interactive loop
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="Interactive retriever (+ optional generator).")
    parser.add_argument("--prefix", default="chat_index", help="Index/doc prefix (chat_index -> chat_index.faiss + chat_index_docs.json)")
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL, help="SentenceTransformers embedding model")
    parser.add_argument("--generator-model", default=DEFAULT_GENERATOR, help="Generator model (text2text pipeline)")
    parser.add_argument("--topk", type=int, default=3, help="Number of docs to retrieve")
    parser.add_argument("--no-generator", action="store_true", help="Disable generator (just show retrieved docs)")
    parser.add_argument("--auto-build", action="store_true", help="If index missing, run run_query.py --rebuild automatically")
    parser.add_argument("--json", action="store_true", help="Enable JSON-structured logging (default: text logs)")
    parser.add_argument("--logs-dir", default="logs", help="Directory to store logs")
    parser.add_argument("--rotation", choices=["daily", "hourly", "size"], default="daily", help="Log rotation policy")
    parser.add_argument("--conf-thresh", type=float, default=0.35,
                        help="Confidence threshold (higher -> stricter). See docs for tuning.")
    parser.add_argument("--rerank", action="store_true", help="Enable cross-encoder re-ranking for top candidates (slower)")
    parser.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL, help="Cross-encoder rerank model")
    parser.add_argument("--rerank-topk", type=int, default=10, help="Number of top candidates to rerank (before returning topk)")
    args = parser.parse_args()

    logger = init_logger("chatbot", logs_dir=args.logs_dir, json_mode=args.json, rotation=args.rotation)
    logger.info("Starting run_query_reload.py (prefix=%s, topk=%d, generator=%s, rerank=%s)",
                args.prefix, args.topk, ("enabled" if not args.no_generator else "disabled"), args.rerank)

    print(f"\n>>> Logging: {'JSON' if args.json else 'text'} | Generator: {'disabled' if args.no_generator else 'enabled'} | Rerank: {args.rerank}")
    if getattr(logger, "logfile", None):
        print("Logfile:", logger.logfile)

    # ensure index and docs exist
    try:
        idx_path, docs_path = ensure_index_exists(args.prefix, args.auto_build, logger)
    except FileNotFoundError as e:
        logger.error(str(e))
        print("ERROR:", e)
        sys.exit(1)

    # load docs mapping
    with open(docs_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    docs_map = normalize_docs(raw)
    logger.info("Loaded %d docs from %s", len(docs_map), docs_path)

    # load embedder model
    logger.info("Loading embedder model: %s", args.model)
    try:
        embedder = SentenceTransformer(args.model)
    except Exception as e:
        logger.exception("Failed to load embedder: %s", e)
        print("Failed to load embedding model:", e)
        sys.exit(1)

    # load FAISS index
    try:
        index = faiss.read_index(idx_path)
        logger.info("Loaded FAISS index: %s (ntotal=%d, metric=%s)", idx_path, int(index.ntotal),
                    getattr(index, "metric_type", "unknown"))
    except Exception as e:
        logger.exception("Failed to load FAISS index: %s", e)
        print("Failed to load FAISS index:", e)
        sys.exit(1)

    # optional reranker (loaded on demand)
    reranker = None
    if args.rerank:
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading reranker model: %s", args.rerank_model)
            reranker = CrossEncoder(args.rerank_model)
            logger.info("Reranker loaded.")
        except Exception as e:
            logger.exception("Failed to load reranker: %s", e)
            logger.warning("Reranker disabled due to error.")
            reranker = None

    # optional generator (transformers pipeline)
    generator = None
    if not args.no_generator:
        if not HF_AVAILABLE:
            logger.warning("transformers not available; generator disabled.")
        else:
            try:
                logger.info("Loading generator pipeline: %s", args.generator_model)
                generator = pipeline("text2text-generation", model=args.generator_model)
                logger.info("Generator ready.")
            except Exception as e:
                logger.exception("Failed to load generator model: %s", e)
                generator = None
                logger.warning("Generator disabled due to error; will show raw docs instead.")

    # main interactive loop
    print(f"Loaded index ({len(docs_map)} docs). Ready to query.\n")
    logger.info("Ready to query.")
    conversation = []

    try:
        while True:
            q = input("Query> ").strip()
            if q == "":
                break
            logger.info("User: %s", q)
            conversation.append(f"User: {q}")

            # retrieve
            results = search_index(q, embedder, index, docs_map, top_k=max(args.topk, args.rerank_topk))

            # optionally rerank top candidates
            if reranker and results:
                top_candidates = results[:args.rerank_topk]
                pairs = [[q, r["text"][:512]] for r in top_candidates]
                try:
                    rerank_scores = reranker.predict(pairs)
                    for r, s in zip(top_candidates, rerank_scores):
                        r["score"] = float(s)
                    top_candidates.sort(key=lambda r: r["score"], reverse=True)
                    # replace initial results with reranked top_candidates + remaining (if any)
                    results = top_candidates + results[args.rerank_topk:]
                except Exception as e:
                    logger.exception("Reranker predict failed: %s", e)

            # defensive: sort descending
            results.sort(key=lambda r: r.get("score", 0.0), reverse=True)

            # no matches
            if not results:
                out = "No results found."
                print("Bot>", out)
                logger.info("Bot: %s", out)
                conversation.append(f"Bot: {out}")
                continue

            # confidence gating
            top_score = results[0]["score"]
            logger.debug("Top score (post-conversion/rerank): %s", top_score)
            if top_score < args.conf_thresh:
                out = "I'm not confident I found a good answer — could you rephrase or give more detail?"
                print("Bot>", out)
                logger.info("Bot: low_confidence (score=%.4f). Asked clarification.", top_score)
                conversation.append(f"Bot: {out}")
                continue

            # show retrieved docs (limit to args.topk)
            display_results = results[:args.topk]
            print("\n--- Retrieved ---")
            for r in display_results:
                filepart = f" ({r['file']})" if r.get("file") else ""
                preview = r["text"][:500].replace("\n", " ")
                print(f"[score {r['score']:.4f}]{filepart}: {preview}\n")
            print("-----------------\n")

            # generate final answer (optional)
            if generator:
                context = "\n\n".join([r["text"][:1500] for r in display_results])
                history = "\n".join(conversation[-6:])
                prompt = (f"Context:\n{context}\n\nConversation:\n{history}\n\n"
                          f"Answer the user's last question concisely and cite sources if possible.\nQuestion: {q}\nAnswer:")
                logger.debug("Generator prompt length: %d", len(prompt))
                try:
                    resp = generator(prompt, max_length=300, do_sample=False)[0]["generated_text"].strip()
                    print("Bot>", resp, "\n")
                    logger.info("Bot: %s", resp)
                    conversation.append(f"Bot: {resp}")
                except Exception as e:
                    logger.exception("Generator error: %s", e)
                    fallback = display_results[0]["text"][:1000]
                    print("Bot> (generator failed) Top doc snippet:\n", fallback)
                    logger.info("Bot (fallback): %s", fallback)
                    conversation.append(f"Bot (fallback): {fallback}")
            else:
                combined = "\n\n".join([r["text"] for r in display_results])
                print("Bot> (retrieved docs)\n", combined[:2000])
                logger.info("Bot (retrieved): %s", combined[:1000])
                conversation.append(f"Bot: {combined[:2000]}")

    except KeyboardInterrupt:
        print("\nExiting.")
        logger.info("Session interrupted by KeyboardInterrupt.")
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        print("Unexpected error:", e)
    finally:
        logger.info("Session ended.")


if __name__ == "__main__":
    main()
