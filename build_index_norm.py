#!/usr/bin/env python3
# build_index_norm.py
import os
import json
import argparse
import shutil
import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kw): return x

def backup_if_exists(prefix):
    idx = prefix + ".faiss"
    docs = prefix + "_docs.json"
    if os.path.exists(idx) or os.path.exists(docs):
        os.makedirs("backups", exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if os.path.exists(idx):
            shutil.copy(idx, os.path.join("backups", f"{os.path.basename(prefix)}_{ts}.faiss.bak"))
        if os.path.exists(docs):
            shutil.copy(docs, os.path.join("backups", f"{os.path.basename(prefix)}_{ts}.docs.json.bak"))
        print("Backups written:\n -", os.path.join("backups", f"{os.path.basename(prefix)}_{ts}.faiss.bak"))
        print(" -", os.path.join("backups", f"{os.path.basename(prefix)}_{ts}.docs.json.bak"))

def read_txt_files(docs_dir):
    files = []
    for fn in sorted(os.listdir(docs_dir)):
        if fn.lower().endswith(".txt"):
            path = os.path.join(docs_dir, fn)
            with open(path, "r", encoding="utf8") as f:
                text = f.read().strip()
            if text:
                files.append((fn, text))
    return files

def chunk_text(text, max_chars, overlap):
    if max_chars <= 0:
        return [(0, text)]
    text = text.strip()
    if len(text) <= max_chars:
        return [(0, text)]
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = start + max_chars
        chunk = text[start:end]
        chunks.append((start, chunk))
        if end >= L:
            break
        start = max(0, end - overlap)
    return chunks

def encode_in_batches(model, texts, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i:i+batch_size]
        embs = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
        embeddings.append(embs.astype("float32"))
    if embeddings:
        return np.vstack(embeddings)
    else:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype="float32")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--docs-dir", default="docs", help="Folder with .txt files")
    p.add_argument("--model", default="all-MiniLM-L6-v2", help="sentence-transformers model")
    p.add_argument("--prefix", default="chat_index", help="index/doc prefix")
    p.add_argument("--chunk-size", type=int, default=1000, help="max chars per chunk (0=disabled)")
    p.add_argument("--overlap", type=int, default=200, help="overlap chars between chunks")
    p.add_argument("--batch-size", type=int, default=32, help="encoding batch size")
    args = p.parse_args()

    if not os.path.isdir(args.docs_dir):
        raise SystemExit(f"Docs directory not found: {args.docs_dir}")

    print("Backing up existing index/docs (if any)...")
    backup_if_exists(args.prefix)

    files = read_txt_files(args.docs_dir)
    if not files:
        raise SystemExit("No .txt files found in docs directory.")

    # Build passages
    passages = []
    mapping = {}
    idx = 0
    for fn, text in files:
        chunks = chunk_text(text, args.chunk_size, args.overlap) if args.chunk_size > 0 else [(0, text)]
        for start_pos, chunk in chunks:
            mapping[str(idx)] = {"content": chunk, "file": fn, "start": start_pos}
            passages.append(chunk)
            idx += 1

    print(f"Encoding {len(passages)} passages with model {args.model} ...")
    model = SentenceTransformer(args.model)
    embeddings = encode_in_batches(model, passages, batch_size=args.batch_size)

    if embeddings.shape[0] != len(passages):
        raise SystemExit("Embeddings count mismatch.")

    # Ensure float32 & normalize (should already be normalized by model if normalize_embeddings=True)
    embeddings = embeddings.astype("float32")
    try:
        faiss.normalize_L2(embeddings)
    except Exception:
        pass

    dim = embeddings.shape[1]
    print("Embedding dim:", dim)

    # build IndexFlatIP for normalized inner-product (cosine)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"Wrote index: {args.prefix}.faiss (ntotal={index.ntotal})")
    faiss.write_index(index, args.prefix + ".faiss")

    # save mapping
    with open(args.prefix + "_docs.json", "w", encoding="utf8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print("Wrote docs mapping:", args.prefix + "_docs.json")

if __name__ == "__main__":
    main()
