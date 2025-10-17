# score_sampler.py
import json, argparse, numpy as np
from sentence_transformers import SentenceTransformer
import faiss

def load_docs(path):
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", default="chat_index.faiss")
    p.add_argument("--docs", default="chat_index_docs.json")
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--queries", default=None, help="Path to newline-separated queries (optional)")
    args = p.parse_args()

    # sample queries if none provided
    default_queries = [
        "what is python?",
        "what is deep learning?",
        "how to adopt a cat?",
        "best exercise for cardio?",
        "how to cook rice?"
    ]
    if args.queries:
        with open(args.queries, "r", encoding="utf8") as f:
            queries = [line.strip() for line in f if line.strip()]
    else:
        queries = default_queries

    model = SentenceTransformer(args.model)
    index = faiss.read_index(args.index)
    raw_docs = load_docs(args.docs)

    top_scores = []
    print("Running sampler for", len(queries), "queries")
    for q in queries:
        q_emb = model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        try:
            faiss.normalize_L2(q_emb)
        except Exception:
            pass
        distances, indices = index.search(q_emb, args.topk)
        # if L2 index, distances are L2 distances -> convert to sim = 1/(1+d)
        metric = getattr(index, "metric_type", None)
        if metric == faiss.METRIC_L2:
            sims = 1.0 / (1.0 + distances[0])
        else:
            sims = distances[0]  # inner product or other
        top_scores.append(float(sims[0]))
        print(f"Q: {q}\n top-{args.topk} scores: {sims.tolist()}\n")

    arr = np.array(top_scores)
    print("Top score stats (mean,median,min,max):", float(arr.mean()), float(np.median(arr)), float(arr.min()), float(arr.max()))
    print("Suggested conf-thresh range:", round(float(arr.min()),3), "to", round(float(arr.max()),3))

if __name__ == "__main__":
    main()
