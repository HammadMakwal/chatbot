# eval_recall.py
import csv, argparse, json
from sentence_transformers import SentenceTransformer
import faiss

def load_docs_map(path):
    with open(path,'r',encoding='utf8') as f:
        return json.load(f)

def normalize_relevant(s):
    return [x.strip() for x in s.split("|") if x.strip()]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt", required=True, help="ground truth CSV (query,relevant_ids)")
    p.add_argument("--index", default="chat_index.faiss")
    p.add_argument("--docs", default="chat_index_docs.json")
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    p.add_argument("--topk", type=int, default=5)
    args = p.parse_args()

    docs_map = load_docs_map(args.docs)
    # map: key -> file (filename may be None)
    key_to_file = {k: v.get("file") for k, v in docs_map.items()}

    model = SentenceTransformer(args.model)
    index = faiss.read_index(args.index)

    queries = []
    gt = []
    with open(args.gt, newline='', encoding='utf8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            q = row[0].strip()
            rels = normalize_relevant(row[1]) if len(row) > 1 else []
            queries.append(q)
            gt.append(rels)

    recall_at = {1:0, 3:0, 5:0}
    for q, rels in zip(queries, gt):
        emb = model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype('float32')
        try:
            faiss.normalize_L2(emb)
        except Exception:
            pass
        distances, indices = index.search(emb, args.topk)
        hits = []
        for idx in indices[0]:
            if int(idx) < 0: continue
            file = key_to_file.get(str(int(idx)))
            hits.append(file)
        for k in recall_at:
            topk_hits = hits[:k]
            # success if any relevant file in topk hits
            if any(r in topk_hits for r in rels):
                recall_at[k] += 1

    n = len(queries)
    for k in sorted(recall_at):
        print(f"Recall@{k}: {recall_at[k]}/{n} = {recall_at[k]/n:.3f}")

if __name__ == "__main__":
    main()
