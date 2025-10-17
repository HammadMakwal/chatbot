#!/usr/bin/env python3
# threshold_sweep.py
import argparse, json
import faiss, numpy as np
from sentence_transformers import SentenceTransformer

def load_docs(path):
    with open(path, 'r', encoding='utf8') as f:
        return json.load(f)

def normalize(emb):
    arr = emb.astype('float32')
    try:
        faiss.normalize_L2(arr)
    except Exception:
        pass
    return arr

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', default='chat_index.faiss')
    p.add_argument('--docs', default='chat_index_docs.json')
    p.add_argument('--model', default='all-MiniLM-L6-v2')
    p.add_argument('--gt', default='eval_ground_truth.csv')
    p.add_argument('--topk', type=int, default=5)
    p.add_argument('--thresholds', default='0.0,0.1,0.25,0.35,0.5')
    args = p.parse_args()

    docs = load_docs(args.docs)
    index = faiss.read_index(args.index)
    model = SentenceTransformer(args.model)

    queries = []
    with open(args.gt, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(',', 1)
            q = parts[0].strip()
            gt = parts[1].strip() if len(parts) > 1 else ""
            queries.append((q, gt))

    thresholds = [float(x) for x in args.thresholds.split(',')]
    out = {}
    for t in thresholds:
        answered = 0
        correct = 0
        for q, gt in queries:
            emb = normalize(model.encode([q], convert_to_numpy=True))
            scores, idxs = index.search(emb, args.topk)
            scores = scores[0].tolist()
            idxs = idxs[0].tolist()
            top_files = [docs.get(str(int(i)), {}).get('file') for i in idxs if int(i) >= 0]
            top_score = scores[0] if scores else -999.0
            if top_score >= t:
                answered += 1
                if gt and gt in top_files:
                    correct += 1
        out[t] = {
            'queries': len(queries),
            'answered': answered,
            'correct': correct,
            'precision_if_answered': (correct/answered if answered else 0.0)
        }
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
