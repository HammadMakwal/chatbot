# check_index.py
import json, faiss
idx = faiss.read_index("chat_index.faiss")
with open("chat_index_docs.json", "r", encoding="utf-8") as f:
    docs = json.load(f)
print("FAISS ntotal:", int(idx.ntotal))
print("Type of docs JSON:", type(docs).__name__)
try:
    print("Docs count (len):", len(docs))
except Exception as e:
    print("Could not determine length of docs json:", e)
# show small sample of the docs JSON (first item keys/preview)
if isinstance(docs, dict):
    first_key = next(iter(docs))
    print("First key in docs JSON:", first_key)
    print("First item preview:", str(docs[first_key])[:300])
elif isinstance(docs, list) and docs:
    print("First item preview:", str(docs[0])[:300])
