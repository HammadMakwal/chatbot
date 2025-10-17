import os
import faiss
import json
from sentence_transformers import SentenceTransformer

# Paths
DOCS_DIR = "docs"
INDEX_FILE = "chat_index.faiss"
DOCS_FILE = "chat_index_docs.json"

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

documents = []
file_names = []

# Read all .txt files in docs/
for fname in os.listdir(DOCS_DIR):
    if fname.endswith(".txt"):
        with open(os.path.join(DOCS_DIR, fname), "r", encoding="utf-8") as f:
            text = f.read().strip()
            documents.append(text)
            file_names.append(fname)

# Convert docs into embeddings
embeddings = model.encode(documents)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, INDEX_FILE)

# Save mapping of docs
with open(DOCS_FILE, "w", encoding="utf-8") as f:
    json.dump({"files": file_names, "docs": documents}, f, indent=2)

print(f"Index built successfully! {len(documents)} documents added.")
