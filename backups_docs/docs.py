# run_query.py
import os
import faiss
import json
from sentence_transformers import SentenceTransformer

# 1. Folder containing your text files
DOCS_FOLDER = "docs"   # create a folder called 'docs' in the same place as this script

# 2. Collect all .txt files
documents = []
file_names = []

for file in os.listdir(DOCS_FOLDER):
    if file.endswith(".txt"):
        path = os.path.join(DOCS_FOLDER, file)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if text:  # skip empty files
                documents.append(text)
                file_names.append(file)

if not documents:
    print("⚠️ No .txt files found in 'docs' folder. Please add some text files.")
    exit()

print(f"Loaded {len(documents)} documents from {DOCS_FOLDER}")

# 3. Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Create embeddings
embeddings = model.encode(documents)

# 5. Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 6. Save FAISS index
faiss.write_index(index, "chat_index.faiss")

# 7. Save documents + file names for lookup
doc_info = [{"file": fn, "content": doc} for fn, doc in zip(file_names, documents)]
with open("chat_index_docs.json", "w", encoding="utf-8") as f:
    json.dump(doc_info, f, indent=2, ensure_ascii=False)

print("✅ Index built and saved: chat_index.faiss + chat_index_docs.json")
