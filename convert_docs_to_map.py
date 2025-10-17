# convert_docs_to_map.py
import json, os, sys, shutil

SRC = "chat_index_docs.json"
BACK = SRC + ".bak"
DOCS_DIR = "docs"

shutil.copyfile(SRC, BACK)
print("Backup saved to:", BACK)

with open(SRC, "r", encoding="utf8") as f:
    raw = json.load(f)

# detect common shapes
if isinstance(raw, dict) and "files" in raw and isinstance(raw["files"], list):
    files = raw["files"]
elif isinstance(raw, list):
    # list of filenames or list of dicts
    # try to detect if list contains filenames (strings)
    if all(isinstance(x, str) for x in raw):
        files = raw
    else:
        # if it's already a list of doc dicts, convert them to mapping
        out = {}
        for i, item in enumerate(raw):
            if isinstance(item, dict):
                content = item.get("content") or item.get("text") or json.dumps(item)
                file = item.get("file") or item.get("filename") or f"item_{i}"
            else:
                content = str(item)
                file = f"item_{i}"
            out[str(i)] = {"content": content, "file": file}
        with open(SRC, "w", encoding="utf8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("Converted list-of-dicts to mapping. Wrote", SRC)
        sys.exit(0)
else:
    print("Unsupported chat_index_docs.json format. See backup:", BACK)
    sys.exit(1)

# Build normalized mapping by reading files from docs/
out = {}
for i, fname in enumerate(files):
    fpath = os.path.join(DOCS_DIR, fname)
    if not os.path.exists(fpath):
        print("Warning: file not found:", fpath, "-> storing empty content")
        content = ""
    else:
        with open(fpath, "r", encoding="utf8", errors="ignore") as ff:
            content = ff.read()
    out[str(i)] = {"content": content, "file": fname}

with open(SRC, "w", encoding="utf8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print("Wrote normalized mapping to", SRC, "with", len(out), "entries.")
