import chromadb
import os
from collections import defaultdict

# ==========================
# Config
# ==========================
COLLECTION_NAME = "imtithal_docs"
CHROMA_PATH = "chroma_db"
OUTPUT_DIR = "chroma_readable"
OUTPUT_FORMAT = "md"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# Init Persistent Chroma
# ==========================
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name=COLLECTION_NAME)

data = collection.get()

documents = data["documents"]
ids = data["ids"]
metadatas = data["metadatas"]

print(f"📦 Loaded {len(documents)} chunks")

# ==========================
# Group chunks by source file
# ==========================
files = defaultdict(list)

for doc, chunk_id, meta in zip(documents, ids, metadatas):
    meta = meta or {}

    source = meta.get("source", "unknown_source")
    page = meta.get("page", -1)
    chunk_index = meta.get("chunk_index", 0)

    files[source].append({
        "id": chunk_id,
        "page": page,
        "chunk_index": chunk_index,
        "text": doc
    })

# ==========================
# Write ONE file per source
# ==========================
for source, chunks in files.items():
    # Sort chunks logically
    chunks.sort(key=lambda x: (x["page"], x["chunk_index"]))

    filename = os.path.basename(source)
    filename = os.path.splitext(filename)[0] + f".{OUTPUT_FORMAT}"
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# Source Document: {source}\n\n")
        f.write(f"Total chunks: {len(chunks)}\n\n")
        f.write("---\n\n")

        for i, chunk in enumerate(chunks, 1):
            f.write(
                f"## Chunk {i}\n"
                f"- **Chunk ID:** {chunk['id']}\n"
                f"- **Page:** {chunk['page']}\n\n"
                f"{chunk['text']}\n\n"
                f"---\n\n"
            )

    print(f"✅ Saved {len(chunks)} chunks → {filepath}")

print("🎯 All documents exported cleanly.")
