import os
import re
import hashlib
import chromadb
import pyarabic.araby as araby
from sentence_transformers import SentenceTransformer
from llama_parse import LlamaParse
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

# ===========================
# Environment
# ===========================
load_dotenv()

# ===========================
# Models & Splitter
# ===========================
embedding_model = SentenceTransformer("BAAI/bge-m3")
text_splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50
)

# ===========================
# ChromaDB
# ===========================
def get_chroma_collection(db_path="./chroma_db", collection_name="imtithal_docs"):
    client = chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection(name=collection_name)

# ===========================
# Utilities
# ===========================
def clean_arabic_pro(text: str) -> str:
    """Advanced Arabic text cleaning."""
    text = araby.strip_tatweel(text)
    text = araby.strip_tashkeel(text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def file_hash(path: str) -> str:
    """Generate hash to detect file changes."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# ===========================
# Inspect Chroma
# ===========================
def show_chroma_summary(collection):
    stored = collection.get(include=["documents", "metadatas"])
    total_chunks = len(stored.get("documents", []))
    print(f"\n📊 Total chunks in ChromaDB: {total_chunks}")

    # Group by file
    files_summary = {}
    for meta in stored.get("metadatas", []):
        src = meta.get("source")
        files_summary[src] = files_summary.get(src, 0) + 1

    if not files_summary:
        print("⚠️ ChromaDB is empty.")
    else:
        print("Files in ChromaDB and number of chunks each:")
        for f, count in files_summary.items():
            print(f" - {os.path.basename(f)}: {count} chunks")

# ===========================
# Indexing Pipeline
# ===========================
def index_files(folder_path: str, collection):
    parser = LlamaParse(
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
        result_type="markdown",
        language="ar",
        use_vendor_multimodal_model=True,
        vendor_multimodal_model_name="gemini-2.5-flash",
        user_prompt=(
            "This is an official SDAIA or NCA regulation. "
            "Extract all text and tables accurately."
        )
    )

    print("\n🚀 Starting ETL indexing pipeline...\n")

    for root, _, files in os.walk(folder_path):
        for filename in files:
            if not filename.lower().endswith((".pdf", ".txt")):
                continue

            file_path = os.path.join(root, filename)
            category = os.path.basename(root)
            current_hash = file_hash(file_path)

            # Check if file already indexed
            existing = collection.get(where={"source": file_path}, include=["metadatas"], limit=1)
            if existing["ids"]:
                old_hash = existing["metadatas"][0].get("hash")
                if old_hash == current_hash:
                    print(f"⏭️ Skipping (already indexed): {filename}")
                    continue
                else:
                    print(f"♻️ File changed, re-indexing: {filename}")
                    collection.delete(where={"source": file_path})

            print(f"📄 Processing file: {filename}")

            try:
                # 1. Parse
                docs = parser.load_data(file_path)
                
                # Check if parsing was successful
                if not docs:
                    print(f" ⚠️ Parsing returned no documents. Skipping.")
                    continue
                
                raw_text = "\n\n".join(doc.text for doc in docs)

                # 2. Clean
                cleaned_text = clean_arabic_pro(raw_text)
                print(f"   Extracted characters: {len(cleaned_text)}")
                
                # CRITICAL CHECK: Verify we have actual content
                if len(cleaned_text) < 10:  # Minimum threshold
                    print(f" ⚠️ Extracted text too short ({len(cleaned_text)} chars). Likely parsing failed. Skipping.")
                    continue

                # 3. Split
                chunks = text_splitter.split_text(cleaned_text)
                print(f"   Generated chunks: {len(chunks)}")
                
                if not chunks:
                    print(" ⚠️ No chunks generated, skipping.")
                    continue
                
                # Filter out empty or whitespace-only chunks
                chunks = [c.strip() for c in chunks if c.strip()]
                if not chunks:
                    print(" ⚠️ All chunks were empty after filtering. Skipping.")
                    continue
                
                print(f"   Valid chunks after filtering: {len(chunks)}")

                # 4. Embed
                embeddings = embedding_model.encode(
                    chunks, batch_size=8, show_progress_bar=False
                ).tolist()

                # 5. Store
                collection.add(
                    documents=chunks,
                    embeddings=embeddings,
                    ids=[f"{file_path}_{i}" for i in range(len(chunks))],
                    metadatas=[
                        {
                            "source": file_path,
                            "category": category,
                            "hash": current_hash
                        } for _ in chunks
                    ]
                )

                print(f" ✅ Indexed successfully with {len(chunks)} chunks")

                # 6. Show current state of Chroma
                show_chroma_summary(collection)

            except Exception as e:
                print(f" ❌ Error processing {filename}: {str(e)}")
                # Print more detailed error info for debugging
                import traceback
                print(f"    Details: {traceback.format_exc()}\n")

# ===========================
# Run Indexing
# ===========================
if __name__ == "__main__":
    collection = get_chroma_collection()
    show_chroma_summary(collection)  # See what's already in Chroma
    index_files("./data", collection)
    
    # Final summary
    print("\n" + "="*50)
    print("INDEXING COMPLETE")
    print("="*50)
    show_chroma_summary(collection)