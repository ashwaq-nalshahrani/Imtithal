import chromadb

def show_first_two_chunks(collection, file_path=None):
    """
    Display the first two chunks from a specific file or from the entire collection.
    
    Args:
        collection: ChromaDB collection object
        file_path: Optional. Full path to the file. If None, shows first two chunks overall.
    """
    if file_path:
        # Get chunks from specific file
        results = collection.get(
            where={"source": file_path},
            include=["documents", "metadatas"],
            limit=2  # Only get first 2 chunks
        )
        print(f"\n{'='*60}")
        print(f"First 2 chunks from: {file_path}")
        print(f"{'='*60}\n")
    else:
        # Get first 2 chunks from entire collection
        results = collection.get(
            include=["documents", "metadatas"],
            limit=2
        )
        print(f"\n{'='*60}")
        print(f"First 2 chunks from collection")
        print(f"{'='*60}\n")
    
    # Display the chunks
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])
    ids = results.get("ids", [])
    
    if not documents:
        print("⚠️ No chunks found!")
        return
    
    for i, (doc, meta, chunk_id) in enumerate(zip(documents, metadatas, ids), 1):
        print(f"{'─'*60}")
        print(f"CHUNK {i}")
        print(f"{'─'*60}")
        print(f"ID: {chunk_id}")
        print(f"Source: {meta.get('source', 'N/A')}")
        print(f"Category: {meta.get('category', 'N/A')}")
        print(f"Length: {len(doc)} characters")
        print(f"\nContent:")
        print(f"{doc}")
        print(f"\n{'─'*60}\n")


def list_available_files(collection):
    """List all files in the collection to help you pick one."""
    stored = collection.get(include=["metadatas"])
    
    files = set()
    for meta in stored.get("metadatas", []):
        files.add(meta.get("source"))
    
    if not files:
        print("⚠️ No files found in collection")
        return []
    
    print(f"\n📁 Available files in collection:")
    for i, f in enumerate(sorted(files), 1):
        print(f"{i}. {f}")
    
    return sorted(files)


# ===========================
# Usage Examples
# ===========================
if __name__ == "__main__":
    # Connect to your ChromaDB
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="imtithal_docs")
    
    # Option 1: List all available files first
    files = list_available_files(collection)
    
    if files:
        print("\n" + "="*60)
        
        # Option 2: Show first 2 chunks from a specific file
        # Replace with your actual file path
        specific_file = files[0]  # Pick the first file
        show_first_two_chunks(collection, file_path=specific_file)
        
        # Option 3: Show first 2 chunks from entire collection
        # show_first_two_chunks(collection)