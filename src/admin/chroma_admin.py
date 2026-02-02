import os
import chromadb

DB_PATH = "./chroma_db"
COLLECTION_NAME = "imtithal_docs"

def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(name=COLLECTION_NAME)

# ===========================
# Option 1: Summary
# ===========================
def show_summary(collection):
    data = collection.get(include=["metadatas"])
    total = len(data.get("metadatas", []))
    print(f"\n📊 Total chunks in ChromaDB: {total}")

    files = {}
    for meta in data.get("metadatas", []):
        src = os.path.basename(meta.get("source", "unknown"))
        files[src] = files.get(src, 0) + 1

    if not files:
        print("⚠️ No data found.")
    else:
        print("\n📁 Files:")
        for f, count in files.items():
            print(f" - {f}: {count} chunks")

# ===========================
# Option 2: Inspect chunks
# ===========================
def inspect_chunks(collection, limit=5):
    data = collection.get(include=["documents", "metadatas"])
    docs = data.get("documents", [])
    metas = data.get("metadatas", [])

    if not docs:
        print("\n⚠️ No chunks to display.")
        return

    print(f"\n🔍 Showing first {min(limit, len(docs))} chunks:\n")

    for i in range(min(limit, len(docs))):
        print("=" * 80)
        print(f"Chunk #{i + 1}")
        print(f"Source: {os.path.basename(metas[i].get('source', 'unknown'))}")
        print("-" * 80)
        print(docs[i][:1500])  # prevent terminal flooding
        print("\n")

# ===========================
# Option 3: Delete everything
# ===========================
def delete_all():
    confirm = input(
        "\n⚠️ This will DELETE the entire Chroma collection. "
        "Type 'DELETE' to confirm: "
    )

    if confirm != "DELETE":
        print("❌ Deletion cancelled.")
        return

    client = chromadb.PersistentClient(path=DB_PATH)

    # Delete collection entirely
    client.delete_collection(name=COLLECTION_NAME)

    # Recreate empty collection
    client.get_or_create_collection(name=COLLECTION_NAME)

    print("🗑️ Collection deleted and recreated successfully.")


# ===========================
# CLI Menu
# ===========================
def menu():
    collection = get_collection()

    while True:
        print("\n" + "=" * 50)
        print("🧠 ChromaDB Admin Tool")
        print("=" * 50)
        print("1️⃣ Show summary")
        print("2️⃣ Inspect chunks")
        print("3️⃣ Delete ALL data")
        print("4️⃣ Exit")

        choice = input("\nChoose an option: ").strip()

        if choice == "1":
            show_summary(collection)
        elif choice == "2":
            inspect_chunks(collection)
        elif choice == "3":
            delete_all()
        elif choice == "4":
            print("👋 Exiting.")
            break
        else:
            print("❌ Invalid option.")

# ===========================
# Run
# ===========================
if __name__ == "__main__":
    menu()
