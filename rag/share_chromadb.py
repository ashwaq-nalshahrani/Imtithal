import os
import zipfile
from pathlib import Path

def compress_chromadb(db_path="./chroma_db", output_zip="chroma_db_backup.zip"):
    """
    Compress ChromaDB folder for sharing.
    """
    print(f"📦 Compressing ChromaDB from {db_path}...")
    
    if not os.path.exists(db_path):
        print(f"❌ Error: {db_path} not found!")
        return
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(db_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(db_path))
                zipf.write(file_path, arcname)
                print(f"  Added: {arcname}")
    
    size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print(f"\n✅ Compressed successfully!")
    print(f"📁 Output: {output_zip}")
    print(f"💾 Size: {size_mb:.2f} MB")
    print(f"\n📤 Share this file with your friends!")


def extract_chromadb(zip_path="chroma_db_backup_new.zip", extract_to="./"):
    """
    Extract ChromaDB from shared zip file.
    """
    print(f"📦 Extracting ChromaDB from {zip_path}...")
    
    if not os.path.exists(zip_path):
        print(f"❌ Error: {zip_path} not found!")
        return
    
    # Check if chroma_db already exists
    db_path = os.path.join(extract_to, "chroma_db")
    if os.path.exists(db_path):
        response = input(f"⚠️  {db_path} already exists. Overwrite? (yes/no): ")
        if response.lower() != 'yes':
            print("Extraction cancelled.")
            return
    
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(extract_to)
        print(f"  Extracted all files to {extract_to}")
    
    print(f"\n✅ Extraction complete!")
    print(f"📁 ChromaDB ready at: {db_path}")
    print(f"\n🚀 You can now run your scripts without re-indexing!")


def check_chromadb_size(db_path="./chroma_db"):
    """
    Check the size of ChromaDB folder.
    """
    if not os.path.exists(db_path):
        print(f"❌ {db_path} not found!")
        return
    
    total_size = 0
    file_count = 0
    
    for root, dirs, files in os.walk(db_path):
        for file in files:
            file_path = os.path.join(root, file)
            total_size += os.path.getsize(file_path)
            file_count += 1
    
    size_mb = total_size / (1024 * 1024)
    size_gb = total_size / (1024 * 1024 * 1024)
    
    print(f"\n📊 ChromaDB Statistics:")
    print(f"  Files: {file_count}")
    print(f"  Total Size: {size_mb:.2f} MB ({size_gb:.3f} GB)")
    
    if size_mb < 100:
        print(f"  💡 Size is reasonable for sharing via cloud storage")
    elif size_mb < 500:
        print(f"  ⚠️  Moderately large - consider cloud storage")
    else:
        print(f"  🚨 Large database - may need specialized sharing method")


# ===========================
# Usage
# ===========================
if __name__ == "__main__":
    print("="*60)
    print("ChromaDB Sharing Tool")
    print("="*60)
    
    print("\nWhat would you like to do?")
    print("1. Compress ChromaDB for sharing (Creator)")
    print("2. Extract received ChromaDB (Friend)")
    print("3. Check ChromaDB size")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        check_chromadb_size()
        print("\n" + "-"*60)
        compress_chromadb()
        print("\n" + "-"*60)
        print("\n📋 Next steps:")
        print("  1. Upload 'chroma_db_backup.zip' to Google Drive/Dropbox/etc")
        print("  2. Share the link with your friends")
        print("  3. They run this script with option 2 to extract")
        
    elif choice == "2":
        zip_file = input("Enter zip file name (default: chroma_db_backup.zip): ").strip()
        if not zip_file:
            zip_file = "chroma_db_backup.zip"
        extract_chromadb(zip_file)
        
    elif choice == "3":
        check_chromadb_size()
        
    else:
        print("Invalid choice!")