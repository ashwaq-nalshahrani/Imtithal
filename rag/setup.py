import os
import requests
import zipfile
from pathlib import Path

# ===========================
# Configuration
# ===========================
# You'll update this URL after uploading to Google Drive/Dropbox
CHROMADB_DOWNLOAD_URL = "https://your-cloud-storage-link/chroma_db_backup.zip"

# Or use Google Drive direct download
# CHROMADB_DOWNLOAD_URL = "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID"

def download_chromadb(url, output_path="chroma_db_backup.zip"):
    """
    Download ChromaDB from cloud storage.
    """
    print(f"📥 Downloading ChromaDB from cloud storage...")
    print(f"   URL: {url}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        progress = (downloaded / total_size) * 100
                        print(f"\r  Progress: {progress:.1f}%", end='')
        
        print(f"\n✅ Download complete: {output_path}")
        return True
        
    except Exception as e:
        print(f"\n❌ Download failed: {str(e)}")
        return False


def extract_chromadb(zip_path="chroma_db_backup.zip"):
    """
    Extract the downloaded ChromaDB.
    """
    print(f"\n📦 Extracting ChromaDB...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zipf.extractall("./")
        
        print(f"✅ Extraction complete!")
        
        # Clean up zip file
        os.remove(zip_path)
        print(f"🗑️  Removed temporary zip file")
        
        return True
        
    except Exception as e:
        print(f"❌ Extraction failed: {str(e)}")
        return False


def setup():
    """
    Main setup function.
    """
    print("="*60)
    print("🚀 Project Setup")
    print("="*60)
    
    # Check if ChromaDB already exists
    if os.path.exists("./chroma_db"):
        print("\n✅ ChromaDB already exists!")
        response = input("Re-download and overwrite? (yes/no): ")
        if response.lower() != 'yes':
            print("Setup cancelled. Using existing ChromaDB.")
            return
        
        # Backup existing
        import shutil
        if os.path.exists("./chroma_db_backup_old"):
            shutil.rmtree("./chroma_db_backup_old")
        shutil.move("./chroma_db", "./chroma_db_backup_old")
        print("📁 Backed up existing ChromaDB to chroma_db_backup_old")
    
    # Download
    if CHROMADB_DOWNLOAD_URL == "https://your-cloud-storage-link/chroma_db_backup.zip":
        print("\n⚠️  CHROMADB_DOWNLOAD_URL not configured!")
        print("Please update the URL in setup.py with your actual cloud storage link.")
        print("\nAlternatively, you can:")
        print("1. Manually download chroma_db_backup.zip")
        print("2. Run: python share_chromadb.py and choose option 2")
        return
    
    if not download_chromadb(CHROMADB_DOWNLOAD_URL):
        return
    
    # Extract
    if not extract_chromadb():
        return
    
    print("\n" + "="*60)
    print("✅ Setup Complete!")
    print("="*60)
    print("\nYou can now run:")
    print("  python inspect_chunks.py    # To view chunks")
    print("  python etl_pipeline.py      # To add more documents")


if __name__ == "__main__":
    setup()