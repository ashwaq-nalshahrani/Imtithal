import os
import chromadb
from sentence_transformers import SentenceTransformer
from llama_parse import LlamaParse
from dotenv import load_dotenv
import google.generativeai as genai
load_dotenv()

# Load BGE-M3 model (Multilingual & High Performance for Arabic)
# Note: It will download ~2.2GB on the first run.
embedding_model = SentenceTransformer('BAAI/bge-m3')

def get_chroma_collection(db_path="./chroma_db", collection_name="imtithal_docs"):
    """Initialize or load the persistent vector database."""
    chroma_client = chromadb.PersistentClient(path=db_path)
    return chroma_client.get_or_create_collection(name=collection_name)

def split_text(text, chunk_size=800, overlap=200):
    """
    Splits text into chunks with a larger overlap to ensure 
    legal/regulatory context is preserved between segments.
    """
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

def index_files_with_llama(folder_path, collection):
    """
    Recursively parses PDFs/Txt files from subfolders (SDAIA/NCA) 
    using LlamaParse and stores them in ChromaDB with metadata.
    """
    parser = LlamaParse(
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
        result_type="markdown", # Preserves table structures in Arabic documents
        language="ar"
    )
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return
# --- السطر الجديد: جلب الملفات التي تمت أرشفتها سابقاً في الجهاز ---
    existing_data = collection.get()
    indexed_files = set([meta['source'] for meta in existing_data['metadatas']]) if existing_data['metadatas'] else set()

    print("Starting Data Preparation (ETL Pipeline)...")

    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith((".pdf", ".txt")):
                # --- السطر الجديد: التخطي الذكي للملفات الموجودة مسبقاً ---
                if filename in indexed_files:
                    print(f"Skipping: {filename} (Already Indexed ✅)")
                    continue
                file_path = os.path.join(root, filename)
                folder_name = os.path.basename(root)
                
                print(f"Processing: {filename} from folder [{folder_name}]...")
                
                # Extract text using LlamaParse
                documents = parser.load_data(file_path)
                
                for doc in documents:
                    chunks = split_text(doc.text)
                    # Create vector embeddings
                    embeddings = embedding_model.encode(chunks).tolist()
                    
                    # Add to ChromaDB with Metadata for Source Citations
                    collection.add(
                        embeddings=embeddings,
                        documents=chunks,
                        metadatas=[{"source": filename, "category": folder_name} for _ in range(len(chunks))],
                        ids=[f"{filename}_{i}" for i in range(len(chunks))]
                    )
                    
    print("Indexing Complete! Your specialized knowledge base is ready.")

def calculate_compliance_score(analysis_text):
    """
    تطلب من Gemini تحويل التحليل النصي إلى درجة مئوية.
    """
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""
    بناءً على التحليل القانوني التالي للفجوات بين سياسة الشركة والأنظمة السعودية:
    ---
    {analysis_text}
    ---
    المطلب:
    قدر نسبة الامتثال الإجمالية للشركة من 100%.
    أعطني النتيجة كـ "رقم فقط" (مثلاً: 75). لا تكتب أي نص إضافي.
    """
    try:
        response = model.generate_content(prompt)
        # تنظيف النص الناتج وتحويله لرقم
        score_str = "".join(filter(str.isdigit, response.text.strip()))
        return int(score_str) if score_str else 0
    except:
        return 0