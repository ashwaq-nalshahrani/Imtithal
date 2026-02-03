import os
import re
import hashlib
import chromadb
import pyarabic.araby as araby
from sentence_transformers import SentenceTransformer
from llama_parse import LlamaParse
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

# ===========================
# Environment
# ===========================
load_dotenv()

# ===========================
# Models
# ===========================
embedding_model = SentenceTransformer("BAAI/bge-m3")

# ===========================
# ChromaDB
# ===========================
class ImtithalRAG:
    def __init__(self, db_path="./chroma_db", collection_name="imtithal_docs"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

        # Check if the collection is empty
        stored = self.collection.get()
        if not stored.get("ids"):
            print("⚠️  Collection is empty. BM25 initialization skipped.")
            self.bm25_initialized = False
        else:
            self.bm25_initialized = True

    def get_collection(self):
        return self.collection

# Initialize the RAG instance
rag_instance = ImtithalRAG()

# ===========================
# Show ChromaDB Summary
# ===========================
def show_chroma_summary(collection, title="CHROMADB STATUS"):
    """Display detailed ChromaDB summary with real-time updates."""

    print(f"\n{'='*70}")
    print(f"📊 {title}")
    print(f"{'='*70}")

    try:
        stored = collection.get(include=["metadatas"])
        total = len(stored.get("ids", []))

        if total == 0:
            print("⚠️  Database is EMPTY - No chunks stored yet")
            print(f"{'='*70}\n")
            return

        print(f"✅ Total chunks in database: {total}")

        # Group by file
        files_summary = {}
        for meta in stored.get("metadatas", []):
            fname = meta.get("file_name", "unknown")
            files_summary[fname] = files_summary.get(fname, 0) + 1

        print(f"\n📁 Files indexed: {len(files_summary)}")
        print(f"{'─'*70}")

        for fname, count in sorted(files_summary.items()):
            print(f"   • {fname:<50} → {count:>3} chunks")

        print(f"{'='*70}\n")

    except Exception as e:
        print(f"❌ Error reading ChromaDB: {e}")
        print(f"{'='*70}\n")

# ===========================
# Utilities
# ===========================
def clean_arabic_pro(text: str) -> str:
    if not text:
        return ""
    text = araby.strip_tatweel(text)
    text = araby.strip_tashkeel(text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    return text.strip()

def file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

# ===========================
# Paragraph Chunking with Percentile
# ===========================
def percentile_chunking(sections, percentile=90):
    """Split sections based on percentile of length"""
    if not sections:
        return []

    lengths = [len(s["text"]) for s in sections]
    threshold = max(500, int(np.percentile(lengths, percentile)))

    final_chunks = []
    for s in sections:
        text = s["text"]
        start = 0
        chunk_index = 0
        while start < len(text):
            end = min(start + threshold, len(text))
            final_chunks.append({
                "subject": s["subject"],
                "text": text[start:end],
                "chunk_index": chunk_index
            })
            start += threshold
            chunk_index += 1

    return final_chunks

# ===========================
# Paragraph Chunking
# ===========================
def extract_paragraph_chunks(text: str, filename: str, target_size=1500):
    raw_paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    if len(raw_paragraphs) < 3:
        raw_paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 50]

    if len(raw_paragraphs) < 3:
        sentences = re.split(r'[.!?؟。]+\s+', text)
        raw_paragraphs = [s.strip() for s in sentences if len(s.strip()) > 50]

    print(f"   📝 Found {len(raw_paragraphs)} text segments")

    sections = []
    current_chunk = ""
    chunk_num = 1

    for para in raw_paragraphs:
        if current_chunk and len(current_chunk) + len(para) > target_size:
            sections.append({
                "subject": f"{filename} - القسم {chunk_num}",
                "text": current_chunk.strip(),
                "level": 1
            })
            current_chunk = para
            chunk_num += 1
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para

    if current_chunk:
        sections.append({
            "subject": f"{filename} - القسم {chunk_num}",
            "text": current_chunk.strip(),
            "level": 1
        })

    print(f"   📦 Grouped into {len(sections)} paragraph-based sections")
    return sections

# ===========================
# Section Extraction
# ===========================
def extract_markdown_headers(text: str):
    sections = []
    lines = text.split('\n')

    current = {"subject": "مقدمة", "text": [], "level": 0, "stack": []}

    for line in lines:
        m = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
        if m:
            if current["text"]:
                text_content = "\n".join(current["text"]).strip()
                if len(text_content) >= 30:
                    sections.append({
                        "subject": current["subject"],
                        "text": text_content,
                        "level": current["level"]
                    })

            level = len(m.group(1))
            header = m.group(2).strip()

            if not current["stack"]:
                stack = [header]
            elif level <= current["level"]:
                stack = current["stack"][:level-1] + [header]
            else:
                stack = current["stack"] + [header]

            current = {"subject": " > ".join(stack), "text": [], "level": level, "stack": stack}
        else:
            if line.strip():
                current["text"].append(line)

    if current["text"]:
        text_content = "\n".join(current["text"]).strip()
        if len(text_content) >= 30:
            sections.append({"subject": current["subject"], "text": text_content, "level": current["level"]})

    return sections

def extract_numbered_sections(text: str):
    sections = []
    patterns = [
        r'^([0-9]+(?:\.[0-9]+)*)\s+(.+)$',
        r'^([٠-٩]+(?:\.[٠-٩]+)*)\s+(.+)$',
        r'^(المادة|القسم|الفصل|الباب)\s+([0-9٠-٩]+)\s*[:\-]?\s*(.*)$',
        r'^([ivxIVX]+)\.\s+(.+)$',
    ]

    lines = text.split('\n')
    current_section = None
    current_text = []

    for line in lines:
        matched = False
        for pattern in patterns:
            m = re.match(pattern, line.strip())
            if m:
                if current_section and current_text:
                    text_content = "\n".join(current_text).strip()
                    if len(text_content) >= 30:
                        sections.append({"subject": current_section, "text": text_content, "level": 1})

                current_section = " ".join([g for g in m.groups() if g]).strip()
                current_text = []
                matched = True
                break

        if not matched and line.strip():
            current_text.append(line)

    if current_section and current_text:
        text_content = "\n".join(current_text).strip()
        if len(text_content) >= 30:
            sections.append({"subject": current_section, "text": text_content, "level": 1})

    return sections

def extract_pattern_sections(text: str):
    sections = []
    patterns = [
        r'^(المادة\s+(?:الأولى|الثانية|الثالثة|الرابعة|الخامسة|[٠-٩0-9]+))',
        r'^(الفصل\s+(?:الأول|الثاني|الثالث|الرابع|الخامس|[٠-٩0-9]+))',
        r'^(الباب\s+(?:الأول|الثاني|الثالث|الرابع|الخامس|[٠-٩0-9]+))',
        r'^(القسم\s+(?:الأول|الثاني|الثالث|الرابع|الخامس|[٠-٩0-9]+))',
        r'^(أولاً|ثانياً|ثالثاً|رابعاً|خامساً|سادساً|سابعاً|ثامناً|تاسعاً|عاشراً)',
        r'^(البند\s+[٠-٩0-9]+)',
    ]

    combined_pattern = '|'.join(f'({p})' for p in patterns)
    lines = text.split('\n')

    current_section = None
    current_text = []

    for line in lines:
        match = re.match(combined_pattern, line.strip())
        if match:
            if current_section and current_text:
                text_content = "\n".join(current_text).strip()
                if len(text_content) >= 30:
                    sections.append({"subject": current_section, "text": text_content, "level": 1})

            current_section = line.strip()
            current_text = []
        else:
            if line.strip():
                current_text.append(line)

    if current_section and current_text:
        text_content = "\n".join(current_text).strip()
        if len(text_content) >= 30:
            sections.append({"subject": current_section, "text": text_content, "level": 1})

    return sections

def extract_sections_from_text(text: str, filename: str):
    print(f"   🔍 Analyzing structure...")

    sections = extract_markdown_headers(text)
    if sections and len(sections) >= 3:
        print(f"   ✓ Found {len(sections)} markdown sections")
        return sections

    sections = extract_numbered_sections(text)
    if sections and len(sections) >= 3:
        print(f"   ✓ Found {len(sections)} numbered sections")
        return sections

    sections = extract_pattern_sections(text)
    if sections and len(sections) >= 3:
        print(f"   ✓ Found {len(sections)} pattern-based sections")
        return sections

    print(f"   ⚠️  No clear structure found, using smart paragraph chunking")
    sections = extract_paragraph_chunks(text, filename, target_size=1500)
    return sections

# ===========================
# Semantic Chunking
# ===========================
def semantic_chunking(sections, model, max_chunk_size=2200, min_chunk_size=700, similarity_threshold=0.68):
    if not sections:
        return []

    if len(sections) <= 2:
        return [{"subject": s["subject"], "text": s["text"]} for s in sections]

    print(f"   🧠 Computing semantic similarities...")

    texts = [s["text"] for s in sections]
    embeddings = model.encode(texts, batch_size=8, show_progress_bar=False)

    chunks = []
    i = 0

    while i < len(sections):
        current = {
            "subject": sections[i]["subject"],
            "text": sections[i]["text"],
            "subjects": [sections[i]["subject"]]
        }
        emb = embeddings[i]
        j = i + 1

        while j < len(sections):
            sim = cosine_similarity([emb], [embeddings[j]])[0][0]
            merged_text = current["text"] + "\n\n" + sections[j]["text"]

            should_merge = (sim >= similarity_threshold and len(merged_text) <= max_chunk_size)

            if should_merge:
                current["text"] = merged_text
                current["subjects"].append(sections[j]["subject"])
                emb = (emb + embeddings[j]) / 2
                j += 1
            else:
                break

        primary_subject = current["subjects"][-1] if current["subjects"] else "غير محدد"
        chunks.append({"subject": primary_subject, "text": current["text"]})
        i = j

    final = []
    for c in chunks:
        if len(c["text"]) < min_chunk_size and final:
            final[-1]["text"] += "\n\n" + c["text"]
            if c["subject"] not in final[-1]["subject"]:
                final[-1]["subject"] += " | " + c["subject"]
        else:
            final.append(c)

    print(f"   ✅ Semantic merging created {len(final)} final chunks")
    return final

# ===========================
# Indexing with Skip Logic
# ===========================
def index_files(folder_path: str, collection, debug_mode=False):
    print("\n🚀 Starting indexing...\n")

    show_chroma_summary(collection, "INITIAL DATABASE STATE")

    total_files = 0
    total_chunks = 0
    skipped_files = 0

    for root, _, files in os.walk(folder_path):
        for f in files:
            if not f.lower().endswith((".pdf", ".txt")):
                continue

            path = os.path.join(root, f)
            category = os.path.basename(root)
            h = file_hash(path)

            print(f"\n{'='*70}")
            print(f"📄 PROCESSING: {f}")
            print(f"{'='*70}")

            # ⭐ CHECK IF ALREADY INDEXED ⭐
            try:
                existing = collection.get(
                    where={"source": path},
                    include=["metadatas"],
                    limit=1
                )

                if existing["ids"] and existing["metadatas"]:
                    stored_hash = existing["metadatas"][0].get("hash")
                    if stored_hash == h:
                        print(f"   ⏭️  Already indexed (file unchanged)")
                        skipped_files += 1
                        continue
                    else:
                        print(f"   🔄 File changed, re-indexing...")
                        # Delete old chunks for this file
                        collection.delete(where={"source": path})
            except Exception as e:
                # If check fails, just proceed with indexing
                pass

            parser = LlamaParse(
                api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
                result_type="markdown",
                language="ar",
                use_vendor_multimodal_model=True,
                vendor_multimodal_model_name="gemini-2.5-flash"
            )

            try:
                docs = parser.load_data(path)
            except Exception as e:
                print(f"   ❌ Parse failed: {e}")
                continue

            if not docs:
                print(f"   ⚠️ No documents returned")
                continue

            raw = "\n\n".join(d.text for d in docs)
            cleaned = clean_arabic_pro(raw)

            print(f"   📏 Extracted {len(cleaned):,} characters")

            if len(cleaned) < 100:
                continue

            sections = extract_sections_from_text(cleaned, f)
            if not sections:
                continue

            print(f"   📑 Extracted {len(sections)} sections")

            # ---------- APPLY PERCENTILE CHUNKING ----------
            sections = percentile_chunking(sections, percentile=75)

            chunks = semantic_chunking(sections, embedding_model)
            if not chunks:
                continue

            print(f"   ✅ Created {len(chunks)} final chunks")

            texts = [c["text"] for c in chunks]
            embeds = embedding_model.encode(texts, batch_size=8, show_progress_bar=False).tolist()

            collection.add(
                documents=texts,
                embeddings=embeds,
                ids=[f"{h}_{i}" for i in range(len(chunks))],
                metadatas=[{
                    "source": path,
                    "file_name": f,
                    "category": category,
                    "subject": c["subject"],
                    "hash": h,
                    "chunk_size": len(c["text"])
                } for c in chunks]
            )

            print(f"   💾 Stored {len(chunks)} chunks")

            total_files += 1
            total_chunks += len(chunks)

            show_chroma_summary(collection, f"AFTER: {f}")

    print(f"\n{'='*70}")
    print(f"🎉 INDEXING COMPLETE")
    print(f"{'='*70}")
    print(f"📁 Files processed: {total_files}")
    print(f"⏭️  Files skipped: {skipped_files}")
    print(f"📦 Total chunks: {total_chunks}")
    print(f"{'='*70}\n")

    show_chroma_summary(collection, "FINAL STATE")

def get_chroma_collection(db_path="./chroma_db", collection_name="imtithal_docs"):
    client = chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection(name=collection_name)

print("✅ Functions loaded! Now run:")
print("collection = get_chroma_collection()")
print("index_files('C:/Users/gjjj3/OneDrive/Documents/GitHub/Imtithal/data', collection)")

collection = get_chroma_collection()
index_files('C:/Users/gjjj3/OneDrive/Documents/GitHub/Imtithal/data', collection)