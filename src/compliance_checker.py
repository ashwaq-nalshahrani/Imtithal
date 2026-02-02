"""
compliance_checker.py
─────────────────────
Uploads a company-project file, chunks it with the same pipeline as utils.py,
then for every chunk retrieves the most relevant regulation passages and asks
the LLM to judge compliance.

Usage (standalone):
    python compliance_checker.py path/to/company_project.pdf
"""

import os
import re
import json
import tempfile
import shutil
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

# ─── reuse your existing utilities ──────────────────────────────────────────
from utils import (
    clean_arabic_pro,
    extract_sections_from_text,
    percentile_chunking,
    semantic_chunking,
)

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
REGULATION_DB_PATH   = "./chroma_db"
REGULATION_COLLECTION = "imtithal_docs"   # your existing regulation index

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
LLM_MODEL            = "llama-3.1-8b-instant"
LLM_TEMPERATURE      = 0.05
LLM_MAX_TOKENS       = 2048

# How many regulation chunks to retrieve & how many to keep after re-rank
RETRIEVE_TOP_K = 15
RERANK_TOP_N   = 5

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBALS (loaded once)
# ═══════════════════════════════════════════════════════════════════════════════
_embedding_model: SentenceTransformer = None
_llm: ChatGroq = None
_reg_collection = None
_bm25 = None
_bm25_docs = None
_bm25_metas = None


def _init():
    """Lazy-load heavy objects exactly once."""
    global _embedding_model, _llm, _reg_collection, _bm25, _bm25_docs, _bm25_metas

    if _embedding_model is not None:
        return                          # already initialised

    print("🧠 Loading embedding model …")
    _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("🤖 Connecting to Groq LLM …")
    _llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )

    print("📂 Opening regulation ChromaDB …")
    client = chromadb.PersistentClient(path=REGULATION_DB_PATH)
    _reg_collection = client.get_collection(name=REGULATION_COLLECTION)
    print(f"   ✅ {_reg_collection.count()} regulation chunks loaded")

    # Build BM25 index over the regulation corpus for hybrid search
    all_data      = _reg_collection.get(include=["documents", "metadatas"])
    _bm25_docs    = all_data["documents"]
    _bm25_metas   = all_data["metadatas"]
    _bm25         = BM25Okapi([doc.split() for doc in _bm25_docs])
    print("   ✅ BM25 keyword index ready\n")


# ═══════════════════════════════════════════════════════════════════════════════
# PARSING (reuses LlamaParse exactly like utils.py index_files)
# ═══════════════════════════════════════════════════════════════════════════════
def parse_file(file_path: str) -> str:
    """Extract raw text from PDF or TXT using LlamaParse (same as utils.py)."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    # PDF → LlamaParse
    from llama_parse import LlamaParse
    parser = LlamaParse(
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
        result_type="markdown",
        language="ar",
        use_vendor_multimodal_model=True,
        vendor_multimodal_model_name="gemini-2.5-flash",
    )
    docs = parser.load_data(file_path)
    if not docs:
        raise ValueError("LlamaParse returned no content for this file.")
    return "\n\n".join(d.text for d in docs)


# ═══════════════════════════════════════════════════════════════════════════════
# CHUNKING (identical pipeline to utils.py)
# ═══════════════════════════════════════════════════════════════════════════════
def chunk_project_text(raw_text: str, filename: str) -> List[Dict[str, str]]:
    """
    Clean → extract sections → percentile chunk → semantic merge.
    Returns list of {"subject": ..., "text": ...} dicts.
    """
    cleaned  = clean_arabic_pro(raw_text)
    print(f"   📏 {len(cleaned):,} characters after cleaning")

    sections = extract_sections_from_text(cleaned, filename)
    if not sections:
        raise ValueError("Could not extract any sections from the file.")

    sections = percentile_chunking(sections, percentile=75)
    chunks   = semantic_chunking(sections, _embedding_model)

    print(f"   📦 {len(chunks)} project chunks created")
    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# HYBRID RETRIEVAL OF REGULATIONS (mirrors rag_new.py HybridRetriever)
# ═══════════════════════════════════════════════════════════════════════════════
def _retrieve_regulations(query: str, top_k: int = RETRIEVE_TOP_K) -> List[Dict]:
    """Semantic + BM25 hybrid search over the regulation ChromaDB."""
    # ── semantic ──
    q_emb = _embedding_model.encode([query])[0]
    sem   = _reg_collection.query(
        query_embeddings=[q_emb.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    max_dist = max(sem["distances"][0]) if sem["distances"][0] else 1.0
    semantic_docs = {}
    for i, did in enumerate(sem["ids"][0]):
        dist  = sem["distances"][0][i]
        score = 1 - (dist / (max_dist + 1e-6))
        semantic_docs[did] = {
            "document": sem["documents"][0][i],
            "metadata": sem["metadatas"][0][i],
            "score":    score,
        }

    # ── BM25 keyword ──
    bm25_scores = _bm25.get_scores(query.split())
    max_bm25    = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
    norm_bm25   = bm25_scores / (max_bm25 + 1e-6)
    top_idx     = np.argsort(norm_bm25)[-top_k:][::-1]

    keyword_docs = {}
    for idx in top_idx:
        keyword_docs[idx] = {
            "document": _bm25_docs[idx],
            "metadata": _bm25_metas[idx],
            "score":    float(norm_bm25[idx]),
        }

    # ── fuse (0.7 semantic + 0.3 keyword) ──
    all_ids  = set(semantic_docs.keys()) | set(keyword_docs.keys())
    combined = []
    for did in all_ids:
        s = semantic_docs.get(did, {}).get("score", 0)
        k = keyword_docs.get(did, {}).get("score", 0)
        info = semantic_docs.get(did, keyword_docs.get(did))
        combined.append({
            "document": info["document"],
            "metadata": info["metadata"],
            "score":    0.7 * s + 0.3 * k,
        })
    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined[:top_k]


def _rerank(query: str, docs: List[Dict], top_n: int = RERANK_TOP_N) -> List[Dict]:
    """Re-rank retrieved regulation chunks by cosine similarity."""
    if not docs:
        return []
    q_emb  = _embedding_model.encode([query])[0]
    d_embs = _embedding_model.encode([d["document"] for d in docs])
    sims   = cosine_similarity([q_emb], d_embs)[0]
    for i, d in enumerate(docs):
        d["rerank_score"] = float(sims[i])
    return sorted(docs, key=lambda x: x["rerank_score"], reverse=True)[:top_n]


# ═══════════════════════════════════════════════════════════════════════════════
# COMPLIANCE PROMPT + LLM CALL
# ═══════════════════════════════════════════════════════════════════════════════
COMPLIANCE_PROMPT = ChatPromptTemplate.from_template("""أنت مستشار امتثال متخصص في لوائح هيئة البيانات والذكاء الاصطناعي (سدايا) والهيئة الوطنية للأمن السيبراني (NCA).

────────────────────────────────
الأنظمة والتنظيمات المرجعية:
────────────────────────────────
{regulations}

────────────────────────────────
النص من مشروع الشركة الذي يحتاج للتقييم:
────────────────────────────────
{project_chunk}

────────────────────────────────
المهمة:
────────────────────────────────
قيّم مدى امتثال النص أعلاه مع الأنظمة المرجعية المذكورة. أجب بـ JSON فقط بالصيغة التالية بلا شرح خارجي وبلا شفرات:

{{
  "verdict": "متوافق" أو "غير متوافق" أو "يحتاج مراجعة",
  "confidence": "عالي" أو "متوسط" أو "منخفض",
  "summary": "ملخص قصير بالعربية عن نتيجة التقييم",
  "matched_regulations": ["المادة / النص التنظيمي المعني 1", "..."],
  "issues": ["وصف المخالفة أو النقطة التي تحتاج مراجعة 1", "..."],
  "recommendations": ["التوصية 1", "..."]
}}""")


def _judge_chunk(project_chunk_text: str, project_subject: str) -> Dict[str, Any]:
    """
    For one project chunk:
      1. Retrieve relevant regulations
      2. Ask LLM to compare and return structured verdict
    """
    # retrieve & rerank regulations relevant to this chunk
    retrieved = _retrieve_regulations(project_chunk_text)
    reranked  = _rerank(project_chunk_text, retrieved, top_n=RERANK_TOP_N)

    if not reranked:
        return {
            "subject":   project_subject,
            "verdict":   "يحتاج مراجعة",
            "confidence":"منخفض",
            "summary":   "لم يتم العثور على لوائح مرتبطة لهذا القسم.",
            "matched_regulations": [],
            "issues":    ["لا توجد لوائح مرتبطة في قاعدة البيانات."],
            "recommendations": ["تحقق من أن قاعدة البيانات تحتوي على اللوائح المعنية."],
            "regulation_sources": [],
        }

    # build regulation context
    reg_parts = []
    reg_sources = []
    for i, doc in enumerate(reranked, 1):
        meta = doc["metadata"]
        fname   = meta.get("file_name", "مصدر غير معروف")
        subject = meta.get("subject", "")
        reg_parts.append(f"[المصدر {i}: {fname} – {subject}]\n{doc['document']}")
        reg_sources.append({"file_name": fname, "subject": subject})

    regulations_text = "\n\n".join(reg_parts)

    # call LLM
    chain  = COMPLIANCE_PROMPT | _llm | StrOutputParser()
    raw    = chain.invoke({
        "regulations":   regulations_text,
        "project_chunk": project_chunk_text,
    })

    # parse JSON (strip any accidental markdown fences)
    raw = raw.strip().replace("```json", "").replace("```", "").strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # fallback: couldn't parse
        result = {
            "verdict":   "يحتاج مراجعة",
            "confidence":"منخفض",
            "summary":   "فشل في تحليل نتيجة الذكاء الاصطناعي. يرجى المراجعة يدوياً.",
            "matched_regulations": [],
            "issues":    ["خطأ في تحليل الإجابة."],
            "recommendations": [],
        }

    result["subject"]            = project_subject
    result["regulation_sources"] = reg_sources
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY: full compliance check on one file
# ═══════════════════════════════════════════════════════════════════════════════
def check_compliance(file_path: str) -> Dict[str, Any]:
    """
    End-to-end compliance check.

    Returns:
        {
            "file_name": str,
            "total_chunks": int,
            "results": [ {verdict per chunk} ],
            "overall_summary": { "compliant": N, "non_compliant": N, "needs_review": N }
        }
    """
    _init()   # ensure models are loaded

    filename = os.path.basename(file_path)
    print(f"\n{'═'*70}")
    print(f"📄 Compliance Check: {filename}")
    print(f"{'═'*70}\n")

    # 1. Parse
    print("📖 Parsing file …")
    raw_text = parse_file(file_path)
    print(f"   ✅ Parsed {len(raw_text):,} characters\n")

    # 2. Chunk
    print("✂️  Chunking …")
    chunks = chunk_project_text(raw_text, filename)
    print(f"   ✅ {len(chunks)} chunks ready\n")

    # 3. Judge each chunk
    results = []
    counts  = {"متوافق": 0, "غير متوافق": 0, "يحتاج مراجعة": 0}

    for i, chunk in enumerate(chunks, 1):
        print(f"⚖️  Judging chunk {i}/{len(chunks)}: {chunk['subject'][:60]} …")
        verdict = _judge_chunk(chunk["text"], chunk.get("subject", f"القسم {i}"))
        results.append(verdict)
        v = verdict.get("verdict", "يحتاج مراجعة")
        counts[v] = counts.get(v, 0) + 1
        print(f"   → {v} (الثقة: {verdict.get('confidence','')})")

    print(f"\n{'═'*70}")
    print(f"✅ Compliance check complete for {filename}")
    print(f"   متوافق: {counts.get('متوافق',0)}  |  غير متوافق: {counts.get('غير متوافق',0)}  |  يحتاج مراجعة: {counts.get('يحتاج مراجعة',0)}")
    print(f"{'═'*70}\n")

    return {
        "file_name":      filename,
        "total_chunks":   len(chunks),
        "results":        results,
        "overall_summary": counts,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python compliance_checker.py <path_to_file.pdf|.txt>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.isfile(path):
        print(f"❌ File not found: {path}")
        sys.exit(1)

    report = check_compliance(path)

    # pretty-print
    print("\n📊 FULL REPORT (JSON):\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))