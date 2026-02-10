"""
Recall@k Evaluator for the new ImtithalRAG (hybrid + reranker)
Evaluates both raw retrieval (HybridRetriever) and reranked top-k.
"""

import os
import json
import csv
from rapidfuzz import fuzz
from rag.rag_engine import ImtithalRAG
from rag.utils import embedding_model, clean_arabic_pro

# ----------------------------
# 1. Load questions
# ----------------------------
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
questions_file = BASE_DIR / "questions_200.json"

with open(questions_file, "r", encoding="utf-8") as f:
    questions = json.load(f)

questions = questions[:20]  # test first 20, remove slicing for full set

if not questions:
    print("⚠️ No questions found")
    exit()

# ----------------------------
# 2. Fuzzy matching
# ----------------------------
def is_relevant(gt_answer, chunk_text, threshold=75):
    gt_clean = clean_arabic_pro(gt_answer)
    chunk_clean = clean_arabic_pro(chunk_text)
    return fuzz.partial_ratio(gt_clean, chunk_clean) >= threshold

# ----------------------------
# 3. Helper: retrieve raw chunks without reranker
# ----------------------------
def retrieve_chunks_raw(rag: ImtithalRAG, query: str, top_k=15):
    """
    Retrieve top-K chunks from HybridRetriever WITHOUT reranking
    """
    retrieved = rag.retriever.retrieve(query, top_k=top_k)
    raw_chunks = [{"document": d["document"], "metadata": d["metadata"], "score": d["score"]} for d in retrieved]
    return raw_chunks

# ----------------------------
# 4. Helper: retrieve reranked top-K chunks
# ----------------------------
def retrieve_chunks_reranked(rag: ImtithalRAG, query: str, top_k=6):
    retrieved = rag.retriever.retrieve(query, top_k=15)
    reranked = rag.reranker.rerank(query, retrieved, top_n=top_k)
    return reranked

# ----------------------------
# 5. Initialize RAG
# ----------------------------
rag = ImtithalRAG()
print("✅ RAG system loaded and ready for retrieval evaluation.")

# ----------------------------
# 6. Evaluation
# ----------------------------
results = []
recall_raw_1, recall_raw_5, recall_raw_10 = [], [], []
recall_rerank_1, recall_rerank_5, recall_rerank_10 = [], [], []

for idx, q in enumerate(questions, 1):
    question = q["question"]
    ground_truth = q["answer"]

    print(f"\n🔍 Question {idx}: {question}")

    try:
        raw_chunks = retrieve_chunks_raw(rag, question, top_k=15)
        reranked_chunks = retrieve_chunks_reranked(rag, question, top_k=6)
    except Exception as e:
        print(f"⚠️ Retrieval error: {e}")
        continue

    if not raw_chunks or not reranked_chunks:
        print("⚠️ No chunks retrieved")
        continue

    raw_texts = [c["document"] for c in raw_chunks]
    rerank_texts = [c["document"] for c in reranked_chunks]



    # --- Recall@k calculation for reranked retrieval ---
    r_rerank_1 = int(any(is_relevant(ground_truth, t) for t in rerank_texts[:1]))
    r_rerank_5 = int(any(is_relevant(ground_truth, t) for t in rerank_texts[:5]))
    r_rerank_10 = int(any(is_relevant(ground_truth, t) for t in rerank_texts[:10]))
    recall_rerank_1.append(r_rerank_1)
    recall_rerank_5.append(r_rerank_5)
    recall_rerank_10.append(r_rerank_10)

    results.append({
        "question": question,
        "ground_truth": ground_truth,
        "recall_rerank@1": r_rerank_1,
        "recall_rerank@5": r_rerank_5,
        "recall_rerank@10": r_rerank_10,
        "top_raw_files": [c["metadata"].get("file_name", "غير معروف") for c in raw_chunks[:5]],
        "top_rerank_files": [c["metadata"].get("file_name", "غير معروف") for c in reranked_chunks[:5]],
    })

# ----------------------------
# 7. Print averages
# ----------------------------
def safe_mean(lst):
    return sum(lst)/len(lst) if lst else 0.0

print("\n===== 📊 Reranked Retrieval Evaluation =====")
print(f"Recall@1  : {safe_mean(recall_rerank_1):.2f}")
print(f"Recall@5  : {safe_mean(recall_rerank_5):.2f}")
print(f"Recall@10 : {safe_mean(recall_rerank_10):.2f}")

# ----------------------------
# 8. Save CSV
# ----------------------------
if results:
    with open("retrieval_evaluation_newrag.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print("\n✅ Results saved to retrieval_evaluation_newrag.csv")
else:
    print("\n⚠️ No results to save")
