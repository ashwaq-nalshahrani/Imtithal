"""
Recall@k Evaluator for SDAIA RAG (raw retrieval, no reranker)
"""

import os
import json
import csv
from rapidfuzz import fuzz
from utils import embedding_model, clean_arabic_pro
from RAG import ImtithalRAG  # Make sure this points to your current RAG class

# ----------------------------
# 1. Load questions
# ----------------------------
questions_file = "C:/Users/gjjj3/OneDrive/Desktop/imtithal-django/imtithal/questions_200.json"

with open(questions_file, "r", encoding="utf-8") as f:
    questions = json.load(f)

questions = questions[:20]  # For testing; remove slicing for full dataset

if not questions:
    print("⚠️ No questions found")
    exit()

# ----------------------------
# 2. Fuzzy matching function
# ----------------------------
def is_relevant(gt_answer, chunk_text, threshold=75):
    """
    Return True if retrieved chunk matches ground truth.
    Uses Arabic preprocessing to normalize text.
    """
    gt_clean = clean_arabic_pro(gt_answer)
    chunk_clean = clean_arabic_pro(chunk_text)
    return fuzz.partial_ratio(gt_clean, chunk_clean) >= threshold

# ----------------------------
# 3. Helper: retrieve raw chunks without reranker
# ----------------------------
def retrieve_chunks_raw(rag: ImtithalRAG, query: str, top_k=10):
    """
    Retrieve top-k chunks directly from ChromaDB, WITHOUT any reranking.
    """
    if not hasattr(rag, "collection"):
        raise AttributeError("RAG object has no 'collection' attribute")
    
    # 1. Encode query
    q_vec = embedding_model.encode([query])[0]

    # 2. Query ChromaDB
    results = rag.collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=top_k,
        include=["documents", "metadatas"]
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    retrieved = [{"document": d, "metadata": m, "score": 0.0} for d, m in zip(docs, metas)]
    return retrieved

# ----------------------------
# 4. Initialize RAG
# ----------------------------
rag = ImtithalRAG()  # uses your current RAG class
print("✅ RAG system loaded and ready for retrieval evaluation.")

# ----------------------------
# 5. Evaluation
# ----------------------------
results = []
recall_at_1 = []
recall_at_5 = []
recall_at_10 = []

for idx, q in enumerate(questions, 1):
    question = q["question"]
    ground_truth = q["answer"]

    print(f"\n🔍 Question {idx}: {question}")

    try:
        retrieved_chunks = retrieve_chunks_raw(rag, question, top_k=10)
    except Exception as e:
        print(f"⚠️ Retrieval error: {e}")
        continue

    if not retrieved_chunks:
        print("⚠️ No chunks retrieved")
        continue

    chunk_texts = [c["document"] for c in retrieved_chunks]

    # Calculate Recall@k
    r1 = int(any(is_relevant(ground_truth, t) for t in chunk_texts[:1]))
    r5 = int(any(is_relevant(ground_truth, t) for t in chunk_texts[:5]))
    r10 = int(any(is_relevant(ground_truth, t) for t in chunk_texts[:10]))

    recall_at_1.append(r1)
    recall_at_5.append(r5)
    recall_at_10.append(r10)

    # Optional: print which chunks matched
    matched_chunks = [t for t in chunk_texts[:10] if is_relevant(ground_truth, t)]
    if matched_chunks:
        print(f"✅ Matching chunk(s) found: {matched_chunks}")
    else:
        print("⚠️ No matching chunks in top 10.")

    results.append({
        "question": question,
        "recall@1": r1,
        "recall@5": r5,
        "recall@10": r10,
        "top_chunk_files": [
            c["metadata"].get("file_name", "غير معروف") for c in retrieved_chunks[:5]
        ]
    })

# ----------------------------
# 6. Print averages
# ----------------------------
def safe_mean(lst):
    return sum(lst) / len(lst) if lst else 0.0

print("\n===== 📊 Retrieval Evaluation =====")
print(f"Recall@1  : {safe_mean(recall_at_1):.2f}")
print(f"Recall@5  : {safe_mean(recall_at_5):.2f}")
print(f"Recall@10 : {safe_mean(recall_at_10):.2f}")

# ----------------------------
# 7. Save CSV
# ----------------------------
if results:
    with open("retrieval_evaluation_raw.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\n✅ Results saved to retrieval_evaluation_raw.csv")
else:
    print("\n⚠️ No results to save")
