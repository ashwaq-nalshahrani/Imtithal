"""
Generic RAG Evaluation Framework — Modified to use SAME ChromaDB path as RAG
Evaluates any RAG chatbot that exposes an ask_rag(query) -> Tuple[str, List[str]] function.
Generates questions from ChromaDB, runs them through the external RAG module, evaluates results.
"""

import os, json, importlib.util
import numpy as np
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# IMPORTANT: Use the SAME Chroma config as RAG (utils.py)
from utils import get_chroma_collection

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION — RAG chatbot module path (same folder: src/RAG.py)
RAG_MODULE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RAG.py")

# ═══════════════════════════════════════════════════════════════
# Dynamic loader for ask_rag
def load_rag_module(module_path):
    abs_path = os.path.abspath(module_path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"RAG module not found: {abs_path}")
    spec = importlib.util.spec_from_file_location("rag_module", abs_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "ask_rag"):
        raise AttributeError(
            f"RAG module '{abs_path}' must have 'ask_rag(query)' function. "
            f"Add a wrapper in RAG.py that calls ImtithalRAG().ask(query)."
        )
    return module.ask_rag

# ═══════════════════════════════════════════════════════════════
# Setup embedding model + ChromaDB
print("Loading embedding model BAAI/bge-m3 ...")
embed_model = SentenceTransformer("BAAI/bge-m3")
print("Model loaded.\n")

# Use SAME collection as RAG
collection = get_chroma_collection()

# Try to print debug info
try:
    # in recent chromadb, list_collections exists on client not collection
    # so we just print name if available
    print(f"Using collection: {getattr(collection, 'name', 'imtithal_docs')}")
except Exception:
    pass

# ═══════════════════════════════════════════════════════════════
# Generate questions from ChromaDB
def generate_questions(num_per_type=3):
    results = collection.get(include=['documents', 'metadatas'])
    documents = results.get('documents', [])
    metadatas = results.get('metadatas', [])
    print(f"Found {len(documents)} chunks in ChromaDB")

    if not documents:
        return []

    by_source = defaultdict(list)
    for doc, meta in zip(documents, metadatas):
        src = "unknown"
        if isinstance(meta, dict):
            src = meta.get("source", "unknown")
        by_source[src].append((doc, meta))

    sources = list(by_source.keys())
    questions = []

    # Factual
    for i in range(min(num_per_type, len(sources))):
        src = sources[i % len(sources)]
        doc, meta = by_source[src][0]
        sentences = [s.strip() for s in doc.split('.') if len(s.strip()) > 20]
        ground_truth = sentences[0] if sentences else doc[:300]
        q = "ما هي النقاط الرئيسية في هذا المستند؟"
        questions.append({
            "id": f"factual_{i+1}",
            "type": "factual",
            "question": q,
            "ground_truth": ground_truth,
            "source_chunks": [doc]
        })

    # Conceptual
    for i in range(min(num_per_type, len(sources))):
        src = sources[(i + num_per_type) % len(sources)]
        doc, meta = by_source[src][min(1, len(by_source[src])-1)]
        if len(doc) < 100:
            continue
        q = "اشرح المفهوم الموضح في المستند"
        questions.append({
            "id": f"conceptual_{i+1}",
            "type": "conceptual",
            "question": q,
            "ground_truth": doc[:500],
            "source_chunks": [doc]
        })

    # Reasoning
    for i in range(min(num_per_type, len(documents)-1)):
        doc1, doc2 = documents[i], documents[i+1]
        meta1, meta2 = metadatas[i], metadatas[i+1]
        topic = "موضوع مشترك"
        if isinstance(meta1, dict):
            topic = meta1.get('category', 'موضوع مشترك')
        q = f"قارن بين المعلومات في مستندين متعلقين بـ {topic}"
        questions.append({
            "id": f"reasoning_{i+1}",
            "type": "reasoning",
            "question": q,
            "ground_truth": f"{doc1[:200]} ... و {doc2[:200]} ...",
            "source_chunks": [doc1, doc2]
        })

    return questions

# ═══════════════════════════════════════════════════════════════
# Evaluation metrics
def calc_correctness(gt, ans):
    if not gt or not ans:
        return 0.0
    sim = cosine_similarity(embed_model.encode([gt]), embed_model.encode([ans]))[0][0]
    return float(np.clip((sim + 1) / 2, 0, 1))

def calc_faithfulness(ans, chunks):
    if not ans or not chunks:
        return 0.0
    combined_emb = embed_model.encode([' '.join(chunks)])
    supported = 0
    parts = [s.strip() for s in ans.split('.') if s.strip()]
    for s in parts:
        if cosine_similarity(embed_model.encode([s]), combined_emb)[0][0] > 0.5:
            supported += 1
    return supported / max(1, len(parts))

def calc_relevance(question, chunks):
    if not chunks or not question:
        return 0.0, []
    q_emb = embed_model.encode([question])
    c_embs = embed_model.encode(chunks)
    sims = cosine_similarity(q_emb, c_embs)[0]
    scores = list(np.clip((sims + 1) / 2, 0, 1))
    return float(np.mean(scores)), scores

def calc_completeness(gt, ans):
    if not gt or not ans:
        return 0.0
    stop = {'في','من','الى','على','عن','هو','هي','ان','ما','لا','هذا','هذه','ذلك','التي','الذي','و','او','لكن','ثم','كما'}
    gt_words = set(gt.split()) - stop
    ans_words = set(ans.split()) - stop
    return min(1.0, len(gt_words & ans_words) / max(1, len(gt_words)))

def calc_accuracy_recall_precision(gt, ans):
    if not gt or not ans:
        return 0.0, 0.0, 0.0
    stop = {'في','من','الى','على','عن','هو','هي','ان','ما','لا','هذا','هذه','ذلك','التي','الذي','و','او','لكن','ثم','كما'}
    gt_words = set(gt.split()) - stop
    ans_words = set(ans.split()) - stop
    overlap = gt_words & ans_words
    union = gt_words | ans_words
    accuracy = len(overlap) / len(union) if union else 0.0
    recall = len(overlap) / len(gt_words) if gt_words else 0.0
    precision = len(overlap) / len(ans_words) if ans_words else 0.0
    return round(accuracy, 3), round(recall, 3), round(precision, 3)

def evaluate_single(qdata, answer, chunks):
    correctness = calc_correctness(qdata["ground_truth"], answer)
    faithfulness = calc_faithfulness(answer, chunks)
    relevance, _ = calc_relevance(qdata["question"], chunks)
    completeness = calc_completeness(qdata["ground_truth"], answer)
    hallucination = 1.0 - faithfulness
    accuracy, recall, precision = calc_accuracy_recall_precision(qdata["ground_truth"], answer)
    overall = (0.25*correctness + 0.2*faithfulness + 0.15*relevance +
               0.1*completeness + 0.1*accuracy + 0.1*recall + 0.1*(1-hallucination))
    return {
        "correctness": round(correctness, 3),
        "faithfulness": round(faithfulness, 3),
        "relevance": round(relevance, 3),
        "completeness": round(completeness, 3),
        "hallucination": round(hallucination, 3),
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "overall": round(overall, 3)
    }

# ═══════════════════════════════════════════════════════════════
# MAIN
def main():
    print("="*70)
    print("  Generic RAG Evaluation Framework (SAME DB as RAG)")
    print("="*70)

    ask_rag = load_rag_module(RAG_MODULE_PATH)
    print("ask_rag function loaded successfully.\n")

    questions = generate_questions(num_per_type=3)
    if not questions:
        print("\n⚠️ No chunks found in ChromaDB. This means indexing has not been run on this DB path.")
        print("Run your indexing script first (the one that fills the same DB used by get_chroma_collection).")
        return

    results = []

    for i, q in enumerate(questions, 1):
        print(f"Q{i}/{len(questions)} [{q['type']}]: {q['question'][:60]}...")
        try:
            ans, used_sources = ask_rag(q['question'])

            # For metrics we need chunks. If your RAG returns filenames only as "sources",
            # we should use the ground truth chunks stored in q['source_chunks'] for faithfulness/calc.
            chunks_for_eval = q.get("source_chunks", [])

            metrics = evaluate_single(q, ans, chunks_for_eval)
            results.append({
                "id": q['id'],
                "type": q['type'],
                "question": q['question'],
                "ground_truth": q['ground_truth'][:200],
                "answer": (ans or "")[:200],
                "used_sources": used_sources,
                "metrics": metrics
            })
            print(f"-> Overall: {metrics['overall']:.3f}")
        except Exception as e:
            print(f"-> ERROR: {e}")

    if not results:
        return

    # Print table
    print("\nEvaluation Table:")
    header = ["ID","Type","Correct","Faith","Rel","Comp","Hall","Acc","Recall","Prec","Overall"]
    print(" | ".join(f"{h:>8}" for h in header))
    print("-"*100)
    for r in results:
        m = r["metrics"]
        print(f"{r['id']:<8} | {r['type']:<8} | {m['correctness']:>6.3f} | {m['faithfulness']:>5.3f} | "
              f"{m['relevance']:>5.3f} | {m['completeness']:>5.3f} | {m['hallucination']:>5.3f} | "
              f"{m['accuracy']:>5.3f} | {m['recall']:>5.3f} | {m['precision']:>5.3f} | {m['overall']:>6.3f}")

    # Save JSON
    output_file = "arabic_rag_evaluation.json"

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    print(f"\n✅ Results saved to {output_file}")

if __name__ == "__main__":
    main()
