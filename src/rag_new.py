"""
SDAIA RAG Assistant - Smart Dynamic Chat History
Run: python rag_smart.py
"""

import os
import re
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import chromadb
from sentence_transformers import SentenceTransformer
import pyarabic.araby as araby
from rank_bm25 import BM25Okapi
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()


class ArabicCleaner:
    @staticmethod
    def clean(text: str) -> str:
        if not text:
            return ""
        text = araby.strip_tatweel(text)
        text = araby.strip_tashkeel(text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        return text.strip()


class QueryClassifier:
    """
    Classifies whether a query is a followup or new question.
    Uses semantic similarity + LLM reasoning.
    Output ONLY affects how the prompt is structured — never gates retrieval.
    """

    def __init__(self, embedding_model, llm):
        self.embedding_model = embedding_model
        self.llm = llm

        self.prompt = ChatPromptTemplate.from_template(
            """أنت محلل أسئلة. حدد نوع السؤال بدقة.

المحادثة السابقة:
{history}

السؤال الجديد: {question}

أجب بـ JSON فقط، بدون شرح، بدون ``` :
{{"is_followup": true أو false, "reason": "سبب قصير"}}"""
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def classify(self, query: str, chat_history: List[Dict]) -> Dict:
        if not chat_history:
            return {"is_followup": False, "reason": "no history"}

        # --- Semantic similarity signal ---
        last_q = chat_history[-1]["question"]
        embs = self.embedding_model.encode([query, last_q])
        sim = float(cosine_similarity([embs[0]], [embs[1]])[0][0])

        # --- Length signal ---
        # Arabic questions tend to be shorter than English equivalents.
        # 6 words covers typical followups like "هل تطبق على القطاع الخاص؟"
        is_short = len(query.split()) <= 6

        # --- Fast path: only short-circuit on HIGH confidence ---
        # Short + any meaningful similarity -> definitely a followup
        if is_short and sim > 0.30:
            return {"is_followup": True, "reason": f"short + sim={sim:.2f}"}
        # Only skip to "new" if the query is genuinely long AND similarity
        # is very low. BGE-M3 on Arabic scores lower than on English,
        # so 0.25 is the real floor for "unrelated topics".
        if not is_short and sim < 0.25:
            return {"is_followup": False, "reason": f"long + sim={sim:.2f}"}

        # --- Ambiguous zone: ask the LLM ---
        history_text = ""
        for turn in chat_history[-3:]:
            history_text += f"س: {turn['question']}\nج: {turn['answer']}\n\n"

        try:
            raw = self.chain.invoke({"history": history_text, "question": query})
            raw = raw.strip().replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)
            return {
                "is_followup": bool(parsed.get("is_followup", False)),
                "reason": parsed.get("reason", "llm")
            }
        except Exception:
            # Safe fallback: treat as followup if short, else new
            return {"is_followup": is_short, "reason": "llm-failed, using length"}


class QueryRewriter:
    """
    For followup questions, produces a standalone query that can be
    searched independently against the vector DB.
    """

    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template(
            """المحادثة السابقة:
{history}

السؤال الحالي: {question}

أعد صياغة السؤال الحالي ليصبح سؤالاً مكتملاً ومستقلاً يمكن البحث عنه.
اكتب السؤال المعاد صياغته فقط، بدون شرح."""
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def rewrite(self, query: str, chat_history: List[Dict]) -> str:
        history_text = ""
        for turn in chat_history[-2:]:
            history_text += f"س: {turn['question']}\nج: {turn['answer']}\n\n"

        try:
            rewritten = self.chain.invoke({"history": history_text, "question": query}).strip()
            if not rewritten or len(rewritten) > 500:
                return f"{chat_history[-1]['question']} {query}"
            return rewritten
        except Exception:
            return f"{chat_history[-1]['question']} {query}"


class HybridRetriever:
    def __init__(self, collection, embedding_model, semantic_weight=0.7, keyword_weight=0.3):
        self.collection = collection
        self.embedding_model = embedding_model
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight

        print("🔧 Setting up BM25 keyword search...")
        all_data = collection.get(include=["documents", "metadatas"])
        self.all_docs = all_data["documents"]
        self.all_metadatas = all_data["metadatas"]

        tokenized_docs = [doc.split() for doc in self.all_docs]
        self.bm25 = BM25Okapi(tokenized_docs)
        print(f"✅ Hybrid retriever ready with {len(self.all_docs)} documents")

    def retrieve(self, query: str, top_k: int = 15) -> List[Dict]:
        query_embedding = self.embedding_model.encode([query])[0]
        semantic_results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        semantic_docs = {}
        if semantic_results["ids"][0]:
            max_dist = max(semantic_results["distances"][0]) or 1.0
            for i, doc_id in enumerate(semantic_results["ids"][0]):
                dist = semantic_results["distances"][0][i]
                score = 1 - (dist / (max_dist + 1e-6))
                semantic_docs[doc_id] = {
                    "document": semantic_results["documents"][0][i],
                    "metadata": semantic_results["metadatas"][0][i],
                    "score": score
                }

        query_tokens = query.split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        normalized_bm25 = bm25_scores / (max_bm25 + 1e-6)
        bm25_top_indices = np.argsort(normalized_bm25)[-top_k:][::-1]

        keyword_docs = {}
        for idx in bm25_top_indices:
            keyword_docs[idx] = {
                "document": self.all_docs[idx],
                "metadata": self.all_metadatas[idx],
                "score": float(normalized_bm25[idx])
            }

        all_doc_ids = set(semantic_docs.keys()) | set(keyword_docs.keys())
        combined = []
        for doc_id in all_doc_ids:
            s_score = semantic_docs.get(doc_id, {}).get("score", 0)
            k_score = keyword_docs.get(doc_id, {}).get("score", 0)
            doc_info = semantic_docs.get(doc_id, keyword_docs.get(doc_id))
            combined.append({
                "id": doc_id,
                "document": doc_info["document"],
                "metadata": doc_info["metadata"],
                "score": self.semantic_weight * s_score + self.keyword_weight * k_score,
            })

        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined[:top_k]


class ReRanker:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def rerank(self, query: str, documents: List[Dict], top_n: int = 6) -> List[Dict]:
        if not documents:
            return []
        query_emb = self.embedding_model.encode([query])[0]
        doc_embs = self.embedding_model.encode([d["document"] for d in documents])
        sims = cosine_similarity([query_emb], doc_embs)[0]
        for i, doc in enumerate(documents):
            doc["rerank_score"] = float(sims[i])
        return sorted(documents, key=lambda x: x["rerank_score"], reverse=True)[:top_n]


class ImtithalRAG:
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "imtithal_docs"):
        print("\n" + "=" * 70)
        print("🚀 Initializing Smart Imtithal RAG Assistant")
        print("=" * 70 + "\n")

        print("📂 Loading ChromaDB collection...")
        client = chromadb.PersistentClient(path=db_path)
        self.collection = client.get_collection(name=collection_name)
        chunk_count = self.collection.count()
        print(f"✅ Loaded {chunk_count} existing chunks\n")

        print("🧠 Loading BGE-M3 embedding model...")
        self.embedding_model = SentenceTransformer("BAAI/bge-m3")
        print("✅ Embedding model loaded\n")

        # max_tokens prevents the repetition loops
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.1-8b-instant",
            temperature=0.05,
            max_tokens=1024
        )

        self.cleaner = ArabicCleaner()
        self.retriever = HybridRetriever(self.collection, self.embedding_model)
        self.reranker = ReRanker(self.embedding_model)

        print("🤖 Initializing query classifier & rewriter...")
        self.classifier = QueryClassifier(self.embedding_model, self.llm)
        self.rewriter = QueryRewriter(self.llm)
        print("✅ Ready\n")

        self.chat_history: List[Dict] = []

        # --- Two prompts: one for new questions, one for followups ---
        # The followup prompt surfaces the previous Q&A so the LLM can
        # cross-reference it against the freshly retrieved source docs.
        # Source docs are ALWAYS the ground truth.

        self.prompt_new = ChatPromptTemplate.from_template(
            """أنت مستشار امتثال ذكي ومتخصص في تحليل الأنظمة واللوائح السعودية.

المعلومات المسترجعة من قاعدة البيانات:
{context}

السؤال: {question}

أجب بدقة واحترافية استناداً إلى المعلومات أعلاه فقط. لا تخترع معلومات غير موجودة.

الإجابة:"""
        )

        self.prompt_followup = ChatPromptTemplate.from_template(
            """أنت مستشار امتثال ذكي ومتخصص في تحليل الأنظمة واللوائح السعودية.

السؤال السابق والإجابة السابقة:
السؤال السابق: {prev_question}
الإجابة السابقة: {prev_answer}

المعلومات المسترجعة من قاعدة البيانات للسؤال الحالي:
{context}

السؤال الحالي: {question}

التعليمات:
- السؤال الحالي يستكمل المحادثة السابقة.
- استخدم المعلومات المسترجعة كمصدر الحقيقة.
- إذا كان السؤال يطلب عدداً، اعدّه من المعلومات المسترجعة.
- إذا كان السؤال يطلب عنصراً محدداً (مثل "الثالثة"), حدده بدقة من المعلومات المسترجعة.
- أجب بإجابة قصيرة ودقيقة.

الإجابة:"""
        )

        print("=" * 70)
        print("✅ Smart Imtithal RAG Assistant Ready!")
        print("=" * 70 + "\n")

    def _build_context(self, docs: List[Dict]) -> tuple:
        """Returns (context_string, sources_list)"""
        parts = []
        sources = []
        for i, doc in enumerate(docs, 1):
            meta = doc["metadata"]
            fname = meta.get("file_name", "مصدر غير معروف")
            subject = meta.get("subject", "")
            entry = f"[المصدر {i}: {fname}]"
            if subject:
                entry += f"\n[الموضوع: {subject}]"
            entry += f"\n{doc['document']}"
            parts.append(entry)
            if fname not in [s["file_name"] for s in sources]:
                sources.append({"file_name": fname, "subject": subject, "category": meta.get("category", "")})
        return "\n\n".join(parts), sources

    def ask(self, question: str, verbose: bool = True) -> Dict:
        cleaned = self.cleaner.clean(question)

        # --- Step 1: Classify ---
        classification = self.classifier.classify(cleaned, self.chat_history)
        is_followup = classification["is_followup"]

        if verbose:
            print(f"\n🔍 نوع السؤال: {'متابعة' if is_followup else 'جديد'} — {classification['reason']}")

        # --- Step 2: Determine the search query ---
        # Followups get rewritten into standalone queries so the retriever
        # can actually find the relevant source docs.
        if is_followup and self.chat_history:
            search_query = self.rewriter.rewrite(cleaned, self.chat_history)
            if verbose:
                print(f"✍️  البحث يستخدم: {search_query}")
        else:
            search_query = cleaned

        # --- Step 3: ALWAYS retrieve. Source docs are ground truth. ---
        retrieved = self.retriever.retrieve(search_query, top_k=15)
        reranked = self.reranker.rerank(search_query, retrieved, top_n=6)
        context, sources = self._build_context(reranked)

        # --- Step 4: Pick the right prompt and invoke ---
        if is_followup and self.chat_history:
            prev = self.chat_history[-1]
            answer = (self.prompt_followup | self.llm | StrOutputParser()).invoke({
                "prev_question": prev["question"],
                "prev_answer": prev["answer"],
                "context": context,
                "question": question,
            })
        else:
            answer = (self.prompt_new | self.llm | StrOutputParser()).invoke({
                "context": context,
                "question": question,
            })

        # --- Step 5: Store ---
        self.chat_history.append({
            "question": question,
            "answer": answer,
            "sources": sources,
        })

        return {"answer": answer, "question": question, "sources": sources}

    def chat_interactive(self):
        print("\n" + "=" * 70)
        print("💬 وضع المحادثة التفاعلية الذكية")
        print("=" * 70)
        print("\nاكتب سؤالك بالعربية")
        print("Commands: /exit, /clear, /help\n")

        while True:
            try:
                question = input("🤔 سؤالك: ").strip()
                if not question:
                    continue
                if question.lower() in ['/exit', '/quit', 'خروج']:
                    print("\n👋 شكراً لاستخدامك مستشار امتثال سدايا!\n")
                    break
                if question.lower() == '/clear':
                    self.chat_history.clear()
                    print("✅ تم مسح المحادثة\n")
                    continue
                if question.lower() == '/help':
                    print("\n/exit  — إنهاء")
                    print("/clear — مسح المحادثة\n")
                    continue

                response = self.ask(question, verbose=True)

                print(f"\n💡 الإجابة:\n{response['answer']}\n")
                if response.get("sources"):
                    print(f"📚 المصادر ({len(response['sources'])}):")
                    for i, src in enumerate(response["sources"], 1):
                        print(f"   {i}. {src['file_name']}")
                        if src.get("subject"):
                            print(f"      الموضوع: {src['subject'][:60]}...")
                    print()

            except KeyboardInterrupt:
                print("\n\n👋 شكراً لاستخدامك مستشار امتثال سدايا!\n")
                break
            except Exception as e:
                print(f"\n❌ خطأ: {e}\n")
                continue


def main():
    rag = ImtithalRAG(db_path="./chroma_db", collection_name="imtithal_docs")
    rag.chat_interactive()


if __name__ == "__main__":
    main()

# =========================================================
# Wrapper for evaluation scripts (expects ask_rag)
# =========================================================
_rag_instance = None

def ask_rag(query: str):
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = ImtithalRAG()
    res = _rag_instance.ask(query)
    return res['answer'], res.get('sources', [])
