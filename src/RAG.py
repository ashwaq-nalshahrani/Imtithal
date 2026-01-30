
import os
from google import genai
from dotenv import load_dotenv

from utils import get_chroma_collection, embedding_model, clean_arabic_pro

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def _cosine_sim(a, b):
    # a,b: 1D lists/arrays
    import math
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += float(x) * float(y)
        na += float(x) * float(x)
        nb += float(y) * float(y)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))

class ImtithalRAG:
    def __init__(self):
        self.collection = get_chroma_collection()

        # ====== إعدادات لتحسين الجودة ======
        self.retrieval_k = 12      # Top-K من Chroma (ارفعيها 10-20 حسب حجم الداتا)
        self.final_k = 6           # كم chunk النهائي اللي يدخل للسياق
        self.max_context_chars = 14000  # حد للسياق عشان ما يتشتت المودل

    def ask(self, query):
        # 1) تنظيف السؤال + embedding
        cleaned_query = clean_arabic_pro(query)
        q_vec = embedding_model.encode([cleaned_query])[0]  # 1D vector

        # 2) Retrieval من Chroma (أكبر من قبل)
        results = self.collection.query(
            query_embeddings=[q_vec.tolist()],
            n_results=self.retrieval_k,
            include=["documents", "metadatas"]
        )

        docs = results["documents"][0] if results.get("documents") else []
        metas = results["metadatas"][0] if results.get("metadatas") else []

        if not docs:
            return "(لا يوجد نص كافٍ في الوثيقة)", []

        # 3) Rerank محلي: نحسب similarity بين السؤال وكل chunk ونختار الأفضل
        # (هذا يحسن النتائج كثير لأن Chroma أحياناً يرجّع chunks شبه قريبة)
        doc_vecs = embedding_model.encode(docs)  # list of vectors
        scored = []
        for i, (d, m, v) in enumerate(zip(docs, metas, doc_vecs)):
            scored.append((_cosine_sim(q_vec, v), i, d, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: self.final_k]

        # 4) بناء السياق: إزالة التكرار + ترتيب بسيط + قص للطول
        seen = set()
        chosen_docs = []
        chosen_sources = set()

        for score, i, d, m in top:
            key = d.strip()
            if not key or key in seen:
                continue
            seen.add(key)

            # اجمع المصدر
            src = os.path.basename(m.get("source", "")) if isinstance(m, dict) else ""
            if src:
                chosen_sources.add(src)

            chosen_docs.append(d.strip())

        context = "\n\n".join(chosen_docs)

        # قص السياق لو كبير
        if len(context) > self.max_context_chars:
            context = context[: self.max_context_chars]

       
# 5) بناء البرومبت
        prompt = f"""
أنت مستشار امتثال ذكي ومتخصص في تحليل الأنظمة والتشريعات واللوائح الصادرة عن الجهات السعودية مثل الهيئة الوطنية للأمن السيبراني وسدايا.

مهمتك هي قراءة استفسار المستخدم بدقة، ثم تحليل النصوص النظامية المرتبطة به، والموجودة في الملفات التالية:

{context}

ابدأ دائمًا إجابتك بالجملة التالية:
"أهلًا بك، أنا مستشار امتثال معتمد، هنا لمساعدتك في فهم الأنظمة والتشريعات بدقة..."

ثم اتبع التعليمات التالية:

1. استخرج الكلمات المفتاحية من سؤال المستخدم لتحديد السياق بدقة.
2. استخدم المعلومات الموجودة في الملفات فقط للإجابة على السؤال.
3. لخص النصوص النظامية ذات العلاقة بشكل واضح ومباشر، وركّز على تقديم خلاصة مفيدة يفهمها المستخدم بسهولة.
4. إذا وُجد نص صريح، فابدأ الإجابة بالعبارة التالية:  
   **"استنادًا إلى الملف [اسم الملف]، فإن التنظيم ينص على ما يلي:"**  
   ثم قدم شرحًا مختصرًا ودقيقًا.
5. إذا لم يوجد نص مباشر، لا تفترض أي إجابة، بل استخدم العبارة التالية:  
   **"لم يتم العثور على نص نظامي صريح حول هذا الموضوع، لكن قد يفيدك الاطلاع على الملف [اسم الملف] لاحتوائه على معلومات مرتبطة بالسياق."**

احرص على أن تكون إجابتك:

- باللغة العربية الفصحى فقط  
- رسمية، دقيقة، ومباشرة  
- مُركّزة على **التلخيص الواضح** و**سهولة الفهم**

سؤال المستخدم:
{query}

إجابة المستشار:
"""

        # 6) Gemini
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.05  # خليه منخفض عشان ما "يخترع"
            )
        )

        answer_text = getattr(response, "text", None) or str(response)
        return answer_text, sorted(list(chosen_sources))

if __name__ == "__main__":
    rag = ImtithalRAG()
    print("✅ نظام امتثال متصل بمحرك Gemini وقاعدة البيانات جاهزة.")

    while True:
        user_q = input("\n🤔 اسأل عن الأنظمة (أو اكتب 'خروج'): ")
        if user_q.lower() in ['خروج', 'exit', 'quit']:
            break
        if not user_q.strip():
            continue

        answer, docs = rag.ask(user_q)
        print(f"\n💡 الإجابة:\n{answer}")
        print(f"\n📚 المصادر المستخدمة: {docs}")


