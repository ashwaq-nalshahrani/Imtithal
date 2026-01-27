import os
import google.generativeai as genai
from dotenv import load_dotenv
# استيراد الأدوات من الملف الخاص بك
from utils import get_chroma_collection, embedding_model, clean_arabic_pro

load_dotenv()

# ===========================
# إعداد Gemini
# ===========================
genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) # تأكدي أن هذا هو اسم المتغير في ملف الـ .env
model = genai.GenerativeModel('models/gemini-2.5-flash') # أو gemini-2.0-flash حسب المتاح لك

class ImtithalRAG:
    def __init__(self):
        self.collection = get_chroma_collection()

    def ask(self, query):
        # 1. تنظيف السؤال وتحويله إلى Vector (باستخدام BGE-M3 من الـ utils)
        cleaned_query = clean_arabic_pro(query)
        query_embedding = embedding_model.encode([cleaned_query]).tolist()

        # 2. البحث في ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=5,
            include=["documents", "metadatas"]
        )

        # تجهيز السياق والمصادر
        context = "\n\n".join(results["documents"][0])
        sources = list(set([os.path.basename(m["source"]) for m in results["metadatas"][0]]))

        # 3. صياغة الـ Prompt لـ Gemini
        prompt = f"""
        أنت محرر نصوص امتثال، ولست محللًا ولا ملخصًا.
        
        ⚠️ قواعد إلزامية (يجب الالتزام بها حرفيًا):
        
        1) استخدم فقط النص الموجود داخل "السياق" أدناه.
        2) يُمنع منعًا باتًا:
           - إضافة أي معلومة غير موجودة في السياق.
           - حذف أي جزء من النص ذي الصلة.
           - تلخيص، إعادة صياغة، أو تفسير النص.
           - دمج معلومات من مصادر أخرى أو من معرفتك العامة.
        3) المطلوب منك فقط:
           - تنظيف النص (تنسيق الأسطر، المسافات، الترقيم).
           - تصحيح مشاكل الترميز أو انعكاس العربية إن وُجدت.
           - الحفاظ على النص كما هو من حيث المعنى والمحتوى.
        4) إذا لم تجد نصًا في السياق يجيب على السؤال، أرجع الجملة التالية حرفيًا فقط دون أي إضافة:
           "عذراً، هذه المعلومة غير متوفرة في الأدلة الحالية".
        
        📌 شكل الإخراج:
        - أعرض النص المنظف فقط.
        - لا تضف عناوين، ولا شروح، ولا نقاط تعداد من عندك.
        - لا تذكر أنك نموذج ذكاء اصطناعي.
        
        السياق (النص المصدر الوحيد المسموح لك باستخدامه):
        {context}
        
        السؤال:
        {query}
        """


        # 4. توليد الإجابة من Gemini
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1 # لضمان الدقة وعدم التوقع
            )
        )

        return response.text, sources

# ===========================
# التشغيل في التيرمنال
# ===========================
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