import os
import time 
from dotenv import load_dotenv
from google import genai 
from utils import embedding_model, get_chroma_collection, index_files 

load_dotenv()

# إعداد عميل Gemini
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY"),
    http_options={'api_version': 'v1alpha'}
)

# 1. مجموعة الأنظمة السعودية (دائمة)
collection = get_chroma_collection()

# 2. إنشاء/استدعاء مجموعة ملفات الشركة (ديناميكية)
# قمنا بإضافة "company_data" كمجموعة ثانية مستقلة
import chromadb
chroma_client = chromadb.PersistentClient(path="./chroma_db")
company_collection = chroma_client.get_or_create_collection(name="company_data")

def ask_imtithal_integrated(query, company_col):
    try:
        query_vec = embedding_model.encode(query).tolist()
        
        # البحث في الأنظمة الرسمية
        reg_results = collection.query(query_embeddings=[query_vec], n_results=5)
        reg_context = "\n\n".join(reg_results['documents'][0])
        
        # البحث في ملفات الشركة (إذا وجدت)
        comp_context = "لا توجد ملفات مرفقة للشركة حالياً."
        if company_col.count() > 0:
            comp_results = company_col.query(query_embeddings=[query_vec], n_results=3)
            comp_context = "\n\n".join(comp_results['documents'][0])
        
        integrated_prompt = f"""
أنت 'إمتثال AI'، المستشار الاستراتيجي المتخصص في أنظمة سدايا والهيئة الوطنية للأمن السيبراني (SDAIA & NCA).  
مهمتك: الإجابة بدقة فقط بناءً على المعلومات الموجودة في قاعدة المعرفة المقدمة ({reg_context}).

🔹 تعليمات دقيقة:
- استخدم فقط المعلومات الواردة في {reg_context}.
- إذا لم تجد إجابة في المعلومات المتوفرة، اكتب صراحة: "لا توجد معلومات متاحة حول هذا الموضوع في قاعدة المعرفة".
- لا تستخدم أي مصادر خارجية، ولا تفترض أي شيء.
- حدد أي التزامات أو قواعد مرتبطة بسؤال المستخدم بوضوح.
- قدم توصية عملية قابلة للتنفيذ فقط إذا كانت مدعومة بالمعلومات المتوفرة.
- استخدم اللغة العربية الرسمية، وكن موجزًا وواضحًا.

سؤال المستخدم:
{query}
"""

        
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=integrated_prompt
        )
        return response.text

    except Exception as e:
        return f"خطأ في الدمج: {str(e)}"

if __name__ == "__main__":
    print("\n" + "="*50)
    print("نظام إمتثال المدمج - جاهز (V2.5)")
    print("="*50)
    
    choice = input("هل تريد إعادة قراءة ملفات الأنظمة الرسمية؟ (y/n): ")
    if choice.lower() == 'y':
        index_files("data", collection)

    print("\nالحالة: متصل بالأنظمة وملفات الشركة. اكتب 'exit' للخروج.")
    
    while True:
        user_input = input("\nاسأل إمتثال (عن الأنظمة أو وضع شركتك): ")
        
        if user_input.lower() in ['exit', 'quit', 'خروج']:
            break
        
        if not user_input.strip(): continue

        print("تهدئة الاتصال لضمان استقرار الخدمة...")
        time.sleep(3) 
        
        print("جاري التحليل المدمج...")
        # تأكدي من تمرير الـ company_collection هنا
        answer = ask_imtithal_integrated(user_input, company_collection)
        
        print("\n" + "-"*30)
        print(f"رد إمتثال AI المدمج:\n{answer}")
        print("-"*30)