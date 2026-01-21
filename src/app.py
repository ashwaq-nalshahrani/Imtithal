import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
# استدعاء الدوال من utils (تأكدي أنكِ أضفتِ دالة حساب النسبة في utils.py)
from utils import embedding_model, get_chroma_collection, calculate_compliance_score
from datetime import datetime

# 1. إعدادات الصفحة
load_dotenv()
st.set_page_config(page_title="إمتثال - AI Compliance", layout="wide", page_icon="🛡️")

# 2. تهيئة الاتصالات
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
collection = get_chroma_collection()

# 3. وظيفة البحث والتحليل المدمج
def get_compliance_analysis(user_query, context_from_db):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
    أنت 'إمتثال AI' - المستشار الاستراتيجي. 
    بناءً على الأنظمة السعودية الرسمية التالية:
    {context_from_db}
    
    أجب على سؤال المستخدم أو حلل ملفه: {user_query}
    - وضح أي 'فجوة عدم امتثال'.
    - قدم توصية عملية.
    """
    response = model.generate_content(prompt)
    return response.text

# --- الواجهة ---
st.title("🛡️ مستشارك الذكي للامتثال (V2.5)")

# منطقة الرفع والتحليل
with st.sidebar:
    st.header("📜 سجل التقارير")
    if "analysis_reports" not in st.session_state:
        st.session_state.analysis_reports = []
    
    if not st.session_state.analysis_reports:
        st.info("لا توجد تقارير بعد.")
    for report in st.session_state.analysis_reports:
        with st.expander(f"📁 {report['filename']}"):
            st.caption(report['date'])
            st.write(report['summary'])

uploaded_file = st.file_uploader("ارفع ملف سياسة الشركة (PDF)", type=['pdf'])

if uploaded_file:
    # --- هذا هو الجزء الذي سألتِ عنه (تم تحديثه بالكود الجديد) ---
    if st.button("بدء تحليل الامتثال"):
        with st.spinner("جاري جلب الأنظمة وتحليل الفجوات وحساب النسبة..."):
            
            # 1. جلب الأنظمة ذات الصلة من ChromaDB
            query_vec = embedding_model.encode(uploaded_file.name).tolist()
            reg_results = collection.query(query_embeddings=[query_vec], n_results=5)
            reg_context = "\n\n".join(reg_results['documents'][0])
            
            # 2. توليد التحليل الحقيقي
            full_prompt = f"""
            أنت مستشار قانوني سعودي. قارن ملف الشركة المرفق (اسم الملف: {uploaded_file.name}) مع الأنظمة التالية:
            {reg_context}
            المطلوب: استخراج الفجوات وتقديم توصيات واضحة.
            """
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(full_prompt)
            real_report = response.text
            
            # 3. حساب النسبة المئوية عبر الدالة الجديدة في utils
            compliance_perc = calculate_compliance_score(real_report)
            
            # 4. عرض النتائج والنسبة
            st.subheader("📊 نتائج تحليل الامتثال")
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.metric(label="نسبة الامتثال", value=f"{compliance_perc}%")
            
            with col2:
                st.progress(compliance_perc / 100)
            
            if compliance_perc < 50:
                st.error("⚠️ مستوى الامتثال منخفض - يتطلب إجراءات تصحيحية عاجلة.")
            elif compliance_perc < 80:
                st.warning("🟠 مستوى الامتثال متوسط - يوجد فجوات تحتاج لمعالجة.")
            else:
                st.success("✅ مستوى الامتثال مرتفع.")

            st.markdown("### 📝 التقرير التفصيلي")
            st.info(real_report)
            
            # 5. حفظ في السجل
            st.session_state.analysis_reports.append({
                "filename": uploaded_file.name,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "summary": f"النسبة: {compliance_perc}% \n {real_report[:150]}..."
            })
    # --- نهاية الجزء المحدث ---

# واجهة الشات (الاستشارة الفورية)
st.divider()
if "messages" not in st.session_state: st.session_state.messages = []

st.subheader("💬 استشارة فورية")
for message in st.session_state.messages:
    with st.chat_message(message["role"]): st.markdown(message["content"])

if prompt := st.chat_input("اسأل عن أي نظام قانوني..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    
    query_vec = embedding_model.encode(prompt).tolist()
    results = collection.query(query_embeddings=[query_vec], n_results=3)
    context = "\n\n".join(results['documents'][0])
    
    answer = get_compliance_analysis(prompt, context)
    with st.chat_message("assistant"): st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})