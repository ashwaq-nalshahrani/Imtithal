import streamlit as st
import sys
import io

from rag_new import ImtithalRAG

# --------------------------------------------------
# Streamlit setup
# --------------------------------------------------
st.set_page_config(
    page_title="SDAIA Imtithal RAG",
    layout="wide"
)

st.title("🧠 SDAIA Imtithal RAG Assistant")
st.caption("Smart Compliance Assistant – Arabic RAG")

# --------------------------------------------------
# Initialize RAG once
# --------------------------------------------------
if "rag" not in st.session_state:
    with st.spinner("🚀 Initializing RAG..."):
        st.session_state.rag = ImtithalRAG(
            db_path="./chroma_db",
            collection_name="imtithal_docs"
        )

if "chat" not in st.session_state:
    st.session_state.chat = []

# --------------------------------------------------
# Input
# --------------------------------------------------
question = st.text_input("🤔 اكتب سؤالك بالعربية:")

ask_btn = st.button("إرسال")

# --------------------------------------------------
# Ask RAG
# --------------------------------------------------
if ask_btn and question.strip():
    # Capture prints (keeps ALL messages your code prints)
    log_buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = log_buffer

    try:
        response = st.session_state.rag.ask(
            question,
            verbose=True  # KEEP VERBOSE OUTPUT
        )
    finally:
        sys.stdout = old_stdout

    logs = log_buffer.getvalue()

    st.session_state.chat.append({
        "question": question,
        "answer": response["answer"],
        "sources": response.get("sources", []),
        "logs": logs
    })

# --------------------------------------------------
# Display chat
# --------------------------------------------------
for turn in st.session_state.chat[::-1]:
    with st.chat_message("user"):
        st.markdown(turn["question"])

    with st.chat_message("assistant"):
        st.markdown(turn["answer"])

        if turn["sources"]:
            with st.expander(f"📚 المصادر ({len(turn['sources'])})"):
                for i, src in enumerate(turn["sources"], 1):
                    st.markdown(f"**{i}. {src['file_name']}**")
                    if src.get("subject"):
                        st.markdown(f"- الموضوع: {src['subject']}")

        with st.expander("🖨️ Logs (Debug)"):
            st.code(turn["logs"], language="text")

# --------------------------------------------------
# Clear chat
# --------------------------------------------------
if st.button("🧹 مسح المحادثة"):
    st.session_state.chat.clear()
    st.session_state.rag.chat_history.clear()
    st.experimental_rerun()
