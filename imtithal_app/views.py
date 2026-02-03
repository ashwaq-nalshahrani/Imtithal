from django.shortcuts import render
from rag.rag_engine import ImtithalRAG
import os

rag_instance = ImtithalRAG(
    db_path="./chroma_db",
    collection_name="imtithal_docs"
)

def home(request):
    # Always start with a fresh chat history
    chat_history = []

    if request.method == "POST":
        question = request.POST.get("question", "").strip()

        if question:
            # Save user message
            chat_history.append({
                "role": "user",
                "content": question
            })

            # Ask RAG
            result = rag_instance.ask(question)

            answer = result.get("answer", "")
            sources = result.get("sources", [])

            # Save assistant message
            chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

    return render(request, "home.html", {
        "chat_history": chat_history,
        "sdaia_pdfs": get_sdaia_pdfs(),
        "nca_pdfs": get_nca_pdfs(),
    })


# helpers
def get_sdaia_pdfs():
    path = "imtithal_app/media/SDAIA"
    return sorted([f for f in os.listdir(path) if f.lower().endswith(".pdf")])

def get_nca_pdfs():
    path = "imtithal_app/media/NCA"
    return sorted([f for f in os.listdir(path) if f.lower().endswith(".pdf")])
