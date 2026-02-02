import sys
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Any

# إضافة المجلد الرئيسي للمشروع إلى مسار النظام لضمان العثور على الموديلات
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# الآن نستدعي المحرك باستخدام المسار الصحيح
from core.rag_engine import rag_instance 

app = FastAPI(title="Imtithal API")
origins = [
    "http://localhost:5173",                            # للمعاينة المحلية أثناء التطوير
    "https://imtithal-ai-guardian.lovable.app",        # رابط موقعك المنشور على Lovable
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[Any]

@app.post("/ask", response_model=ChatResponse)
async def ask_question(request: ChatRequest):
    try:
        # تنفيذ الاستعلام عبر المحرك
        result = rag_instance.ask_simple(request.question)
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        print(f"❌ Error in RAG Pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ready"}