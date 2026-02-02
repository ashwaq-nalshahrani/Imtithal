from pydantic import BaseModel
from typing import List, Optional

# شكل الطلب القادم من الفرونت إند (السؤال)
class QueryRequest(BaseModel):
    question: str

# شكل الرد الذي سيرسل للفرونت إند (الإجابة والمصادر)
class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[str]] = []