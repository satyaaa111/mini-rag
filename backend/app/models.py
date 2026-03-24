from pydantic import BaseModel
from typing import List, Optional

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    is_empty_state: bool

class UploadResponse(BaseModel):
    message: str
    chunks_processed: int

class ChatRequest(BaseModel):
    query: str
    history: List[dict]