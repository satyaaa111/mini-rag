from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    query: str
    session_id: str
    history: List[dict] = []  # [{"role": "user", "content": "..."}, ...]

class UploadResponse(BaseModel):
    message: str
    chunks_processed: int

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]  # [{ "text": "...", "source": "file.txt" }]
    is_empty_state: bool