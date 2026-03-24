from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import ChatRequest, ChatResponse, UploadResponse
from .rag_service import rag_service
from .config import DATA_UPLOADS
import os
import uuid

app = FastAPI()

# CORS for Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(DATA_UPLOADS, exist_ok=True)

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    file_path = os.path.join(DATA_UPLOADS, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    try:
        chunks = rag_service.ingest_file(file_path)
        return {"message": "File processed successfully", "chunks_processed": chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    answer, sources, is_empty = rag_service.process_query(request.query, request.history)
    
    if is_empty:
        return {
            "answer": "I am specialized in RAG, please first upload any document to answer queries.",
            "sources": [],
            "is_empty_state": True
        }
    
    return {
        "answer": answer,
        "sources": sources,
        "is_empty_state": False
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)