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
    try:
        file_ext = file.filename.split(".")[-1]
        unique_name = f"{uuid.uuid4()}.{file_ext}"
        file_path = os.path.join(DATA_UPLOADS, unique_name)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        chunks = rag_service.ingest_file(file_path)

        return {
            "message": "File processed successfully",
            "chunks_processed": chunks
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        answer, sources, is_empty = rag_service.process_query(
            request.query,
            request.history
        )

        if is_empty:
            return {
                "answer": "Please upload documents first to enable RAG responses.",
                "sources": [],
                "is_empty_state": True
            }

        return {
            "answer": answer,
            "sources": sources,
            "is_empty_state": False
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)