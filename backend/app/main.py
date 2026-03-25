from http.client import HTTPException

from fastapi import FastAPI, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from .rag_service import rag_service
from .models import ChatRequest

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory history store (Use Redis for production)
chat_histories = {}

@app.post("/upload")
async def upload(file: UploadFile = File(...), session_id: str = Header(None)):
    file_path = f"data/uploads/{file.filename}"
    with open(file_path, "wb") as f: f.write(await file.read())
    
    chunks = rag_service.ingest_file(file_path, session_id)
    return {"chunks": chunks}

@app.post("/chat")
async def chat(request: ChatRequest, session_id: str = Header(None)):
    # 1. Clean up the session_id (FastAPI sometimes passes it as a string "None")
    if not session_id or session_id == "None":
        return {
            "answer": "Error: No session ID provided in headers.",
            "sources": [],
            "is_empty_state": True
        }

    # 2. Handle 'quit' command
    if request.query.lower() == "quit":
        if session_id in chat_histories:
            chat_histories[session_id] = []
        if session_id in rag_service.sessions:
            del rag_service.sessions[session_id]
        return {
            "answer": "Session reset. History cleared.", 
            "sources": [], 
            "is_empty_state": False 
        }

    # 3. Process the query
    # Note: We use the history sent from the frontend (request.history) 
    # to keep it stateless and consistent with your frontend logic.
    answer, sources, is_empty = rag_service.process_query(
        request.query, 
        request.history, 
        session_id
    )

    # 4. Return ALL fields required by ChatResponse
    return {
        "answer": answer,
        "sources": sources,
        "is_empty_state": is_empty
    }
@app.post("/reset-all")
async def reset_all():
    try:
        # Clear the RAG Service (FAISS + Sessions)
        rag_service.reset_all_data()
        
        # Clear Global Chat History
        global chat_histories
        chat_histories = {}
        
        return {"message": "All data cleared successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)