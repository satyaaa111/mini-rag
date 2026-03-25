import os
import uuid
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import ollama

from .config import *
from .history_aware import get_standalone_query, generate_multi_queries

class RAGService:
    def __init__(self):
        self.embed_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.rerank_model = CrossEncoder(RERANK_MODEL)
        self.vector_store = None
        # In-memory storage for session-specific BM25 and metadata
        self.sessions = {} 

    def ingest_file(self, file_path: str, session_id: str):
        loader = TextLoader(file_path, encoding="utf-8") if file_path.endswith('.txt') else PyPDFLoader(file_path)
        chunks = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).split_documents(loader.load())
        
        texts = [c.page_content.replace("\n", " ").strip() for c in chunks]
        metas = [{"source": os.path.basename(file_path), "session_id": session_id} for _ in texts]

        # Add to FAISS (Persistent/Global but filtered by session_id)
        if self.vector_store is None:
            self.vector_store = FAISS.from_texts(texts, self.embed_model, metadatas=metas)
        else:
            self.vector_store.add_texts(texts, metadatas=metas)

        # Update Session-specific BM25
        if session_id not in self.sessions:
            self.sessions[session_id] = {"tokens": [], "metas": []}
        
        new_tokens = [t.split() for t in texts]
        self.sessions[session_id]["tokens"].extend(new_tokens)
        self.sessions[session_id]["metas"].extend(metas)
        self.sessions[session_id]["bm25"] = BM25Okapi(self.sessions[session_id]["tokens"])
        
        return len(chunks)

    # def hybrid_retrieval(self, query: str, session_id: str):
    #     # 1. Vector Search (Filtered by session)
    #     vec_results = self.vector_store.similarity_search(
    #         query, k=TOP_N_INITIAL, filter={"session_id": session_id}
    #     )
        
    #     # 2. BM25 Search
    #     session_data = self.sessions.get(session_id)
    #     if not session_data or "bm25" not in session_data:
    #         return vec_results

    #     scores = session_data["bm25"].get_scores(query.split())
    #     if np.max(scores) == 0 and not vec_results:
    #         return []
    #     top_indices = np.argsort(scores)[-TOP_N_INITIAL:][::-1]
    #     bm25_results = []
    #     for idx in top_indices:
    #         if session_data["metas"][idx]["session_id"] == session_id:
    #             bm25_results.append(type('obj', (object,), {
    #                 'page_content': " ".join(session_data["tokens"][idx]),
    #                 'metadata': session_data["metas"][idx]
    #             }))
        
    #     # 3. Manual Weighted Ensemble (Simple version of LangChain Ensemble)
    #     # For simplicity in this logic, we combine and deduplicate
    #     combined = {res.page_content: res for res in (vec_results + bm25_results)}
    #     return list(combined.values())
    def hybrid_retrieval(self, query: str, session_id: str):
    # 1. Vector search with scores
        vec_results = self.vector_store.similarity_search_with_score(
            query, k=TOP_N_INITIAL, filter={"session_id": session_id}
        )

        # Normalize vector scores (convert distance → similarity if needed)
        vec_dict = {}
        for doc, score in vec_results:
            vec_dict[doc.page_content] = {
                "doc": doc,
                "vec_score": 1 / (1 + score)  # simple normalization
            }

        # 2. BM25 search
        session_data = self.sessions.get(session_id)
        if not session_data or "bm25" not in session_data:
            return [v["doc"] for v in vec_dict.values()]

        scores = session_data["bm25"].get_scores(query.split())

        bm25_dict = {}
        for idx, score in enumerate(scores):
            if session_data["metas"][idx]["session_id"] == session_id:
                content = " ".join(session_data["tokens"][idx])
                bm25_dict[content] = score

        # Normalize BM25 scores
        max_bm25 = max(bm25_dict.values()) if bm25_dict else 1
        for k in bm25_dict:
            bm25_dict[k] /= max_bm25

        # 3. Combine with weights
        final_results = {}

        all_keys = set(vec_dict.keys()) | set(bm25_dict.keys())

        for key in all_keys:
            vec_score = vec_dict.get(key, {}).get("vec_score", 0)
            bm25_score = bm25_dict.get(key, 0)

            final_score = 0.7 * vec_score + 0.3 * bm25_score

            doc = vec_dict.get(key, {}).get("doc") or type('obj', (object,), {
                'page_content': key,
                'metadata': {}
            })

            final_results[key] = (doc, final_score)

        # 4. Sort by final score
        ranked = sorted(final_results.values(), key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in ranked[:TOP_N_INITIAL]]

    def rrf_merge(self, query_results_list):
        # Reciprocal Rank Fusion
        fused_scores = {}
        k = 60 # Standard constant for RRF
        for results in query_results_list:
            for rank, doc in enumerate(results):
                content = doc.page_content
                if content not in fused_scores:
                    fused_scores[content] = [doc, 0]
                fused_scores[content][1] += 1 / (rank + k)
        
        reranked = sorted(fused_scores.values(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in reranked[:TOP_M_RRF]]

    def process_query(self, query: str, history: list, session_id: str):
        # 🚨 EXIT EARLY if no documents uploaded for THIS session
        if session_id not in self.sessions:
            return "Please upload documents for this session first.", [], True

        standalone = get_standalone_query(query, history)
        queries = generate_multi_queries(standalone, count=MULTI_QUERY_COUNT)
        all_query_results = [self.hybrid_retrieval(q, session_id) for q in queries]
        merged_candidates = self.rrf_merge(all_query_results)
        
        if not merged_candidates:
            return "I don't have information in the provided documents.", [], False

        # 3. Reranking with RELEVANCE THRESHOLD
        pairs = [[standalone, c.page_content] for c in merged_candidates]
        scores = self.rerank_model.predict(pairs)
        
        # Filter by score
        # Adjustable based on model's sensitivity
        RELEVANCE_THRESHOLD = 0.08
        
        scored_results = sorted(zip(merged_candidates, scores), key=lambda x: x[1], reverse=True)

        print("\n===== RERANK DEBUG =====")
        for doc, score in scored_results[:4]:   # top 4
            print(f"Score: {score:.4f}")
            print(f"Text: {doc.page_content[:200]}")
            print("------------------------")

        top_chunks = [item[0] for item in scored_results][:TOP_K_FINAL]

        # if not top_chunks:
        #     return "I am designed to assist you on relevant questions. Please ask within the context of the documents provided.", [], False

        context = "\n\n".join([c.page_content for c in top_chunks])
        answer = self.generate_answer(standalone, context, history)

        sources = [{"text": c.page_content, "source": c.metadata.get("source")} for c in top_chunks]
        return answer, sources, False

    def generate_answer(self, query: str, context: str, history: list):
        system_prompt = """ROLE: You are a strict document assistant. Follow the RULES in every answer you generate.
                            RULES:
                            1. Use ONLY the provided Context to answer.
                            2. If the user greets you, respond politely but do not retrieve documents.
                            3. If the Context does not contain the answer, or the question is unrelated to the context, 
                            state: "I am designed to assist you on relevant questions. Please ask within the context of the documents provided."
                            4. Do NOT use your internal knowledge to answer general questions (e.g., about animals, food, music, films, sexual things, philosophy, general politics, physics, chemistry, mathemetics etc).
                            5. Answer should not be long unnecessarily. 
                            6. Avoid Unsupported claims that is if provided data does not give enough information to answer then do not give gibberish or forceful answers.
                            """
        messages = [{'role': 'system', 'content': system_prompt}] + history
        messages.append({'role': 'user', 'content': f"Context:\n{context}\n\nQuestion: {query}"})
        
        response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
        return response['message']['content']
    
    def reset_all_data(self):
        """Wipes all persistent and in-memory data."""
        # 1. Clear In-Memory Data
        self.sessions = {}
        self.vector_store = None
        
        # 2. Delete Persistent Folders
        folders_to_clear = [DATA_INDEX, DATA_UPLOADS, DATA_RAW]
        for folder in folders_to_clear:
            if os.path.exists(folder):
                try:
                    # Remove folder and recreate it empty
                    shutil.rmtree(folder)
                    os.makedirs(folder, exist_ok=True)
                except Exception as e:
                    print(f"Error clearing {folder}: {e}")
        
        print("☢️ Global Data Reset Complete.")
        return True

rag_service = RAGService()