import os
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
import ollama
from .config import *

class RAGService:
    def __init__(self):
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)
        self.rerank_model = CrossEncoder(RERANK_MODEL)
        self.vector_store = None
        self.bm25_index = None
        self.corpus_tokens = []
        self.doc_metadata = [] # Store source info
        self.load_index()

    def load_index(self):
        if os.path.exists(DATA_INDEX):
            try:
                self.vector_store = FAISS.load_local(DATA_INDEX, self.embed_model, allow_dangerous_deserialization=True)
                # Note: BM25 needs to be rebuilt or saved separately. For simplicity, we rebuild BM25 on start if needed.
                # In a production app, save BM25 pickle. Here we assume vector store exists implies docs exist.
            except:
                self.vector_store = None

    def ingest_file(self, file_path: str):
        # 1. Load
        loader = TextLoader(file_path) if file_path.endswith('.txt') else PyPDFLoader(file_path)
        docs = loader.load()
        
        # 2. Chunk
        splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_documents(docs)
        
        # 3. Embed & Store Vector
        texts = [c.page_content for c in chunks]
        metas = [{"source": os.path.basename(file_path), "id": i} for i, c in enumerate(chunks)]
        
        if self.vector_store:
            self.vector_store.add_texts(texts, metas)
        else:
            self.vector_store = FAISS.from_texts(texts, self.embed_model, metas)
        
        # 4. Update BM25 (Simple implementation: rebuild corpus from all known texts)
        # In production, append to BM25. Here we retrieve all texts from FAISS to rebuild BM25
        all_docs = self.vector_store.similarity_search("", k=10000) # Fetch all
        self.corpus_tokens = [doc.page_content.split() for doc in all_docs]
        self.doc_metadata = [{"source": doc.metadata["source"]} for doc in all_docs]
        self.bm25_index = BM25Okapi(self.corpus_tokens)
        
        # 5. Save
        self.vector_store.save_local(DATA_INDEX)
        return len(chunks)

    def generate_multi_queries(self, query: str):
        prompt = f"""Generate 3 distinct variations of the following question to improve search retrieval. 
        Original: {query}
        Output only the 3 variations, one per line."""
        response = ollama.chat(model=OLLAMA_MODEL, messages=[{'role': 'user', 'content': prompt}])
        variations = response['message']['content'].strip().split('\n')
        return [query] + variations[:3]

    def hybrid_search(self, query: str):
        # Vector Search
        vec_docs = self.vector_store.similarity_search(query, k=TOP_N_INITIAL)
        
        # Keyword Search (BM25)
        query_tokens = query.split()
        bm25_scores = self.bm25_index.get_scores(query_tokens)
        top_indices = np.argsort(bm25_scores)[-TOP_N_INITIAL:][::-1]
        
        bm25_docs = []
        for idx in top_indices:
            if idx < len(self.corpus_tokens):
                # Reconstruct doc object roughly
                content = " ".join(self.corpus_tokens[idx])
                bm25_docs.append(type('obj', (object,), {'page_content': content, 'metadata': self.doc_metadata[idx]}))
        
        return vec_docs, bm25_docs

    def reciprocal_rank_fusion(self, vec_docs, bm25_docs, k=60):
        # RRF Implementation
        score_map = {}
        for i, doc in enumerate(vec_docs):
            key = doc.page_content[:100] # Simple hash
            score_map[key] = score_map.get(key, 0) + 1 / (k + i)
            if key not in score_map: # Store doc ref
                score_map[key] = {'score': 0, 'doc': doc}
            score_map[key]['score'] = score_map.get(key, 0) + 1 / (k + i)
            
        # Simplified RRF for demo stability
        all_docs = vec_docs + bm25_docs
        # Remove duplicates based on content
        unique_docs = []
        seen = set()
        for d in all_docs:
            if d.page_content[:50] not in seen:
                seen.add(d.page_content[:50])
                unique_docs.append(d)
        return unique_docs[:TOP_M_RRF]

    def rerank_chunks(self, query: str, chunks):
        if not chunks: return []
        pairs = [[query, c.page_content] for c in chunks]
        scores = self.rerank_model.predict(pairs)
        scored_chunks = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [c[0] for c in scored_chunks[:TOP_K_FINAL]]

    def generate_answer(self, query: str, context: str, history: list):
        system_prompt = """You are a construction assistant. Answer ONLY based on the provided context. 
        If the answer is not in the context, say "I don't have information on that in the provided documents."
        Do not use general knowledge."""
        
        messages = [{'role': 'system', 'content': system_prompt}]
        for h in history:
            messages.append(h)
        messages.append({'role': 'user', 'content': f"Context:\n{context}\n\nQuestion: {query}"})
        
        response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
        return response['message']['content']

    def process_query(self, query: str, history: list):
        if not self.vector_store or not self.bm25_index:
            return None, [], True
        
        # 1. Multi-Query
        queries = self.generate_multi_queries(query)
        
        # 2. Retrieval & Fusion
        all_candidates = []
        for q in queries:
            vec, bm25 = self.hybrid_search(q)
            fused = self.reciprocal_rank_fusion(vec, bm25)
            all_candidates.extend(fused)
        
        # 3. Rerank
        final_chunks = self.rerank_chunks(query, all_candidates)
        
        # 4. Generate
        context_text = "\n\n".join([c.page_content for c in final_chunks])
        sources = [{"text": c.page_content, "source": c.metadata.get("source", "unknown")} for c in final_chunks]
        answer = self.generate_answer(query, context_text, history)
        
        return answer, sources, False

rag_service = RAGService()