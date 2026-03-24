import os
import pickle
import numpy as np

from sentence_transformers import CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from rank_bm25 import BM25Okapi
import ollama

from .config import *


class RAGService:
    def __init__(self):
        # ✅ FIX: Use LangChain embedding wrapper (NOT SentenceTransformer)
        self.embed_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        self.rerank_model = CrossEncoder(RERANK_MODEL)

        self.vector_store = None
        self.bm25_index = None
        self.corpus_tokens = []
        self.doc_metadata = []

        self.bm25_path = os.path.join(DATA_INDEX, "bm25_index.pkl")

        self.load_index()

    def load_index(self):
        if os.path.exists(DATA_INDEX) and os.path.exists(self.bm25_path):
            try:
                self.vector_store = FAISS.load_local(
                    DATA_INDEX,
                    self.embed_model,
                    allow_dangerous_deserialization=True
                )

                with open(self.bm25_path, "rb") as f:
                    bm25_data = pickle.load(f)
                    self.corpus_tokens = bm25_data["corpus"]
                    self.doc_metadata = bm25_data["metadata"]

                self.bm25_index = BM25Okapi(self.corpus_tokens)
                print("Index loaded successfully.")

            except Exception as e:
                print(f"Error loading index: {e}")
                self.vector_store = None
        else:
            print("⚠️ No index found. Please run indexer.py first.")

    def save_index(self):
        if self.vector_store:
            self.vector_store.save_local(DATA_INDEX)

            with open(self.bm25_path, "wb") as f:
                pickle.dump(
                    {
                        "corpus": self.corpus_tokens,
                        "metadata": self.doc_metadata
                    },
                    f
                )

    def ingest_file(self, file_path: str):
        # ✅ FIX: Encoding for txt
        if file_path.endswith('.txt'):
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            loader = PyPDFLoader(file_path)

        docs = loader.load()

        # ✅ FIX: Use Recursive splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        chunks = splitter.split_documents(docs)

        texts = [c.page_content.replace("\n", " ").strip() for c in chunks]

        metas = [
            {"source": os.path.basename(file_path), "chunk_id": i}
            for i in range(len(chunks))
        ]

        # ✅ FIX: FAISS API
        if self.vector_store:
            self.vector_store.add_texts(texts, metadatas=metas)
        else:
            self.vector_store = FAISS.from_texts(
                texts,
                self.embed_model,
                metadatas=metas
            )

        # BM25 update
        new_tokens = [text.split() for text in texts]
        self.corpus_tokens.extend(new_tokens)
        self.doc_metadata.extend(metas)
        self.bm25_index = BM25Okapi(self.corpus_tokens)

        self.save_index()
        return len(chunks)

    def hybrid_search(self, query: str):
        vec_docs = self.vector_store.similarity_search(query, k=TOP_N_INITIAL)

        query_tokens = query.split()
        bm25_scores = self.bm25_index.get_scores(query_tokens)

        top_indices = np.argsort(bm25_scores)[-TOP_N_INITIAL:][::-1]

        bm25_docs = []
        for idx in top_indices:
            if idx < len(self.corpus_tokens):
                content = " ".join(self.corpus_tokens[idx])

                bm25_docs.append(
                    type('obj', (object,), {
                        'page_content': content,
                        'metadata': self.doc_metadata[idx]
                    })
                )

        return vec_docs, bm25_docs

    def rerank_chunks(self, query: str, chunks):
        if not chunks:
            return []

        pairs = [[query, c.page_content] for c in chunks]

        try:
            scores = self.rerank_model.predict(pairs)
            ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
            return [c[0] for c in ranked[:TOP_K_FINAL]]
        except:
            return chunks[:TOP_K_FINAL]

    def generate_answer(self, query: str, context: str, history: list):
        system_prompt = """You are a construction assistant.
Answer ONLY from the provided context.
If not found, say: "I don't have information in the provided documents."
"""

        messages = [{'role': 'system', 'content': system_prompt}]
        messages.extend(history)

        messages.append({
            'role': 'user',
            'content': f"Context:\n{context}\n\nQuestion: {query}"
        })

        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=messages
            )
            return response['message']['content']

        except Exception as e:
            return f"LLM Error: {str(e)}"

    def process_query(self, query: str, history: list):
        if not self.vector_store or not self.bm25_index:
            return None, [], True

        vec_docs, bm25_docs = self.hybrid_search(query)

        candidates = vec_docs + bm25_docs

        final_chunks = self.rerank_chunks(query, candidates)

        context = "\n\n".join([c.page_content for c in final_chunks])

        sources = [
            {
                "text": c.page_content,
                "source": c.metadata.get("source", "unknown")
            }
            for c in final_chunks
        ]

        answer = self.generate_answer(query, context, history)

        return answer, sources, False


rag_service = RAGService()