import os
import pickle

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from rank_bm25 import BM25Okapi

from .config import *


def build_initial_index():
    print(f"Starting Initial Indexing using {EMBEDDING_MODEL}...")

    #Initialize Embedding Model (FIXED)
    embed_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    #Load Documents
    documents = []
    raw_path = DATA_RAW

    if not os.path.exists(raw_path):
        os.makedirs(raw_path)
        print(f"{raw_path} not found. Created empty folder. Please add documents there.")
        return

    for filename in os.listdir(raw_path):
        file_path = os.path.join(raw_path, filename)

        try:
            if filename.endswith('.txt'):
                loader = TextLoader(file_path, encoding="utf-8")

            elif filename.endswith('.pdf'):
                loader = PyPDFLoader(file_path)

            else:
                continue

            docs = loader.load()

            # 🔥 Clean text (important for embeddings)
            for doc in docs:
                doc.page_content = doc.page_content.replace("\n", " ").strip()

            documents.extend(docs)
            print(f"Loaded: {filename}")

        except Exception as e:
            print(f"Error loading {filename}: {e}")

    if not documents:
        print("No documents found in data/raw/. Cannot build index.")
        return

    #Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    #Prepare texts + metadata
    texts = [c.page_content for c in chunks]

    metas = [
        {
            "source": c.metadata.get("source", "unknown"),
            "chunk_id": i
        }
        for i, c in enumerate(chunks)
    ]

    #FAISS Index
    vector_store = FAISS.from_texts(
        texts,
        embed_model,
        metadatas=metas
    )

    # BM25 Index
    corpus_tokens = [text.split() for text in texts]
    bm25_index = BM25Okapi(corpus_tokens)

    #Save Index
    os.makedirs(DATA_INDEX, exist_ok=True)

    vector_store.save_local(DATA_INDEX)

    bm25_path = os.path.join(DATA_INDEX, "bm25_index.pkl")

    with open(bm25_path, "wb") as f:
        pickle.dump(
            {
                "bm25": bm25_index,
                "corpus": corpus_tokens,
                "metadata": metas
            },
            f
        )

    print(f"FAISS index saved to: {DATA_INDEX}")
    print(f"BM25 index saved to: {bm25_path}")


if __name__ == "__main__":
    build_initial_index()