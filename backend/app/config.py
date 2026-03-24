CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_N_INITIAL = 5       # Per query per method
TOP_M_RRF = 15          # After Fusion
TOP_K_FINAL = 5         # After Reranking
MULTI_QUERY_COUNT = 3   # Variations to generate

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-small"
OLLAMA_MODEL = "phi3"

DATA_RAW = "data/raw"
DATA_UPLOADS = "data/uploads"
DATA_INDEX = "data/index"