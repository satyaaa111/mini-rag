CHUNK_SIZE = 350
CHUNK_OVERLAP = 50
TOP_N_INITIAL = 4 #After hybrid search
TOP_M_RRF = 10 #After reciprocal rank fusion and before re-ranking
TOP_K_FINAL = 4 #after re-ranking
MULTI_QUERY_COUNT = 3

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OLLAMA_MODEL = "phi3"

DATA_RAW = "data/raw"
DATA_UPLOADS = "data/uploads"
DATA_INDEX = "data/index"