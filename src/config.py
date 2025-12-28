import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "documents"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
RETRIEVER_K = 5