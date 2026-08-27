import os

# global configuration options
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "huggingface").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")

try:
    LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "15"))
except ValueError:
    LLM_TIMEOUT = 15.0

try:
    LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
except ValueError:
    LLM_MAX_RETRIES = 2

# Path settings
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_CSV_PATH = os.path.join(BASE_DIR, "Facets Assignment.csv")
ENRICHED_CSV_PATH = os.path.join(DATA_DIR, "facets_enriched.csv")

# Local fallback model settings
LOCAL_FALLBACK_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Retrieval settings
RETRIEVAL_K = 10
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
