import os
import yaml
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# Load env
load_dotenv()

# Load YAML
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

cfg = load_config()

# --- ENVIRONMENT VARIABLES ---
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# --- DATABASE CONFIG ---
CHROMA_PATH = cfg["database"]["chroma_path"]
COLLECTION_NAME = cfg["database"]["collection_name"]
STORAGE_NODES_PATH = cfg['database'].get('storage_nodes_path', './storage_nodes')

# --- APP CONFIG ---
CACHE_FILE = cfg["app"]["cache_file"]

# --- DEBUG CONFIG ---
# We use .get() with defaults to prevent errors if the key is missing
DEBUG_ENABLED = cfg.get("debug", {}).get("enabled", False)
DEBUG_OUTPUT_DIR = cfg.get("debug", {}).get("output_dir", "debug_converted")

# --- MODEL CONFIG ---
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", cfg["models"]["ollama_url"])
MODEL_NAME = cfg["models"]["llm_model"]

# --- SPLITTER CONFIG ---
CHUNK_SIZE = cfg["splitter"]["chunk_size"]
CHUNK_OVERLAP = cfg["splitter"]["chunk_overlap"]

def get_embedding_model():
    model_name = cfg["models"]["embedding_model"]
    print(f"Loading embeddings: {model_name}")
    return HuggingFaceEmbedding(model_name=model_name, trust_remote_code=True)
