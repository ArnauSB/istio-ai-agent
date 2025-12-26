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

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
CHROMA_PATH = cfg["database"]["chroma_path"]
COLLECTION_NAME = cfg["database"]["collection_name"]
CACHE_FILE = cfg["app"]["cache_file"]
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", cfg["models"]["ollama_url"])
MODEL_NAME = cfg["models"]["llm_model"]

def get_embedding_model():
    model_name = cfg["models"]["embedding_model"]
    print(f"🔄 Cargando modelo de embeddings: {model_name}")
    return HuggingFaceEmbedding(model_name=model_name)

# Settings.embed_model = get_embedding_model()
# Settings.chunk_size = cfg["splitter"]["chunk_size"]
