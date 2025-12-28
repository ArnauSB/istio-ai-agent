import os
import sys
import logging
import re
import uvicorn

from typing import List, Optional, Dict
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import nest_asyncio
import chromadb
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter, FilterCondition

# --- IMPORTS FOR HYBRID SEARCH & CHAT ENGINE ---
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.chat_engine import ContextChatEngine

import config
import prompts

os.environ["TOKENIZERS_PARALLELISM"] = "false"
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- GLOBAL STATE ---
vector_index = None 
bm25_retriever = None
# This dictionary stores the MEMORY for each user, ensuring context is kept
session_store: Dict[str, ChatMemoryBuffer] = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str

class Source(BaseModel):
    repo: str
    file: str
    url: Optional[str] = None
    score: float

class ChatResponse(BaseModel):
    response: str
    sources: List[Source] = []

# --- CUSTOM HYBRID RETRIEVER ---
class CustomHybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever, target_version):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.target_version = target_version
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # Get Vector Results (Filtered by Chroma logic)
        vector_nodes = self.vector_retriever.retrieve(query_bundle)
        
        # Get BM25 Results (Unfiltered)
        bm25_nodes = self.bm25_retriever.retrieve(query_bundle)
        
        # Filter BM25 Results manually
        filtered_bm25_nodes = []
        for node in bm25_nodes:
            ver = node.node.metadata.get("system_version", "unknown")
            if ver == self.target_version or ver == "any":
                filtered_bm25_nodes.append(node)
        
        # Combine Results (RRF Algorithm)
        combined_dict = {}
        
        # Process Vector (Baseline)
        for rank, node in enumerate(vector_nodes):
            node.score = node.score or 0.0
            combined_dict[node.node.node_id] = node
            
        # Process BM25 (Boost)
        for rank, node in enumerate(filtered_bm25_nodes):
            if node.node.node_id in combined_dict:
                combined_dict[node.node.node_id].score += 0.2
            else:
                node.score = 0.5 - (rank * 0.01) 
                combined_dict[node.node.node_id] = node
                
        final_nodes = list(combined_dict.values())
        final_nodes.sort(key=lambda x: x.score, reverse=True)
        return final_nodes[:5]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_index, bm25_retriever
    logger.info("--- Starting API Server ---")
    
    try:
        Settings.embed_model = config.get_embedding_model()
        Settings.llm = Ollama(
            model=config.MODEL_NAME, 
            base_url=config.OLLAMA_URL, 
            request_timeout=300.0,
            temperature=0.1 
        )

        logger.info(f"Connecting to ChromaDB at {config.CHROMA_PATH}...")
        db = chromadb.PersistentClient(path=config.CHROMA_PATH)
        chroma_collection = db.get_collection(config.COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # --- LOAD LOCAL DOCSTORE FOR BM25 (Decoupled from Chroma) ---
        if os.path.exists(config.STORAGE_NODES_PATH):
            logger.info(f"Loading nodes for BM25 index from {config.STORAGE_NODES_PATH}...")
            
            # Use separate storage context for local files to force disk read
            local_storage = StorageContext.from_defaults(
                persist_dir=config.STORAGE_NODES_PATH
            )
            
            all_nodes = list(local_storage.docstore.docs.values())
            logger.info(f"Loaded {len(all_nodes)} nodes for BM25.")

            if len(all_nodes) > 0:
                logger.info("Building BM25 Keyword Index (RAM)...")
                bm25_retriever = BM25Retriever.from_defaults(
                    nodes=all_nodes, 
                    similarity_top_k=5
                )
            else:
                logger.warning("Docstore loaded but returned 0 nodes. Check ingest_code.py.")
                bm25_retriever = None
        else:
            logger.warning(f"{config.STORAGE_NODES_PATH} not found. Hybrid search disabled.")
            bm25_retriever = None

        # Initialize Vector Index (From Chroma)
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=config.get_embedding_model()
        )
        
        logger.info(f"System Ready! Hybrid Search: {'Active' if bm25_retriever else 'Inactive'}")
        
    except Exception as e:
        logger.error(f"Fatal error during startup: {e}")
        sys.exit(1)
        
    yield
    logger.info("--- Shutting down API Server ---")

app = FastAPI(lifespan=lifespan, title="Istio AI Agent API")
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    if os.path.exists("static/index.html"):
        return FileResponse('static/index.html')
    return {"message": "Istio Agent API Running (Index not found)"}

# --- CHAT ENGINE FACTORY (Where Context + Search meet) ---
def get_chat_engine(session_id: str, filters=None, target_version="1.28"):
    if not vector_index:
        raise HTTPException(status_code=503, detail="System not ready")

    # RETRIEVE PERSISTENT MEMORY
    if session_id not in session_store:
        logger.info(f"Creating new memory for session: {session_id}")
        session_store[session_id] = ChatMemoryBuffer.from_defaults(token_limit=8000)
    
    # We grab the memory object that holds the history for this user
    user_memory = session_store[session_id]

    # CREATE RETRIEVER (Dynamic per request)
    vector_retriever = vector_index.as_retriever(
        similarity_top_k=6,
        filters=filters
    )

    if bm25_retriever:
        final_retriever = CustomHybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            target_version=target_version
        )
    else:
        final_retriever = vector_retriever

    # CREATE ENGINE WITH PERSISTENT MEMORY
    # We inject the 'user_memory' here. This ensures the engine sees past chats
    # even though the engine itself is newly created.
    return ContextChatEngine.from_defaults(
        retriever=final_retriever,
        llm=Settings.llm,
        memory=user_memory, 
        system_prompt=prompts.ISTIO_SYSTEM_PROMPT
    )

def detect_version_intent(user_message: str):
    active_versions = config.cfg['system']['active_versions']
    default_ver = config.cfg['system']['default_version']
    
    try:
        sorted_vers = sorted(active_versions, key=lambda x: int(x.split('.')[1]), reverse=True)
        oldest_supported = sorted_vers[-1]
    except:
        oldest_supported = active_versions[-1]

    match = re.search(r'\b1\.(\d+)\b', user_message)
    if match:
        asked_minor = int(match.group(1))
        
        oldest_minor = int(oldest_supported.split('.')[1])
        if asked_minor < oldest_minor:
             return oldest_supported, f"Note: You asked for 1.{asked_minor}, answering with {oldest_supported}."

        req = f"1.{asked_minor}"
        if req in active_versions: return req, None
        return default_ver, f"Note: Using default {default_ver}."
        
    return default_ver, None

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    target_version, warning_msg = detect_version_intent(request.message)
    
    filters = MetadataFilters(
        filters=[
            ExactMatchFilter(key="system_version", value=target_version),
            ExactMatchFilter(key="system_version", value="any")
        ],
        condition=FilterCondition.OR 
    )
    
    # Get the engine configured for this user + this version
    user_engine = get_chat_engine(request.session_id, filters, target_version)
    
    try:
        response = await user_engine.achat(request.message)
        
        source_list = []
        seen_paths = set()

        for node in response.source_nodes:
            repo = node.metadata.get('repo_name', 'istio')
            url = node.metadata.get('source', '')
            title = node.metadata.get('title')
            file_path = node.metadata.get('file_path', 'N/A')
            clean_path = re.sub(r'.*data_[^/]+/', '', file_path)
            
            unique_id = url if node.metadata.get('type') == 'github_issue' else file_path
            display = f"Issue: {title}" if node.metadata.get('type') == 'github_issue' else clean_path

            if unique_id not in seen_paths:
                source_list.append(Source(
                    repo=repo,
                    file=display,
                    url=url if url else None,
                    score=float(node.score or 0)
                ))
                seen_paths.add(unique_id)

        final_text = str(response.response)
        if warning_msg:
            final_text = f"{warning_msg}\n\n{final_text}"

        return ChatResponse(response=final_text, sources=source_list)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/api/reset")
async def reset_chat(request: ChatRequest): 
    if request.session_id in session_store:
        del session_store[request.session_id]
        return {"status": "memory_cleared"}
    return {"status": "no_session_found"}

@app.get("/health")
async def health_check():
    return {"status": "ok", "mode": "hybrid" if bm25_retriever else "vector_only", "active_sessions": len(session_store)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
