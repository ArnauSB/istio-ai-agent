import os
import sys
import time
import logging
import re
import uvicorn
import math
import json
import threading

from collections import OrderedDict
from typing import List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import nest_asyncio
import chromadb

# --- CORE IMPORTS ---
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter, FilterCondition
from llama_index.core.chat_engine import ContextChatEngine

# --- RETRIEVAL & RERANKING IMPORTS ---
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever, BaseRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank

import config
import prompts

os.environ["TOKENIZERS_PARALLELISM"] = "false"
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- SESSION STORE CONFIG ---
# Idle sessions expire after this many seconds; total is capped to bound memory.
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "1000"))


class SessionStore:
    """Bounded, TTL-based store for per-session chat memory.

    Prevents unbounded memory growth: idle sessions expire after
    ``ttl`` seconds and the total count is capped at ``max_sessions``
    (least-recently-used evicted first). A lock guards access since the
    ASGI app may serve requests concurrently.
    """

    def __init__(self, ttl: int, max_sessions: int):
        self._ttl = ttl
        self._max = max_sessions
        # session_id -> (last_seen_monotonic, ChatMemoryBuffer)
        self._data: "OrderedDict[str, tuple]" = OrderedDict()
        self._lock = threading.Lock()

    def _purge_expired(self, now: float):
        expired = [sid for sid, (ts, _) in self._data.items() if now - ts > self._ttl]
        for sid in expired:
            del self._data[sid]
            logger.info(f"Expired idle session: {sid}")

    def get_or_create(self, session_id: str) -> ChatMemoryBuffer:
        now = time.monotonic()
        with self._lock:
            self._purge_expired(now)
            if session_id in self._data:
                _, memory = self._data.pop(session_id)  # pop+reinsert = mark as MRU
            else:
                logger.info(f"Creating new memory for session: {session_id}")
                memory = ChatMemoryBuffer.from_defaults(token_limit=8000)
            self._data[session_id] = (now, memory)
            # Enforce the size cap (OrderedDict keeps LRU at the front)
            while len(self._data) > self._max:
                evicted, _ = self._data.popitem(last=False)
                logger.info(f"Evicting least-recently-used session: {evicted}")
            return memory

    def remove(self, session_id: str) -> bool:
        with self._lock:
            return self._data.pop(session_id, None) is not None

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


# --- GLOBAL STATE ---
vector_index = None
bm25_retriever = None
session_store = SessionStore(SESSION_TTL_SECONDS, MAX_SESSIONS)

# --- RERANKER ---
# This decides which document is best, whether it came from Vector or BM25.
reranker = SentenceTransformerRerank(
    model="BAAI/bge-reranker-v2-m3", 
    top_n=5
)

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_index, bm25_retriever
    logger.info("--- Starting API Server ---")
    
    try:
        embed_model = config.get_embedding_model() 
        Settings.embed_model = embed_model
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
        
        # --- LOAD BM25 INDEX ---
        if os.path.exists(config.STORAGE_NODES_PATH):
            logger.info(f"Loading nodes for BM25 from {config.STORAGE_NODES_PATH}...")
            local_storage = StorageContext.from_defaults(
                persist_dir=config.STORAGE_NODES_PATH
            )
            all_nodes = list(local_storage.docstore.docs.values())
            
            if len(all_nodes) > 0:
                bm25_retriever = BM25Retriever.from_defaults(
                    nodes=all_nodes,
                    # Fetch extra via keywords: results are version-filtered
                    # post-retrieval, so over-fetch to keep a healthy candidate pool.
                    similarity_top_k=25
                )
            else:
                logger.warning("Docstore loaded but returned 0 nodes. Check ingest_code.py.")
                bm25_retriever = None
        else:
            logger.warning(f"{config.STORAGE_NODES_PATH} not found. Hybrid search disabled.")
            bm25_retriever = None

        # Initialize Vector Index
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=embed_model
        )
        
        logger.info(f"System Ready! Hybrid Mode: {'Active' if bm25_retriever else 'Vector Only'}")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
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
    return {"message": "Istio Agent API Running"}

class VersionFilteredRetriever(BaseRetriever):
    """Wraps a retriever and keeps only nodes matching the requested version.

    BM25 has no native metadata filtering, so without this it would surface
    docs from every ingested Istio version regardless of the query's target.
    We keep nodes whose ``system_version`` matches ``target_version`` or is
    ``"any"`` (version-agnostic docs such as GitHub issues).
    """

    def __init__(self, inner: BaseRetriever, target_version: str):
        self._inner = inner
        self._target = target_version
        super().__init__()

    def _keep(self, nodes):
        return [
            n for n in nodes
            if n.node.metadata.get("system_version") in (self._target, "any")
        ]

    def _retrieve(self, query_bundle):
        return self._keep(self._inner.retrieve(query_bundle))

    async def _aretrieve(self, query_bundle):
        return self._keep(await self._inner.aretrieve(query_bundle))


# --- CHAT ENGINE FACTORY (Hybrid + Reranker) ---
def get_chat_engine(session_id: str, filters=None, target_version="1.28"):
    if not vector_index:
        raise HTTPException(status_code=503, detail="System not ready")

    # Grab (or lazily create) the memory object that holds this user's history.
    # The store expires idle sessions and caps the total to bound memory use.
    user_memory = session_store.get_or_create(session_id)

    # 1. Setup Vector Retriever
    vector_retriever = vector_index.as_retriever(
        similarity_top_k=10,
        filters=filters
    )

    # 2. Setup Hybrid Retriever (Fusion)
    if bm25_retriever:
        # BM25 can't filter on metadata, so wrap it to drop other-version docs
        # before fusion — otherwise keyword hits leak across Istio versions.
        version_bm25 = VersionFilteredRetriever(bm25_retriever, target_version)
        # QueryFusionRetriever combines results from both retrievers
        final_retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, version_bm25],
            similarity_top_k=15, # Total candidates to send to Reranker
            num_queries=1,       # Only use the original query (no query generation)
            mode="simple"        # Simple merge, let Reranker sort it out
        )
    else:
        final_retriever = vector_retriever

    # 3. Create Engine with Reranker
    return ContextChatEngine.from_defaults(
        retriever=final_retriever,
        node_postprocessors=[reranker], # <--- Reranker selects the best 5 from the 15 candidates
        llm=Settings.llm,
        memory=user_memory, 
        system_prompt=prompts.ISTIO_SYSTEM_PROMPT
    )

def detect_version_intent(user_message: str):
    active_versions = config.cfg['system']['active_versions']
    default_ver = config.cfg['system']['default_version']
    
    # Sort versions to find the oldest supported one dynamically
    try:
        sorted_vers = sorted(active_versions, key=lambda x: int(x.split('.')[1]), reverse=True)
        oldest_supported = sorted_vers[-1]
    except (ValueError, IndexError):
        oldest_supported = active_versions[-1]

    # Find a candidate minor version. We accept two kinds of mentions:
    #   1. Anchored: a "1.x" sitting near a version keyword (istio/version/release/
    #      upgrade/migrate). High confidence, so we honor any minor and reconcile it
    #      below (out-of-range -> oldest, unknown -> default with a note).
    #   2. Bare: a plain "1.x" anywhere. Low confidence, since pasted configs and
    #      semver strings are full of stray "1.x". We only trust it when it exactly
    #      matches a supported version; otherwise we ignore it and use the default.
    anchored = re.search(
        r'(?i)\b(?:istio|versions?|release|upgrade|migrat\w*)\b[^\d]{0,20}\bv?1\.(\d+)\b',
        user_message,
    )
    bare = re.search(r'\bv?1\.(\d+)\b', user_message)

    if anchored:
        asked_minor = int(anchored.group(1))
        # Protect against asking for ancient versions (e.g., 1.15)
        if asked_minor < int(oldest_supported.split('.')[1]):
            return oldest_supported, f"Note: Asked 1.{asked_minor}, answering with {oldest_supported}."

        req = f"1.{asked_minor}"
        # Return matched version if valid, otherwise fallback to default
        return (req, None) if req in active_versions else (default_ver, f"Note: Using default {default_ver}.")

    if bare:
        req = f"1.{int(bare.group(1))}"
        if req in active_versions:
            return req, None

    return default_ver, None

MAX_FILE_SIZE = 5 * 1024 * 1024
ALLOWED_EXTENSIONS = {'.yaml', '.yml', '.go', '.md', '.txt', '.json'}

@app.post("/api/chat")
async def chat_endpoint(
    message: str = Form(...),
    session_id: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    # 1. Validation Logic
    if file:
        # Size Validation
        file.file.seek(0, os.SEEK_END)
        file_size = file.file.tell()
        file.file.seek(0) 
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large.")

        # Extension Validation
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type ({ext}). Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
    target_version, warning_msg = detect_version_intent(message)
    
    # Process attached file if present
    file_context = ""
    if file:
        try:
            content = await file.read()
            # Decode content (assuming text-based files like YAML/Go/MD)
            file_text = content.decode("utf-8")
            file_context = f"\n\n[Attached File: {file.filename}]\n---\n{file_text}\n---\n"
        except Exception as e:
            logger.error(f"Error reading uploaded file: {e}")
            warning_msg = f" (Warning: Could not read file {file.filename})"

    full_query = f"{file_context}{message}"
    
    filters = MetadataFilters(
        filters=[
            ExactMatchFilter(key="system_version", value=target_version),
            ExactMatchFilter(key="system_version", value="any")
        ],
        condition=FilterCondition.OR 
    )
    
    user_engine = get_chat_engine(session_id, filters, target_version)
    
    # 2. Streaming Logic
    try:
        # Stream response
        response_stream = await user_engine.astream_chat(full_query)

        async def event_generator():
            # A. Send Warning Message first
            if warning_msg:
                yield f"{warning_msg}\n\n"

            # B. Yield tokens one by one as LLM generates them
            async for token in response_stream.async_response_gen():
                yield token

            # C. Process Sources
            source_list = []
            seen_paths = set()
            
            # response_stream.source_nodes contains the retrieved nodes
            for node in response_stream.source_nodes:
                if len(source_list) >= 5: break
                
                repo = node.metadata.get('repo_name', 'istio')
                url = node.metadata.get('source', '')
                title = node.metadata.get('title')
                path = node.metadata.get('file_path', 'N/A')

                # Calculate the sigmoid score
                raw_score = float(node.score or 0)
                sigmoid_score = 1 / (1 + math.exp(-raw_score))
                
                # Clean up the path for display (removes data_versions/x.xx/ prefix)
                clean_path = re.sub(r'.*data_[^/]+/', '', path)

                # Determine unique ID to prevent duplicate sources
                unique_id = url if node.metadata.get('type') == 'github_issue' else path

                # Determine what to show in the UI
                display = f"Issue: {title}" if node.metadata.get('type') == 'github_issue' else clean_path

                if unique_id not in seen_paths:
                    source_list.append({
                        "repo": repo,
                        "file": display,
                        "url": url,
                        "score": sigmoid_score
                    })
                    seen_paths.add(unique_id)
            
            # D. Send Sources as a special footer packet
            # We use a delimiter "__SOURCES__:" to let frontend know this isn't text
            yield f"\n\n__SOURCES__:{json.dumps(source_list)}"

        return StreamingResponse(event_generator(), media_type="text/plain")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/api/reset")
async def reset_chat(session_id: str = Form(...)):
    if session_store.remove(session_id):
        return {"status": "memory_cleared"}
    return {"status": "no_session_found"}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.png")

@app.get("/health")
async def health_check():
    return {"status": "ok", "mode": "hybrid_rerank", "active_sessions": len(session_store)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
