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
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter, FilterCondition

import config
import prompts

os.environ["TOKENIZERS_PARALLELISM"] = "false"
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

vector_index = None 
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_index 
    logger.info("--- Starting API Server ---")
    
    try:
        Settings.embed_model = config.get_embedding_model()
        
        Settings.llm = Ollama(
            model=config.MODEL_NAME, 
            base_url=config.OLLAMA_URL, 
            request_timeout=600.0,
            temperature=0.1 
        )

        logger.info(f"Connecting to ChromaDB at {config.CHROMA_PATH}...")
        db = chromadb.PersistentClient(path=config.CHROMA_PATH)
        chroma_collection = db.get_collection(config.COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=config.get_embedding_model()
        )
        
        logger.info(f"System Ready! Serving model: {config.MODEL_NAME}")
        
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

def get_chat_engine(session_id: str, filters=None):
    if not vector_index:
        raise HTTPException(status_code=503, detail="System not ready (Index not loaded)")

    # Retrieve or Create Memory
    if session_id not in session_store:
        logger.info(f"Creating new memory for session: {session_id}")
        session_store[session_id] = ChatMemoryBuffer.from_defaults(token_limit=8000)
    
    user_memory = session_store[session_id]

    # Create Engine with DYNAMIC filters but PERSISTENT memory
    return vector_index.as_chat_engine(
        chat_mode="context",
        memory=user_memory,
        similarity_top_k=7,
        filters=filters,
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
            warning = f"Note: You asked for **Istio 1.{asked_minor}**, but the oldest version I have indexed is **1.{oldest_minor}**. Answering based on 1.{oldest_minor}."
            return oldest_supported, warning
            
        requested_ver = f"1.{asked_minor}"
        if requested_ver in active_versions:
            return requested_ver, None
        else:
            warning = f"Note: Istio {requested_ver} is not indexed. Using default version ({default_ver})."
            return default_ver, warning
            
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
    
    user_engine = get_chat_engine(request.session_id, filters)
    
    try:
        context_prompt = f"Context: You are answering based on Istio Version {target_version} documentation/code.\n"
        full_message = context_prompt + request.message
        
        response = await user_engine.achat(full_message)
        
        source_list = []
        seen_paths = set()

        for node in response.source_nodes:
            repo = node.metadata.get('repo_name', 'istio')
            url = node.metadata.get('source', '')
            title = node.metadata.get('title')
            file_path = node.metadata.get('file_path', 'N/A')
            
            clean_path = re.sub(r'.*data_[^/]+/', '', file_path)

            if node.metadata.get('type') == 'github_issue' and title:
                display_name = f"Issue: {title}"
                unique_id = url
            else:
                display_name = clean_path
                unique_id = file_path

            if unique_id not in seen_paths:
                source_list.append(Source(
                    repo=repo,
                    file=display_name,
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
        # Simply deleting the memory object resets the context
        del session_store[request.session_id]
        return {"status": "memory_cleared"}
    return {"status": "no_session_found"}

@app.get("/health")
async def health_check():
    return {
        "status": "ok", 
        "model": config.MODEL_NAME,
        "database": "connected" if vector_index else "loading",
        "active_sessions": len(session_store)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
