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

import config
import prompts

os.environ["TOKENIZERS_PARALLELISM"] = "false"
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

vector_index = None 
session_store: Dict[str, object] = {}

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
        # Global Settings
        Settings.embed_model = config.get_embedding_model()
        
        Settings.llm = Ollama(
            model=config.MODEL_NAME, 
            base_url=config.OLLAMA_URL, 
            request_timeout=300.0,
            temperature=0.2 
        )

        # Database Connection
        logger.info(f"Connecting to ChromaDB at {config.CHROMA_PATH}...")
        db = chromadb.PersistentClient(path=config.CHROMA_PATH)
        chroma_collection = db.get_collection(config.COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # Load Index
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

def get_chat_engine(session_id: str):
    """
    Creates or retrieves a specific chat engine for a user.
    """
    if not vector_index:
        raise HTTPException(status_code=503, detail="System not ready (Index not loaded)")

    if session_id not in session_store:
        logger.info(f"Creating new session: {session_id}")
        # Create a fresh memory buffer for this user
        memory = ChatMemoryBuffer.from_defaults(token_limit=4000)
        
        # Create their personal engine connected to the shared index
        session_store[session_id] = vector_index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            similarity_top_k=5,
            system_prompt=prompts.ISTIO_SYSTEM_PROMPT
        )
    
    return session_store[session_id]

@app.get("/")
async def read_root():
    if os.path.exists("static/index.html"):
        return FileResponse('static/index.html')
    return {"message": "Istio Agent API Running"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    user_engine = get_chat_engine(request.session_id)
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

        return ChatResponse(response=str(response.response), sources=source_list)
    except Exception as e:
        logger.error(f"Error: {e}")
        return ChatResponse(response="Sorry, I encountered an error.", sources=[])
    
@app.post("/api/reset")
async def reset_chat(request: ChatRequest): 
    if request.session_id in session_store:
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
    print("Starting Istio Agent API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)