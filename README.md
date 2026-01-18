# 🤖 Istio AI Agent (Experimental v0.2)

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.11+-yellow.svg)
![Status](https://img.shields.io/badge/Status-Beta-green.svg)

An Advanced AI Assistant specifically designed for **Istio Service Mesh**. 

Unlike generic chatbots, this agent uses **Advanced RAG (Retrieval-Augmented Generation)** with a **Hybrid Search** strategy. It ingests Istio source code, documentation, and GitHub Issues locally, uses a Cross-Encoder to verify relevance, and streams answers in real-time.

> **⚠️ Disclaimer:** This is a Proof of Concept. Do not use in critical production environments without verification.

## ✨ Key Features

- **🧠 Advanced Retrieval:** Uses **Hybrid Search** (Vector Semantic Search + BM25 Keyword Search) to find both high-level concepts and specific error codes.
- **🎯 AI Reranking:** A dedicated **Cross-Encoder Model** evaluates retrieved documents to filter out hallucinations and irrelevant matches.
- **⚡ Real-Time Streaming:** Answers are streamed token-by-token for an instant, responsive experience.
- **📎 Context Aware:** Supports **File Uploads** (YAML, Go, Logs). Drag & drop a `virtualservice.yaml` or a log file, and the AI will analyze it.
- **🔍 Precision Citations:** Displays the exact source file and a **Relevance Score (0-100%)** for every piece of information used.
- **🛡️ 100% Local Privacy:** Runs locally using **Ollama** and local vector embeddings. No data leaves your machine.

## 🛠️ Architecture

The pipeline implements a "Retrieve & Rerank" strategy:

1.  **Ingestion:** Code & Docs are split and embedded into **ChromaDB**.
2.  **Retrieval:** * **Vector Search:** Finds semantic meaning (e.g., "traffic splitting").
    * **BM25:** Finds exact keyword matches (e.g., "Error 503-UC").
3.  **Reranking:** The `cross-encoder/ms-marco-MiniLM-L-6-v2` model reads the top 30 candidates and selects the **Top 10** most relevant chunks.
4.  **Generation:** **Ollama** (Llama3/GPT-OSS) generates the answer using the curated context.

**Tech Stack:**
- **LLM:** Ollama (default: `gpt-oss:20b`)
- **Orchestration:** LlamaIndex
- **Reranker:** SentenceTransformers (`ms-marco-MiniLM-L-6-v2`)
- **Vector DB:** ChromaDB
- **Backend:** FastAPI (Async Streaming)
- **Frontend:** HTML5/JS (No frameworks, pure WebSocket-like streaming)

## 🚀 Getting Started

### Prerequisites

1. **Python 3.11+** installed.
2. **[Ollama](https://ollama.com/)** installed and running.
3. Pull the model:
```bash
ollama pull gpt-oss:20b
```

(Note: You can change the model in config.py)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ArnauSB/istio-ai-agent.git
cd istio-ai-agent
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
GITHUB_TOKEN=your_github_token_here
```

### Building the Knowledge Base

Before running the chat, you need to download and index the data.

1. Ingest Code & Docs:
```bash
python ingest_code.py
```

This clones the Istio repositories and creates the vector embeddings.

2. Ingest GitHub Issues (Optional):
```bash
python ingest_issues.py
```

Downloads solved issues from the last year to learn from real-world problems.

### Running the Agent

Start the API server (Backend + Frontend):
```bash
python -m uvicorn api:app --reload --loop asyncio
```

Open your browser at: http://localhost:8000

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
