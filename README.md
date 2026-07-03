# 🤖 Istio AI Agent (Experimental v0.3)

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.12+-yellow.svg)
![Status](https://img.shields.io/badge/Status-Beta-green.svg)

An Advanced AI Assistant specifically designed for **Istio Service Mesh**. 

Unlike generic chatbots, this agent uses **Advanced RAG (Retrieval-Augmented Generation)** with a **Hybrid Search** strategy. It ingests Istio source code, documentation, and GitHub Issues locally, uses a Cross-Encoder to verify relevance, and streams answers in real-time.

> **⚠️ Disclaimer:** This is a Proof of Concept. Do not use in critical production environments without verification.

## ✨ Key Features

- **🧠 Advanced Retrieval:** Uses **Hybrid Search** (Vector Semantic Search + BM25 Keyword Search) to find both high-level concepts and specific error codes.
- **🎯 AI Reranking:** A dedicated **Cross-Encoder Model** evaluates retrieved documents to filter out hallucinations and irrelevant matches.
- **🔀 Version Aware:** Indexes multiple Istio releases (1.26–1.28) side by side, detects the version you are asking about ("how do I do X in istio 1.27?"), and answers only from that version's docs and code.
- **⚡ Real-Time Streaming:** Answers are streamed token-by-token for an instant, responsive experience.
- **📎 Context Aware:** Supports **File Uploads** (YAML, Go, Logs). Drag & drop a `virtualservice.yaml` or a log file, and the AI will analyze it.
- **🔍 Precision Citations:** Displays the exact source file and a **Relevance Score (0-100%)** for every piece of information used.
- **🛡️ 100% Local Privacy:** Runs locally using **Ollama** and local vector embeddings. No data leaves your machine.

## 🛠️ Architecture

The pipeline implements a "Retrieve & Rerank" strategy:

1.  **Ingestion:** Code & Docs are split and embedded into **ChromaDB**.
2.  **Retrieval:** * **Vector Search:** Finds semantic meaning (e.g., "traffic splitting").
    * **BM25:** Finds exact keyword matches (e.g., "Error 503-UC").
3.  **Reranking:** The `BAAI/bge-reranker-v2-m3` cross-encoder reads the fused candidates and selects the **Top 5** most relevant chunks.
4.  **Generation:** **Ollama** (Llama3/GPT-OSS) generates the answer using the curated context.

**Tech Stack:**
- **LLM:** Ollama (default: `gpt-oss:20b`)
- **Orchestration:** LlamaIndex
- **Reranker:** SentenceTransformers (`BAAI/bge-reranker-v2-m3`)
- **Vector DB:** ChromaDB
- **Backend:** FastAPI (Async Streaming)
- **Frontend:** HTML5/JS (No frameworks, pure WebSocket-like streaming)

## 🚀 Getting Started

### Prerequisites

1. **Python 3.12+** installed.
2. **[Ollama](https://ollama.com/)** installed and running.
3. Pull the model:
```bash
ollama pull gpt-oss:20b
```

(Note: You can change the model in `config.yaml`)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ArnauSB/istio-ai-agent.git
cd istio-ai-agent
```

2. Create a virtual environment:
```bash
make venv
source venv/bin/activate
```

3. Install dependencies (pinned versions):
```bash
make install
```

4. Configure environment — create a `.env` file in the repo root (only needed for issue ingestion):
```bash
GITHUB_TOKEN=your_github_token_here
```

### Building the Knowledge Base

Before running the chat, you need to download and index the data.

1. Ingest Code & Docs:
```bash
make ingest-code
```

This clones the Istio repositories and creates the vector embeddings. **Warning:** it resets the existing vector DB first.

2. Ingest GitHub Issues (Optional):
```bash
make ingest-issues
```

Downloads solved issues from the last year to learn from real-world problems.

### Running the Agent

Start the API server (Backend + Frontend):
```bash
make run
```

Open your browser at: http://localhost:8000

## 🧪 Development

```bash
make install-dev   # runtime deps + dev tools (ruff)
make test          # run the unit test suite (offline, no GPU/network needed)
make lint          # lint with ruff
make lint-fix      # auto-fix what ruff can
make help          # list all targets
```

Every PR to `main` runs the lint and test jobs automatically via GitHub Actions
([.github/workflows/tests.yml](.github/workflows/tests.yml)).

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
