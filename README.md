# 🤖 Istio AI Agent (Experimental v0.1)

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.11+-yellow.svg)
![Status](https://img.shields.io/badge/Status-Alpha-orange.svg)

An AI-powered assistant specifically designed for **Istio Service Mesh**. 

Unlike generic LLMs, this agent uses **RAG (Retrieval-Augmented Generation)** to ingest the latest Istio source code, documentation, and GitHub Issues locally. It provides answers with **direct citations** to the files it used.

> **⚠️ Disclaimer:** This is a v0.1 Proof of Concept. Do not use in critical production environments without verification.

## ✨ Features

- **🧠 Domain Specific:** Trained on `istio/istio`, `istio/api`, and `envoyproxy/envoy`.
- **🔍 Source Citations:** Tells you exactly which file or GitHub Issue was used to generate the answer.
- **🛡️ Data Privacy:** Runs 100% locally using **Ollama** and local vector embeddings. No data leaves your machine.
- **💾 Session Memory:** Remembers context per user session (isolated memory).

## 🛠️ Architecture
- **LLM:** Ollama (default: `gpt-oss`)
- **Orchestration:** LlamaIndex
- **Vector Database:** ChromaDB (Local persistent storage)
- **Backend:** FastAPI (Python)
- **Frontend:** HTML5/JS (No frameworks required)

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
git clone [https://github.com/ArnauSB/istio-ai-agent.git](https://github.com/ArnauSB/istio-ai-agent.git)
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
