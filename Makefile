# Istio AI Agent — common developer tasks.
# Override PYTHON to point at a specific interpreter, e.g.:
#   make test PYTHON=./venv/bin/python
PYTHON ?= python3

.PHONY: help venv install test run ingest ingest-code ingest-issues clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

venv: ## Create a local virtualenv in ./venv
	$(PYTHON) -m venv venv
	@echo "Run: source venv/bin/activate"

install: ## Install pinned dependencies
	$(PYTHON) -m pip install -r requirements.txt

test: ## Run the unit test suite
	$(PYTHON) -m unittest discover -s tests -v -b

run: ## Start the API server (http://localhost:8000)
	$(PYTHON) api.py

ingest: ingest-code ingest-issues ## Rebuild the full knowledge base

ingest-code: ## Clone repos and index code/docs (destructive: resets the DB)
	$(PYTHON) ingest_code.py

ingest-issues: ## Fetch GitHub issues and append them to the index
	$(PYTHON) ingest_issues.py

clean: ## Remove caches and build artifacts (keeps the vector DB)
	find . -type d -name __pycache__ -not -path './venv/*' -not -path './data_versions/*' -exec rm -rf {} +
	rm -rf .pytest_cache
