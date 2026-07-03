import time
import json
import sys
import os
from datetime import datetime, timedelta
import nest_asyncio
from github import Github, Auth, RateLimitExceededException, GithubException
from llama_index.core import VectorStoreIndex, Document, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

import config 

nest_asyncio.apply()

def get_repo_slug_from_config(target_name="istio-core"):
    repos = config.cfg['github'].get('repositories', [])
    for repo in repos:
        if repo['name'] == target_name:
            url = repo['url']
            return url.replace("https://github.com/", "").replace(".git", "")
    return None

# GH API limits at 1000 issues
def fetch_github_issues(token, repo_name, days_back=365, limit=1000):
    print(f"\n--- Starting DEEP download for {repo_name} (Issues + Comments) ---")
    
    auth = Auth.Token(token)
    g = Github(auth=auth)
    
    since_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    print(f"Searching for issues created after: {since_date} with >5 comments")
    
    # This ensures we only fetch "deep" discussions useful for troubleshooting
    query = f"repo:{repo_name} is:issue created:>{since_date} comments:>5"
    
    issues_data = []
    
    try:
        results = g.search_issues(query=query, sort='created', order='desc')
        total_found = results.totalCount
        print(f"GitHub found {total_found} matching issues.")
        
        count = 0
        for issue in results:
            if count >= limit:
                print(f"Limit of {limit} reached.")
                break

            try:
                full_conversation = (
                    f"Title: {issue.title}\n"
                    f"Date: {issue.created_at}\n"
                    f"Status: {issue.state.upper()}\n"
                    f"Original Issue:\n{issue.body}\n"
                )

                # Download all comments
                if issue.comments > 0:
                    full_conversation += "\n--- CONVERSATION / COMMENTS ---\n"
                    
                    comments = issue.get_comments()
                    
                    # Limit to 50 comments
                    for idx, comment in enumerate(comments):
                        if idx > 50: 
                            full_conversation += "\n[... truncated long thread ...]"
                            break
                        
                        full_conversation += (
                            f"\n[Comment by {comment.user.login} on {comment.created_at}]\n"
                            f"{comment.body}\n"
                            f"----------------------------------------\n"
                        )

                full_conversation += f"\nUrl: {issue.html_url}"

                # Save document
                issues_data.append({
                    "text": full_conversation,
                    "metadata": {
                        "source": issue.html_url,
                        "title": issue.title,
                        "created_at": str(issue.created_at),
                        "repo_name": repo_name,
                        "type": "github_issue",
                        "state": issue.state,
                        "system_version": "any"
                    }
                })

                count += 1
                
                if count % 10 == 0:
                    sys.stdout.write(f"\rProcessed {count}/{limit} issues (with comments)...")
                    sys.stdout.flush()

            except RateLimitExceededException:
                print("\nRate Limit Hit! Sleeping for 60 seconds...")
                time.sleep(60)
                continue
            except GithubException as e:
                if e.status == 403:
                     print("\nSecondary Rate Limit Hit! Sleeping for 60 seconds...")
                     time.sleep(60)
                else:
                    print(f"Skipping broken issue: {e}")

    except RateLimitExceededException:
        print("\nGitHub API rate limit exceeded drastically. Saving progress...")
    except Exception as e:
        print(f"\nUnexpected error in search: {e}")

    print(f"\nDownload finished: {len(issues_data)} issues ready.")
    return issues_data

def run_ingestion():
    target_repo_slug = get_repo_slug_from_config("istio-core")
    
    if not target_repo_slug:
        print("Error: Could not find 'istio-core' in config.yaml.")
        return

    print(f"Targeting repository: {target_repo_slug}")

    cache_file = "data_versions/istio_issues_cache.json" 
    
    # Note: If you change the query, you MUST delete the old cache file manually or it will load old data!
    if os.path.exists(cache_file):
        print(f"Loading from local cache ({cache_file})...")
        with open(cache_file) as f:
            raw_data = json.load(f)
        print(f"-> {len(raw_data)} issues retrieved from disk.")
    else:
        if not config.GITHUB_TOKEN:
            print("Error: GITHUB_TOKEN missing in .env file.")
            return
        
        print("No cache found. Contacting GitHub (This will take a while due to comments)...")
        raw_data = fetch_github_issues(
            config.GITHUB_TOKEN, 
            target_repo_slug, 
            days_back=365,
            limit=1000 
        )
        
        if len(raw_data) > 0:
            with open(cache_file, 'w') as f:
                json.dump(raw_data, f)
            print("Cache saved to disk.")
        else:
            print("No issues downloaded.")
            return

    if not raw_data:
        return

    # Documents
    documents = []
    print("Processing documents for embedding...")
    for d in raw_data:
        text_content = d['text'] if d['text'] else "No content"
        documents.append(Document(text=text_content, metadata=d['metadata']))
    
    # ChromaDB
    print(f"Connecting to ChromaDB at {config.CHROMA_PATH}...")
    db = chromadb.PersistentClient(path=config.CHROMA_PATH)
    chroma_collection = db.get_or_create_collection(config.COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Load the existing nodes from 'ingest_code.py' so we don't overwrite them.
    if os.path.exists(config.STORAGE_NODES_PATH):
        print(f"Loading existing docstore from {config.STORAGE_NODES_PATH} to append issues...")
        try:
            storage_context = StorageContext.from_defaults(
                persist_dir=config.STORAGE_NODES_PATH, 
                vector_store=vector_store
            )
        except Exception as e:
            print(f"Error loading existing docstore: {e}. Starting with fresh context.")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
    else:
        print("No existing docstore found. Creating new context.")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Embeddings
    Settings.embed_model = config.get_embedding_model()

    print(f"Generating vectors for {len(documents)} deep issues...")
    
    # --- HYBRID SEARCH PERSISTENCE ---
    VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context,
        show_progress=True
    )
    
    print(f"Persisting nodes for BM25 at {config.STORAGE_NODES_PATH}...")
    if not os.path.exists(config.STORAGE_NODES_PATH):
        os.makedirs(config.STORAGE_NODES_PATH)
        
    storage_context.persist(persist_dir=config.STORAGE_NODES_PATH)

    print("\nDeep Issues ingestion completed successfully!")

if __name__ == "__main__":
    run_ingestion()
