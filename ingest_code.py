import os
import sys
import yaml
import fnmatch
from git import Repo
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter 
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import nest_asyncio

import config 

nest_asyncio.apply()

def load_exclusions():
    """
    Loads exclusion patterns from config.yaml.
    """
    try:
        with open("config.yaml", "r") as f:
            data = yaml.safe_load(f)
            return data.get("exclude_patterns", [])
    except FileNotFoundError:
        print("config.yaml not found, using default exclusions.")
        return ["**/.git/**", "**/*_test.go", "**/vendor/**", "**/testdata/**"]

EXCLUDE_PATTERNS = load_exclusions()

def is_excluded(file_path):
    """
    Checks if the file path matches any exclusion pattern defined in config.yaml.
    """
    # Normalize path to avoid slash issues
    normalized_path = os.path.normpath(file_path)
    
    for pattern in EXCLUDE_PATTERNS:
        # Check against the pattern
        if fnmatch.fnmatch(normalized_path, pattern):
            return True
        # Also check just the filename for simpler patterns
        if fnmatch.fnmatch(os.path.basename(normalized_path), pattern):
            return True
            
    return False

def clone_repos():
    """
    Iterates through the list of repositories in config.yaml and clones them.
    Supports specific branches (e.g. 'release-1.28').
    """
    repo_list = config.cfg['github']['repositories']
    
    for repo_conf in repo_list:
        name = repo_conf['name']
        url = repo_conf['url']
        local_path = repo_conf['local_path']
        branch = repo_conf.get('branch', 'master') 
        
        print(f"\n--- Processing Repo: {name} ---")
        
        if os.path.exists(local_path):
            print(f"Status: {name} already exists at {local_path}.")
            print(f"Info: Skipping clone. Ensure the local folder is on branch '{branch}'.")
        else:
            print(f"Action: Cloning {name} from {url}...")
            print(f"Target Branch: {branch}")
            try:
                Repo.clone_from(url, local_path, depth=1, branch=branch) 
                print("Status: Clone successful.")
            except Exception as e:
                print(f"Error cloning {name}: {e}")

def get_allowed_files(repo_path, extensions):
    """
    Walks the directory and returns a list of files that:
    1. Match the allowed extensions.
    2. Do NOT match the exclusion patterns.
    """
    allowed_files = []
    
    print(f"Scanning {repo_path} with exclusion filters...")
    
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            full_path = os.path.join(root, file)
            
            # Check Extension
            if not any(file.endswith(ext) for ext in extensions):
                continue
                
            # Check Exclusions
            if is_excluded(full_path):
                # print(f"Skipping: {file}") # Uncomment to debug
                continue
            
            allowed_files.append(full_path)
            
    return allowed_files

def ingest_code():
    # Download all repositories
    clone_repos()

    print("\nScanning files from ALL repositories...")
    
    all_documents = []
    repo_list = config.cfg['github']['repositories']
    allowed_extensions = config.cfg['github']['extensions']
    
    # Load files from each repo folder
    for repo_conf in repo_list:
        local_path = repo_conf['local_path']
        repo_name = repo_conf['name']
        
        if not os.path.exists(local_path):
            print(f"Warning: Path {local_path} not found. Skipping {repo_name}.")
            continue

        # Get Clean List of Files
        file_paths = get_allowed_files(local_path, allowed_extensions)
        print(f"Selected {len(file_paths)} valid files in {repo_name}")
        
        if not file_paths:
            continue

        # Load ONLY these specific files
        reader = SimpleDirectoryReader(input_files=file_paths)
        docs = reader.load_data()
        
        # Add metadata
        for d in docs:
            d.metadata["repo_name"] = repo_name
            # Add file path relative to repo root for cleaner display later
            d.metadata["file_path"] = d.metadata.get("file_path", "").replace(local_path, "")
            
        all_documents.extend(docs)

    print(f"\nTOTAL: Loaded {len(all_documents)} clean files across all repositories.")
    
    if len(all_documents) == 0:
        print("Error: No documents found. Check your paths and extensions.")
        return

    print("Preparing fragmentation...")

    # Configure Global Settings
    Settings.embed_model = config.get_embedding_model()
    Settings.text_splitter = SentenceSplitter(
        chunk_size=config.cfg['splitter']['chunk_size'],
        chunk_overlap=config.cfg['splitter']['chunk_overlap']
    )

    # Connect to ChromaDB
    print(f"Connecting to ChromaDB at {config.CHROMA_PATH}...")
    db = chromadb.PersistentClient(path=config.CHROMA_PATH)
    chroma_collection = db.get_or_create_collection(config.COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Generate Embeddings
    print("Generating Embeddings and indexing... (This will take time)")
    VectorStoreIndex.from_documents(
        all_documents,
        storage_context=storage_context,
        show_progress=True
    )
    print("Ingestion complete.")

if __name__ == "__main__":
    confirm = input("Are you sure you want to index ALL Istio repositories defined in config? (y/n): ")
    if confirm.lower() in ["y", "yes", "s"]:
        ingest_code()
    else:
        print("Operation cancelled.")
