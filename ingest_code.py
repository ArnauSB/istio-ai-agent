import os
import shutil
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
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f).get("github", {}).get("exclude_patterns", [])
    except:
        return []

EXCLUDE_PATTERNS = load_exclusions()

def is_excluded(file_path):
    normalized_path = os.path.normpath(file_path)
    for pattern in EXCLUDE_PATTERNS:
        if fnmatch.fnmatch(normalized_path, pattern): return True
        if fnmatch.fnmatch(os.path.basename(normalized_path), pattern): return True
    return False

def reset_database():
    """
    Deletes the existing database and node storage to prevent duplication.
    """
    folders_to_clean = [config.CHROMA_PATH, config.STORAGE_NODES_PATH]
    
    print("\n🧹 Cleaning up old databases...")
    for folder in folders_to_clean:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
                print(f"   - Deleted: {folder}")
            except Exception as e:
                print(f"   - Error deleting {folder}: {e}")
    print("✨ Database is clean.\n")

def ingest_code():
    # CLEAN UP
    reset_database()

    # READ CONFIG
    system_versions = config.cfg['system']['active_versions']
    repo_list = config.cfg['github']['repositories']
    allowed_extensions = config.cfg['github']['extensions']
    
    all_documents = []

    print(f"Starting Multi-Version Ingestion: {system_versions}")

    # ITERATE "USER FACING" VERSIONS
    for system_ver in system_versions:
        print(f"\n=== Processing System Version: {system_ver} ===")
        
        for repo_conf in repo_list:
            repo_name = repo_conf['name']
            url = repo_conf['url']
            
            # --- BRANCH RESOLUTION LOGIC ---
            git_branch = repo_conf.get('version_maps', {}).get(system_ver)
            if not git_branch:
                git_branch = repo_conf.get('branch') # Fallback for global
                
            if not git_branch:
                print(f"Skipping {repo_name}: No mapping found for {system_ver}")
                continue

            version_path = os.path.join("data_versions", system_ver, repo_name)
            
            # --- CLONE / PULL ---
            if os.path.exists(version_path):
                print(f"Using existing folder: {version_path}")
            else:
                print(f"Cloning {repo_name} ({git_branch}) into {version_path}...")
                try:
                    Repo.clone_from(url, version_path, depth=1, branch=git_branch)
                except Exception as e:
                    print(f"Error cloning {repo_name}: {e}")
                    continue

            # --- FILTER FILES ---
            file_paths = []
            for root, dirs, files in os.walk(version_path):
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                for file in files:
                    full_path = os.path.join(root, file)
                    if not any(file.endswith(ext) for ext in allowed_extensions): continue
                    if is_excluded(full_path): continue
                    file_paths.append(full_path)

            if not file_paths: continue

            # --- LOAD & TAG ---
            print(f"Ingesting {len(file_paths)} files from {repo_name}...")
            reader = SimpleDirectoryReader(input_files=file_paths)
            docs = reader.load_data()
            
            for d in docs:
                d.metadata["repo_name"] = repo_name
                d.metadata["system_version"] = system_ver 
                d.metadata["git_branch"] = git_branch
                clean_rel_path = os.path.relpath(d.metadata.get("file_path"), version_path)
                d.metadata["file_path"] = f"{repo_name}/{clean_rel_path}" 

            all_documents.extend(docs)

    # INDEXING WITH MANUAL NODE SAVING
    print(f"\nTOTAL: Loaded {len(all_documents)} docs across all versions.")
    
    print(f"Connecting to ChromaDB at {config.CHROMA_PATH}...")
    db = chromadb.PersistentClient(path=config.CHROMA_PATH)
    chroma_collection = db.get_or_create_collection(config.COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Initialize Storage Context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    Settings.embed_model = config.get_embedding_model()
    
    print("Parsing nodes manually to ensure they are saved to disk...")
    parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    Settings.text_splitter = parser
    
    # Manually create nodes
    nodes = parser.get_nodes_from_documents(all_documents)
    
    # Manually add them to the docstore (This forces saving to JSON later)
    storage_context.docstore.add_documents(nodes)
    print(f"-> Created {len(nodes)} nodes in memory.")

    print("Generating Embeddings (ChromaDB)...")
    # Create Index from NODES (not documents)
    VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)
    
    print(f"Persisting nodes for BM25 at {config.STORAGE_NODES_PATH}...")
    if not os.path.exists(config.STORAGE_NODES_PATH):
        os.makedirs(config.STORAGE_NODES_PATH)
        
    # Save to disk
    storage_context.persist(persist_dir=config.STORAGE_NODES_PATH)

    print("Multi-version Ingestion Complete!")

if __name__ == "__main__":
    ingest_code()
