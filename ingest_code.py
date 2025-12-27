import os
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

def ingest_code():
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
            
            # Fallback for playgrounds (Global repos)
            if not git_branch:
                git_branch = repo_conf.get('branch')
                
            if not git_branch:
                print(f"Skipping {repo_name}: No mapping found for {system_ver}")
                continue

            # Path: data_versions/1.28/istio-core
            version_path = os.path.join("data_versions", system_ver, repo_name)
            
            # --- CLONE / PULL ---
            if os.path.exists(version_path):
                print(f"Using existing folder: {version_path}. Pulling latest changes...")
                try:
                    repo = Repo(version_path)
                    repo.remotes.origin.pull()
                except Exception as e:
                    print(f"Could not pull latest changes: {e}")
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

    # INDEX
    print(f"\nTOTAL: Loaded {len(all_documents)} docs across all versions.")
    
    print(f"Connecting to ChromaDB at {config.CHROMA_PATH}...")
    db = chromadb.PersistentClient(path=config.CHROMA_PATH)
    chroma_collection = db.get_or_create_collection(config.COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    Settings.embed_model = config.get_embedding_model()
    Settings.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

    print("Generating Embeddings...")
    VectorStoreIndex.from_documents(all_documents, storage_context=storage_context, show_progress=True)
    print("Multi-version Ingestion Complete!")

if __name__ == "__main__":
    ingest_code()
