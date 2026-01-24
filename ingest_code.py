import os
import shutil
import yaml
import fnmatch
import pypandoc
from git import Repo
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings, Document
from llama_index.core.node_parser import SentenceSplitter 
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import nest_asyncio
import config 

nest_asyncio.apply()

# --- CONFIGURATION ---
DEBUG_ENABLED = config.DEBUG_ENABLED
DEBUG_OUTPUT_DIR = config.DEBUG_OUTPUT_DIR

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
    folders_to_clean = [config.CHROMA_PATH, config.STORAGE_NODES_PATH, DEBUG_OUTPUT_DIR]

    print("\nCleaning up old databases and debug files...")
    for folder in folders_to_clean:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
                print(f"   - Deleted: {folder}")
            except Exception as e:
                print(f"   - Error deleting {folder}: {e}")
    print("Database is clean.\n")

def save_debug_file(content, relative_path):
    """
    Saves the converted Markdown content to disk so the user can inspect it.
    Only runs if debug.enabled is True in config.
    """
    if not DEBUG_ENABLED:
        return

    # Construct full path: debug_converted/1.28/envoy-docs/path/to/file.md
    full_path = os.path.join(DEBUG_OUTPUT_DIR, relative_path)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)

def convert_rst_to_md_pypandoc(file_path):
    """
    Robust converter using Pypandoc (wraps Pandoc binary).
    Uses --quiet to suppress 'Reference not found' warnings.
    """
    try:
        # Pypandoc handles the file reading and conversion
        output = pypandoc.convert_file(
            file_path, 
            'md', 
            format='rst', 
            extra_args=['--quiet']
        )
        return output
    except Exception as e:
        print(f"Pandoc error on {file_path}: {e}")
        return ""

def convert_proto_to_md(proto_content, filename):
    """
    Wraps Protobuf definitions in a Markdown code block.
    """
    return f"# {filename}\n\n```protobuf\n{proto_content}\n```"

def ingest_code():
    # CLEAN UP
    reset_database()

    # READ CONFIG
    system_versions = config.cfg['system']['active_versions']
    repo_list = config.cfg['github']['repositories']
    allowed_extensions = config.cfg['github']['extensions']
    
    all_documents = []

    print(f"Starting Multi-Version Ingestion: {system_versions}")
    if DEBUG_ENABLED:
        print(f"Debug mode enabled. Saving converted files to '{DEBUG_OUTPUT_DIR}/'")

    # ITERATE "USER FACING" VERSIONS
    for system_ver in system_versions:
        print(f"\n=== Processing System Version: {system_ver} ===")
        
        for repo_conf in repo_list:
            repo_name = repo_conf['name']
            url = repo_conf['url']
            repo_subdir = repo_conf.get('subdir', '')
            
            # --- BRANCH RESOLUTION LOGIC ---
            git_branch = repo_conf.get('version_maps', {}).get(system_ver)
            if not git_branch:
                git_branch = repo_conf.get('branch')
                
            if not git_branch:
                print(f"Skipping {repo_name}: No mapping found for {system_ver}")
                continue

            base_path = os.path.join("data_versions", system_ver, repo_name)
            
            # --- CLONE ---
            if not os.path.exists(base_path):
                print(f"Cloning {repo_name} ({git_branch}) into {base_path}...")
                try:
                    Repo.clone_from(url, base_path, depth=1, branch=git_branch)
                except Exception as e:
                    print(f"Error cloning {repo_name}: {e}")
                    continue
            else:
                print(f"Using existing folder: {base_path}")

            # --- DETERMINE SCAN PATH ---
            scan_path = os.path.join(base_path, repo_subdir) if repo_subdir else base_path
            if not os.path.exists(scan_path):
                print(f"Warning: Subdirectory {scan_path} not found. Skipping.")
                continue

            # --- COLLECT FILES ---
            files_for_standard_loader = []
            envoy_rst_docs = []
            envoy_proto_docs = []

            for root, dirs, files in os.walk(scan_path):
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                for file in files:
                    full_path = os.path.join(root, file)
                    if is_excluded(full_path): continue
                    
                    ext = os.path.splitext(file)[1]

                    # --- SPECIAL LOGIC: ENVOY DOCS ---
                    if repo_name == "envoy-docs":
                        if ext == ".rst":
                            envoy_rst_docs.append(full_path)
                        elif ext == ".proto":
                            # Exclude /v2/ paths
                            normalized_path = full_path.replace(os.sep, "/")
                            if "/v2/" in normalized_path:
                                continue 
                            
                            envoy_proto_docs.append(full_path)
                        
                        # Continue explicitly skips any other extension (like .md) for Envoy
                        continue

                    # --- STANDARD LOGIC ---
                    if ext not in allowed_extensions: continue
                    files_for_standard_loader.append(full_path)

            # 1. Process Standard Files
            if files_for_standard_loader:
                print(f"[{repo_name}] Loading {len(files_for_standard_loader)} standard files...")
                reader = SimpleDirectoryReader(input_files=files_for_standard_loader)
                docs = reader.load_data()
                
                for d in docs:
                    d.metadata["repo_name"] = repo_name
                    d.metadata["system_version"] = system_ver
                    d.metadata["git_branch"] = git_branch
                    # Calc relative path from the repo root (not the subdir) to keep links valid
                    clean_rel_path = os.path.relpath(d.metadata.get("file_path"), base_path)
                    d.metadata["file_path"] = f"{repo_name}/{clean_rel_path}"
                
                all_documents.extend(docs)

            # 2. Process Envoy RST -> MD
            if envoy_rst_docs:
                print(f"[{repo_name}] Converting {len(envoy_rst_docs)} RST files to Markdown...")
                for rst_path in envoy_rst_docs:
                    md_content = convert_rst_to_md_pypandoc(rst_path)
                    if not md_content:
                        continue

                    if DEBUG_ENABLED:
                        rel_path_from_version = os.path.relpath(rst_path, os.path.dirname(base_path))
                        save_rel_path = rel_path_from_version.replace('.rst', '.md')
                        save_debug_file(md_content, save_rel_path)
                        
                    # Create Document Manually
                    rel_path = os.path.relpath(rst_path, base_path)
                    doc = Document(
                        text=md_content,
                        metadata={
                            "file_path": f"{repo_name}/{rel_path.replace('.rst', '.md')}",
                            "file_name": os.path.basename(rst_path),
                            "repo_name": repo_name,
                            "system_version": system_ver,
                            "git_branch": git_branch,
                            "original_format": "rst"
                        }
                    )
                    all_documents.append(doc)

            # 3. Process Envoy PROTO -> MD
            if envoy_proto_docs:
                print(f"[{repo_name}] Converting {len(envoy_proto_docs)} PROTO files to Markdown...")
                for proto_path in envoy_proto_docs:
                    try:
                        with open(proto_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        md_content = convert_proto_to_md(content, os.path.basename(proto_path))

                        if DEBUG_ENABLED:
                            rel_path_from_version = os.path.relpath(proto_path, os.path.dirname(base_path))
                            save_rel_path = rel_path_from_version + ".md" 
                            save_debug_file(md_content, save_rel_path)
                        
                        rel_path = os.path.relpath(proto_path, base_path)
                        doc = Document(
                            text=md_content,
                            metadata={
                                "file_path": f"{repo_name}/{rel_path}.md", # Fake extension for UI clarity
                                "file_name": os.path.basename(proto_path),
                                "repo_name": repo_name,
                                "system_version": system_ver,
                                "git_branch": git_branch,
                                "original_format": "proto"
                            }
                        )
                        all_documents.append(doc)
                    except Exception as e:
                        print(f"Failed to convert {proto_path}: {e}")

    # --- INDEXING ---
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
    print(f"-> Created {len(nodes)} nodes.")

    print("Generating Embeddings...")
    # Create Index from NODES (not documents)
    VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)
    
    print(f"Persisting nodes for BM25 at {config.STORAGE_NODES_PATH}...")
    if not os.path.exists(config.STORAGE_NODES_PATH):
        os.makedirs(config.STORAGE_NODES_PATH)
        
    # Save to disk
    storage_context.persist(persist_dir=config.STORAGE_NODES_PATH)

    print(f"Ingestion Complete!")

if __name__ == "__main__":
    ingest_code()
