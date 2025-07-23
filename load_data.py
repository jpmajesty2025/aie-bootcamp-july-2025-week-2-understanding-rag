import os
import re
import json
import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
import mmh3
from tqdm import tqdm

import openai
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import tiktoken
from scipy.sparse import csr_matrix

# --- Config ---
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")

REPO_PATHS = [
    r"C:\Projects\langchain",
    r"C:\Projects\fastapi"
]

BATCH_SIZE_EMBED = 16
BATCH_SIZE_MILVUS = 100
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 1536
SPARSE_MODEL = "naver/splade-cocondenser-ensembledistil"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# --- Utility Functions ---
def discover_files(base_paths: List[str]) -> List[Path]:
    """Discover files to process using dynamic inclusion/exclusion rules."""
    include_exts = {'.py', '.js', '.ts', '.jsx', '.tsx', '.ipynb', '.md', '.mdx', '.rst', '.txt'}
    important_dirs = {'src', 'lib', 'docs', 'examples', 'cookbook'}
    files = []
    for base in base_paths:
        for root, dirs, fs in os.walk(base):
            # Exclude build/test/artifact dirs
            if any(x in root for x in ['__pycache__', 'node_modules', 'dist', 'build', 'test', 'tests']):
                continue
            for f in fs:
                ext = os.path.splitext(f)[1].lower()
                if ext in include_exts:
                    files.append(Path(root) / f)
                elif any(d in root for d in important_dirs):
                    files.append(Path(root) / f)
    return files

def should_exclude_chunk_for_security(chunk_text: str, file_path: str) -> bool:
    sensitive_indicators = [
        r"api[_-]?key\s*[:=]", r"secret[_-]?key\s*[:=]", r"password\s*[:=]",
        r"token\s*[:=]", r"sk-[a-zA-Z0-9]+", r"pk-[a-zA-Z0-9]+",
        r"bearer\s+[a-zA-Z0-9]+", r"oauth[_-]?token"
    ]
    for pattern in sensitive_indicators:
        if re.search(pattern, chunk_text, re.IGNORECASE):
            return True
    if any(s in file_path.lower() for s in ['.env', 'secrets', 'credentials']):
        return True
    return False

def mmh3_hash(text: str) -> int:
    return mmh3.hash(text, signed=False)

# --- Chunking Strategies ---
def chunk_by_lines(text: str, n: int = 30) -> List[str]:
    lines = text.splitlines()
    return ['\n'.join(lines[i:i+n]) for i in range(0, len(lines), n)]

def chunk_markdown_by_header(text: str) -> List[str]:
    chunks = []
    current = []
    for line in text.splitlines():
        if line.strip().startswith('#') and current:
            chunks.append('\n'.join(current))
            current = []
        current.append(line)
    if current:
        chunks.append('\n'.join(current))
    return chunks

# --- Truncate Chunks to Token Limit ---
encoding = tiktoken.get_encoding("cl100k_base")

def truncate_chunk(chunk: str, max_tokens: int = 2000) -> str:
    tokens = encoding.encode(chunk, disallowed_special=())
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        return encoding.decode(tokens)
    return chunk

# Set up a logger for clamping info
clamp_logger = logging.getLogger("sparse_clamp")
clamp_handler = logging.FileHandler("sparse_clamp.log")
clamp_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
clamp_logger.addHandler(clamp_handler)
clamp_logger.setLevel(logging.INFO)

def dense_to_sparse(vec, threshold=1e-6):
    """Convert a dense vector to a scipy.sparse.csr_matrix with thresholding and non-negativity. Log clamping stats."""
    total = len(vec)
    clamped = sum(1 for v in vec if v < 0)
    sparse_vec = [v if v > threshold else 0.0 for v in vec]  # Only keep positive values above threshold
    # Log the fraction of values clamped
    if total > 0:
        clamp_logger.info(f"Clamped {clamped}/{total} ({clamped/total:.2%}) negative values to zero in sparse vector.")
    return csr_matrix([sparse_vec])

# --- Embedding Functions ---
def batch_openai_embed(texts: List[str], model: str = EMBEDDING_MODEL, dim: int = EMBEDDING_DIM, max_retries: int = 3) -> List[Optional[List[float]]]:
    openai.api_key = OPENAI_API_KEY
    results = [None] * len(texts)
    for i in range(0, len(texts), BATCH_SIZE_EMBED):
        batch = texts[i:i+BATCH_SIZE_EMBED]
        for attempt in range(max_retries):
            try:
                response = openai.embeddings.create(
                    model=model,
                    input=batch,
                    dimensions=dim
                )
                for j, emb in enumerate(response.data):
                    results[i+j] = emb.embedding
                break
            except Exception as e:
                logging.warning(f"OpenAI embedding batch failed (attempt {attempt+1}): {e}")
                time.sleep(2 ** attempt)
        else:
            logging.error(f"Failed to embed batch: {batch}")
    return results

def batch_splade_embed(texts: List[str], model: SentenceTransformer) -> List[csr_matrix]:
    dense_vecs = model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
    return [dense_to_sparse(vec) for vec in dense_vecs]

# --- Milvus Insert ---
def batch_insert(collection_name: str, records: List[Dict[str, Any]], vector_field: str, batch_size: int = BATCH_SIZE_MILVUS):
    collection = Collection(collection_name)
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        try:
            data = [
                [r['primary_key'] for r in batch],
                [r[vector_field] for r in batch],
                [r['text'] for r in batch],
                [json.dumps(r['metadata']) for r in batch]
            ]
            collection.insert(data)
        except Exception as e:
            logging.error(f"Milvus insert failed: {e} (batch {i}-{i+batch_size})")

# --- Main Data Loading Logic ---
def process_and_load():
    # Connect to Milvus
    connections.connect(uri=MILVUS_URI, token=MILVUS_TOKEN)
    # Load SPLADE model for sparse embeddings
    splade_model = SentenceTransformer(SPARSE_MODEL)
    files = discover_files(REPO_PATHS)
    logging.info(f"Discovered {len(files)} files to process.")
    dense_records, sparse_records = [], []
    for file_path in tqdm(files, desc="Processing files"):
        try:
            with open(file_path, encoding='utf-8', errors='ignore') as f:
                text = f.read()
        except Exception as e:
            logging.warning(f"Failed to read {file_path}: {e}")
            continue
        ext = file_path.suffix.lower()
        # Choose chunking strategy
        if ext in {'.md', '.mdx', '.rst', '.txt'}:
            chunks = chunk_markdown_by_header(text)
            chunking_strategy = 'markdown_header'
            content_type = 'documentation'
        else:
            chunks = chunk_by_lines(text, n=30)
            chunking_strategy = 'by_lines'
            content_type = 'code'
        # Filter and prepare records
        filtered_chunks = []
        for chunk in chunks:
            if not chunk.strip():
                continue
            if should_exclude_chunk_for_security(chunk, str(file_path)):
                logging.info(f"Excluded sensitive chunk in {file_path}")
                continue
            # Truncate long chunks to avoid OpenAI token limit
            chunk = truncate_chunk(chunk)
            filtered_chunks.append(chunk)
        if not filtered_chunks:
            continue
        # Prepare metadata
        metadatas = [
            {
                "repo_name": file_path.parts[2] if len(file_path.parts) > 2 else str(file_path),
                "file_path": str(file_path),
                "chunking_strategy": chunking_strategy,
                "content_type": content_type,
                "chunk_size": len(chunk),
                "language": ext.lstrip('.'),
            }
            for chunk in filtered_chunks
        ]
        # Embedding
        dense_vecs = batch_openai_embed(filtered_chunks)
        sparse_vecs = batch_splade_embed(filtered_chunks, splade_model)
        # Prepare Milvus records
        for chunk, meta, dense, sparse in zip(filtered_chunks, metadatas, dense_vecs, sparse_vecs):
            if dense is not None:
                dense_records.append({
                    "primary_key": mmh3_hash(chunk),
                    "dense_vector": dense,
                    "text": chunk,
                    "metadata": meta
                })
            if sparse is not None:
                sparse_records.append({
                    "primary_key": mmh3_hash(chunk),
                    "sparse_vector": sparse,  # Now a csr_matrix
                    "text": chunk,
                    "metadata": meta
                })
        # Batch insert periodically
        if len(dense_records) >= BATCH_SIZE_MILVUS:
            batch_insert('code_dense' if content_type == 'code' else 'docs_dense', dense_records, 'dense_vector')
            dense_records.clear()
        if len(sparse_records) >= BATCH_SIZE_MILVUS:
            batch_insert('code_sparse' if content_type == 'code' else 'docs_sparse', sparse_records, 'sparse_vector')
            sparse_records.clear()
    # Final insert
    if dense_records:
        batch_insert('code_dense', dense_records, 'dense_vector')
    if sparse_records:
        batch_insert('code_sparse', sparse_records, 'sparse_vector')
    logging.info("Data loading complete.")

if __name__ == "__main__":
    process_and_load() 