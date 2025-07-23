from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from pymilvus import connections, Collection
import logging

load_dotenv()

MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")

app = FastAPI()

# --- Pydantic Models ---
class SearchRequest(BaseModel):
    query: str
    limit: int = 5
    include_code: bool = True

class SearchResult(BaseModel):
    rank: int
    text: str
    relevance_score: float
    source_metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[SearchResult]
    strategy_breakdown: Dict[str, int]

# --- Milvus Connection ---
def get_collection(name: str) -> Collection:
    connections.connect(uri=MILVUS_URI, token=MILVUS_TOKEN)
    return Collection(name)

# --- Hybrid Search Logic (no reranking yet) ---
def hybrid_search(query: str, limit: int = 5, include_code: bool = True):
    # For now, just do a simple search in both dense and sparse collections, combine, deduplicate, and return top N by score
    results = []
    collections = []
    if include_code:
        collections += ["code_dense", "code_sparse"]
    collections += ["docs_dense", "docs_sparse"]
    seen = set()
    for coll_name in collections:
        coll = get_collection(coll_name)
        # For now, use a simple vector search with a dummy query vector (since we don't have embedding here)
        # In production, you would embed the query and search by vector
        # We'll just use a placeholder search for now
        try:
            # Placeholder: search by text field using a simple filter (not vector search)
            expr = f'text like "%{query}%"'
            res = coll.query(expr, output_fields=["primary_key", "text", "metadata"])
            for r in res:
                pk = r["primary_key"]
                if pk in seen:
                    continue
                seen.add(pk)
                results.append({
                    "text": r["text"],
                    "relevance_score": 1.0,  # Placeholder
                    "source_metadata": r["metadata"],
                    "strategy": coll_name
                })
        except Exception as e:
            logging.warning(f"Milvus search failed for {coll_name}: {e}")
    # Sort by score (all 1.0 for now), return top N
    results = results[:limit]
    # Add rank and breakdown
    for i, r in enumerate(results):
        r["rank"] = i + 1
    strategy_breakdown = {}
    for r in results:
        strategy_breakdown[r["strategy"]] = strategy_breakdown.get(r["strategy"], 0) + 1
    return {
        "query": query,
        "total_results": len(results),
        "results": [SearchResult(**{k: v for k, v in r.items() if k != "strategy"}) for r in results],
        "strategy_breakdown": strategy_breakdown
    }

# --- Endpoints ---
@app.post("/search", response_model=SearchResponse)
async def search_endpoint(request: SearchRequest):
    return hybrid_search(request.query, request.limit, request.include_code)

@app.post("/search/semantic")
async def semantic_search_endpoint(request: SearchRequest):
    raise HTTPException(status_code=501, detail="Semantic search endpoint not yet implemented. Use /search for full hybrid search.")

@app.post("/search/keyword")
async def keyword_search_endpoint(request: SearchRequest):
    raise HTTPException(status_code=501, detail="Keyword search endpoint not yet implemented. Use /search for full hybrid search.")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "collections": ["code_dense", "code_sparse", "docs_dense", "docs_sparse"]} 