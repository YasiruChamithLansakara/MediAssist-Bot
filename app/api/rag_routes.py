# -------------------------------------------
#   RAG API ROUTES (MediAssist)
#   FAISS-powered semantic search endpoints
#   SQLite RAG removed — all queries served
#   directly from the FAISS store singleton.
# -------------------------------------------

from fastapi import APIRouter, Query, HTTPException
from typing import Dict, Any

from app.ml.faiss_store import get_faiss_store

router = APIRouter(prefix="/api/rag", tags=["RAG"])


# =====================================================
# HEALTH CHECK FOR RAG SYSTEM
# =====================================================
@router.get("/status")
def rag_status():
    """Returns FAISS index status (replaces old SQLite RAG status)."""
    return get_faiss_store().status()


# =====================================================
# VECTOR SEARCH ENDPOINT (MAIN FEATURE)
# =====================================================
@router.get("/search")
def rag_search(
    query: str = Query(..., description="User query for semantic search"),
    top_k: int = Query(5, ge=1, le=20, description="Number of results"),
):
    """
    Perform FAISS semantic search over the drug knowledge base.
    Results are ranked by cosine similarity (inner-product on L2-normalised vectors).
    """
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        store = get_faiss_store()
        if not store.is_ready():
            raise HTTPException(
                status_code=503,
                detail="FAISS index not ready — server is still starting up or dataset is missing",
            )
        results = store.search(query, top_k=top_k)
        return {
            "query": query,
            "top_k": top_k,
            "results_count": len(results),
            "results": results,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG search failed: {str(exc)}")


# =====================================================
# SINGLE DRUG CONTEXT LOOKUP
# =====================================================
@router.get("/drug/{drug_name}")
def get_drug_context(drug_name: str):
    """Retrieve semantic context for a specific drug name via FAISS."""
    try:
        store = get_faiss_store()
        if not store.is_ready():
            raise HTTPException(status_code=503, detail="FAISS index not ready")
        results = store.search(drug_name, top_k=3)
        return {"drug": drug_name, "results": results}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Drug lookup failed: {str(exc)}")


# =====================================================
# INDEX BUILD (DISABLED — rebuild at startup instead)
# =====================================================
@router.post("/index")
def build_index(payload: Dict[str, Any]):
    """
    Manual re-indexing is not supported via API.
    The FAISS index is built automatically at server startup from the drug dataset.
    Restart the server (or delete data/faiss_index/) to trigger a rebuild.
    """
    raise HTTPException(
        status_code=501,
        detail=(
            "Manual re-indexing via API is disabled. "
            "Restart the server (or delete data/faiss_index/) to rebuild the FAISS index."
        ),
    )
