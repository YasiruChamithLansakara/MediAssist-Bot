# -------------------------------------------
#   RAG API ROUTES (MediAssist)
#   FAISS-powered semantic search endpoints
# -------------------------------------------

from fastapi import APIRouter, Query, HTTPException
from typing import List, Dict, Any

from app.services.rag_service import RAGService

router = APIRouter(prefix="/api/rag", tags=["RAG"])

# Global RAG instance
rag_service = RAGService()


# =====================================================
# HEALTH CHECK FOR RAG SYSTEM
# =====================================================
@router.get("/status")
def rag_status():
    """
    Returns current RAG system status
    """
    return rag_service.get_status()


# =====================================================
# VECTOR SEARCH ENDPOINT (MAIN FEATURE)
# =====================================================
@router.get("/search")
def rag_search(
    query: str = Query(..., description="User query for semantic search"),
    top_k: int = Query(5, ge=1, le=20, description="Number of results")
):
    """
    Perform FAISS-based semantic search over drug database
    """

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        results = rag_service.retrieve_context(query, top_k=top_k)

        return {
            "query": query,
            "top_k": top_k,
            "results_count": len(results),
            "results": results
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RAG search failed: {str(e)}"
        )


# =====================================================
# VECTORIZE / BUILD INDEX ENDPOINT
# =====================================================
@router.post("/index")
def build_index(payload: Dict[str, Any]):
    """
    Build FAISS index from drug dataset

    Expected payload:
    {
        "drugs": [...]
    }
    """

    drugs = payload.get("drugs", [])

    if not drugs:
        raise HTTPException(status_code=400, detail="No drug data provided")

    try:
        count = rag_service.vectorize_drugs(drugs)

        return {
            "message": "FAISS index built successfully",
            "indexed_documents": count
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Indexing failed: {str(e)}"
        )


# =====================================================
# SINGLE DRUG CONTEXT LOOKUP (OPTIONAL FEATURE)
# =====================================================
@router.get("/drug/{drug_name}")
def get_drug_context(drug_name: str):
    """
    Retrieve context for a specific drug name
    """

    try:
        results = rag_service.retrieve_context(drug_name, top_k=3)

        return {
            "drug": drug_name,
            "results": results
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Drug lookup failed: {str(e)}"
        )