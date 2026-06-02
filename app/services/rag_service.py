"""
RAG Service — compatibility shim (SQLite RAG removed).

All semantic search is now handled exclusively by the FAISS store
(app.ml.faiss_store). This module is kept as a no-op shim so that
existing imports in main.py, llm_service.py, and tests continue to
work without errors. It does NOT load sentence-transformers, OpenAI,
or any SQLite database.

To perform semantic search use:
    from app.ml.faiss_store import get_faiss_store
    results = get_faiss_store().search(query, top_k=5)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("mediassist.rag")

# Always disabled — FAISS is the sole vector backend
RAG_ENABLED = False


class RAGService:
    """
    No-op compatibility stub.

    The previous implementation used SQLite as a vector store with an
    OpenAI / sentence-transformers embedder loaded at request time,
    causing a ~56-second startup delay. That code has been removed.

    All callers should use app.ml.faiss_store.get_faiss_store() instead.
    """

    def __init__(self):
        self.enabled = False
        self.embedder = None
        self.vector_db = None

    def is_available(self) -> bool:
        return False

    def retrieve_context(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Always returns empty — use get_faiss_store().search() instead."""
        return []

    def vectorize_drugs(self, drugs: List[Dict[str, Any]]) -> int:
        """No-op — FAISS index is built at startup from the drug dataset."""
        return 0

    def get_status(self) -> Dict[str, Any]:
        return {
            "rag_enabled": False,
            "provider": None,
            "dimension": None,
            "vector_count": 0,
            "db_path": None,
            "note": "SQLite RAG removed; semantic search is FAISS-only.",
        }


# ── module-level helpers (backward-compat) ────────────────────────────────────

_singleton: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """Return the (disabled) RAGService singleton."""
    global _singleton
    if _singleton is None:
        _singleton = RAGService()
    return _singleton


def is_rag_available() -> bool:
    """Always False — FAISS is used instead."""
    return False


def retrieve_drug_context(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """No-op stub — use get_faiss_store().search() directly."""
    return []


def vectorize_drug_knowledge(drugs: List[Dict[str, Any]]) -> int:
    """No-op stub."""
    return 0


def get_rag_status() -> Dict[str, Any]:
    """Return disabled status dict."""
    return get_rag_service().get_status()
