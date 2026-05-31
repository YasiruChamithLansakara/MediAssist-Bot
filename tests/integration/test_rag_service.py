"""
Tests for the RAG / semantic search layer.

SQLite RAG has been removed. All semantic search is now handled by the
FAISS store (app.ml.faiss_store).  This file tests:
  1. The RAGService compatibility stub — always disabled, no-op.
  2. The FAISS store interface — build, search, save/load.
"""

import os
import tempfile
import pytest

from app.services.rag_service import (
    RAGService,
    get_rag_service,
    is_rag_available,
    retrieve_drug_context,
    vectorize_drug_knowledge,
)


# ─────────────────────────────────────────────────────────────────────────────
# RAGService stub tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRAGServiceStub:
    """RAGService is a no-op stub — SQLite RAG has been removed."""

    def test_rag_service_init(self):
        """Stub initialises without error."""
        rag = RAGService()
        assert rag is not None

    def test_rag_service_disabled(self):
        """Stub is always disabled."""
        rag = RAGService()
        assert rag.is_available() is False

    def test_rag_service_status(self):
        """Status dict has the expected shape."""
        rag = RAGService()
        status = rag.get_status()
        assert isinstance(status, dict)
        assert "rag_enabled" in status
        assert "provider" in status
        assert status["rag_enabled"] is False

    def test_retrieve_context_returns_empty(self):
        """Stub always returns an empty list."""
        rag = RAGService()
        results = rag.retrieve_context("diabetes medication", top_k=5)
        assert results == []

    def test_vectorize_drugs_returns_zero(self):
        """Stub always returns 0."""
        rag = RAGService()
        count = rag.vectorize_drugs([{"name": "Aspirin"}])
        assert count == 0


class TestRAGHelpers:
    """Module-level helper functions are backward-compatible stubs."""

    def test_is_rag_available_false(self):
        available = is_rag_available()
        assert available is False

    def test_get_rag_service_returns_instance(self):
        rag = get_rag_service()
        assert rag is not None
        assert hasattr(rag, "is_available")
        assert hasattr(rag, "retrieve_context")
        assert hasattr(rag, "vectorize_drugs")

    def test_retrieve_drug_context_empty(self):
        results = retrieve_drug_context("aspirin", top_k=3)
        assert results == []

    def test_vectorize_drug_knowledge_zero(self):
        count = vectorize_drug_knowledge([{"name": "Test"}])
        assert count == 0


# ─────────────────────────────────────────────────────────────────────────────
# FAISS store tests
# ─────────────────────────────────────────────────────────────────────────────

try:
    import faiss  # noqa: F401
    from app.ml.faiss_store import FAISSStore
    from app.ml.embeddings import is_available as embeddings_available
    _FAISS_TEST_DEPS = True
except ImportError:
    _FAISS_TEST_DEPS = False


@pytest.mark.skipif(
    not _FAISS_TEST_DEPS,
    reason="faiss-cpu or sentence-transformers not installed",
)
class TestFAISSStore:
    """FAISS store: build, search, persist."""

    SAMPLE_DRUGS = [
        {
            "drug_id": "met001",
            "generic_name": "metformin",
            "generic_name_clean": "metformin",
            "brand_names": "Glucophage",
            "drug_class": "Biguanide",
            "indications": "Type 2 diabetes management",
            "warnings": "Lactic acidosis risk; hold before contrast imaging",
            "contraindications": "Kidney disease, liver failure",
            "disease_category": "diabetes",
        },
        {
            "drug_id": "lis001",
            "generic_name": "lisinopril",
            "generic_name_clean": "lisinopril",
            "brand_names": "Zestril",
            "drug_class": "ACE inhibitor",
            "indications": "Hypertension and heart failure",
            "warnings": "Dry cough, angioedema",
            "contraindications": "Pregnancy",
            "disease_category": "hypertension",
        },
        {
            "drug_id": "asp001",
            "generic_name": "aspirin",
            "generic_name_clean": "aspirin",
            "brand_names": "Bayer",
            "drug_class": "NSAID / antiplatelet",
            "indications": "Pain, fever, antiplatelet therapy",
            "warnings": "Bleeding risk",
            "contraindications": "Active peptic ulcer",
            "disease_category": "heart_disease",
        },
    ]

    def test_faiss_store_init(self):
        """Store can be instantiated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FAISSStore(index_path=tmpdir)
            assert store is not None
            assert not store.is_ready()

    def test_faiss_store_build(self):
        """Build succeeds when embeddings are available."""
        if not embeddings_available():
            pytest.skip("No embedding backend available")

        with tempfile.TemporaryDirectory() as tmpdir:
            store = FAISSStore(index_path=tmpdir)
            count = store.build(self.SAMPLE_DRUGS)
            assert count == len(self.SAMPLE_DRUGS)
            assert store.is_ready()
            assert store.vector_count() == len(self.SAMPLE_DRUGS)

    def test_faiss_store_search(self):
        """Search returns relevant results."""
        if not embeddings_available():
            pytest.skip("No embedding backend available")

        with tempfile.TemporaryDirectory() as tmpdir:
            store = FAISSStore(index_path=tmpdir)
            store.build(self.SAMPLE_DRUGS)

            results = store.search("diabetes sugar medication", top_k=2)
            assert isinstance(results, list)
            assert len(results) > 0
            # Metformin should rank highest for a diabetes query
            top = results[0]
            assert "drug_name" in top
            assert "similarity" in top
            assert 0.0 <= top["similarity"] <= 1.0

    def test_faiss_store_search_empty_when_not_ready(self):
        """Search on un-built store returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FAISSStore(index_path=tmpdir)
            results = store.search("aspirin", top_k=3)
            assert results == []

    def test_faiss_store_save_and_load(self):
        """Index survives a save → load round-trip."""
        if not embeddings_available():
            pytest.skip("No embedding backend available")

        with tempfile.TemporaryDirectory() as tmpdir:
            store = FAISSStore(index_path=tmpdir)
            store.build(self.SAMPLE_DRUGS)
            assert store.save() is True

            store2 = FAISSStore(index_path=tmpdir)
            assert store2.load() is True
            assert store2.is_ready()
            assert store2.vector_count() == len(self.SAMPLE_DRUGS)

            results = store2.search("blood pressure hypertension", top_k=1)
            assert len(results) > 0

    def test_faiss_status_dict(self):
        """status() returns expected keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FAISSStore(index_path=tmpdir)
            st = store.status()
            assert "faiss_available" in st
            assert "index_ready" in st
            assert "vector_count" in st
            assert "dimension" in st


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
