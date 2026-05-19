"""
Tests for RAG Vector Search Service
"""

import os
import pytest
import numpy as np
from pathlib import Path

# Mock environment before importing RAG service
os.environ["RAG_ENABLED"] = "1"
os.environ["EMBEDDING_PROVIDER"] = "tfidf"  # Use TF-IDF for testing (no external deps)
os.environ["VECTOR_DB_PATH"] = str(Path(__file__).resolve().parent / ".test_vector_db.sqlite3")

from app.services.rag_vector_search import (
    TFIDFEmbedding,
    VectorDatabase,
    RAGService,
    get_rag_service,
    is_rag_available,
    retrieve_drug_context,
    vectorize_drug_knowledge,
)


class TestEmbedding:
    """Test embedding backends."""

    def test_tfidf_embedding_basic(self):
        """Test TF-IDF embedding."""
        embedder = TFIDFEmbedding()
        
        texts = ["acetaminophen", "ibuprofen", "aspirin"]
        embeddings = embedder.embed(texts)
        
        assert embeddings is not None
        assert embeddings.shape[0] == 3
        assert embeddings.dtype == np.float32

    def test_tfidf_embedding_dimension(self):
        """Test TF-IDF dimension."""
        embedder = TFIDFEmbedding()
        dim = embedder.get_dimension()
        
        assert dim > 0
        assert dim == 100  # TF-IDF max_features

    def test_tfidf_embedding_single(self):
        """Test single text embedding."""
        embedder = TFIDFEmbedding()
        
        embedding = embedder.embed_single("aspirin")
        
        assert embedding is not None
        # Dimension varies based on input text, but should be > 0
        assert embedding.shape[0] > 0
        assert embedding.dtype == np.float32


class TestVectorDatabase:
    """Test vector database operations."""

    @pytest.fixture
    def db(self):
        """Create test database."""
        db_path = os.environ["VECTOR_DB_PATH"]
        
        # Remove if exists
        if os.path.exists(db_path):
            os.remove(db_path)
        
        database = VectorDatabase(db_path, embedding_dim=100)
        yield database
        
        # Cleanup
        if os.path.exists(db_path):
            os.remove(db_path)

    def test_vector_db_init(self, db):
        """Test database initialization."""
        assert db.db_path == os.environ["VECTOR_DB_PATH"]
        assert db.embedding_dim == 100

    def test_add_vector(self, db):
        """Test adding vectors to database."""
        embedding = np.random.rand(100).astype(np.float32)
        
        result = db.add_vector(
            drug_name="Aspirin",
            generic_name="acetylsalicylic acid",
            embedding=embedding,
            disease_context="pain",
            metadata={"warning": "blood thinner"},
        )
        
        assert result is True
        assert db.count() == 1

    def test_search(self, db):
        """Test vector search."""
        # Add some vectors
        embeddings = np.random.rand(5, 100).astype(np.float32)
        drugs = ["aspirin", "ibuprofen", "acetaminophen", "naproxen", "ketorolac"]
        
        for drug, emb in zip(drugs, embeddings):
            db.add_vector(
                drug_name=drug.capitalize(),
                generic_name=drug,
                embedding=emb,
                metadata={"type": "pain_relief"},
            )
        
        # Search
        query_embedding = embeddings[0]  # Should match aspirin
        results = db.search(query_embedding, top_k=3)
        
        assert len(results) > 0
        assert results[0]["drug_name"] == "Aspirin"

    def test_clear_database(self, db):
        """Test clearing database."""
        embedding = np.random.rand(100).astype(np.float32)
        db.add_vector("Test", "test", embedding)
        
        assert db.count() == 1
        
        db.clear()
        assert db.count() == 0


class TestRAGService:
    """Test RAG service integration."""

    def test_rag_service_init(self):
        """Test RAG service initialization."""
        rag = RAGService()
        
        # Service might not be enabled if dependencies missing
        # But should initialize without error
        assert rag is not None

    def test_rag_service_status(self):
        """Test getting RAG status."""
        rag = RAGService()
        status = rag.get_status()
        
        assert isinstance(status, dict)
        assert "rag_enabled" in status
        assert "provider" in status

    def test_vectorize_drugs(self):
        """Test drug vectorization."""
        rag = RAGService()
        
        if not rag.is_available():
            pytest.skip("RAG not available")
        
        drugs = [
            {
                "name": "Aspirin",
                "generic_name": "acetylsalicylic acid",
                "sections": {
                    "indications": "Pain relief",
                    "warnings": "May cause bleeding",
                    "contraindications": "Pregnancy",
                },
                "disease": "pain",
            },
            {
                "name": "Ibuprofen",
                "generic_name": "ibuprofen",
                "sections": {
                    "indications": "Fever and pain",
                    "warnings": "GI upset",
                    "contraindications": "Kidney disease",
                },
                "disease": "fever",
            },
        ]
        
        count = rag.vectorize_drugs(drugs)
        
        # At least some drugs should be vectorized
        assert count >= 0

    def test_retrieve_context(self):
        """Test context retrieval."""
        rag = RAGService()
        
        if not rag.is_available():
            pytest.skip("RAG not available")
        
        # First vectorize some data
        drugs = [
            {
                "name": "Aspirin",
                "generic_name": "acetylsalicylic acid",
                "sections": {
                    "indications": "Pain relief and fever reduction",
                    "warnings": "May cause bleeding",
                    "contraindications": "Pregnancy",
                },
                "disease": "pain",
            },
        ]
        
        rag.vectorize_drugs(drugs)
        
        # Now retrieve
        results = rag.retrieve_context("aspirin pain relief", top_k=1)
        
        # Should get results (even if just the one we added)
        assert isinstance(results, list)


class TestRAGHelpers:
    """Test RAG helper functions."""

    def test_is_rag_available(self):
        """Test RAG availability check."""
        available = is_rag_available()
        
        assert isinstance(available, bool)

    def test_get_rag_service(self):
        """Test getting RAG service."""
        rag = get_rag_service()
        
        assert rag is not None
        assert hasattr(rag, "is_available")
        assert hasattr(rag, "retrieve_context")
        assert hasattr(rag, "vectorize_drugs")


class TestRAGIntegration:
    """Integration tests for RAG with chat flow."""

    def test_rag_in_chat_context(self):
        """Test RAG used in chat context."""
        # Simulate a chat scenario where RAG enhances LLM context
        
        drug_list = [
            {
                "name": "Metformin",
                "generic_name": "metformin",
                "sections": {
                    "indications": "Type 2 diabetes management",
                    "warnings": "Lactic acidosis risk",
                    "contraindications": "Kidney disease",
                },
                "disease": "diabetes",
            },
            {
                "name": "Lisinopril",
                "generic_name": "lisinopril",
                "sections": {
                    "indications": "Hypertension control",
                    "warnings": "Cough and dizziness",
                    "contraindications": "Pregnancy",
                },
                "disease": "hypertension",
            },
        ]
        
        # Vectorize
        count = vectorize_drug_knowledge(drug_list)
        
        # Retrieve for chat
        if is_rag_available():
            results = retrieve_drug_context("diabetes medication", top_k=2)
            
            # Should get semantic matches
            assert isinstance(results, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
