"""
RAG Vector Search Service: Retrieval-Augmented Generation with vector embeddings.

Features:
- Drug knowledge base vectorization
- Semantic similarity search
- Multiple embedding backends (OpenAI, local)
- Efficient vector storage with SQLite
- Fallback to TF-IDF if embeddings unavailable
- Context-aware drug retrieval for LLM grounding
"""

from __future__ import annotations

import os
import json
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import hashlib

import numpy as np

logger = logging.getLogger("mediassist.rag")

# Configuration
RAG_ENABLED = os.getenv("RAG_ENABLED", "1").lower() in ("1", "true", "yes")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower()  # openai, local, tfidf
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))  # For text-embedding-3-small
LOCAL_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
VECTOR_DB_PATH = os.getenv(
    "VECTOR_DB_PATH",
    str(Path(__file__).resolve().parents[2] / "data" / "vector_db.sqlite3"),
)
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))

# Global state
_embedder = None
_vector_db = None
_embedder_lock = threading.Lock()
_db_lock = threading.Lock()


# =====================================================================
# EMBEDDING BACKENDS
# =====================================================================

class EmbeddingBackend:
    """Base class for embedding providers."""

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts. Returns (n, dimension) array."""
        raise NotImplementedError

    def embed_single(self, text: str) -> Optional[np.ndarray]:
        """Embed a single text. Returns (dimension,) array, or None on failure."""
        result = self.embed([text])
        if result is None:          # ← was crashing: len(None) has no len()
            return None
        return result[0] if len(result) > 0 else None

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        raise NotImplementedError


class OpenAIEmbedding(EmbeddingBackend):
    """OpenAI embedding backend."""

    def __init__(self, api_key: str, model: str = EMBEDDING_MODEL):
        self.api_key = api_key
        self.model = model
        self.client = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize OpenAI client."""
        try:
            try:
                openai = __import__("openai")
            except ImportError:
                logger.warning("OpenAI library not installed. Embeddings disabled.")
                self.client = None
                return

            # use imported openai module and set api key
            openai.api_key = self.api_key
            self.client = openai
            logger.info(f"OpenAI embedding initialized with model {self.model}")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI client: {e}")
            self.client = None

    def embed(self, texts: List[str]) -> Optional[np.ndarray]:
        """Embed texts using OpenAI API."""
        if not self.client:
            return None

        try:
            response = self.client.Embedding.create(
                input=texts,
                model=self.model,
            )
            embeddings = [item["embedding"] for item in response["data"]]
            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            return None

    def get_dimension(self) -> int:
        """Get dimension (OpenAI text-embedding-3-small is 1536)."""
        return EMBEDDING_DIMENSION


class LocalEmbedding(EmbeddingBackend):
    """Local embedding backend using sentence-transformers."""

    def __init__(self, model_name: str = LOCAL_MODEL):
        self.model_name = model_name
        self.model = None
        self._init_model()

    def _init_model(self) -> None:
        """Initialize local embedding model."""
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
            # get_embedding_dimension() is the current API name (≥3.x)
            # fall back to the deprecated name for older versions
            if hasattr(self.model, "get_embedding_dimension"):
                dim = self.model.get_embedding_dimension()
            else:
                dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Local embedding initialized with {self.model_name} (dim={dim})")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. Local embeddings disabled. "
                "Install with: pip install sentence-transformers"
            )
            self.model = None
        except Exception as e:
            logger.warning(f"Failed to initialize local embedding model: {e}")
            self.model = None

    def embed(self, texts: List[str]) -> Optional[np.ndarray]:
        """Embed texts using local model."""
        if not self.model:
            return None

        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Local embedding error: {e}")
            return None

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        if not self.model:
            return 384  # all-MiniLM-L6-v2 default
        if hasattr(self.model, "get_embedding_dimension"):
            return self.model.get_embedding_dimension()
        return self.model.get_sentence_embedding_dimension()


class TFIDFEmbedding(EmbeddingBackend):
    """TF-IDF fallback embedding."""

    def __init__(self):
        self.vectorizer = None
        self._init_vectorizer()

    def _init_vectorizer(self) -> None:
        """Initialize TF-IDF vectorizer."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words="english",
                lowercase=True,
                analyzer="char",
                ngram_range=(2, 3),
            )
            logger.info("TF-IDF embedding initialized (fallback mode)")
        except ImportError:
            logger.warning("scikit-learn not installed. TF-IDF disabled.")
            self.vectorizer = None

    def embed(self, texts: List[str]) -> Optional[np.ndarray]:
        """Embed texts using TF-IDF."""
        if not self.vectorizer:
            return None

        try:
            embeddings = self.vectorizer.fit_transform(texts).toarray()
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"TF-IDF embedding error: {e}")
            return None

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return 100  # TF-IDF max_features


# =====================================================================
# VECTOR DATABASE (SQLite)
# =====================================================================

class VectorDatabase:
    """SQLite-based vector database for drug embeddings."""

    def __init__(self, db_path: str, embedding_dim: int):
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database."""
        try:
            import sqlite3

            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS drug_vectors (
                    id INTEGER PRIMARY KEY,
                    drug_hash TEXT UNIQUE,
                    drug_name TEXT,
                    generic_name TEXT,
                    disease_context TEXT,
                    embedding BLOB,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # Create index on drug_hash
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_drug_hash ON drug_vectors(drug_hash)"
            )

            conn.commit()
            conn.close()
            logger.info(f"Vector database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {e}")

    def _drug_hash(self, drug_name: str, generic_name: str) -> str:
        """Create hash for drug identity."""
        key = f"{drug_name}:{generic_name}".lower()
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def add_vector(
        self,
        drug_name: str,
        generic_name: str,
        embedding: np.ndarray,
        disease_context: str = "",
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """Add drug vector to database."""
        try:
            import sqlite3

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            drug_hash = self._drug_hash(drug_name, generic_name)
            embedding_bytes = embedding.astype(np.float32).tobytes()
            metadata_str = json.dumps(metadata or {})

            cursor.execute(
                """
                INSERT OR REPLACE INTO drug_vectors 
                (drug_hash, drug_name, generic_name, disease_context, embedding, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (drug_hash, drug_name, generic_name, disease_context, embedding_bytes, metadata_str),
            )

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Failed to add vector: {e}")
            return False

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar drugs using cosine similarity."""
        try:
            import sqlite3

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT drug_name, generic_name, embedding, metadata FROM drug_vectors")
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return []

            # Calculate similarity scores
            results = []
            for drug_name, generic_name, embedding_bytes, metadata_str in rows:
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                # Cosine similarity
                similarity = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding) + 1e-8
                )

                if similarity >= SIMILARITY_THRESHOLD:
                    metadata = json.loads(metadata_str) if metadata_str else {}
                    results.append({
                        "drug_name": drug_name,
                        "generic_name": generic_name,
                        "similarity": float(similarity),
                        "metadata": metadata,
                    })

            # Sort by similarity and return top-k
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:top_k]
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def clear(self) -> bool:
        """Clear all vectors from database."""
        try:
            import sqlite3

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM drug_vectors")
            conn.commit()
            conn.close()
            logger.info("Vector database cleared")
            return True
        except Exception as e:
            logger.error(f"Failed to clear database: {e}")
            return False

    def count(self) -> int:
        """Get number of vectors in database."""
        try:
            import sqlite3

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM drug_vectors")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            logger.error(f"Failed to get count: {e}")
            return 0


# =====================================================================
# RAG SERVICE
# =====================================================================

class RAGService:
    """RAG Service for vector search and context retrieval."""

    def __init__(self):
        self.embedder = None
        self.vector_db = None
        self.enabled = False
        self._init()

    def _init(self) -> None:
        """Initialize RAG components."""
        if not RAG_ENABLED:
            logger.info("RAG disabled via RAG_ENABLED config")
            return

        # Initialize embedder — prefer sentence-transformers (free/local),
        # then OpenAI (paid), then TF-IDF (non-semantic fallback).
        try:
            if EMBEDDING_PROVIDER == "openai":
                api_key = os.getenv("LLM_API_KEY", "").strip()
                if api_key:
                    candidate = OpenAIEmbedding(api_key, EMBEDDING_MODEL)
                    # Verify it can actually embed (OpenAI package might be missing)
                    if candidate.client is not None:
                        self.embedder = candidate
                    else:
                        logger.warning(
                            "OpenAI embedder has no client (package not installed?); "
                            "falling back to sentence-transformers"
                        )
                        self.embedder = LocalEmbedding(LOCAL_MODEL)
                else:
                    logger.warning("No LLM_API_KEY — falling back to sentence-transformers")
                    self.embedder = LocalEmbedding(LOCAL_MODEL)
            elif EMBEDDING_PROVIDER == "local":
                self.embedder = LocalEmbedding(LOCAL_MODEL)
            else:
                self.embedder = TFIDFEmbedding()

            # Final check: can the selected embedder actually produce a vector?
            if self.embedder is not None:
                probe = self.embedder.embed(["test"])
                if probe is None:
                    logger.warning(
                        "Selected embedder (%s) returned None on probe — "
                        "falling back to sentence-transformers",
                        type(self.embedder).__name__,
                    )
                    self.embedder = LocalEmbedding(LOCAL_MODEL)
                    probe = self.embedder.embed(["test"])

                if probe is not None and len(probe) > 0:
                    dim = len(probe[0])
                    self.vector_db = VectorDatabase(VECTOR_DB_PATH, dim)
                    self.enabled = True
                    logger.info(
                        "RAG initialized: backend=%s, dimension=%d, db=%s",
                        type(self.embedder).__name__, dim, VECTOR_DB_PATH,
                    )
                else:
                    logger.warning("All embedding backends unavailable — RAG disabled")
        except Exception as e:
            logger.warning(f"RAG initialization failed: {e}")
            self.enabled = False

    def is_available(self) -> bool:
        """Check if RAG service is available."""
        return self.enabled

    def vectorize_drugs(self, drugs: List[Dict[str, Any]]) -> int:
        """
        Vectorize drug data and store in database.

        Args:
            drugs: List of drug dictionaries with 'name', 'generic_name', 'sections'

        Returns:
            Number of drugs successfully vectorized
        """
        if not self.enabled or not self.embedder:
            return 0

        count = 0
        for drug in drugs:
            try:
                # Prepare text for embedding
                drug_name = drug.get("generic_name") or drug.get("name", "Unknown")
                text_parts = [drug_name]

                # Add key information for semantic understanding
                sections = drug.get("sections", {})
                if sections.get("indications"):
                    text_parts.append(f"Used for: {sections['indications'][:200]}")
                if sections.get("contraindications"):
                    text_parts.append(f"Avoid if: {sections['contraindications'][:200]}")

                embedding_text = " | ".join(text_parts)

                # Get embedding
                embedding = self.embedder.embed_single(embedding_text)
                if embedding is None:
                    continue

                # Store in database
                metadata = {
                    "indications": sections.get("indications", "")[:100],
                    "warnings": sections.get("warnings", "")[:100],
                }

                if self.vector_db.add_vector(
                    drug_name=drug.get("name", drug_name),
                    generic_name=drug_name,
                    embedding=embedding,
                    disease_context=drug.get("disease", ""),
                    metadata=metadata,
                ):
                    count += 1
            except Exception as e:
                logger.debug(f"Failed to vectorize drug {drug.get('name', 'Unknown')}: {e}")
                continue

        logger.info(f"Vectorized {count}/{len(drugs)} drugs")
        return count

    def retrieve_context(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
        """
        Retrieve relevant drug context for a query.

        Args:
            query: User question or medication name
            top_k: Number of results to return

        Returns:
            List of relevant drugs with similarity scores
        """
        if not self.enabled or not self.embedder:
            return []

        try:
            # Get query embedding
            query_embedding = self.embedder.embed_single(query)
            if query_embedding is None:
                logger.warning(f"Failed to embed query: {query}")
                return []

            # Search vector database
            results = self.vector_db.search(query_embedding, top_k=top_k)
            logger.debug(f"RAG retrieved {len(results)} results for query: {query}")
            return results
        except Exception as e:
            logger.error(f"RAG retrieval error: {e}")
            return []

    def get_status(self) -> Dict[str, Any]:
        """Get RAG service status."""
        return {
            "rag_enabled": self.enabled,
            "provider": EMBEDDING_PROVIDER,
            "dimension": self.embedder.get_dimension() if self.embedder else None,
            "vector_count": self.vector_db.count() if self.vector_db else 0,
            "db_path": VECTOR_DB_PATH,
        }


# =====================================================================
# GLOBAL INSTANCE & HELPERS
# =====================================================================

def get_rag_service() -> RAGService:
    """Get or create RAG service singleton."""
    global _vector_db
    with _embedder_lock:
        if _vector_db is None:
            _vector_db = RAGService()
    return _vector_db


def is_rag_available() -> bool:
    """Check if RAG is available."""
    return get_rag_service().is_available()


def retrieve_drug_context(query: str, top_k: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
    """Helper: Retrieve context for a query."""
    rag = get_rag_service()
    return rag.retrieve_context(query, top_k=top_k)


def vectorize_drug_knowledge(drugs: List[Dict[str, Any]]) -> int:
    """Helper: Vectorize drug knowledge base."""
    rag = get_rag_service()
    return rag.vectorize_drugs(drugs)


def get_rag_status() -> Dict[str, Any]:
    """Helper: Get RAG status."""
    rag = get_rag_service()
    return rag.get_status()
