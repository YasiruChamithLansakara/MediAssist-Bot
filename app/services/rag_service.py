from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional

from app.ml.faiss_store import FAISSStore
from app.ml.embeddings import EmbeddingService

logger = logging.getLogger("mediassist.rag")


# =====================================================
# RAG SERVICE (FAISS + EMBEDDINGS)
# =====================================================
class RAGService:
    """
    FAISS-based Retrieval Augmented Generation system
    for MediAssist drug knowledge base.
    """

    def __init__(self):
        self.embedding_service = EmbeddingService(
            api_key=os.getenv("LLM_API_KEY")
        )

        self.vector_store = FAISSStore(dim=1536)

        self.initialized: bool = False
        self._cache_size: int = 0

    # =====================================================
    # INDEX BUILDING
    # =====================================================
    def vectorize_drugs(self, drugs: List[Dict[str, Any]]) -> int:
        """
        Build FAISS index from drug dataset
        """

        if not drugs:
            logger.warning("No drug data provided for vectorization")
            return 0

        embeddings: List[List[float]] = []
        metadata: List[Dict[str, Any]] = []

        for drug in drugs:

            drug_name = (
                drug.get("generic_name")
                or drug.get("generic_name_clean")
                or drug.get("name")
                or "Unknown"
            )

            text = self._build_drug_text(drug, drug_name)

            try:
                embedding = self.embedding_service.embed(text)
            except Exception as e:
                logger.error(f"Embedding failed for {drug_name}: {e}")
                continue

            embeddings.append(embedding)

            metadata.append(
                {
                    "drug_name": drug_name,
                    "generic": drug.get("generic_name"),
                    "indications": drug.get("indications", ""),
                    "warnings": drug.get("warnings", ""),
                    "side_effects": drug.get("side_effects_all", ""),
                    "raw_text": text,
                }
            )

        if not embeddings:
            logger.error("No embeddings generated")
            return 0

        self.vector_store.add(embeddings, metadata)

        self.initialized = True
        self._cache_size = len(embeddings)

        logger.info(f"FAISS index built with {self._cache_size} drugs")

        return self._cache_size

    # =====================================================
    # TEXT BUILDER (IMPORTANT FOR QUALITY)
    # =====================================================
    def _build_drug_text(self, drug: Dict[str, Any], name: str) -> str:
        return f"""
        Drug Name: {name}
        Indications: {drug.get('indications', 'Not available')}
        Warnings: {drug.get('warnings', 'Not available')}
        Side Effects: {drug.get('side_effects_all', 'Not available')}
        Dosage: {drug.get('dosage_and_administration', 'Not available')}
        Drug Class: {drug.get('drug_class', 'Unknown')}
        Route: {drug.get('route', 'Unknown')}
        """

    # =====================================================
    # SEARCH (FAISS RETRIEVAL)
    # =====================================================
    def retrieve_context(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:

        if not self.initialized:
            logger.warning("FAISS not initialized")
            return []

        try:
            query_embedding = self.embedding_service.embed(query)
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            return []

        results = self.vector_store.search(query_embedding, top_k)

        return self._format_results(results)

    # =====================================================
    # FORMAT OUTPUT (CLEAN API RESPONSE)
    # =====================================================
    def _format_results(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        formatted = []

        for r in results:
            formatted.append(
                {
                    "drug_name": r.get("drug_name"),
                    "score": r.get("score"),
                    "indications": r.get("indications"),
                    "warnings": r.get("warnings"),
                    "side_effects": r.get("side_effects"),
                }
            )

        return formatted

    # =====================================================
    # STATUS
    # =====================================================
    def get_status(self) -> Dict[str, Any]:
        return {
            "faiss_enabled": True,
            "initialized": self.initialized,
            "vector_count": self._cache_size,
        }