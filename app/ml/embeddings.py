# -------------------------------------------
#   EMBEDDING SERVICE (MediAssist)
#   Supports OpenAI + safe fallback handling
#   Used for FAISS + RAG pipeline
# -------------------------------------------

from __future__ import annotations

import os
import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger("mediassist.embeddings")


class EmbeddingService:
    """
    Handles text → vector embeddings for RAG + FAISS.

    Default:
        - OpenAI embeddings (text-embedding-3-small)
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.client = None
        self.model = "text-embedding-3-small"

        self._init_client()

    # -------------------------------------
    # INITIALIZE OPENAI CLIENT
    # -------------------------------------
    def _init_client(self):
        try:
            from openai import OpenAI

            if not self.api_key:
                logger.warning("No OpenAI API key found for embeddings")
                return

            self.client = OpenAI(api_key=self.api_key)
            logger.info("EmbeddingService initialized (OpenAI)")
        except ImportError:
            logger.error(
                "OpenAI package not installed. Run: pip install openai"
            )
            self.client = None
        except Exception as e:
            logger.error(f"Embedding init failed: {e}")
            self.client = None

    # -------------------------------------
    # EMBED SINGLE TEXT
    # -------------------------------------
    def embed(self, text: str) -> Optional[List[float]]:
        """
        Convert single text → embedding vector
        """
        if not self.client:
            return self._fallback_embedding(text)

        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )

            vector = response.data[0].embedding
            return vector

        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return self._fallback_embedding(text)

    # -------------------------------------
    # EMBED BATCH TEXTS
    # -------------------------------------
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Convert multiple texts → embeddings
        """
        if not self.client:
            return [self._fallback_embedding(t) for t in texts]

        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )

            return [item.embedding for item in response.data]

        except Exception as e:
            logger.error(f"Batch embedding error: {e}")
            return [self._fallback_embedding(t) for t in texts]

    # -------------------------------------
    # FALLBACK EMBEDDING (SAFE MODE)
    # -------------------------------------
    def _fallback_embedding(self, text: str) -> List[float]:
        """
        Lightweight fallback when OpenAI is unavailable.
        NOT semantic — only structural placeholder.
        """

        np.random.seed(abs(hash(text)) % (2**32))
        vector = np.random.rand(1536).astype(np.float32)

        return vector.tolist()

    # -------------------------------------
    # GET VECTOR DIMENSION
    # -------------------------------------
    def get_dimension(self) -> int:
        """
        Return embedding dimension (must match FAISS)
        """
        return 1536