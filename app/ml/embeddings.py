"""
Embedding Service — MediAssist
Provides text → vector embeddings for FAISS + RAG pipeline.

Backend priority (auto-detected at startup):
  1. sentence-transformers / all-MiniLM-L6-v2  (local, free, semantic — PREFERRED)
  2. OpenAI text-embedding-3-small              (cloud, paid — requires LLM_API_KEY)
  3. TF-IDF character n-gram fallback           (no GPU/network required, NOT semantic)

Set EMBEDDING_BACKEND=openai  to force OpenAI.
Set EMBEDDING_BACKEND=tfidf   to force TF-IDF (testing only).
Default: tries sentence-transformers first, then OpenAI, then TF-IDF.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np

logger = logging.getLogger("mediassist.embeddings")

# ── env config ──────────────────────────────────────────────────────────────
_BACKEND_ENV = os.getenv("EMBEDDING_BACKEND", "auto").strip().lower()
_ST_MODEL_NAME = os.getenv(
    "ST_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)
_OPENAI_KEY = os.getenv("LLM_API_KEY", "").strip()
_OPENAI_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Dimensions per backend
_ST_DIM = 384           # all-MiniLM-L6-v2
_OPENAI_DIM = 1536      # text-embedding-3-small
_TFIDF_DIM = 256        # max_features


# ════════════════════════════════════════════════════════════════════════════
# BACKEND CLASSES
# ════════════════════════════════════════════════════════════════════════════

class _SentenceTransformersBackend:
    """Local, free, semantic embeddings via sentence-transformers."""

    dim = _ST_DIM

    def __init__(self):
        self._model = None
        self._ready = False
        self._try_load()

    def _try_load(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(_ST_MODEL_NAME)
            self._ready = True
            logger.info(
                "EmbeddingService: sentence-transformers loaded (%s, dim=%d)",
                _ST_MODEL_NAME, self.dim,
            )
        except ImportError:
            logger.warning(
                "sentence-transformers not installed — run: "
                "pip install sentence-transformers"
            )
        except Exception as exc:
            logger.warning("sentence-transformers load failed: %s", exc)

    @property
    def ready(self) -> bool:
        return self._ready

    def embed(self, texts: List[str]) -> Optional[np.ndarray]:
        if not self._ready or not self._model:
            return None
        try:
            vecs = self._model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=64,
            )
            return vecs.astype(np.float32)
        except Exception as exc:
            logger.error("sentence-transformers embed error: %s", exc)
            return None


class _OpenAIBackend:
    """Cloud embeddings via OpenAI API."""

    dim = _OPENAI_DIM

    def __init__(self):
        self._client = None
        self._ready = False
        self._try_init()

    def _try_init(self):
        if not _OPENAI_KEY:
            logger.debug("No LLM_API_KEY — OpenAI embeddings skipped")
            return
        try:
            import importlib
            openai = importlib.import_module("openai")
            self._client = openai.OpenAI(api_key=_OPENAI_KEY)
            self._ready = True
            logger.info(
                "EmbeddingService: OpenAI backend ready (model=%s, dim=%d)",
                _OPENAI_MODEL, self.dim,
            )
        except ImportError:
            logger.warning("openai package not installed")
        except Exception as exc:
            logger.warning("OpenAI embedding init failed: %s", exc)

    @property
    def ready(self) -> bool:
        return self._ready

    def embed(self, texts: List[str]) -> Optional[np.ndarray]:
        if not self._ready or not self._client:
            return None
        try:
            resp = self._client.embeddings.create(
                model=_OPENAI_MODEL, input=texts
            )
            vecs = [item.embedding for item in resp.data]
            return np.array(vecs, dtype=np.float32)
        except Exception as exc:
            logger.error("OpenAI embed error: %s", exc)
            return None


class _TFIDFBackend:
    """
    Character n-gram TF-IDF fallback.
    NOT semantically meaningful — use only when nothing else is available.
    """

    dim = _TFIDF_DIM

    def __init__(self):
        self._vec = None
        self._ready = False
        self._try_init()

    def _try_init(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vec = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 4),
                max_features=self.dim,
                sublinear_tf=True,
            )
            self._ready = True
            logger.info("EmbeddingService: TF-IDF fallback backend ready")
        except ImportError:
            logger.warning("scikit-learn not installed — TF-IDF unavailable")

    @property
    def ready(self) -> bool:
        return self._ready

    def embed(self, texts: List[str]) -> Optional[np.ndarray]:
        if not self._ready or not self._vec:
            return None
        try:
            mat = self._vec.fit_transform(texts).toarray()
            return mat.astype(np.float32)
        except Exception as exc:
            logger.error("TF-IDF embed error: %s", exc)
            return None


# ════════════════════════════════════════════════════════════════════════════
# SERVICE (auto-selects best available backend)
# ════════════════════════════════════════════════════════════════════════════

class EmbeddingService:
    """
    Unified embedding service.  Picks the best available backend at startup.

    Usage:
        svc = EmbeddingService()
        vec = svc.embed("metformin 500mg")          # → list[float] | None
        vecs = svc.embed_batch(["...", "..."])       # → list[list[float]]
        dim = svc.get_dimension()                    # → int
    """

    def __init__(self, api_key: Optional[str] = None):
        # Allow caller to override API key
        if api_key:
            os.environ["LLM_API_KEY"] = api_key

        self._backend = self._select_backend()

    def _select_backend(self):
        if _BACKEND_ENV == "openai":
            b = _OpenAIBackend()
            if b.ready:
                return b
            logger.warning("OpenAI backend requested but unavailable; falling back")

        if _BACKEND_ENV == "tfidf":
            return _TFIDFBackend()

        # auto — try sentence-transformers first (free, local, semantic)
        b_st = _SentenceTransformersBackend()
        if b_st.ready:
            return b_st

        # then OpenAI
        b_oa = _OpenAIBackend()
        if b_oa.ready:
            return b_oa

        # last resort — TF-IDF
        logger.warning(
            "No semantic embedding backend available. "
            "Install sentence-transformers for proper FAISS search: "
            "pip install sentence-transformers"
        )
        return _TFIDFBackend()

    # ── public API ───────────────────────────────────────────────────────

    def embed(self, text: str) -> Optional[List[float]]:
        """Embed a single text → list[float] | None"""
        vecs = self._backend.embed([text])
        if vecs is None or len(vecs) == 0:
            return None
        return vecs[0].tolist()

    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Embed multiple texts → list of vectors (None for failed items)"""
        if not texts:
            return []
        vecs = self._backend.embed(texts)
        if vecs is None:
            return [None] * len(texts)
        return [row.tolist() for row in vecs]

    def get_dimension(self) -> int:
        return self._backend.dim

    def is_ready(self) -> bool:
        return self._backend.ready

    @property
    def backend_name(self) -> str:
        return type(self._backend).__name__


# ════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL HELPERS  (used by faiss_store.py and rag_service.py)
# ════════════════════════════════════════════════════════════════════════════

_service: Optional[EmbeddingService] = None


def _get_service() -> EmbeddingService:
    global _service
    if _service is None:
        _service = EmbeddingService()
    return _service


def embed_single(text: str) -> Optional[np.ndarray]:
    """Return (dim,) float32 array, or None."""
    vec = _get_service().embed(text)
    if vec is None:
        return None
    return np.array(vec, dtype=np.float32)


def embed_texts(texts: List[str]) -> Optional[np.ndarray]:
    """Return (n, dim) float32 array, or None."""
    svc = _get_service()
    vecs = svc._backend.embed(texts)
    if vecs is None:
        return None
    return np.array(vecs, dtype=np.float32)


def normalize_vectors(arr: np.ndarray) -> np.ndarray:
    """L2-normalise rows in place (required for FAISS cosine similarity)."""
    arr = np.array(arr, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (arr / norms).astype(np.float32)


def warmup_embeddings() -> None:
    """
    Pre-load sentence-transformers in a background thread at server startup.

    When FAISS loads from disk it skips building vectors (no model needed).
    Without this warmup the embedding model loads lazily on the FIRST semantic
    search call — triggering 30+ HuggingFace HTTP checks inside a live request
    and causing a ~50-second response / client timeout.
    """
    import logging
    import threading

    log = logging.getLogger("mediassist.embeddings")

    def _load() -> None:
        log.info("EmbeddingService: warming up sentence-transformers in background …")
        try:
            svc = _get_service()
            if svc.is_ready():
                embed_single("warmup")   # forces tokenizer + model fully into memory
                log.info(
                    "EmbeddingService: %s ready (dim=%d)",
                    svc.backend_name,
                    svc.get_dimension(),
                )
            else:
                log.warning("EmbeddingService: no semantic backend available")
        except Exception as exc:
            log.warning("EmbeddingService warmup failed: %s", exc)

    threading.Thread(target=_load, daemon=True, name="embeddings-warmup").start()


def is_available() -> bool:
    """Return True if a semantic (non-TF-IDF) embedding backend is ready."""
    try:
        svc = _get_service()
        return svc.is_ready() and "TFIDF" not in svc.backend_name
    except Exception:
        return False


def is_available() -> bool:
    """True when a *semantic* backend (sentence-transformers or OpenAI) is active."""
    svc = _get_service()
    return svc.is_ready() and not isinstance(svc._backend, _TFIDFBackend)
