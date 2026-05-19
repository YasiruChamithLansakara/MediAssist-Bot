from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np

logger = logging.getLogger("mediassist.embeddings")

LOCAL_MODEL_NAME = os.getenv(
    "LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# ── sentence-transformers backend ─────────────────────────────────────────

_st_model = None
_ST_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    _st_model = SentenceTransformer(LOCAL_MODEL_NAME)
    # sentence-transformers renamed the method; use the new name when available
    try:
        _EMBEDDING_DIM = _st_model.get_embedding_dimension()
    except Exception:
        _EMBEDDING_DIM = _st_model.get_sentence_embedding_dimension()
    _ST_AVAILABLE = True
    logger.info("sentence-transformers loaded: %s (dim=%d)", LOCAL_MODEL_NAME, _EMBEDDING_DIM)
except ImportError:
    logger.warning(
        "sentence-transformers not installed. "
        "Run: pip install sentence-transformers"
    )
except Exception as exc:
    logger.warning("sentence-transformers load failed: %s", exc)


def embed_texts(texts: List[str]) -> Optional[np.ndarray]:
    """
    Embed a list of texts.
    Returns float32 array (n, dim), or None if no backend available.
    Primary: sentence-transformers   Fallback: TF-IDF
    """
    if not texts:
        return None

    if _ST_AVAILABLE and _st_model is not None:
        try:
            vecs = _st_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return vecs.astype(np.float32)
        except Exception as exc:
            logger.warning("sentence-transformers encode failed: %s", exc)

    return _tfidf_embed(texts)


def embed_single(text: str) -> Optional[np.ndarray]:
    """Embed one text string. Returns 1-D float32 array or None."""
    result = embed_texts([text])
    return result[0] if result is not None else None


def embedding_dim() -> int:
    """Return the dimension of the current embedding backend."""
    if _ST_AVAILABLE and _st_model is not None:
        return _st_model.get_sentence_embedding_dimension()
    return _TFIDF_DIM


def is_available() -> bool:
    if _ST_AVAILABLE:
        return True
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401
        return True
    except ImportError:
        return False


# ── TF-IDF fallback ───────────────────────────────────────────────────────

_TFIDF_DIM = 512
_tfidf_vectorizer = None
_TFIDF_READY = False


def _tfidf_embed(texts: List[str]) -> Optional[np.ndarray]:
    global _tfidf_vectorizer, _TFIDF_READY
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        if _tfidf_vectorizer is None:
            _tfidf_vectorizer = TfidfVectorizer(
                max_features=_TFIDF_DIM,
                analyzer="char_wb",
                ngram_range=(2, 4),
                lowercase=True,
            )
        mat = _tfidf_vectorizer.fit_transform(texts).toarray()
        _TFIDF_READY = True
        return mat.astype(np.float32)
    except ImportError:
        logger.warning("scikit-learn not installed. No embedding backend available.")
        return None
    except Exception as exc:
        logger.warning("TF-IDF embed failed: %s", exc)
        return None


def normalize_vectors(vecs: np.ndarray) -> np.ndarray:
    """L2-normalise rows so cosine similarity = inner product."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (vecs / norms).astype(np.float32)
