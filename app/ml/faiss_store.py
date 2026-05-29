# Improve by Nazifa
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.ml.embeddings import embed_single, embed_texts, normalize_vectors, is_available

logger = logging.getLogger("mediassist.faiss")

DEFAULT_INDEX_PATH = os.getenv(
    "FAISS_INDEX_PATH",
    str(Path(__file__).resolve().parents[2] / "data" / "faiss_index"),
)

_FAISS_AVAILABLE = False
try:
    import faiss
    _FAISS_AVAILABLE = True
    logger.info("FAISS loaded successfully")
except ImportError:
    logger.warning("faiss-cpu not installed. Run: pip install faiss-cpu")


class FAISSStore:
    """
    FAISS vector store for drug knowledge base.

    Stores drug vectors using IndexFlatIP (inner product).
    Vectors are L2-normalised before insertion so inner product = cosine similarity.

    Usage:
        store = FAISSStore()
        store.build(drug_list)          # index all drugs
        results = store.search("diabetes medication", top_k=5)
        store.save()                    # persist to disk
        store.load()                    # restore from disk
    """

    def __init__(self, index_path: str = DEFAULT_INDEX_PATH):
        self.index_path = index_path
        self._index = None
        self._metadata: List[Dict[str, Any]] = []
        self._dim: int = 0
        self._lock = threading.Lock()
        self._ready = False

    # ── build ────────────────────────────────────────────────────────────

    def build(self, drugs: List[Dict[str, Any]]) -> int:
        """
        Vectorise drug records and build the FAISS index.

        Each drug dict is expected to have at minimum:
            drug_id, generic_name, generic_name_clean,
            indications, warnings, contraindications, drug_class

        Returns number of drugs successfully indexed.
        """
        if not _FAISS_AVAILABLE:
            logger.error("FAISS not available — cannot build index")
            return 0
        if not is_available():
            logger.error("No embedding backend available — cannot build index")
            return 0

        texts, meta = self._prepare_drug_texts(drugs)
        if not texts:
            logger.warning("No drug texts to embed")
            return 0

        logger.info("Embedding %d drugs …", len(texts))
        vecs = embed_texts(texts)
        if vecs is None:
            logger.error("Embedding returned None")
            return 0

        vecs = normalize_vectors(vecs)
        dim = vecs.shape[1]

        with self._lock:
            self._dim = dim
            self._index = faiss.IndexFlatIP(dim)
            self._index.add(vecs)
            self._metadata = meta
            self._ready = True

        logger.info("FAISS index built: %d vectors, dim=%d", len(meta), dim)
        return len(meta)

    # ── search ───────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Semantic search over indexed drugs.

        Returns list of dicts:
            { drug_id, drug_name, generic_name, similarity, metadata }
        """
        if not self._ready or self._index is None:
            return []

        q_vec = embed_single(query)
        if q_vec is None:
            return []

        q_vec = normalize_vectors(q_vec.reshape(1, -1))

        k = min(top_k, self._index.ntotal)
        if k == 0:
            return []

        with self._lock:
            scores, indices = self._index.search(q_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            meta = self._metadata[idx]
            results.append({
                "drug_id":     meta.get("drug_id", ""),
                "drug_name":   meta.get("drug_name", ""),
                "generic_name": meta.get("generic_name", ""),
                "similarity":  round(float(score), 4),
                "metadata":    meta,
            })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results

    # ── persist ──────────────────────────────────────────────────────────

    def save(self) -> bool:
        if not self._ready or self._index is None:
            logger.warning("Index not ready — nothing to save")
            return False
        try:
            os.makedirs(self.index_path, exist_ok=True)
            faiss.write_index(self._index, os.path.join(self.index_path, "index.faiss"))
            with open(os.path.join(self.index_path, "metadata.json"), "w") as f:
                json.dump({"dim": self._dim, "metadata": self._metadata}, f)
            logger.info("FAISS index saved to %s", self.index_path)
            return True
        except Exception as exc:
            logger.error("FAISS save failed: %s", exc)
            return False

    def load(self) -> bool:
        index_file = os.path.join(self.index_path, "index.faiss")
        meta_file  = os.path.join(self.index_path, "metadata.json")
        if not os.path.exists(index_file) or not os.path.exists(meta_file):
            return False
        try:
            with self._lock:
                self._index = faiss.read_index(index_file)
                with open(meta_file) as f:
                    data = json.load(f)
                self._dim = data["dim"]
                self._metadata = data["metadata"]
                self._ready = True
            logger.info("FAISS index loaded from %s (%d vectors)", self.index_path, len(self._metadata))
            return True
        except Exception as exc:
            logger.error("FAISS load failed: %s", exc)
            return False

    # ── helpers ──────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self._ready

    def vector_count(self) -> int:
        return len(self._metadata) if self._ready else 0

    def status(self) -> Dict[str, Any]:
        return {
            "faiss_available": _FAISS_AVAILABLE,
            "index_ready":     self._ready,
            "vector_count":    self.vector_count(),
            "dimension":       self._dim,
            "index_path":      self.index_path,
        }

    # ── internal ─────────────────────────────────────────────────────────

    @staticmethod
    def _prepare_drug_texts(
        drugs: List[Dict[str, Any]],
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Build one embedding text per drug and a matching metadata record.

        Text format gives the model maximum signal:
            "<generic_name> | <drug_class> | indications: <...> | warnings: <...>"
        """
        texts: List[str] = []
        meta:  List[Dict[str, Any]] = []

        for row in drugs:
            name = (
                row.get("generic_name_clean")
                or row.get("generic_name")
                or ""
            ).strip()
            if not name:
                continue

            parts = [name]
            if row.get("drug_class"):
                parts.append(row["drug_class"].strip())
            if row.get("indications"):
                parts.append("indications: " + str(row["indications"])[:300])
            if row.get("warnings"):
                parts.append("warnings: " + str(row["warnings"])[:200])
            if row.get("contraindications"):
                parts.append("contraindications: " + str(row["contraindications"])[:200])

            texts.append(" | ".join(parts))
            meta.append({
                "drug_id":      row.get("drug_id", ""),
                "drug_name":    row.get("brand_names", name).split(",")[0].strip() or name,
                "generic_name": name,
                "drug_class":   row.get("drug_class", ""),
                "indications":  str(row.get("indications", ""))[:150],
                "disease":      row.get("disease_category", ""),
            })

        return texts, meta


# ── singleton ─────────────────────────────────────────────────────────────

_store: Optional[FAISSStore] = None
_store_lock = threading.Lock()


def get_faiss_store() -> FAISSStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = FAISSStore()
    return _store
