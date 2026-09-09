# Improve by Nazifa
"""
Hybrid retrieval over the drug knowledge base.
==============================================

Two changes over the original single-vector-per-drug design, both driven by a
measured failure: the query "blood pressure hypertension" returned
phenylephrine, midodrine and levophed — three *vasopressors*, the semantic
opposite of what was asked, and the whole index scored a 40% hit rate.

1. SECTION CHUNKING.  Each drug used to be one vector built from its name plus
   300 characters of indications and 200 of warnings. Long label boilerplate
   dominated the vector and the drug's own name barely registered. Each drug
   is now several short, focused chunks — identity, indications, warnings,
   dosage — so "warnings for X" matches a warnings chunk instead of a blend of
   the whole label.

2. HYBRID SEARCH.  Dense vectors handle "medicine for high blood pressure";
   BM25 handles "amlodipine". Neither alone serves both. Results are combined
   with Reciprocal Rank Fusion, which needs no score calibration between the
   two very differently-scaled rankers.

Results are grouped by drug, so `top_k` means k drugs — not k chunks of the
same drug.
"""

from __future__ import annotations

import json
import logging
import os
import re
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

# How many chunks to pull from each ranker before fusing.
DENSE_CANDIDATES = int(os.getenv("RETRIEVAL_DENSE_K", "40"))
LEXICAL_CANDIDATES = int(os.getenv("RETRIEVAL_LEXICAL_K", "40"))
# RRF damping. 60 is the value from the original Cormack et al. paper and is
# insensitive enough that tuning it is rarely worth the effort.
RRF_K = int(os.getenv("RETRIEVAL_RRF_K", "60"))

_FAISS_AVAILABLE = False
try:
    import faiss
    _FAISS_AVAILABLE = True
    logger.info("FAISS loaded successfully")
except ImportError:
    logger.warning("faiss-cpu not installed. Run: pip install faiss-cpu")

_BM25_AVAILABLE = False
try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    BM25Okapi = None  # type: ignore[assignment]
    logger.warning("rank_bm25 not installed — lexical half of hybrid search disabled")

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _shorten(value: Any, limit: int) -> str:
    """Collapse whitespace and cut to a sentence boundary where possible."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    stop = cut.rfind(". ")
    return (cut[: stop + 1] if stop > limit * 0.5 else cut).strip()


class FAISSStore:
    """
    Hybrid (dense + lexical) store over section-level drug chunks.

    Usage:
        store = FAISSStore()
        store.build(drug_records)
        results = store.search("medicine for high blood pressure", top_k=5)
        store.save()
        store.load()
    """

    def __init__(self, index_path: str = DEFAULT_INDEX_PATH):
        self.index_path = index_path
        self._index = None
        self._metadata: List[Dict[str, Any]] = []
        self._texts: List[str] = []
        self._bm25 = None
        self._dim: int = 0
        self._lock = threading.Lock()
        self._ready = False

    # ── build ────────────────────────────────────────────────────────────

    def build(self, drugs: List[Dict[str, Any]]) -> int:
        """
        Chunk, embed and index the drug records.

        Returns the number of *chunks* indexed (several per drug).
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

        drug_count = len({m["drug_id"] for m in meta if m.get("drug_id")})
        logger.info("Embedding %d chunks across %d drugs …", len(texts), drug_count)
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
            self._texts = texts
            self._bm25 = BM25Okapi([_tokenize(t) for t in texts]) if _BM25_AVAILABLE else None
            self._ready = True

        logger.info(
            "Index built: %d chunks / %d drugs, dim=%d, lexical=%s",
            len(meta), drug_count, dim, "on" if self._bm25 else "off",
        )
        return len(meta)

    # ── search ───────────────────────────────────────────────────────────

    def _dense_ranking(self, query: str, k: int) -> List[int]:
        q_vec = embed_single(query)
        if q_vec is None or self._index is None:
            return []
        q_vec = normalize_vectors(q_vec.reshape(1, -1))
        k = min(k, self._index.ntotal)
        if k == 0:
            return []
        _scores, indices = self._index.search(q_vec, k)
        return [int(i) for i in indices[0] if i >= 0]

    def _lexical_ranking(self, query: str, k: int) -> List[int]:
        if self._bm25 is None:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        if scores is None or len(scores) == 0:
            return []
        top = np.argsort(scores)[::-1][:k]
        return [int(i) for i in top if scores[i] > 0]

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Hybrid search, returning at most `top_k` distinct drugs.

        Each result carries the chunk that matched, so callers can tell a
        warnings hit from an indications hit:
            { drug_id, drug_name, generic_name, section, similarity, metadata }
        """
        if not self._ready or self._index is None:
            return []

        # Hold the lock across the whole read: a concurrent build() swaps the
        # index, metadata, texts and BM25 model together, and a partial read
        # across that swap would pair one index's ids with another's metadata.
        with self._lock:
            dense = self._dense_ranking(query, DENSE_CANDIDATES)
            lexical = self._lexical_ranking(query, LEXICAL_CANDIDATES)

            # Reciprocal Rank Fusion: the two rankers' raw scores are on
            # incomparable scales (cosine vs BM25), but their ranks are not.
            fused: Dict[int, float] = {}
            for ranking in (dense, lexical):
                for rank, idx in enumerate(ranking):
                    fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

            if not fused:
                return []

            # Best chunk per drug, so top_k counts drugs rather than sections.
            best_per_drug: Dict[str, Tuple[float, int]] = {}
            for idx, score in fused.items():
                if idx >= len(self._metadata):
                    continue
                meta = self._metadata[idx]
                drug_key = meta.get("drug_id") or meta.get("generic_name") or str(idx)
                current = best_per_drug.get(drug_key)
                if current is None or score > current[0]:
                    best_per_drug[drug_key] = (score, idx)

            ordered = sorted(best_per_drug.values(), key=lambda pair: pair[0], reverse=True)[:top_k]

            results: List[Dict[str, Any]] = []
            for score, idx in ordered:
                meta = self._metadata[idx]
                results.append({
                    "drug_id":      meta.get("drug_id", ""),
                    "drug_name":    meta.get("drug_name", ""),
                    "generic_name": meta.get("generic_name", ""),
                    "section":      meta.get("section", ""),
                    "similarity":   round(float(score), 6),
                    "metadata":     meta,
                })

        return results

    # ── persist ──────────────────────────────────────────────────────────

    @staticmethod
    def _current_csv_path() -> str:
        """Return the canonical absolute path of the active drug CSV."""
        return str(
            os.path.abspath(
                os.getenv(
                    "DRUG_DATASET_PATH",
                    str(Path(__file__).resolve().parents[2]
                        / "data" / "processed" / "drug_knowledge_bot_ready_clean.csv")
                )
            )
        )

    # Bump when the chunking scheme changes, so a cached index built by the
    # old single-vector-per-drug code is rebuilt instead of silently reused.
    SCHEMA_VERSION = 2

    def save(self) -> bool:
        if not self._ready or self._index is None:
            logger.warning("Index not ready — nothing to save")
            return False
        try:
            os.makedirs(self.index_path, exist_ok=True)
            faiss.write_index(self._index, os.path.join(self.index_path, "index.faiss"))
            with open(os.path.join(self.index_path, "metadata.json"), "w") as f:
                json.dump({
                    "dim":            self._dim,
                    "metadata":       self._metadata,
                    "texts":          self._texts,     # BM25 is rebuilt from these
                    "source_csv":     self._current_csv_path(),
                    "schema_version": self.SCHEMA_VERSION,
                }, f)
            logger.info("Index saved to %s", self.index_path)
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
            with open(meta_file) as f:
                data = json.load(f)

            if int(data.get("schema_version", 1)) != self.SCHEMA_VERSION:
                logger.info(
                    "Cached index uses chunking schema v%s, this build expects v%s — rebuilding.",
                    data.get("schema_version", 1), self.SCHEMA_VERSION,
                )
                return False

            # Invalidate the cached index if it was built from a different CSV.
            # This makes DRUG_DATASET_PATH changes take effect automatically.
            stored_csv  = data.get("source_csv", "")
            current_csv = self._current_csv_path()
            if stored_csv and stored_csv != current_csv:
                logger.info(
                    "FAISS index was built from a different CSV — rebuilding.\n"
                    "  stored : %s\n  current: %s", stored_csv, current_csv,
                )
                return False

            with self._lock:
                self._index = faiss.read_index(index_file)
                self._dim = data["dim"]
                self._metadata = data["metadata"]
                self._texts = data.get("texts", [])
                self._bm25 = (
                    BM25Okapi([_tokenize(t) for t in self._texts])
                    if (_BM25_AVAILABLE and self._texts) else None
                )
                self._ready = True
            logger.info(
                "Index loaded from %s (%d chunks, lexical=%s)",
                self.index_path, len(self._metadata), "on" if self._bm25 else "off",
            )
            return True
        except Exception as exc:
            logger.error("FAISS load failed: %s", exc)
            return False

    # ── helpers ──────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self._ready

    def vector_count(self) -> int:
        return len(self._metadata) if self._ready else 0

    def drug_count(self) -> int:
        if not self._ready:
            return 0
        return len({m.get("drug_id") for m in self._metadata if m.get("drug_id")})

    def status(self) -> Dict[str, Any]:
        return {
            "faiss_available": _FAISS_AVAILABLE,
            "lexical_available": _BM25_AVAILABLE and self._bm25 is not None,
            "retrieval_mode": "hybrid_dense_bm25" if self._bm25 else "dense_only",
            "index_ready":     self._ready,
            "vector_count":    self.vector_count(),
            "drug_count":      self.drug_count(),
            "dimension":       self._dim,
            "index_path":      self.index_path,
            "schema_version":  self.SCHEMA_VERSION,
        }

    # ── internal ─────────────────────────────────────────────────────────

    @staticmethod
    def _prepare_drug_texts(
        drugs: List[Dict[str, Any]],
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Turn each drug row into several short, single-purpose chunks.

        Every chunk is prefixed with the drug name so the name stays a strong
        signal in the vector no matter which section the chunk carries — the
        precise thing the old 500-character blob destroyed.
        """
        texts: List[str] = []
        meta:  List[Dict[str, Any]] = []

        # Section -> (source column, character budget). Budgets are tight on
        # purpose: a chunk should be about one thing.
        sections: List[Tuple[str, str, int]] = [
            ("indications",       "indications", 220),
            ("warnings",          "warnings", 220),
            ("dosage",            "dosage_and_administration", 180),
            ("contraindications", "contraindications", 180),
        ]

        for row in drugs:
            name = (
                row.get("generic_name_clean")
                or row.get("generic_name")
                or ""
            ).strip()
            if not name:
                continue

            drug_class = re.sub(r"\s+", " ", str(row.get("drug_class") or "")).strip()
            brands = str(row.get("brand_names") or "").split(",")
            brand = brands[0].strip() if brands else ""

            base_meta = {
                "drug_id":      row.get("drug_id", ""),
                # The generic name, NOT the first brand. Indexing the brand
                # made results read as "cvs arthritis pain relief" instead of
                # "ibuprofen", and made every eval query miss on name.
                "drug_name":    name,
                "generic_name": name,
                "brand_name":   brand,
                "drug_class":   drug_class,
            }

            # Identity chunk: what this drug IS. Short by design, so a query
            # naming a drug or its class lands here rather than in prose.
            identity_bits = [name]
            if brand and brand.lower() != name.lower():
                identity_bits.append(f"brand {brand}")
            if drug_class:
                identity_bits.append(drug_class)
            identity_bits.append(_shorten(row.get("indications"), 90))
            texts.append(" | ".join(b for b in identity_bits if b))
            meta.append({**base_meta, "section": "identity",
                         "text": _shorten(row.get("indications"), 150)})

            for section, column, budget in sections:
                body = _shorten(row.get(column), budget)
                if not body:
                    continue
                texts.append(f"{name} — {section}: {body}")
                meta.append({**base_meta, "section": section, "text": body})

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
