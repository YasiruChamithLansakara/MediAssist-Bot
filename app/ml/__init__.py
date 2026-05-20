"""Machine-learning helpers for embeddings and FAISS-backed similarity search."""

from app.ml.embeddings import embed_single, embed_texts, embedding_dim, is_available, normalize_vectors
from app.ml.faiss_store import FAISSStore, get_faiss_store

__all__ = [
	"FAISSStore",
	"embed_single",
	"embed_texts",
	"embedding_dim",
	"get_faiss_store",
	"is_available",
	"normalize_vectors",
]
