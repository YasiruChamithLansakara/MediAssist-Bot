"""Test FAISS index load and semantic search."""

from app.ml.faiss_store import get_faiss_store

store = get_faiss_store()
print("Before load:")
print(f"  Index ready: {store.is_ready()}")
print(f"  Vector count: {store.vector_count()}")
print()

print("Loading index from disk...")
loaded = store.load()
print(f"Load successful: {loaded}")
print()

print("After load:")
print(f"  Index ready: {store.is_ready()}")
print(f"  Vector count: {store.vector_count()}")
print(f"  Dimension: {store.status()['dimension']}")
print()

print("Testing semantic search...")
results = store.search("diabetes medication", top_k=5)
print("Search results for 'diabetes medication':")
for r in results[:3]:
    print(f"  - {r['generic_name']}: similarity={r['similarity']}")
