"""
Build full FAISS index for all drugs in the database.

This script:
1. Loads all 3868 drug records
2. Embeds them using sentence-transformers (or TF-IDF fallback)
3. Builds FAISS index with normalized vectors
4. Persists index and metadata to disk
"""

import time
import sys
from pathlib import Path

def build_faiss_index():
    print("📦 Building FAISS index for all drugs...")
    print("⏱️  This may take 5-15 minutes depending on embedding backend.")
    print()

    from app.ml.faiss_store import get_faiss_store
    from app.services.drug_lookup import init_store
    import app.services.drug_lookup as lookup_module

    start = time.time()

    # Initialize drug store
    print("📖 Loading drug database...")
    init_store()

    # Get FAISS store and build index
    store = get_faiss_store()
    df = lookup_module._df
    if df is not None and len(df) > 0:
        print(f"📊 Loaded {len(df):,} drug records from CSV")
        print()
        print("🔄 Embedding and indexing...")
        count = store.build(df.to_dict('records'))
        store.save()
        elapsed = time.time() - start
        
        status = store.status()
        print()
        print("✅ FAISS index built successfully!")
        print(f"  Vectors: {count:,}")
        print(f"  Dimension: {status['dimension']}")
        print(f"  Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"  Location: {status['index_path']}")
        print()
        return True
    else:
        print("❌ No drug data found")
        return False

if __name__ == "__main__":
    try:
        success = build_faiss_index()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
