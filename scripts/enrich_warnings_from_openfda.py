"""
Optional script to enrich missing warnings data from openFDA API.

This fetches additional drug data from the FDA's public API for records
where the 'warnings' field is empty in the current dataset.

Usage:
    python scripts/enrich_warnings_from_openfda.py
"""

import requests
import pandas as pd
import time
from pathlib import Path
from typing import Optional

# Configuration
DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "drug_knowledge_bot_ready_clean.csv"
OUTPUT_PATH = DATASET_PATH.with_stem(DATASET_PATH.stem + "_warnings_enriched")

OPENFDA_URL = "https://api.fda.gov/drug/label.json"
BATCH_SIZE = 100
RATE_LIMIT_DELAY = 1  # seconds between requests


def fetch_drug_warnings(generic_name: str) -> Optional[str]:
    """
    Fetch warnings from openFDA API for a given generic drug name.
    
    Returns the 'warnings' field if found, else None.
    """
    try:
        params = {
            "search": f'openfda.generic_name:"{generic_name}"',
            "limit": 1,
        }
        resp = requests.get(OPENFDA_URL, params=params, timeout=5)
        resp.raise_for_status()
        
        data = resp.json()
        if data.get("results"):
            result = data["results"][0]
            warnings = result.get("warnings", [None])[0]
            if isinstance(warnings, list):
                warnings = " ".join(warnings)
            return str(warnings) if warnings else None
        return None
    except Exception as e:
        print(f"  ⚠️ Error fetching {generic_name}: {e}")
        return None


def enrich_dataset():
    """Enrich missing warnings with openFDA data."""
    print(f"Loading dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH, dtype=str, keep_default_na=False)
    
    # Identify rows with missing warnings
    missing_mask = df["warnings"].isna() | (df["warnings"].str.strip() == "")
    missing_count = missing_mask.sum()
    print(f"Found {missing_count} records with missing warnings out of {len(df)} total.")
    
    if missing_count == 0:
        print("No missing warnings to enrich. Exiting.")
        return
    
    # Attempt to fetch from openFDA
    print("\nFetching from openFDA API (this may take a while)...")
    enriched_count = 0
    
    for idx, row in df[missing_mask].iterrows():
        generic_name = row.get("generic_name_clean") or row.get("generic_name")
        if not generic_name or pd.isna(generic_name):
            continue
        
        print(f"  [{enriched_count + 1}/{missing_count}] Fetching {generic_name}...", end=" ")
        warnings = fetch_drug_warnings(generic_name)
        
        if warnings:
            df.at[idx, "warnings"] = warnings
            enriched_count += 1
            print(f"✓ ({len(warnings)} chars)")
        else:
            print("✗ (no data)")
        
        time.sleep(RATE_LIMIT_DELAY)
    
    print(f"\n✅ Enriched {enriched_count} records.")
    
    # Save enriched dataset
    print(f"\nSaving enriched dataset to {OUTPUT_PATH}...")
    df.to_csv(OUTPUT_PATH, index=False)
    print("✅ Done! You can update the dataset by renaming the output file.")
    print(f"   mv {OUTPUT_PATH} {DATASET_PATH}")


if __name__ == "__main__":
    if not DATASET_PATH.exists():
        print(f"❌ Dataset not found: {DATASET_PATH}")
        exit(1)
    
    try:
        enrich_dataset()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user.")
        exit(1)
