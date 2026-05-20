#!/usr/bin/env python
"""
Build comprehensive drug synonym mapping from the dataset.
Extracts all generic names, brand names, and creates a lookup index.
Run once: py scripts/build_drug_synonyms.py
"""
import json
import re
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "drug_knowledge_bot_ready_clean.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "drug_synonyms.json"

def normalize_name(name: str) -> str:
    """Normalize drug name for comparison"""
    if not name:
        return ""
    s = str(name).lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)  # Remove special chars
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_brand_names(brand_str: str) -> list[str]:
    """Split comma/pipe-separated brand names"""
    if not brand_str:
        return []
    # Split by comma, pipe, or semicolon
    parts = re.split(r"[,\|;]+", str(brand_str))
    return [p.strip() for p in parts if p.strip()]

def build_synonym_map():
    """Build comprehensive synonym mapping"""
    print(f"Loading dataset from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    synonym_map = {}  # normalized_name -> {generic, drug_id, brands}
    brand_to_generic = {}  # brand -> generic name
    
    print(f"Processing {len(df)} drugs...")
    
    for idx, row in df.iterrows():
        generic = (row.get("generic_name") or "").strip()
        drug_id = (row.get("drug_id") or "").strip()
        brands_raw = row.get("brand_names") or ""
        
        if not generic:
            continue
        
        generic_norm = normalize_name(generic)
        
        # Index generic name
        if generic_norm not in synonym_map:
            synonym_map[generic_norm] = {
                "generic_name": generic,
                "drug_id": drug_id,
                "brands": [],
                "sources": ["generic"],
            }
        
        # Parse and index brand names
        brands = parse_brand_names(brands_raw)
        for brand in brands:
            brand_norm = normalize_name(brand)
            if brand_norm and brand_norm != generic_norm:
                synonym_map[generic_norm]["brands"].append(brand)
                # Create reverse mapping: brand -> generic
                brand_to_generic[brand_norm] = generic
    
    # Merge generic and brand indices
    for brand_norm, generic_name in brand_to_generic.items():
        if brand_norm not in synonym_map:
            generic_norm = normalize_name(generic_name)
            if generic_norm in synonym_map:
                synonym_map[brand_norm] = synonym_map[generic_norm]
    
    # Add international names (hard-coded, can expand)
    international_map = {
        "salbutamol": "albuterol",
        "paracetamol": "acetaminophen",
        "panadol": "acetaminophen",
        "tylenol": "acetaminophen",
    }
    
    for intl_norm, generic in international_map.items():
        generic_norm = normalize_name(generic)
        if generic_norm in synonym_map and intl_norm not in synonym_map:
            synonym_map[intl_norm] = synonym_map[generic_norm]
    
    result = {
        "metadata": {
            "total_drugs": len(df),
            "total_synonyms": len(synonym_map),
            "description": "Drug synonym mapping: all names point to canonical generic name and drug_id"
        },
        "synonyms": synonym_map
    }
    
    print(f"\nBuilt synonym map:")
    print(f"  ✓ Total drugs: {len(df)}")
    print(f"  ✓ Total synonym entries: {len(synonym_map)}")
    print(f"  ✓ Example: 'salbutamol' -> {synonym_map.get('salbutamol', {}).get('generic_name')}")
    
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✓ Synonym map saved to: {OUTPUT_PATH}")
    return result

if __name__ == "__main__":
    build_synonym_map()
