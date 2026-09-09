#!/usr/bin/env python3
"""
Backfill essential chronic-disease drugs missing from the processed dataset.
==========================================================================

Why this exists
---------------
An audit of `drug_knowledge_bot_ready_clean.csv` found that several of the
most-prescribed drugs for the six supported conditions were absent — most
seriously **lisinopril** and **losartan**, two first-line hypertension
medicines. They are present in `data/raw/openfda/`, so this is a gap in the
dataset build, not in the source data.

The consequence was not a missing answer but a WRONG one: before the
ingredient-identity guard was added to `drug_lookup.py`, looking up
"lisinopril" fuzzy-matched **fosinopril** at confidence 0.80 and the chat
layer served that drug's dosing and warnings.

What it does
------------
1. Reads the processed CSV and computes the ingredient keys already covered.
2. Streams each openFDA label file, keeping only single-ingredient
   prescription labels whose generic name IS one of the missing essentials.
3. Picks the most complete label per drug (most populated sections).
4. Appends rows in the existing 23-column schema.

Usage
-----
    python scripts/add_missing_essential_drugs.py --dry-run
    python scripts/add_missing_essential_drugs.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.drug_lookup import ALIASES, ingredient_key  # noqa: E402

CSV_PATH = PROJECT_ROOT / "data" / "processed" / "drug_knowledge_bot_ready_clean.csv"
OPENFDA_DIR = PROJECT_ROOT / "data" / "raw" / "openfda"

# First-line medicines for the six supported conditions, drawn from the WHO
# Model List of Essential Medicines. Anything here that the dataset lacks is
# a coverage bug worth fixing.
ESSENTIAL_BY_DISEASE: Dict[str, List[str]] = {
    "hypertension": [
        "lisinopril", "enalapril", "ramipril", "losartan", "valsartan",
        "amlodipine", "nifedipine", "hydrochlorothiazide", "chlorthalidone",
        "atenolol", "metoprolol", "bisoprolol", "spironolactone",
    ],
    "diabetes": [
        "metformin", "glibenclamide", "gliclazide", "glimepiride",
        "sitagliptin", "empagliflozin", "insulin human", "insulin glargine",
    ],
    "asthma": [
        "salbutamol", "albuterol", "beclomethasone", "budesonide",
        "fluticasone", "ipratropium", "montelukast", "prednisolone",
    ],
    "heart disease": [
        "aspirin", "clopidogrel", "atorvastatin", "simvastatin",
        "bisoprolol", "furosemide", "warfarin", "isosorbide dinitrate",
        "digoxin", "glyceryl trinitrate",
    ],
    "arthritis": [
        "ibuprofen", "naproxen", "diclofenac", "methotrexate",
        "prednisolone", "hydroxychloroquine", "paracetamol", "acetaminophen",
    ],
    "migraine": [
        "sumatriptan", "propranolol", "amitriptyline", "topiramate",
        "ergotamine",
    ],
}

# Sections we want populated, in openFDA field order of preference.
_SECTION_SOURCES = {
    "indications": ["indications_and_usage", "purpose"],
    "dosage_and_administration": ["dosage_and_administration"],
    "warnings": ["warnings_and_cautions", "warnings", "boxed_warning"],
    "contraindications": ["contraindications"],
    "adverse": ["adverse_reactions"],
}

_WS_RE = re.compile(r"\s+")


def _clean(value: Any, limit: int = 6000) -> str:
    """openFDA sections arrive as single-element lists of long strings."""
    if not value:
        return ""
    if isinstance(value, list):
        value = " ".join(str(v) for v in value if v)
    text = _WS_RE.sub(" ", str(value)).strip()
    return text[:limit]


def _first_section(record: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        text = _clean(record.get(key))
        if text:
            return text
    return ""


def _completeness(record: Dict[str, Any]) -> int:
    """How many of the sections we care about this label actually fills."""
    return sum(1 for keys in _SECTION_SOURCES.values() if _first_section(record, keys))


def _is_single_ingredient(generic_names: List[str]) -> bool:
    if len(generic_names) != 1:
        return False
    name = generic_names[0].lower()
    return not re.search(r"[,/]|\band\b|\bwith\b", name)


def load_existing(csv_path: Path) -> tuple[pd.DataFrame, set[str]]:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    covered = set()
    for name in df["generic_name_clean"].tolist() + df["generic_name"].tolist():
        key = ingredient_key(name)
        if key:
            covered.add(key)
    return df, covered


def find_missing(covered: set[str]) -> List[str]:
    wanted: List[str] = []
    for drugs in ESSENTIAL_BY_DISEASE.values():
        for drug in drugs:
            # An INN that the lookup already aliases to a covered drug is not
            # a gap: salbutamol -> albuterol and paracetamol -> acetaminophen
            # both resolve today, so re-adding them would only create
            # duplicate rows competing for the same ingredient key.
            resolved = ALIASES.get(drug.strip().lower(), drug)
            key = ingredient_key(resolved)
            if key and key not in covered and key not in wanted:
                wanted.append(key)
    return wanted


def scan_openfda(missing: List[str], verbose: bool = True) -> Dict[str, Dict[str, Any]]:
    """Return the most complete single-ingredient label found per missing drug."""
    wanted = set(missing)
    best: Dict[str, Dict[str, Any]] = {}

    files = sorted(OPENFDA_DIR.glob("drug-label-*.json"))
    if not files:
        raise SystemExit(f"No openFDA label files under {OPENFDA_DIR}")

    for path in files:
        if verbose:
            print(f"  scanning {path.name} …", flush=True)
        with open(path) as handle:
            payload = json.load(handle)

        for record in payload.get("results", []):
            openfda = record.get("openfda") or {}
            generics = [g.strip() for g in (openfda.get("generic_name") or []) if g.strip()]
            if not _is_single_ingredient(generics):
                continue

            key = ingredient_key(generics[0])
            if key not in wanted:
                continue

            score = _completeness(record)
            if score == 0:
                continue
            if key not in best or score > best[key]["_score"]:
                best[key] = {"_score": score, "_record": record, "_openfda": openfda}

        del payload  # a label file is ~600 MB parsed; free it before the next

    return best


def build_row(key: str, bundle: Dict[str, Any], drug_id: str) -> Dict[str, str]:
    record = bundle["_record"]
    openfda = bundle["_openfda"]

    brands = [b.strip() for b in (openfda.get("brand_name") or []) if b.strip()]
    routes = [r.strip().title() for r in (openfda.get("route") or []) if r.strip()]
    pharm_class = [
        c.strip()
        for c in (openfda.get("pharm_class_epc") or openfda.get("pharm_class_moa") or [])
        if c.strip()
    ]
    generic = (openfda.get("generic_name") or [key])[0].strip().lower()

    return {
        "drug_id": drug_id,
        "generic_name": generic,
        "generic_name_clean": generic,
        "brand_names": ", ".join(dict.fromkeys(brands))[:500],
        "drug_class": "; ".join(dict.fromkeys(pharm_class))[:300],
        "route": ", ".join(dict.fromkeys(routes)),
        "indications": _first_section(record, _SECTION_SOURCES["indications"]),
        "dosage_and_administration": _first_section(record, _SECTION_SOURCES["dosage_and_administration"]),
        "warnings": _first_section(record, _SECTION_SOURCES["warnings"]),
        "contraindications": _first_section(record, _SECTION_SOURCES["contraindications"]),
        # Side-effect buckets come from the separate MedDRA enrichment step,
        # so they are left empty rather than guessed at here.
        "side_effects_all": _first_section(record, _SECTION_SOURCES["adverse"])[:2000],
        "side_effects_label_confirmed": "",
        "sources": "openFDA drug label (backfilled by add_missing_essential_drugs.py)",
        "last_updated": str(record.get("effective_time", "") or ""),
        "top_label_confirmed_side_effects": "",
        "top_all_side_effects": "",
        "side_effect_count_label_confirmed": "0",
        "side_effect_count_all": "0",
        "common_side_effects": "",
        "less_common_side_effects": "",
        "rare_side_effects": "",
        "postmarketing_side_effects": "",
        "unknown_frequency_side_effects": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report gaps, write nothing")
    parser.add_argument("--csv", default=str(CSV_PATH))
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df, covered = load_existing(csv_path)
    print(f"Dataset: {len(df):,} drugs, {len(covered):,} ingredient keys")

    missing = find_missing(covered)
    if not missing:
        print("No essential drugs missing — nothing to do.")
        return

    print(f"\nMissing essential medicines ({len(missing)}):")
    for key in missing:
        print(f"  - {key}")

    if args.dry_run:
        print("\n--dry-run: stopping before the openFDA scan.")
        return

    print(f"\nScanning openFDA labels in {OPENFDA_DIR} …")
    found = scan_openfda(missing)

    still_missing = [k for k in missing if k not in found]
    if still_missing:
        print(f"\nNot found in openFDA (skipped): {', '.join(still_missing)}")
    if not found:
        print("Nothing to add.")
        return

    existing_ids = set(df["drug_id"])
    next_num = 1
    rows: List[Dict[str, str]] = []
    for key in missing:
        if key not in found:
            continue
        while f"DRUG_9{next_num:05d}" in existing_ids:
            next_num += 1
        drug_id = f"DRUG_9{next_num:05d}"
        existing_ids.add(drug_id)
        next_num += 1
        rows.append(build_row(key, found[key], drug_id))

    additions = pd.DataFrame(rows, columns=list(df.columns))
    combined = pd.concat([df, additions], ignore_index=True)

    backup = csv_path.with_suffix(".csv.bak")
    csv_path.replace(backup)
    combined.to_csv(csv_path, index=False)

    print(f"\nAdded {len(rows)} drugs:")
    for row in rows:
        print(
            f"  {row['drug_id']}  {row['generic_name']:<24}"
            f" indications={len(row['indications']):>5}"
            f" warnings={len(row['warnings']):>5}"
        )
    print(f"\nWrote {csv_path}  ({len(combined):,} rows)")
    print(f"Previous version saved to {backup.name}")
    print("\nNext: delete data/faiss_index/ so the vector index rebuilds on startup.")


if __name__ == "__main__":
    main()
