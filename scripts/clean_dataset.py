#!/usr/bin/env python3
"""
Canonicalise the drug knowledge base.
=====================================

Three defects found in the processed dataset on 2026-09-09, all of which
change what a patient is told:

1. DUPLICATE INGREDIENTS (346 keys / 874 rows).  "lidocaine" had 8 rows and
   "menthol" 9 — each a different product with a different label. `lookup_drug`
   picks the row with the fewest ingredients and breaks ties by file order, so
   which warnings a patient saw was effectively arbitrary and would change if
   the CSV were ever re-sorted. Rows sharing an ingredient key are merged into
   one canonical entry built from the most complete label, with brand names and
   routes unioned so nothing is lost.

2. NON-MEDICINE PRODUCTS.  openFDA labels cover alcohol prep pads, compressed
   air, hand sanitiser and unbranded "analgesic" products. They are not
   medicines a patient asks about, and because their names literally contain
   query words they outranked real drugs in lexical search — "asthma inhaler
   breathing" returned "breathing air" and "inhalant". Removed by an explicit
   denylist, never by a broad pattern that might catch a real drug.

3. BRAND-LIST ARTEFACTS.  31 rows carried a literal ", ... (+7 more)" suffix
   inside the brand_names field, which then surfaced in the UI.

Usage:
    python scripts/clean_dataset.py --dry-run
    python scripts/clean_dataset.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.drug_lookup import ingredient_key, ingredient_set  # noqa: E402

CSV_PATH = PROJECT_ROOT / "data" / "processed" / "drug_knowledge_bot_ready_clean.csv"

# Products that are not medicines for this application's purpose. Matched on
# the exact normalised generic name, or as a whole-word prefix where the family
# is unambiguous. Kept explicit: a regex broad enough to catch "alcohol pad"
# also catches "alcohol", which is a real ingredient in real drug labels.
NON_MEDICINE_EXACT = {
    "air", "air compressed", "compressed air", "medical air", "breathing air",
    "oxygen", "nitrogen", "nitrogen liquid",
    "alcohol liquid", "alcohol pad", "alcohol pads", "alcohol prep pad",
    "alcohol prep pads", "alcohol wipe", "alcohol wipes", "alcohol swab",
    "alcohol swabs", "isopropyl alcohol", "isopropyl rubbing alcohol",
    "ethyl alcohol", "ethanol",
    "analgesic", "antiseptic", "antiseptic wipe", "antiseptic towelette",
    "hand sanitizer", "hand sanitiser", "hand antiseptic",
    "first aid", "first aid antiseptic", "inhalant", "inhalants",
    "ammonia inhalants", "ammonia", "smelling salts",
    "sunscreen", "sunblock", "cholesterol", "pain relief", "reuseable pain",
    "water", "purified water", "sterile water", "saline", "normal saline",
}

# Sections that decide which duplicate row is the most complete, weighted by
# how much they matter to a patient reading about their prescription.
_QUALITY_FIELDS = {
    "warnings": 3.0,
    "indications": 2.5,
    "dosage_and_administration": 2.0,
    "contraindications": 2.0,
    "common_side_effects": 1.5,
    "drug_class": 1.0,
    "side_effects_all": 0.5,
}

_BRAND_ARTEFACT_RE = re.compile(r",?\s*\.\.\.\s*\(\+\d+\s*more\)\s*$", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def _text(row: Dict[str, Any], col: str) -> str:
    return _WS_RE.sub(" ", str(row.get(col, "") or "")).strip()


def completeness(row: Dict[str, Any]) -> float:
    """Weighted richness of a row's clinical content."""
    score = 0.0
    for col, weight in _QUALITY_FIELDS.items():
        length = len(_text(row, col))
        if length:
            # Diminishing returns: a 6,000-char warning is not twice as useful
            # as a 3,000-char one, but an empty field is a real loss.
            score += weight * min(length / 500.0, 4.0)
    return round(score, 3)


def _union_list(values: List[str], limit: int = 25) -> str:
    seen: Dict[str, None] = {}
    for value in values:
        cleaned = _BRAND_ARTEFACT_RE.sub("", str(value or ""))
        for part in re.split(r"[,;|]", cleaned):
            part = _WS_RE.sub(" ", part).strip()
            if part and part.lower() not in {"...", "more"} and part.lower() not in seen:
                seen[part.lower()] = None
                seen.setdefault(part.lower(), None)
                if len(seen) >= limit:
                    break
    # Preserve original casing of first occurrence
    out: List[str] = []
    taken: set[str] = set()
    for value in values:
        cleaned = _BRAND_ARTEFACT_RE.sub("", str(value or ""))
        for part in re.split(r"[,;|]", cleaned):
            part = _WS_RE.sub(" ", part).strip()
            key = part.lower()
            if part and key in seen and key not in taken:
                taken.add(key)
                out.append(part)
                if len(out) >= limit:
                    return ", ".join(out)
    return ", ".join(out)


def display_name(rows: List[Dict[str, Any]]) -> str:
    """
    The name a patient should see: the base ingredient (INN), not a salt form.

    People write "metformin" and "amlodipine" on a prescription; openFDA
    records "metformin hydrochloride" and "amlodipine besylate". Answering
    with the salt form is technically right and practically confusing, so the
    shortest name that still resolves to the same ingredient wins. The exact
    product name stays in `generic_name`, and merged products are recorded in
    `sources`.
    """
    key = ingredient_key(rows[0].get("generic_name_clean") or rows[0].get("generic_name") or "")
    candidates = [
        _text(row, "generic_name_clean") or _text(row, "generic_name")
        for row in rows
    ]
    candidates = [c for c in candidates if c]
    if not candidates:
        return key

    # An exact INN match beats everything.
    for candidate in candidates:
        if candidate.strip().lower() == key:
            return candidate.strip().lower()

    # Otherwise the shortest name carrying the same ingredient identity —
    # falling back to the key itself when it is a clean single ingredient.
    same = [c for c in candidates if ingredient_key(c) == key]
    if key and " " not in key and len(key) >= 4:
        return key
    return min(same or candidates, key=len).strip().lower()


def merge_group(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collapse rows that name the same ingredient into one canonical record.

    The richest row wins the clinical text; brand names and routes are unioned
    across the whole group; any field the winner left empty is backfilled from
    the next-richest row that has it.
    """
    ordered = sorted(rows, key=completeness, reverse=True)
    winner = dict(ordered[0])

    for column in winner:
        if _text(winner, column):
            continue
        for candidate in ordered[1:]:
            value = _text(candidate, column)
            if value:
                winner[column] = candidate[column]
                break

    winner["brand_names"] = _union_list([r.get("brand_names", "") for r in ordered])
    winner["route"] = _union_list([r.get("route", "") for r in ordered], limit=8)
    winner["generic_name_clean"] = display_name(ordered)

    sources = _union_list([r.get("sources", "") for r in ordered], limit=6)
    if len(ordered) > 1:
        sources = (sources + f" | merged from {len(ordered)} openFDA products").strip(" |")
    winner["sources"] = sources
    return winner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    parser.add_argument("--csv", default=str(CSV_PATH))
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    original = len(df)
    print(f"Loaded {original:,} rows from {csv_path.name}")

    # ---- 1. drop non-medicine products ----------------------------------
    names = df["generic_name_clean"].str.strip().str.lower()
    is_junk = names.isin(NON_MEDICINE_EXACT)
    dropped = df[is_junk]
    df = df[~is_junk].copy()
    print(f"\n1. Non-medicine products removed: {len(dropped)}")
    for name in sorted(set(dropped["generic_name_clean"]))[:12]:
        print(f"     - {name}")
    if len(dropped) > 12:
        print(f"     … and {len(dropped) - 12} more")

    # ---- 2. merge duplicate ingredients ---------------------------------
    df["_key"] = df["generic_name_clean"].map(ingredient_key)
    groups = df.groupby("_key", sort=False)

    merged_rows: List[Dict[str, Any]] = []
    merged_count = 0
    examples: List[str] = []

    for key, group in groups:
        records = group.to_dict("records")
        if not key:
            merged_rows.extend(records)
            continue
        if len(records) == 1:
            # Single rows get the same INN display treatment, so the whole
            # dataset is consistent: "amlodipine", not "amlodipine besylate".
            row = dict(records[0])
            row["generic_name_clean"] = display_name(records)
            merged_rows.append(row)
            continue
        merged_count += len(records) - 1
        if len(examples) < 8:
            examples.append(f"{key} ({len(records)} rows → 1)")
        merged_rows.append(merge_group(records))

    out = pd.DataFrame(merged_rows)
    out = out.drop(columns=["_key"], errors="ignore")
    print(f"\n2. Duplicate ingredient rows merged away: {merged_count}")
    for example in examples:
        print(f"     - {example}")

    # ---- 3. strip brand artefacts ---------------------------------------
    before_artefacts = out["brand_names"].str.contains(r"\(\+\d+\s*more\)", regex=True).sum()
    out["brand_names"] = out["brand_names"].map(lambda v: _BRAND_ARTEFACT_RE.sub("", str(v)).strip(" ,"))
    print(f"\n3. Brand-list artefacts stripped: {before_artefacts}")

    # ---- report ----------------------------------------------------------
    print(f"\nRows: {original:,} → {len(out):,}")
    keys = out["generic_name_clean"].map(ingredient_key)
    remaining_dupes = sum(1 for _k, c in keys.value_counts().items() if c > 1)
    print(f"Ingredient keys still duplicated: {remaining_dupes}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    backup = csv_path.with_suffix(".csv.pre-clean")
    csv_path.replace(backup)
    out.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")
    print(f"Backup: {backup.name}")
    print("\nNext: delete data/faiss_index/ so the retrieval index rebuilds.")


if __name__ == "__main__":
    main()
