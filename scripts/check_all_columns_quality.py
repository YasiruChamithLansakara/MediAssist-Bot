from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

CSV_PATH = Path("data/processed/drug_knowledge_bot_ready_clean.csv")
OUT_DIR = Path("data/quality_reports")

# Columns to check
COLUMNS_TO_CHECK = [
    "warnings",
    "indications",
    "dosage_and_administration",
    "contraindications",
    "common_side_effects",
    "less_common_side_effects",
    "rare_side_effects",
    "postmarketing_side_effects",
    "unknown_frequency_side_effects",
]

# Warnings labels (only meaningful for the warnings column)
SECTION_LABELS = [
    "Liver Warning",
    "Allergy Alert",
    "Do not use",
    "Ask a doctor before use",
    "Ask a doctor or pharmacist before use",
    "Stop use and ask a doctor if",
    "Pregnancy / Breastfeeding",
]

# ✅ Non-capturing groups (no pandas warning)
BAD_ENCODING_RE = re.compile(r"(?:câ|â€¦|Â|Ã|â€™|â€œ|â€|�)", re.IGNORECASE)

MAX_CHARS_DEFAULT = 1200


def count_label_repeats(text: str) -> Dict[str, int]:
    t = text.lower()
    out: Dict[str, int] = {}
    for label in SECTION_LABELS:
        out[label] = t.count(label.lower() + ":")
    return out


def has_repeated_block(text: str, window_words: int = 18) -> bool:
    """
    Detect repeated long phrase blocks using word windows.
    Good generic detector for "repeat_text" across all columns.
    """
    words = re.findall(r"[a-z0-9]+", text.lower())
    if len(words) < window_words * 2:
        return False

    seen = set()
    for i in range(0, len(words) - window_words):
        chunk = " ".join(words[i : i + window_words])
        if chunk in seen:
            return True
        seen.add(chunk)
    return False


def column_quality_report(
    df: pd.DataFrame,
    col: str,
    *,
    max_chars: int = MAX_CHARS_DEFAULT,
    window_words: int = 18,
    check_label_repeats: bool = False,
) -> Tuple[Dict[str, int], Dict[str, pd.DataFrame]]:
    """
    Returns:
      summary dict
      offenders dict (name -> dataframe)
    """
    s = df[col].fillna("").astype(str)

    non_empty_mask = s.str.strip() != ""
    non_empty_count = int(non_empty_mask.sum())

    bad_encoding_mask = s.str.contains(BAD_ENCODING_RE, regex=True, na=False)
    bad_encoding_rows = df[bad_encoding_mask]

    too_long_mask = s.str.len() > max_chars
    too_long_rows = df[too_long_mask]

    repeated_block_rows_idx: List[int] = []
    for idx, txt in s.items():
        if not txt.strip():
            continue
        if has_repeated_block(txt, window_words=window_words):
            repeated_block_rows_idx.append(idx)

    repeated_block_rows = df.loc[repeated_block_rows_idx] if repeated_block_rows_idx else df.iloc[0:0]

    label_repeat_rows = df.iloc[0:0]
    label_repeat_export = df.iloc[0:0]

    label_repeat_count = 0
    if check_label_repeats:
        hits = []
        for idx, txt in s.items():
            if not txt.strip():
                continue
            counts = count_label_repeats(txt)
            if any(v > 1 for v in counts.values()):
                hits.append((idx, counts))

        label_repeat_count = len(hits)

        if hits:
            rows = []
            for idx, counts in hits[:300]:
                row = df.loc[idx].to_dict()
                row["label_counts"] = str(counts)
                rows.append(row)
            label_repeat_export = pd.DataFrame(rows)
            label_repeat_rows = df.loc[[i for i, _ in hits]]

    summary = {
        "total_rows": int(len(df)),
        "non_empty": non_empty_count,
        "bad_encoding": int(len(bad_encoding_rows)),
        "repeated_long_block": int(len(repeated_block_rows)),
        "too_long": int(len(too_long_rows)),
    }
    if check_label_repeats:
        summary["label_repeats"] = int(label_repeat_count)

    offenders = {
        "bad_encoding": bad_encoding_rows,
        "too_long": too_long_rows,
        "repeated_blocks": repeated_block_rows,
    }
    if check_label_repeats:
        offenders["label_repeats_export"] = label_repeat_export
        offenders["label_repeats_rows"] = label_repeat_rows

    return summary, offenders


def main() -> None:
    df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== DATASET QUALITY REPORT ===")
    print(f"CSV: {CSV_PATH}")
    print(f"Total rows: {len(df):,}")
    print()

    for col in COLUMNS_TO_CHECK:
        if col not in df.columns:
            print(f"[SKIP] Column not found: {col}")
            continue

        is_warnings = (col == "warnings")
        summary, offenders = column_quality_report(
            df,
            col,
            max_chars=MAX_CHARS_DEFAULT,
            window_words=18,
            check_label_repeats=is_warnings,
        )

        print(f"--- {col.upper()} ---")
        print(f"Non-empty: {summary['non_empty']:,}")
        print(f"Encoding garbage rows: {summary['bad_encoding']:,}")
        if is_warnings:
            print(f"Label repeated rows: {summary.get('label_repeats', 0):,}")
        print(f"Repeated long-block rows: {summary['repeated_long_block']:,}")
        print(f"Too long rows (>{MAX_CHARS_DEFAULT}): {summary['too_long']:,}")
        print()

        # Write offenders
        prefix = col.lower()

        offenders["bad_encoding"].to_csv(OUT_DIR / f"{prefix}_bad_encoding.csv", index=False)
        offenders["too_long"].to_csv(OUT_DIR / f"{prefix}_too_long.csv", index=False)

        if len(offenders["repeated_blocks"]) > 0:
            offenders["repeated_blocks"].to_csv(OUT_DIR / f"{prefix}_repeated_blocks.csv", index=False)
        else:
            # ensure old files don't confuse you
            (OUT_DIR / f"{prefix}_repeated_blocks.csv").write_text("", encoding="utf-8")

        if is_warnings:
            # export label repeat details
            if len(offenders["label_repeats_export"]) > 0:
                offenders["label_repeats_export"].to_csv(OUT_DIR / f"{prefix}_label_repeats.csv", index=False)
            else:
                (OUT_DIR / f"{prefix}_label_repeats.csv").write_text("", encoding="utf-8")

    print(f"Reports written to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
