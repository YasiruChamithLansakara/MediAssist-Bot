from __future__ import annotations
import re
import pandas as pd
from pathlib import Path

CSV_PATH = Path("data/processed/drug_knowledge_bot_ready_clean.csv")

SECTION_LABELS = [
    "Liver Warning",
    "Allergy Alert",
    "Do not use",
    "Ask a doctor before use",
    "Ask a doctor or pharmacist before use",
    "Stop use and ask a doctor if",
    "Pregnancy / Breastfeeding",
]

BAD_ENCODING_RE = re.compile(r"(câ|â€¦|Â|�)", re.IGNORECASE)

def count_label_repeats(text: str) -> dict:
    t = text.lower()
    out = {}
    for label in SECTION_LABELS:
        out[label] = t.count(label.lower() + ":")
    return out

def has_repeated_block(text: str, window_words: int = 18) -> bool:
    # Detect repeated long phrase blocks using word windows (more robust than raw string)
    words = re.findall(r"[a-z0-9]+", text.lower())
    if len(words) < window_words * 2:
        return False
    seen = set()
    for i in range(0, len(words) - window_words):
        chunk = " ".join(words[i:i+window_words])
        if chunk in seen:
            return True
        seen.add(chunk)
    return False

def main():
    df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)
    if "warnings" not in df.columns:
        raise ValueError("No warnings column found")

    warnings = df["warnings"].fillna("").astype(str)

    # 1) encoding issues
    bad_encoding_rows = df[warnings.str.contains(BAD_ENCODING_RE)]

    # 2) label repeats
    repeat_label_rows = []
    for idx, w in warnings.items():
        if not w.strip():
            continue
        counts = count_label_repeats(w)
        if any(v > 1 for v in counts.values()):
            repeat_label_rows.append((idx, counts))

    # 3) repeated long blocks
    repeated_block_rows = []
    for idx, w in warnings.items():
        if not w.strip():
            continue
        if has_repeated_block(w, window_words=18):
            repeated_block_rows.append(idx)

    # 4) length check
    too_long_rows = df[warnings.str.len() > 1200]

    print("=== WARNINGS QUALITY REPORT ===")
    print(f"Total rows: {len(df):,}")
    print(f"Non-empty warnings: {(warnings.str.strip() != '').sum():,}")
    print()
    print(f"Encoding garbage rows: {len(bad_encoding_rows)}")
    print(f"Label repeated rows: {len(repeat_label_rows)}")
    print(f"Repeated long-block rows: {len(repeated_block_rows)}")
    print(f"Too long rows (>1200): {len(too_long_rows)}")
    print()

    # Save offenders for inspection
    out_dir = Path("data/quality_reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    bad_encoding_rows.to_csv(out_dir / "warnings_bad_encoding.csv", index=False)
    too_long_rows.to_csv(out_dir / "warnings_too_long.csv", index=False)

    if repeat_label_rows:
        rows = []
        for idx, counts in repeat_label_rows[:200]:  # limit export
            row = df.loc[idx].to_dict()
            row["label_counts"] = str(counts)
            rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / "warnings_label_repeats.csv", index=False)

    if repeated_block_rows:
        df.loc[repeated_block_rows].to_csv(out_dir / "warnings_repeated_blocks.csv", index=False)

    print(f"Reports written to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
