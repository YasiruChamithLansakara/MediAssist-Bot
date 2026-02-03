# scripts/build_side_effects_frequency_buckets.py
# ------------------------------------------------------------
# Builds a human-friendly side-effects summary with frequency buckets:
#   - common
#   - less_common
#   - rare
#   - postmarketing
#
# INPUTS:
#   1) data/processed/drug_side_effects_meddra.csv
#      Columns: generic_name_clean, side_effect, term_type, source
#      (Already includes sources: all / label_confirmed)
#
#   2) data/raw/meddra/meddra_frequency.tsv
#      (Your renamed SIDER frequency file; plain TSV, NOT gz)
#
# OUTPUT:
#   data/processed/drug_side_effects_summary_buckets.csv
#
# Notes:
# - Uses PT only
# - Uses label_confirmed only (safest)
# - Uses frequency categories from meddra_frequency.tsv
# - If a side effect has multiple frequency records, we keep the most common (best priority).
# - Side effects without frequency fall into "unknown" bucket (optional column).
# ------------------------------------------------------------

from pathlib import Path
import pandas as pd
import re

SE_PATH = Path("data/processed/drug_side_effects_meddra.csv")
FREQ_PATH = Path("data/raw/meddra/meddra_frequency.tsv")
OUT_PATH = Path("data/processed/drug_side_effects_summary_buckets.csv")

# How many to keep per bucket (adjust for your UI)
TOP_K_COMMON = 12
TOP_K_LESS_COMMON = 10
TOP_K_RARE = 8
TOP_K_POST = 8
TOP_K_UNKNOWN = 8  # optional

def norm_term(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def bucket_from_freq(freq: str) -> str:
    """Map SIDER/MeDRA frequency strings into 4 UX buckets."""
    if not isinstance(freq, str):
        return "unknown"
    f = freq.strip().lower()
    if f in {"very common", "common", "frequent"}:
        return "common"
    if f in {"uncommon", "infrequent"}:
        return "less_common"
    if f in {"rare", "very rare"}:
        return "rare"
    if f == "postmarketing":
        return "postmarketing"
    # exact % like "2%" or "0.1%" -> treat as common/less_common using thresholds (optional)
    if f.endswith("%"):
        try:
            val = float(f[:-1])
            if val >= 1.0:
                return "common"
            if 0.1 <= val < 1.0:
                return "less_common"
            return "rare"
        except Exception:
            return "unknown"
    return "unknown"

# priority (lower = more important)
FREQ_PRIORITY = {
    "very common": 1,
    "common": 2,
    "frequent": 3,
    "uncommon": 4,
    "infrequent": 4,
    "rare": 5,
    "very rare": 6,
    "postmarketing": 7,
}

def main():
    if not SE_PATH.exists():
        raise FileNotFoundError(f"Missing side effects file: {SE_PATH.resolve()}")
    if not FREQ_PATH.exists():
        raise FileNotFoundError(f"Missing frequency file: {FREQ_PATH.resolve()}")

    # ---------- Load side effects (PT + label_confirmed only) ----------
    se = pd.read_csv(SE_PATH, dtype=str, low_memory=False)
    required_se = {"generic_name_clean", "side_effect", "term_type", "source"}
    missing_se = required_se - set(se.columns)
    if missing_se:
        raise ValueError(f"Side effects file missing columns: {missing_se}")

    se["term_type"] = se["term_type"].astype(str).str.upper().str.strip()
    se["source"] = se["source"].astype(str).str.strip()
    se["generic_name_clean"] = se["generic_name_clean"].astype(str).str.strip()
    se["side_effect_norm"] = se["side_effect"].apply(norm_term)

    se = se[
        (se["term_type"] == "PT") &
        (se["source"] == "label_confirmed") &
        (se["generic_name_clean"] != "") &
        (se["side_effect_norm"] != "")
    ].copy()

    # ---------- Load frequency file (SIDER meddra_freq format, no header) ----------
    # Your format description says columns:
    # 1&2 STITCH ids, 3 umls, 4 placebo, 5 freq, 6 low, 7 high, 8-10 MedDRA info
    freq = pd.read_csv(
        FREQ_PATH,
        sep="\t",
        header=None,
        dtype=str,
        low_memory=False
    )

    # SIDER freq file typically has 10 columns; we only need:
    # col 4 = frequency description, col 9/10 = side effect name (last col)
    # We'll safely pick: frequency = col[4], side_effect = last column
    if freq.shape[1] < 6:
        raise ValueError("Frequency file does not look like SIDER meddra_freq format (too few columns).")

    freq_desc_col = 4
    side_effect_col = freq.shape[1] - 1  # last column

    freq_df = pd.DataFrame({
        "freq_category": freq.iloc[:, freq_desc_col].fillna("").astype(str).str.strip(),
        "side_effect_norm": freq.iloc[:, side_effect_col].fillna("").astype(str).apply(norm_term),
    })

    freq_df = freq_df[(freq_df["freq_category"] != "") & (freq_df["side_effect_norm"] != "")]

    # Priority for selecting "best" frequency when multiple exist per term
    freq_df["priority"] = freq_df["freq_category"].str.lower().map(FREQ_PRIORITY).fillna(99)

    # Keep best frequency per side effect term
    freq_best = (
        freq_df.sort_values("priority")
        .drop_duplicates("side_effect_norm", keep="first")
        .drop(columns=["priority"])
    )

    # ---------- Merge frequency into side effects ----------
    se = se.merge(freq_best, on="side_effect_norm", how="left")

    se["bucket"] = se["freq_category"].apply(bucket_from_freq)

    # ---------- Build bucketed lists per drug ----------
    def top_k(series: pd.Series, k: int) -> str:
        vals = sorted(set(series.dropna().tolist()))
        return ", ".join(vals[:k])

    grouped = se.groupby(["generic_name_clean", "bucket"])["side_effect_norm"].apply(list).reset_index()

    # Expand lists and aggregate top-k per bucket
    rows = []
    for drug, sub in grouped.groupby("generic_name_clean"):
        buckets = {"common": "", "less_common": "", "rare": "", "postmarketing": "", "unknown": ""}
        for _, r in sub.iterrows():
            b = r["bucket"]
            items = r["side_effect_norm"]
            if b == "common":
                buckets["common"] = top_k(pd.Series(items), TOP_K_COMMON)
            elif b == "less_common":
                buckets["less_common"] = top_k(pd.Series(items), TOP_K_LESS_COMMON)
            elif b == "rare":
                buckets["rare"] = top_k(pd.Series(items), TOP_K_RARE)
            elif b == "postmarketing":
                buckets["postmarketing"] = top_k(pd.Series(items), TOP_K_POST)
            else:
                buckets["unknown"] = top_k(pd.Series(items), TOP_K_UNKNOWN)

        rows.append({
            "generic_name_clean": drug,
            "common_side_effects": buckets["common"],
            "less_common_side_effects": buckets["less_common"],
            "rare_side_effects": buckets["rare"],
            "postmarketing_side_effects": buckets["postmarketing"],
            "unknown_frequency_side_effects": buckets["unknown"],
        })

    out = pd.DataFrame(rows).sort_values("generic_name_clean").reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False, encoding="utf-8")

    print("✅ Frequency-bucket summary created")
    print(f"Drugs included: {len(out):,}")
    print(f"Saved to: {OUT_PATH.resolve()}")
    print("\nSample rows:")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
