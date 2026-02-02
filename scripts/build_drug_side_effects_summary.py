import pandas as pd
from pathlib import Path

# ================= PATHS =================
BASE_DIR = Path("data")

MEDDRA_FREQ_PATH = BASE_DIR / "raw/meddra/meddra_frequency.tsv"
SIDE_EFFECTS_PATH = BASE_DIR / "processed/drug_side_effects_meddra.csv"
OUTPUT_PATH = BASE_DIR / "processed/drug_side_effects_summary.csv"

# ================= LOAD SIDE EFFECTS =================
se = pd.read_csv(SIDE_EFFECTS_PATH)

# PT-only + label-confirmed
se = se[
    (se["term_type"] == "PT") &
    (se["source"] == "label_confirmed")
].copy()

se["side_effect"] = se["side_effect"].str.lower().str.strip()

# ================= LOAD FREQUENCY FILE =================
freq = pd.read_csv(
    MEDDRA_FREQ_PATH,
    sep="\t",
    header=None,
    names=[
        "stitch_id_flat",
        "stitch_id_stereo",
        "umls_label",
        "placebo",
        "freq_category",
        "freq_low",
        "freq_high",
        "meddra_type",
        "meddra_umls",
        "side_effect"
    ],
    low_memory=False
)

freq["side_effect"] = freq["side_effect"].str.lower().str.strip()

# ================= FREQUENCY PRIORITY =================
priority_map = {
    "very common": 1,
    "common": 2,
    "frequent": 3,
    "uncommon": 4,
    "rare": 5,
    "very rare": 6,
    "postmarketing": 7
}

freq["priority"] = freq["freq_category"].map(priority_map).fillna(99)

# Keep BEST frequency per side effect
freq_best = (
    freq.sort_values("priority")
        .drop_duplicates("side_effect", keep="first")
        [["side_effect", "freq_category", "priority"]]
)

# ================= MERGE FREQUENCY =================
se = se.merge(freq_best, on="side_effect", how="left")

se["priority"] = se["priority"].fillna(99)

# ================= BUILD SUMMARY =================
summary = (
    se.sort_values(["generic_name_clean", "priority", "side_effect"])
      .groupby("generic_name_clean")
      .agg(
          top_label_confirmed_side_effects=(
              "side_effect",
              lambda x: ", ".join(x.tolist())
          ),
          side_effect_count_label_confirmed=("side_effect", "count")
      )
      .reset_index()
)

# ================= SAVE =================
summary.to_csv(OUTPUT_PATH, index=False)

print("✅ PT-only label-confirmed summary WITH frequency ranking created")
print(f"Drugs included: {summary.shape[0]}")
print(f"Saved to: {OUTPUT_PATH}")
