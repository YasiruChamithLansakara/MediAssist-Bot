import pandas as pd

# -----------------------------
# Paths
# -----------------------------
CORE_PATH = "data/processed/drug_knowledge_bot_ready.csv"
SE_PATH = "data/processed/drug_side_effects_summary_buckets.csv"
OUT_PATH = "data/processed/drug_knowledge_bot_ready_final.csv"

# -----------------------------
# Load datasets
# -----------------------------
core = pd.read_csv(CORE_PATH)
side_fx = pd.read_csv(SE_PATH)

print(f"✅ Core drugs loaded: {len(core):,}")
print(f"✅ Side effects summaries loaded: {len(side_fx):,}")

# -----------------------------
# Select only needed SE columns
# -----------------------------
side_fx = side_fx[
    [
        "generic_name_clean",
        "common_side_effects",
        "less_common_side_effects",
        "rare_side_effects",
        "postmarketing_side_effects",
        "unknown_frequency_side_effects",
    ]
]

# -----------------------------
# Merge
# -----------------------------
merged = core.merge(
    side_fx,
    on="generic_name_clean",
    how="left",
)

# -----------------------------
# Fill missing side effects (bot-safe)
# -----------------------------
SE_COLS = [
    "common_side_effects",
    "less_common_side_effects",
    "rare_side_effects",
    "postmarketing_side_effects",
    "unknown_frequency_side_effects",
]

for col in SE_COLS:
    merged[col] = merged[col].fillna("")

# -----------------------------
# Save
# -----------------------------
merged.to_csv(OUT_PATH, index=False)

# -----------------------------
# Report
# -----------------------------
with_fx = (merged["common_side_effects"] != "").sum()

print("\n✅ Final bot-ready dataset created")
print(f"Core rows: {len(core):,}")
print(f"Drugs with side effects info: {with_fx:,}")
print(f"Saved to: {OUT_PATH}")
