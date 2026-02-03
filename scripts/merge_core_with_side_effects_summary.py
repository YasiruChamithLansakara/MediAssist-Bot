# scripts/merge_core_with_side_effects_summary.py
# ------------------------------------------------------------
# Merge:
#   Core drug table  +  Side-effects summary
#
# INPUT:
#   data/processed/drug_knowledge_project_ready_final.csv
#   data/processed/drug_side_effects_summary.csv
#
# OUTPUT:
#   data/processed/drug_knowledge_bot_ready.csv
#
# Join key:
#   generic_name_clean
# ------------------------------------------------------------

from pathlib import Path
import pandas as pd

CORE_PATH = Path("data/processed/drug_knowledge_project_ready_final.csv")
SE_SUMMARY_PATH = Path("data/processed/drug_side_effects_summary.csv")
OUT_PATH = Path("data/processed/drug_knowledge_bot_ready.csv")

def main():
    if not CORE_PATH.exists():
        raise FileNotFoundError(f"Missing core file: {CORE_PATH.resolve()}")
    if not SE_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing side-effects summary file: {SE_SUMMARY_PATH.resolve()}")

    core = pd.read_csv(CORE_PATH, dtype=str, low_memory=False)
    se = pd.read_csv(SE_SUMMARY_PATH, dtype=str, low_memory=False)

    # Basic validation
    if "generic_name_clean" not in core.columns:
        raise ValueError("Core file must contain column: generic_name_clean")
    if "generic_name_clean" not in se.columns:
        raise ValueError("Side-effects summary must contain column: generic_name_clean")

    # Clean join keys
    core["generic_name_clean"] = core["generic_name_clean"].fillna("").astype(str).str.strip()
    se["generic_name_clean"] = se["generic_name_clean"].fillna("").astype(str).str.strip()

    # Ensure expected summary columns exist (create if missing)
    expected_cols = [
        "top_label_confirmed_side_effects",
        "top_all_side_effects",
        "side_effect_count_label_confirmed",
        "side_effect_count_all",
    ]
    for c in expected_cols:
        if c not in se.columns:
            se[c] = ""

    # Merge (left join keeps all core drugs)
    merged = core.merge(
        se[["generic_name_clean"] + expected_cols],
        on="generic_name_clean",
        how="left"
    )

    # Fill missing side-effects info for drugs not covered by SIDER/MeDRA
    merged["top_label_confirmed_side_effects"] = merged["top_label_confirmed_side_effects"].fillna("")
    merged["top_all_side_effects"] = merged["top_all_side_effects"].fillna("")
    merged["side_effect_count_label_confirmed"] = merged["side_effect_count_label_confirmed"].fillna(0)
    merged["side_effect_count_all"] = merged["side_effect_count_all"].fillna(0)

    # Make counts numeric if possible
    merged["side_effect_count_label_confirmed"] = pd.to_numeric(
        merged["side_effect_count_label_confirmed"], errors="coerce"
    ).fillna(0).astype(int)

    merged["side_effect_count_all"] = pd.to_numeric(
        merged["side_effect_count_all"], errors="coerce"
    ).fillna(0).astype(int)

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_PATH, index=False, encoding="utf-8")

    print("✅ Merged bot-ready dataset created")
    print(f"Core rows:   {len(core):,}")
    print(f"SE rows:     {len(se):,}")
    print(f"Final rows:  {len(merged):,}")
    print(f"Saved to:    {OUT_PATH.resolve()}")

    covered = (merged["side_effect_count_all"] > 0).sum()
    print(f"Drugs with side effects summary: {covered:,} / {len(merged):,}")

if __name__ == "__main__":
    main()
