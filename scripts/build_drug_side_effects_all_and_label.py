import re
from pathlib import Path
import pandas as pd

# ---------------- Paths (edit only if needed) ----------------
CORE_PATH = Path("data/processed/drug_knowledge_project_ready_final.csv")

DRUG_NAMES_PATH = Path("data/raw/meddra/drug_names.tsv")

MEDDRA_ALL_SE_PATH = Path("data/raw/meddra/meddra_all_side_effects.tsv")
MEDDRA_ALL_LABEL_SE_PATH = Path("data/raw/meddra/meddra_label_confirmed_side_effects.tsv")  # <-- label-confirmed style

OUT_PATH = Path("data/processed/drug_side_effects_meddra.csv")
# -------------------------------------------------------------

def clean_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.strip().lower()
    name = re.sub(r"^\(.*?\)\s*", "", name)
    name = re.sub(r"^\d+(\.\d+)?%\s*", "", name)
    name = re.sub(r"\bto deliver\b", "", name)
    name = re.sub(r"[^a-z0-9\s\-\/\+\.]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    if name == "0xygen" or name.startswith("0xygen"):
        name = "oxygen" + name[len("0xygen"):]
    return name

def load_drug_names() -> dict:
    dn = pd.read_csv(DRUG_NAMES_PATH, sep="\t", header=None, dtype=str, low_memory=False)
    if dn.shape[1] < 2:
        raise ValueError("drug_names.tsv must have at least 2 columns: STITCH_ID, drug_name")
    dn = dn.iloc[:, :2].copy()
    dn.columns = ["stitch_id", "drug_name"]
    dn["stitch_id"] = dn["stitch_id"].fillna("").astype(str).str.strip()
    dn["drug_name"] = dn["drug_name"].fillna("").astype(str).str.strip()
    dn["generic_name_clean"] = dn["drug_name"].apply(clean_name)
    dn = dn[(dn["stitch_id"] != "") & (dn["generic_name_clean"] != "")]
    return dict(zip(dn["stitch_id"], dn["generic_name_clean"]))

def map_stitch_to_name(stitch_flat: str, stitch_stereo: str, id_to_name: dict) -> str:
    n = id_to_name.get(str(stitch_flat).strip(), "")
    if not n:
        n = id_to_name.get(str(stitch_stereo).strip(), "")
    return n

def main():
    # Core drugs
    core = pd.read_csv(CORE_PATH, dtype=str, low_memory=False)
    if "generic_name_clean" not in core.columns:
        raise ValueError("Core file must contain generic_name_clean")
    core["generic_name_clean"] = core["generic_name_clean"].fillna("").astype(str).apply(clean_name)
    core_set = set(core["generic_name_clean"].unique())
    print(f"✅ Core drugs loaded: {len(core_set):,}")

    # Mapping
    id_to_name = load_drug_names()
    print(f"✅ SIDER drug_names loaded: {len(id_to_name):,} unique stitch IDs")

    # ---------- ALL side effects ----------
    all_df = pd.read_csv(MEDDRA_ALL_SE_PATH, sep="\t", header=None, dtype=str, low_memory=False)
    if all_df.shape[1] < 6:
        raise ValueError("meddra_all_se.tsv must have 6 columns (SIDER format)")
    all_df = all_df.iloc[:, :6].copy()
    all_df.columns = ["stitch_flat", "stitch_stereo", "umls_on_label", "term_type", "umls_meddra", "side_effect"]

    all_df["generic_name_clean"] = all_df.apply(
        lambda r: map_stitch_to_name(r["stitch_flat"], r["stitch_stereo"], id_to_name),
        axis=1
    )
    all_df["source"] = "all"
    all_df = all_df[["generic_name_clean", "side_effect", "term_type", "source"]]
    all_df = all_df[(all_df["generic_name_clean"] != "") & (all_df["side_effect"].fillna("").astype(str).str.strip() != "")]

    # ---------- LABEL-CONFIRMED side effects ----------
    label_df = pd.read_csv(MEDDRA_ALL_LABEL_SE_PATH, sep="\t", header=None, dtype=str, low_memory=False)
    if label_df.shape[1] < 7:
        raise ValueError("meddra_all_label_se.tsv must have 7 columns (SIDER format with label source)")
    label_df = label_df.iloc[:, :7].copy()
    label_df.columns = ["label_source", "stitch_flat", "stitch_stereo", "umls_on_label", "term_type", "umls_meddra", "side_effect"]

    label_df["generic_name_clean"] = label_df.apply(
        lambda r: map_stitch_to_name(r["stitch_flat"], r["stitch_stereo"], id_to_name),
        axis=1
    )
    label_df["source"] = "label_confirmed"
    label_df = label_df[["generic_name_clean", "side_effect", "term_type", "source"]]
    label_df = label_df[(label_df["generic_name_clean"] != "") & (label_df["side_effect"].fillna("").astype(str).str.strip() != "")]

    # ---------- Combine + filter to core ----------
    combined = pd.concat([all_df, label_df], ignore_index=True)
    combined["side_effect"] = combined["side_effect"].astype(str).str.strip()
    combined["term_type"] = combined["term_type"].astype(str).str.strip()

    combined = combined.drop_duplicates(subset=["generic_name_clean", "side_effect", "term_type", "source"])
    combined = combined[combined["generic_name_clean"].isin(core_set)].copy()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_PATH, index=False, encoding="utf-8")

    print("\n✅ drug_side_effects_meddra.csv created (ALL + LABEL)")
    print(f"Rows: {len(combined):,}")
    print(f"Drugs with any side effects: {combined['generic_name_clean'].nunique():,}")
    print("Breakdown by source:")
    print(combined["source"].value_counts())

if __name__ == "__main__":
    main()
