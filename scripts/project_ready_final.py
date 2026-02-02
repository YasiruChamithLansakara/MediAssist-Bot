# scripts/project_ready_final.py
# ------------------------------------------------------------
# Single source of truth FINAL script
# INPUT : data/processed/drug_knowledge.csv  (raw master)
# OUTPUT: data/processed/drug_knowledge_project_ready_final.csv
#
# Strong filters remove:
#  - herbal/homeopathic examples (abies nigra, absinthium)
#  - acne cleanser/toner/pads, scar gel, rash, anti-chafing
#  - cosmetic marketing brand rows (above original woman)
# Keeps:
#  - true medicines (DrugBank class OR label-confirmed SE)
# ------------------------------------------------------------

import re
from pathlib import Path
from datetime import date
import pandas as pd

IN_PATH = Path("data/processed/drug_knowledge.csv")
OUT_PATH = Path("data/processed/drug_knowledge_project_ready_final.csv")

TEXT_COLS = [
    "indications",
    "dosage_and_administration",
    "warnings",
    "contraindications",
    "side_effects_all",
    "side_effects_label_confirmed",
]

MAX_CHARS = {
    "indications": 2500,
    "dosage_and_administration": 3500,
    "warnings": 3000,
    "contraindications": 2000,
}

MAX_SIDE_EFFECTS = {
    "side_effects_all": 250,
    "side_effects_label_confirmed": 200,
}

ALLOWED_ROUTES = {
    "ORAL", "INTRAVENOUS", "INTRAMUSCULAR", "SUBCUTANEOUS",
    "INHALATION", "RESPIRATORY (INHALATION)",
    "NASAL", "OPHTHALMIC", "OTIC",
    "RECTAL", "VAGINAL", "TRANSDERMAL",
    "BUCCAL", "SUBLINGUAL",
    "TOPICAL",
}

# Strong blacklist: add anything you see in the sheet that is not medicine
BLACKLIST = {
    # cosmetics / consumer-care
    "above original", "woman", "beauty", "makeup", "foundation", "lipstick", "mascara",
    "sunscreen", "spf", "moisturizer", "lotion", "cream", "shampoo", "soap",
    "hand sanitizer", "sanitizer", "disinfectant",
    "deodorant", "antiperspirant",
    "mouthwash", "oral rinse", "tooth", "teeth", "dental", "toothpaste", "toothbrush",
    "cleanser", "toner", "face wash", "facial",
    "scar gel", "rash", "anti-chafing", "chafing", "overnight healing",

    # acne consumer products
    "acne cleanser", "acne toner", "acne treatment pads",

    # supplements/herbal/homeopathic
    "homeopathic", "herbal", "supplement", "dietary", "extract", "essential oil",
    "vitamin", "multivitamin", "mineral", "probiotic", "omega", "fish oil",
    "seed", "leaf", "root", "bark", "flower", "fruit",

    # explicit bad rows you saw
    "abies nigra", "absinthium",
}

# nicotine cessation removed (not in your chronic disease target)
NICOTINE_TERMS = {"nicotine", "patch", "patches", "gum", "lozenge", "lozenges", "quit smoking", "cessation"}

MIXTURE_PATTERNS = [r"\s-\s", r";", r"\b and \b", r"\b with \b", r"\b plus \b", r"\b combination \b"]
BAD_GENERIC_PATTERNS = [r"\bto deliver\b", r"^\(.*?\)\s*", r"^\d+(\.\d+)?%\s*"]

MAX_WORDS = 3
MAX_LEN = 32


def fix_mojibake(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    rep = {
        "â€¢": "•", "â€“": "-", "â€”": "-", "â€™": "'", "â€œ": '"', "â€�": '"',
        "Â": "", "\u00a0": " "
    }
    for b, g in rep.items():
        text = text.replace(b, g)
    text = text.replace("•", "-")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_generic(name: str) -> str:
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


def limit_text(s: str, max_chars: int) -> str:
    if not isinstance(s, str) or not s.strip():
        return ""
    return s.strip()[:max_chars]


def limit_csv_list(text: str, max_items: int) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    items = [t.strip() for t in text.split(",") if t.strip()]
    seen, out = set(), []
    for it in items:
        k = it.lower()
        if k not in seen:
            seen.add(k)
            out.append(it)
        if len(out) >= max_items:
            break
    return ", ".join(out)


def contains_any(text: str, keywords: set[str]) -> bool:
    if not isinstance(text, str) or not text:
        return False
    t = text.lower()
    return any(k in t for k in keywords)


def route_ok(r: str) -> bool:
    if not isinstance(r, str) or not r.strip():
        return False
    parts = [p.strip().upper() for p in r.split(",") if p.strip()]
    return any(p in ALLOWED_ROUTES for p in parts)


def looks_bad_generic(name: str) -> bool:
    if not isinstance(name, str) or not name.strip():
        return True
    for p in BAD_GENERIC_PATTERNS:
        if re.search(p, name, flags=re.IGNORECASE):
            return True
    return False


def is_single_medicine_name(name: str) -> bool:
    if not isinstance(name, str) or not name.strip():
        return False
    if len(name) > MAX_LEN:
        return False
    if any(re.search(p, name, flags=re.IGNORECASE) for p in MIXTURE_PATTERNS):
        return False
    if len(name.split()) > MAX_WORDS:
        return False
    if len(re.findall(r"\d", name)) >= 3:
        return False
    if name.startswith("above "):
        return False
    return True


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing input: {IN_PATH.resolve()}")

    df = pd.read_csv(IN_PATH, encoding="utf-8", low_memory=False)

    # Ensure columns exist
    for c in TEXT_COLS:
        if c not in df.columns:
            df[c] = ""
    for c in ["brand_names", "drug_class", "route", "sources"]:
        if c not in df.columns:
            df[c] = ""

    # Clean encoding
    for c in TEXT_COLS:
        df[c] = df[c].apply(fix_mojibake)

    # Clean generic names
    df["generic_name_clean"] = df["generic_name"].astype(str).apply(clean_generic)

    # Trim
    for c, m in MAX_CHARS.items():
        df[c] = df[c].apply(lambda x: limit_text(x, m))

    # Limit SE lists
    for c, n in MAX_SIDE_EFFECTS.items():
        df[c] = df[c].apply(lambda x: limit_csv_list(x, n))

    gn = df["generic_name_clean"].astype(str)
    brand = df["brand_names"].astype(str)
    ind = df["indications"].astype(str)
    drug_class = df["drug_class"].astype(str)
    label_se = df["side_effects_label_confirmed"].astype(str)
    route = df["route"].astype(str)

    # Filters
    mask = ~gn.apply(looks_bad_generic)
    mask &= route.apply(route_ok)
    mask &= gn.apply(is_single_medicine_name)

    # blacklist in multiple fields
    blk = gn.apply(lambda x: contains_any(x, BLACKLIST)) | brand.apply(lambda x: contains_any(x, BLACKLIST)) | ind.apply(lambda x: contains_any(x, BLACKLIST))
    mask &= ~blk

    # nicotine removal
    nic = gn.apply(lambda x: contains_any(x, NICOTINE_TERMS)) | brand.apply(lambda x: contains_any(x, NICOTINE_TERMS)) | ind.apply(lambda x: contains_any(x, NICOTINE_TERMS))
    mask &= ~nic

    # core rule
    drug_class_ok = drug_class.str.strip().ne("")
    label_ok = label_se.str.strip().ne("")
    mask &= (drug_class_ok | label_ok)

    out = df.loc[mask].copy()

    out["last_updated"] = str(date.today())
    out = out.sort_values("generic_name_clean").reset_index(drop=True)
    out["drug_id"] = out.index.map(lambda i: f"DRUG_{i+1:06d}")

    cols = [
        "drug_id", "generic_name", "generic_name_clean", "brand_names", "drug_class", "route",
        "indications", "dosage_and_administration", "warnings", "contraindications",
        "side_effects_all", "side_effects_label_confirmed", "sources", "last_updated"
    ]
    out = out[cols]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False, encoding="utf-8")

    print("✅ Final project-ready dataset created (single final script)")
    print(f"Input rows : {len(df):,}")
    print(f"Output rows: {len(out):,}")
    print(f"Saved to   : {OUT_PATH.resolve()}")
    print("\nSample generic_name_clean:")
    print(out["generic_name_clean"].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
