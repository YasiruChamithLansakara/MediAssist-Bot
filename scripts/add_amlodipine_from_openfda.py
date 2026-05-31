
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

DATASET = Path(__file__).resolve().parents[1] / "data" / "processed" / "drug_knowledge_bot_ready_clean.csv"
OUTPUT = DATASET
QUERY = "amlodipine"
OPENFDA = "https://api.fda.gov/drug/label.json"

print(f"Loading dataset: {DATASET}")
df = pd.read_csv(DATASET, dtype=str, keep_default_na=False)
exists = df[ df['generic_name'].str.lower().str.contains(QUERY, na=False) | df['generic_name_clean'].str.lower().str.contains(QUERY, na=False) ]
if not exists.empty:
    print("Amlodipine already present in dataset. No changes made.")
    print(exists[['drug_id','generic_name','generic_name_clean']].head())
    exit(0)

print("Querying openFDA for amlodipine...")
params = {"search": f'openfda.generic_name:"{QUERY}"', "limit": 1}
resp = requests.get(OPENFDA, params=params, timeout=10)
resp.raise_for_status()
data = resp.json()
results = data.get('results') or []
if not results:
    print("No results from openFDA for amlodipine.")
    exit(1)
rec = results[0]
openfda = rec.get('openfda', {})

def join_field(key):
    v = openfda.get(key) or []
    return ", ".join(v) if v else ""

brand_names = join_field('brand_name')
pharm_class = join_field('pharm_class_epc') or join_field('pharm_class_pe') or join_field('pharm_class')
route = join_field('route')

generic = (openfda.get('generic_name') or [QUERY])[0]
clean = generic.lower().strip()

indications = " ".join(rec.get('indications_and_usage') or [])
dosage = " ".join(rec.get('dosage_and_administration') or [])
warnings = " ".join(rec.get('warnings') or [])
contra = " ".join(rec.get('contraindications') or [])
adverse = " ".join(rec.get('adverse_reactions') or [])
post = " ".join(rec.get('postmarketing_experience') or [])
side_all = ", ".join(filter(None, [adverse, post]))

# Prepare a new drug_id
existing_ids = df['drug_id'].str.extract(r'DRUG_(\d+)', expand=False).dropna().astype(int)
next_id = int(existing_ids.max()) + 1 if not existing_ids.empty else 1
new_id = f"DRUG_{next_id:06d}"

row = {
    'drug_id': new_id,
    'generic_name': generic,
    'generic_name_clean': clean,
    'brand_names': brand_names,
    'drug_class': pharm_class,
    'route': route,
    'indications': indications,
    'dosage_and_administration': dosage,
    'warnings': warnings,
    'contraindications': contra,
    'side_effects_all': side_all,
    'side_effects_label_confirmed': '',
    'sources': 'openFDA',
    'last_updated': rec.get('effective_time') or datetime.utcnow().strftime('%Y-%m-%d'),
    'top_label_confirmed_side_effects': '',
    'top_all_side_effects': '',
    'side_effect_count_label_confirmed': 0,
    'side_effect_count_all': 0,
    'common_side_effects': '',
    'less_common_side_effects': '',
    'rare_side_effects': '',
    'postmarketing_side_effects': '',
    'unknown_frequency_side_effects': '',
}

print(f"Appending new record for {generic} as {new_id}...")
df = pd.concat([df, pd.DataFrame([row])], ignore_index=True, sort=False)
print(f"Saving updated dataset to {OUTPUT} (backup original as .bak)...")
Backup = OUTPUT.with_suffix('.bak')
OUTPUT.rename(Backup)
df.to_csv(OUTPUT, index=False)
print("Done. You may need to restart the app to reload the dataset.")
