import xml.etree.ElementTree as ET
import pandas as pd

XML_PATH = "data/raw/drugbank/full_database.xml"
OUT_CSV = "data/raw/drugbank/drugbank_drug_classes.csv"

print("🔍 Parsing DrugBank XML...")

tree = ET.parse(XML_PATH)
root = tree.getroot()

ns = {"db": "http://www.drugbank.ca"}

rows = []

for drug in root.findall("db:drug", ns):
    name_elem = drug.find("db:name", ns)
    class_elem = drug.find("db:classification/db:direct-parent", ns)

    if name_elem is None:
        continue

    drug_name = name_elem.text.strip().lower()

    drug_class = ""
    if class_elem is not None and class_elem.text:
        drug_class = class_elem.text.strip()

    rows.append({
        "generic_name": drug_name,
        "drug_class": drug_class
    })

df = pd.DataFrame(rows)
df = df.drop_duplicates(subset=["generic_name"])

df.to_csv(OUT_CSV, index=False, encoding="utf-8")

print(f"✅ DrugBank CSV created: {OUT_CSV}")
print(f"📦 Total drugs extracted: {len(df)}")
