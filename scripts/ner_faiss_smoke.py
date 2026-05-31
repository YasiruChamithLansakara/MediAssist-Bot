from __future__ import annotations

import csv
import logging
import os
from pathlib import Path

from app.services.ner_service import extract_medical_entities, get_drug_names
from app.ml.faiss_store import FAISSStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smoke")

SAMPLE_TEXT = "Metformin 500 mg oral twice daily\nParacetamol 500 mg prn"
DATA_CSV = Path(__file__).resolve().parents[1] / "data" / "processed" / "drug_knowledge_bot_ready_clean.csv"


def ner_smoke():
    print("\n--- NER Smoke Test ---")
    res = extract_medical_entities(SAMPLE_TEXT, disease="diabetes", age=45, max_drugs=8, min_confidence=0.5)
    print("NER extraction completed:", isinstance(res, dict))
    drugs = res.get("drugs", [])
    confidences = [float(d.get("confidence", 0) or 0) for d in drugs]
    avg_confidence = (sum(confidences) / len(confidences)) if confidences else 0.0
    print("Validated drug entities:", len(drugs))
    print("Average confidence:", round(avg_confidence, 3))
    print("Resolved drug names count:", len(get_drug_names(res)))


def build_and_search(sample_queries=("metformin", "paracetamol"), max_rows=200):
    print("\n--- FAISS Build & Search Smoke Test ---")
    if not DATA_CSV.exists():
        print("Data CSV not found:", DATA_CSV)
        return

    drugs = []
    with open(DATA_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            drugs.append(row)

    store = FAISSStore()
    count = store.build(drugs)
    print("Indexed vectors:", count)
    store.save()

    for q in sample_queries:
        print(f"\nSearch: {q}")
        results = store.search(q, top_k=5)
        if not results:
            print(" No results")
        for r in results:
            print(f" - {r['generic_name']} (similarity={r['similarity']})")


if __name__ == '__main__':
    ner_smoke()
    build_and_search()
