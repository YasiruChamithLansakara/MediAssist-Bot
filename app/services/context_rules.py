from __future__ import annotations

from typing import Any, Dict, Optional, List


SUPPORTED_DISEASES = [
    "diabetes",
    "hypertension",
    "asthma",
    "heart disease",
    "arthritis",
]


def _age_group(age: int) -> str:
    if age < 12:
        return "pediatric"
    if age < 18:
        return "adolescent"
    if age < 65:
        return "adult"
    return "elderly"


def _contains(text: str, *keywords: str) -> bool:
    t = (text or "").lower()
    return any(k.lower() in t for k in keywords)


def build_context_highlights(*, disease: str, age: int, match: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Starter rule-based logic:
    - age group warnings (pediatric/elderly)
    - disease-specific caution hints based on drug_class + warnings text

    IMPORTANT: non-prescriptive. “May require caution” only.
    """
    ag = _age_group(int(age))
    highlights: Dict[str, List[str]] = {"age": [], "disease": [], "general": []}

    if match:
        drug_class = (match.get("drug_class") or "").strip()
        warnings = (match.get("warnings") or "").strip()
        contraindications = (match.get("contraindications") or "").strip()

        # Age-related signals
        if ag == "elderly" and (_contains(warnings, "elderly") or _contains(warnings, "older patients")):
            highlights["age"].append("Warnings mention elderly/older patients — extra caution may be needed.")
        if ag in {"pediatric", "adolescent"} and _contains(warnings, "pediatric", "children", "child"):
            highlights["age"].append("Warnings mention children/pediatric use — verify suitability for this age.")
        if _contains(warnings, "pregnancy", "lactation", "breastfeeding"):
            highlights["age"].append("Warnings mention pregnancy/lactation — only relevant if applicable.")

        # Disease-specific starter rules (safe wording)
        d = disease.lower().strip()

        # Hypertension / heart disease
        if d in {"hypertension", "heart disease"}:
            if _contains(drug_class, "nsaid") or _contains(warnings, "cardiovascular", "thrombotic", "heart failure", "hypertension"):
                highlights["disease"].append(
                    "For hypertension/heart disease: some medicines (e.g., NSAIDs) may increase cardiovascular risk or affect blood pressure — review warnings."
                )

        # Asthma
        if d == "asthma":
            if _contains(warnings, "bronchospasm", "asthma") or _contains(drug_class, "beta blocker", "nsaid"):
                highlights["disease"].append(
                    "For asthma: some medicines may worsen breathing in sensitive individuals — review warnings for bronchospasm/asthma mentions."
                )

        # Diabetes
        if d == "diabetes":
            if _contains(warnings, "blood glucose", "hyperglycemia") or _contains(drug_class, "corticosteroid"):
                highlights["disease"].append(
                    "For diabetes: some medicines may affect blood glucose — review warnings if glucose changes are mentioned."
                )

        # Arthritis
        if d == "arthritis":
            if _contains(drug_class, "nsaid") or _contains(warnings, "gi bleeding", "ulceration", "perforation"):
                highlights["disease"].append(
                    "For arthritis: pain/anti-inflammatory drugs may carry GI risks — review warnings for bleeding/ulceration risk."
                )

        # General
        if contraindications:
            highlights["general"].append("Contraindications section exists — check if any apply to the user’s condition.")
        if warnings:
            highlights["general"].append("Warnings section exists — review key safety notes before use.")

    else:
        highlights["general"].append(
            "No drug matched yet. Provide a drug name (generic or brand) for disease/age-based highlights."
        )

    # remove empty categories
    highlights = {k: v for k, v in highlights.items() if v}

    return {
        "age_group": ag,
        "highlights": highlights,
        "recommended_sections": ["warnings", "contraindications", "dosage_and_administration"],
        "supported_diseases": SUPPORTED_DISEASES,
    }