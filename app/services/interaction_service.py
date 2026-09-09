"""
Drug–drug interaction checking.
===============================

The highest-value capability the assistant was missing. A chronic patient on
four medicines is exactly the person an interaction endangers, and the
pipeline already extracts every drug on a prescription from one photograph —
so the information needed to warn them was being thrown away.

Why the rules are curated rather than mined
-------------------------------------------
openFDA labels carry a `drug_interactions` section, but it is free prose:
"may increase, prolong, or intensify the sedative action of other central
nervous system depressants…". Extracting pairwise facts from that is an
error-prone NLP problem, and a *fabricated* interaction warning in a
medication tool is worse than none — it erodes trust in the warnings that are
real, and could push someone to stop a drug they need.

So this module uses an explicit, auditable table of well-established
interactions, each carrying its mechanism and a citation. Everything it says
can be traced to a source and reviewed by a pharmacist. It is deliberately
small and conservative: high-confidence, clinically significant interactions
relevant to the six supported conditions.

The dataset's own `drug_class` column is used as a fallback, but it is empty
or inconsistent for many drugs (lisinopril, warfarin and metoprolol all have
none), which is why class membership is declared here instead.

LIMITS — stated plainly because they matter
-------------------------------------------
* Not exhaustive. A "no interactions found" result means *this table* found
  none, NOT that the combination is safe. Every response says so.
* No dose, renal function, age or genotype adjustment.
* Never tells anyone to stop or change a medicine — only to ask a pharmacist.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set

from app.services.drug_lookup import base_ingredient, ingredient_set

# ---------------------------------------------------------------------------
# CLASS MEMBERSHIP
# ---------------------------------------------------------------------------
# ingredient -> class tags. Curated for the medicines that actually appear in
# the six supported conditions, plus the common interactors patients also take.
DRUG_CLASSES: Dict[str, Set[str]] = {
    # --- cardiovascular ---
    "lisinopril":          {"ace_inhibitor", "antihypertensive"},
    "enalapril":           {"ace_inhibitor", "antihypertensive"},
    "ramipril":            {"ace_inhibitor", "antihypertensive"},
    "captopril":           {"ace_inhibitor", "antihypertensive"},
    "perindopril":         {"ace_inhibitor", "antihypertensive"},
    "losartan":            {"arb", "antihypertensive"},
    "valsartan":           {"arb", "antihypertensive"},
    "irbesartan":          {"arb", "antihypertensive"},
    "telmisartan":         {"arb", "antihypertensive"},
    "amlodipine":          {"calcium_channel_blocker", "antihypertensive"},
    "nifedipine":          {"calcium_channel_blocker", "antihypertensive"},
    "diltiazem":           {"calcium_channel_blocker", "antihypertensive", "cyp3a4_inhibitor"},
    "verapamil":           {"calcium_channel_blocker", "antihypertensive", "cyp3a4_inhibitor"},
    "atenolol":            {"beta_blocker", "beta_blocker_selective", "antihypertensive"},
    "metoprolol":          {"beta_blocker", "beta_blocker_selective", "antihypertensive"},
    "bisoprolol":          {"beta_blocker", "beta_blocker_selective", "antihypertensive"},
    "propranolol":         {"beta_blocker", "beta_blocker_nonselective", "antihypertensive"},
    "carvedilol":          {"beta_blocker", "beta_blocker_nonselective", "antihypertensive"},
    "labetalol":           {"beta_blocker", "beta_blocker_nonselective", "antihypertensive"},
    "hydrochlorothiazide": {"thiazide_diuretic", "diuretic", "antihypertensive", "potassium_lowering"},
    "chlorthalidone":      {"thiazide_diuretic", "diuretic", "antihypertensive", "potassium_lowering"},
    "furosemide":          {"loop_diuretic", "diuretic", "antihypertensive", "potassium_lowering"},
    "spironolactone":      {"potassium_sparing_diuretic", "diuretic", "potassium_raising"},
    "amiloride":           {"potassium_sparing_diuretic", "diuretic", "potassium_raising"},
    "digoxin":             {"digoxin", "narrow_therapeutic_index"},
    "potassium chloride":  {"potassium_supplement", "potassium_raising"},

    # --- antithrombotic ---
    "warfarin":            {"anticoagulant", "vitamin_k_antagonist", "narrow_therapeutic_index", "bleeding_risk"},
    "apixaban":            {"anticoagulant", "doac", "bleeding_risk"},
    "rivaroxaban":         {"anticoagulant", "doac", "bleeding_risk"},
    "dabigatran":          {"anticoagulant", "doac", "bleeding_risk"},
    "heparin":             {"anticoagulant", "bleeding_risk"},
    "clopidogrel":         {"antiplatelet", "bleeding_risk"},
    "ticagrelor":          {"antiplatelet", "bleeding_risk"},
    "aspirin":             {"antiplatelet", "nsaid", "bleeding_risk", "gi_irritant"},

    # --- lipids ---
    "atorvastatin":        {"statin", "cyp3a4_substrate", "myopathy_risk"},
    "simvastatin":         {"statin", "cyp3a4_substrate", "myopathy_risk"},
    "lovastatin":          {"statin", "cyp3a4_substrate", "myopathy_risk"},
    "rosuvastatin":        {"statin", "myopathy_risk"},
    "pravastatin":         {"statin", "myopathy_risk"},
    "gemfibrozil":         {"fibrate", "myopathy_risk"},
    "fenofibrate":         {"fibrate", "myopathy_risk"},

    # --- analgesia / anti-inflammatory ---
    "ibuprofen":           {"nsaid", "gi_irritant", "bleeding_risk", "renal_risk"},
    "naproxen":            {"nsaid", "gi_irritant", "bleeding_risk", "renal_risk"},
    "diclofenac":          {"nsaid", "gi_irritant", "bleeding_risk", "renal_risk"},
    "meloxicam":           {"nsaid", "gi_irritant", "bleeding_risk", "renal_risk"},
    "celecoxib":           {"nsaid", "renal_risk"},
    "indomethacin":        {"nsaid", "gi_irritant", "bleeding_risk", "renal_risk"},
    "ketorolac":           {"nsaid", "gi_irritant", "bleeding_risk", "renal_risk"},
    "acetaminophen":       {"analgesic_non_nsaid", "hepatotoxic_in_overdose"},
    "prednisolone":        {"corticosteroid", "gi_irritant"},
    "prednisone":          {"corticosteroid", "gi_irritant"},
    "methotrexate":        {"dmard", "narrow_therapeutic_index", "renal_cleared"},
    "hydroxychloroquine":  {"dmard"},

    # --- diabetes ---
    "metformin":           {"biguanide", "antidiabetic", "lactic_acidosis_risk"},
    "glipizide":           {"sulfonylurea", "antidiabetic", "hypoglycaemia_risk"},
    "glyburide":           {"sulfonylurea", "antidiabetic", "hypoglycaemia_risk"},
    "glimepiride":         {"sulfonylurea", "antidiabetic", "hypoglycaemia_risk"},
    "gliclazide":          {"sulfonylurea", "antidiabetic", "hypoglycaemia_risk"},
    "sitagliptin":         {"dpp4_inhibitor", "antidiabetic"},
    "empagliflozin":       {"sglt2_inhibitor", "antidiabetic"},
    "insulin":             {"insulin", "antidiabetic", "hypoglycaemia_risk"},
    "insulin human":       {"insulin", "antidiabetic", "hypoglycaemia_risk"},
    "insulin glargine":    {"insulin", "antidiabetic", "hypoglycaemia_risk"},

    # --- respiratory ---
    "albuterol":           {"beta_agonist", "bronchodilator"},
    "salmeterol":          {"beta_agonist", "bronchodilator"},
    "formoterol":          {"beta_agonist", "bronchodilator"},
    "ipratropium":         {"anticholinergic", "bronchodilator"},
    "montelukast":         {"leukotriene_antagonist"},
    "fluticasone":         {"inhaled_corticosteroid"},
    "budesonide":          {"inhaled_corticosteroid"},
    "beclomethasone":      {"inhaled_corticosteroid"},
    "theophylline":        {"xanthine", "narrow_therapeutic_index"},

    # --- migraine / CNS ---
    "sumatriptan":         {"triptan", "serotonergic"},
    "rizatriptan":         {"triptan", "serotonergic"},
    "zolmitriptan":        {"triptan", "serotonergic"},
    "amitriptyline":       {"tricyclic", "serotonergic", "anticholinergic"},
    "topiramate":          {"anticonvulsant"},
    "sertraline":          {"ssri", "serotonergic", "bleeding_risk"},
    "fluoxetine":          {"ssri", "serotonergic", "bleeding_risk"},
    "citalopram":          {"ssri", "serotonergic", "bleeding_risk"},
    "escitalopram":        {"ssri", "serotonergic", "bleeding_risk"},
    "paroxetine":          {"ssri", "serotonergic", "bleeding_risk"},
    "venlafaxine":         {"snri", "serotonergic", "bleeding_risk"},
    "duloxetine":          {"snri", "serotonergic", "bleeding_risk"},
    "ergotamine":          {"ergot", "vasoconstrictor"},

    # --- common interactors patients also take ---
    "clarithromycin":      {"macrolide", "cyp3a4_inhibitor"},
    "erythromycin":        {"macrolide", "cyp3a4_inhibitor"},
    "azithromycin":        {"macrolide"},
    "ketoconazole":        {"azole_antifungal", "cyp3a4_inhibitor"},
    "itraconazole":        {"azole_antifungal", "cyp3a4_inhibitor"},
    "fluconazole":         {"azole_antifungal", "cyp3a4_inhibitor"},
    "ciprofloxacin":       {"fluoroquinolone"},
    "omeprazole":          {"ppi"},
    "pantoprazole":        {"ppi"},
    "levothyroxine":       {"thyroid_hormone", "narrow_therapeutic_index"},
}

# Patterns applied to the dataset's own drug_class text when an ingredient is
# not in the curated table above. Lower confidence, and marked as such.
_CLASS_PATTERNS: List[tuple[str, Set[str]]] = [
    (r"\bnsaid|nonsteroidal\b",                    {"nsaid", "gi_irritant", "bleeding_risk"}),
    (r"\bstatin|hmg-?coa\b",                       {"statin", "myopathy_risk"}),
    (r"\bace inhibitor|angiotensin.converting\b",  {"ace_inhibitor", "antihypertensive"}),
    (r"\bangiotensin receptor|\barb\b",            {"arb", "antihypertensive"}),
    (r"\bbeta.?blocker|adrenergic blocker\b",      {"beta_blocker", "antihypertensive"}),
    (r"\bcalcium channel\b",                       {"calcium_channel_blocker", "antihypertensive"}),
    (r"\bdiuretic\b",                              {"diuretic", "antihypertensive"}),
    (r"\baldosterone antagonist\b",                {"potassium_sparing_diuretic", "potassium_raising"}),
    (r"\banticoagulant\b",                         {"anticoagulant", "bleeding_risk"}),
    (r"\bantiplatelet\b",                          {"antiplatelet", "bleeding_risk"}),
    (r"\btriptan|5-?ht1\b",                        {"triptan", "serotonergic"}),
    (r"\bssri|serotonin reuptake\b",               {"ssri", "serotonergic", "bleeding_risk"}),
    (r"\bsulfonylurea\b",                          {"sulfonylurea", "hypoglycaemia_risk"}),
    (r"\bbiguanide\b",                             {"biguanide", "lactic_acidosis_risk"}),
    (r"\bcorticosteroid\b",                        {"corticosteroid"}),
    (r"\bmacrolide\b",                             {"macrolide", "cyp3a4_inhibitor"}),
]

SEVERITY_ORDER = {"major": 0, "moderate": 1, "minor": 2}


@dataclass(frozen=True)
class InteractionRule:
    """One documented interaction between two drug classes."""
    left: str
    right: str
    severity: str          # major | moderate | minor
    title: str
    mechanism: str         # why it happens, in plain language
    advice: str            # what the patient should DO — always "ask", never "stop"
    source: str
    same_class: bool = False   # True when the rule is about duplicate therapy


# ---------------------------------------------------------------------------
# THE TABLE
# ---------------------------------------------------------------------------
INTERACTION_RULES: List[InteractionRule] = [
    InteractionRule(
        left="ace_inhibitor", right="potassium_raising", severity="major",
        title="Risk of high potassium",
        mechanism=(
            "ACE inhibitors reduce how much potassium the kidneys remove. Combined with a "
            "potassium-sparing diuretic or a potassium supplement, potassium can rise to a "
            "level that affects heart rhythm."
        ),
        advice=(
            "This combination is prescribed deliberately in some conditions, with blood tests "
            "to monitor potassium. Ask your doctor or pharmacist whether your potassium is "
            "being checked."
        ),
        source="FDA label — ACE inhibitor class warnings (hyperkalemia)",
    ),
    InteractionRule(
        left="arb", right="potassium_raising", severity="major",
        title="Risk of high potassium",
        mechanism=(
            "Angiotensin receptor blockers reduce potassium excretion. Taken with a "
            "potassium-sparing diuretic or supplement, potassium can rise too far."
        ),
        advice="Ask whether your potassium level is being monitored with blood tests.",
        source="FDA label — ARB class warnings (hyperkalemia)",
    ),
    InteractionRule(
        left="anticoagulant", right="nsaid", severity="major",
        title="Greatly increased bleeding risk",
        mechanism=(
            "Anticoagulants slow clotting, while anti-inflammatory painkillers irritate the "
            "stomach lining and reduce platelet function. Together the risk of serious "
            "stomach bleeding is much higher than with either alone."
        ),
        advice=(
            "Tell your pharmacist before taking any over-the-counter painkiller — including "
            "ibuprofen or aspirin bought without a prescription. Paracetamol is often "
            "suggested instead, but confirm that with them."
        ),
        source="FDA label — warfarin/DOAC bleeding warnings; NSAID GI warnings",
    ),
    InteractionRule(
        left="anticoagulant", right="antiplatelet", severity="major",
        title="Two blood-thinning medicines together",
        mechanism=(
            "Both reduce the blood's ability to clot, by different mechanisms. The bleeding "
            "risk adds up."
        ),
        advice=(
            "Sometimes prescribed together on purpose after a stent or heart attack. If both "
            "came from different doctors, make sure each one knows about the other."
        ),
        source="FDA label — anticoagulant and antiplatelet bleeding warnings",
    ),
    InteractionRule(
        left="anticoagulant", right="cyp3a4_inhibitor", severity="major",
        title="Blood thinner may become too strong",
        mechanism=(
            "Some antibiotics and antifungals slow the liver enzymes that clear the "
            "anticoagulant, so more of it stays in the blood and the thinning effect increases."
        ),
        advice=(
            "If you have been prescribed a short antibiotic or antifungal course, tell the "
            "prescriber you take a blood thinner — extra INR checks may be needed."
        ),
        source="FDA label — warfarin drug interactions (CYP inhibition)",
    ),
    InteractionRule(
        left="statin", right="cyp3a4_inhibitor", severity="major",
        title="Higher risk of muscle damage",
        mechanism=(
            "Certain antibiotics, antifungals and heart medicines block the enzyme that "
            "clears some statins, raising statin levels and with them the risk of muscle "
            "injury (rhabdomyolysis)."
        ),
        advice=(
            "Report unexplained muscle pain, tenderness or dark urine promptly. Mention your "
            "statin whenever you are prescribed a new antibiotic."
        ),
        source="FDA label — simvastatin/atorvastatin CYP3A4 interaction warnings",
    ),
    InteractionRule(
        left="statin", right="fibrate", severity="major",
        title="Higher risk of muscle damage",
        mechanism="Both lower cholesterol but each can affect muscle; together the risk of muscle injury rises.",
        advice="Report muscle pain or weakness to your doctor promptly.",
        source="FDA label — statin/fibrate myopathy warnings",
    ),
    InteractionRule(
        left="ace_inhibitor", right="nsaid", severity="moderate",
        title="Blood pressure control may weaken, and kidneys are under more strain",
        mechanism=(
            "Anti-inflammatory painkillers cause fluid retention and narrow the blood vessels "
            "in the kidney, which works against the blood-pressure medicine and reduces "
            "kidney blood flow."
        ),
        advice=(
            "Occasional use is usually manageable; regular use is what matters. Ask your "
            "pharmacist which painkiller suits someone on blood-pressure treatment."
        ),
        source="FDA label — NSAID warnings (renal effects, antihypertensive interference)",
    ),
    InteractionRule(
        left="arb", right="nsaid", severity="moderate",
        title="Blood pressure control may weaken, and kidneys are under more strain",
        mechanism="Anti-inflammatory painkillers reduce kidney blood flow and counteract blood-pressure medicines.",
        advice="Ask your pharmacist before using anti-inflammatory painkillers regularly.",
        source="FDA label — NSAID renal and antihypertensive warnings",
    ),
    InteractionRule(
        left="diuretic", right="nsaid", severity="moderate",
        title="Water tablet may work less well",
        mechanism="Anti-inflammatory painkillers cause the body to hold on to salt and water, opposing the diuretic.",
        advice="Mention regular painkiller use at your next review, especially if swelling returns.",
        source="FDA label — NSAID warnings (fluid retention)",
    ),
    InteractionRule(
        left="beta_blocker_nonselective", right="beta_agonist", severity="major",
        title="Reliever inhaler may not work properly",
        mechanism=(
            "Non-selective beta-blockers block the same receptors the reliever inhaler works "
            "through, and can themselves tighten the airways."
        ),
        advice=(
            "Important if you have asthma. Make sure whoever prescribed the beta-blocker "
            "knows about your inhaler — a different medicine may be preferred."
        ),
        source="FDA label — propranolol contraindication in bronchospastic disease",
    ),
    InteractionRule(
        left="beta_blocker", right="sulfonylurea", severity="moderate",
        title="Warning signs of low blood sugar may be hidden",
        mechanism=(
            "Beta-blockers can mask the racing heart and tremor that normally warn of a "
            "hypo, so low blood sugar may go unnoticed until it is more severe."
        ),
        advice="Check your blood sugar as advised rather than relying on how you feel.",
        source="FDA label — beta-blocker warnings (masking of hypoglycemia)",
    ),
    InteractionRule(
        left="beta_blocker", right="insulin", severity="moderate",
        title="Warning signs of low blood sugar may be hidden",
        mechanism="Beta-blockers blunt the tremor and rapid heartbeat that usually signal a hypo.",
        advice="Monitor blood glucose as advised rather than waiting to feel symptoms.",
        source="FDA label — beta-blocker warnings (masking of hypoglycemia)",
    ),
    InteractionRule(
        left="triptan", right="serotonergic", severity="moderate",
        title="Small risk of serotonin syndrome",
        mechanism=(
            "Migraine triptans and antidepressants both raise serotonin activity. Rarely the "
            "combination causes agitation, sweating, tremor, fast heartbeat and confusion."
        ),
        advice=(
            "Often prescribed together and usually tolerated. Seek urgent advice if you feel "
            "agitated, shivery, sweaty and confused after a dose.",
        )[0],
        source="FDA label — triptan warnings (serotonin syndrome)",
    ),
    InteractionRule(
        left="ssri", right="nsaid", severity="moderate",
        title="Increased risk of stomach bleeding",
        mechanism="SSRIs reduce platelet function; anti-inflammatory painkillers irritate the stomach lining.",
        advice="Ask your pharmacist about stomach protection if you need painkillers regularly.",
        source="FDA label — SSRI warnings (abnormal bleeding with NSAIDs)",
    ),
    InteractionRule(
        left="digoxin", right="potassium_lowering", severity="major",
        title="Water tablet can make digoxin more toxic",
        mechanism=(
            "Diuretics lower potassium, and low potassium makes the heart more sensitive to "
            "digoxin — the level in the blood need not change for toxicity to develop."
        ),
        advice="Report nausea, visual changes or a very slow pulse. Keep blood-test appointments.",
        source="FDA label — digoxin warnings (hypokalemia and toxicity)",
    ),
    InteractionRule(
        left="methotrexate", right="nsaid", severity="major",
        title="Methotrexate may build up in the body",
        mechanism="Anti-inflammatory painkillers reduce how fast the kidneys clear methotrexate.",
        advice=(
            "Low-dose weekly methotrexate with occasional painkillers is common, but check "
            "with your rheumatology team before regular use."
        ),
        source="FDA label — methotrexate drug interactions (NSAIDs)",
    ),
    InteractionRule(
        left="corticosteroid", right="nsaid", severity="moderate",
        title="Increased risk of stomach ulcer",
        mechanism="Steroids and anti-inflammatory painkillers each irritate the stomach lining; together the risk multiplies.",
        advice="Ask whether you need a stomach-protecting medicine while on both.",
        source="FDA label — corticosteroid and NSAID GI warnings",
    ),
    InteractionRule(
        left="ace_inhibitor", right="lithium", severity="moderate",
        title="Lithium level may rise",
        mechanism="ACE inhibitors reduce lithium clearance by the kidneys.",
        advice="Lithium levels should be monitored more closely.",
        source="FDA label — ACE inhibitor drug interactions (lithium)",
    ),

    # --- duplicate therapy ---
    InteractionRule(
        left="nsaid", right="nsaid", severity="major", same_class=True,
        title="Two anti-inflammatory painkillers at once",
        mechanism=(
            "Taking two NSAIDs together multiplies the risk of stomach bleeding and kidney "
            "strain without improving pain relief. Aspirin counts as one of them."
        ),
        advice=(
            "Check whether one of these came from a pharmacy shelf rather than a "
            "prescription — combination cold and pain remedies often contain an NSAID."
        ),
        source="FDA label — NSAID class warnings (concomitant NSAID use)",
    ),
    InteractionRule(
        left="ace_inhibitor", right="arb", severity="major", same_class=True,
        title="Two medicines acting on the same blood-pressure pathway",
        mechanism=(
            "ACE inhibitors and ARBs act on the same system. Combining them raises the risk "
            "of kidney problems, low blood pressure and high potassium without added benefit "
            "for most people."
        ),
        advice="Confirm with your doctor that both are meant to be taken together.",
        source="FDA label — dual RAAS blockade warnings",
    ),
    InteractionRule(
        left="benzodiazepine", right="opioid", severity="major",
        title="Dangerous sedation risk",
        mechanism="Both slow breathing and cause sedation; together they can suppress breathing.",
        advice="This combination carries a boxed warning. Confirm both prescribers know.",
        source="FDA boxed warning — benzodiazepine and opioid co-prescribing",
    ),
]


# ---------------------------------------------------------------------------
# ENGINE
# ---------------------------------------------------------------------------

@dataclass
class ResolvedDrug:
    """A drug the checker was able to place into classes."""
    name: str
    display: str
    classes: Set[str] = field(default_factory=set)
    confident: bool = True


def _classes_from_dataset(drug_class_text: str) -> Set[str]:
    """Best-effort class tags from the dataset's free-text drug_class column."""
    text = (drug_class_text or "").lower()
    tags: Set[str] = set()
    for pattern, mapped in _CLASS_PATTERNS:
        if re.search(pattern, text):
            tags |= mapped
    return tags


def resolve_drug(name: str, drug_class_text: str = "") -> ResolvedDrug:
    """
    Map a drug name onto interaction classes.

    Salt forms are reduced first, so "amlodipine besylate" and "warfarin
    sodium" resolve like their base ingredients.
    """
    display = str(name or "").strip()
    if not display:
        return ResolvedDrug(name="", display="", classes=set(), confident=False)

    tags: Set[str] = set()
    confident = False

    # A combination product contributes every ingredient it contains.
    for ingredient in ingredient_set(display) or {base_ingredient(display)}:
        curated = DRUG_CLASSES.get(ingredient)
        if curated:
            tags |= curated
            confident = True

    if not tags:
        tags = _classes_from_dataset(drug_class_text)

    return ResolvedDrug(
        name=base_ingredient(display) or display.lower(),
        display=display,
        classes=tags,
        confident=confident,
    )


def _rule_matches(rule: InteractionRule, a: ResolvedDrug, b: ResolvedDrug) -> bool:
    if rule.same_class:
        return rule.left in a.classes and rule.right in b.classes and a.name != b.name
    return (
        (rule.left in a.classes and rule.right in b.classes)
        or (rule.left in b.classes and rule.right in a.classes)
    )


def check_interactions(
    drugs: Iterable[Any],
    *,
    disease: str = "",
) -> Dict[str, Any]:
    """
    Check every pair among the supplied drugs.

    `drugs` may be plain names, or dicts carrying `generic_name_clean` /
    `drug_class` (i.e. lookup matches), so callers can pass results straight
    from the prescription pipeline.
    """
    resolved: List[ResolvedDrug] = []
    seen: Set[str] = set()

    for item in drugs or []:
        if isinstance(item, dict):
            name = (
                item.get("generic_name_clean")
                or item.get("generic_name")
                or item.get("drug")
                or item.get("name")
                or ""
            )
            class_text = item.get("drug_class", "")
        else:
            name, class_text = str(item or ""), ""

        drug = resolve_drug(name, class_text)
        if not drug.name or drug.name in seen:
            continue
        seen.add(drug.name)
        resolved.append(drug)

    findings: List[Dict[str, Any]] = []
    for a, b in itertools.combinations(resolved, 2):
        for rule in INTERACTION_RULES:
            if not _rule_matches(rule, a, b):
                continue
            findings.append({
                "drugs": [a.display, b.display],
                "severity": rule.severity,
                "title": rule.title,
                "mechanism": rule.mechanism,
                "advice": rule.advice,
                "source": rule.source,
                "duplicate_therapy": rule.same_class,
                # False when either drug's classes came from the dataset's
                # free-text column rather than the curated table.
                "high_confidence": a.confident and b.confident,
            })
            break  # one finding per pair — the first (most severe) rule wins

    findings.sort(key=lambda f: SEVERITY_ORDER.get(f["severity"], 9))

    unclassified = [d.display for d in resolved if not d.classes]

    return {
        "checked": [d.display for d in resolved],
        "pairs_checked": max(0, len(resolved) * (len(resolved) - 1) // 2),
        "interactions": findings,
        "count": len(findings),
        "highest_severity": findings[0]["severity"] if findings else None,
        "unclassified_drugs": unclassified,
        "disclaimer": (
            "This check covers a curated set of well-established interactions only. "
            "No interaction found does NOT mean the combination is safe — it means "
            "nothing was found in this list. It does not account for your dose, kidney "
            "function, age or other conditions. Never stop or change a medicine based on "
            "this; your pharmacist can review everything you take together."
        ),
        "coverage_note": (
            f"{len(unclassified)} of {len(resolved)} medicines could not be classified "
            "and were not checked." if unclassified else ""
        ),
    }


def interaction_status() -> Dict[str, Any]:
    """Surfaced in /api/meta so the coverage of this feature is visible."""
    return {
        "rules": len(INTERACTION_RULES),
        "classified_ingredients": len(DRUG_CLASSES),
        "severities": sorted(SEVERITY_ORDER, key=SEVERITY_ORDER.get),
        "curated": True,
    }
