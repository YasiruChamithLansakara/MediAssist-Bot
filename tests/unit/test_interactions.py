"""
Drug–drug interaction checking.

Two properties matter more than coverage here:

  * every reported interaction must be real (a fabricated warning in a
    medication tool destroys trust in the warnings that are real), and
  * an empty result must never read as "safe".
"""

from __future__ import annotations

import pytest

from app.services.interaction_service import (
    INTERACTION_RULES,
    check_interactions,
    interaction_status,
    resolve_drug,
)


# --------------------------------------------------------- known interactions
KNOWN = [
    (["warfarin", "ibuprofen"], "major", "bleeding"),
    (["lisinopril", "spironolactone"], "major", "potassium"),
    (["propranolol", "albuterol"], "major", "inhaler"),
    (["atorvastatin", "clarithromycin"], "major", "muscle"),
    (["ibuprofen", "naproxen"], "major", "anti-inflammatory"),
    (["digoxin", "furosemide"], "major", "digoxin"),
    (["sertraline", "ibuprofen"], "moderate", "bleeding"),
    (["sumatriptan", "sertraline"], "moderate", "serotonin"),
    (["metoprolol", "glipizide"], "moderate", "blood sugar"),
]


@pytest.mark.parametrize("drugs,severity,keyword", KNOWN)
def test_known_interactions_are_detected(drugs, severity, keyword):
    report = check_interactions(drugs)
    assert report["count"] >= 1, f"missed interaction for {drugs}"
    assert report["highest_severity"] == severity
    blob = " ".join(
        f"{f['title']} {f['mechanism']}" for f in report["interactions"]
    ).lower()
    assert keyword in blob, f"{keyword!r} not explained for {drugs}"


def test_salt_forms_resolve_like_their_base_ingredient():
    """A prescription says 'warfarin sodium'; the rules are keyed on 'warfarin'."""
    salted = check_interactions(["warfarin sodium", "ibuprofen"])
    plain = check_interactions(["warfarin", "ibuprofen"])
    assert salted["count"] == plain["count"] == 1


def test_unrelated_drugs_report_nothing():
    report = check_interactions(["metformin", "amlodipine"])
    assert report["count"] == 0
    assert report["highest_severity"] is None


def test_triple_whammy_reports_every_pair():
    """ACE inhibitor + NSAID + diuretic is three interacting pairs, not one."""
    report = check_interactions(["lisinopril", "ibuprofen", "spironolactone"])
    assert report["count"] == 3
    assert report["highest_severity"] == "major"


def test_findings_are_ordered_most_severe_first():
    report = check_interactions(["lisinopril", "ibuprofen", "spironolactone"])
    severities = [f["severity"] for f in report["interactions"]]
    assert severities == sorted(severities, key=lambda s: {"major": 0, "moderate": 1, "minor": 2}[s])


# ------------------------------------------------------------- safety framing
def test_empty_result_still_carries_a_disclaimer():
    """'No interactions found' must never be read as 'safe'."""
    report = check_interactions(["metformin", "amlodipine"])
    assert report["disclaimer"]
    assert "does not mean" in report["disclaimer"].lower()


def test_no_rule_tells_a_patient_to_stop_a_medicine():
    """
    The assistant is non-prescribing by design. Advice may say "ask" — never
    "stop", "reduce" or "switch".
    """
    forbidden = ("stop taking", "discontinue", "reduce your dose", "switch to", "you should take")
    for rule in INTERACTION_RULES:
        text = f"{rule.advice} {rule.mechanism}".lower()
        for phrase in forbidden:
            assert phrase not in text, f"{rule.title!r} advises: {phrase!r}"


def test_every_rule_cites_a_source():
    for rule in INTERACTION_RULES:
        assert rule.source.strip(), f"{rule.title!r} has no source"
        assert rule.severity in {"major", "moderate", "minor"}


# ------------------------------------------------------------------- coverage
def test_unclassified_drugs_are_reported_not_hidden():
    """A drug the table cannot classify must be declared, not silently skipped."""
    report = check_interactions(["warfarin", "zynophrenidol"])
    assert "zynophrenidol" in [d.lower() for d in report["unclassified_drugs"]]
    assert report["coverage_note"]


def test_single_drug_is_not_an_error():
    report = check_interactions(["warfarin"])
    assert report["count"] == 0
    assert report["pairs_checked"] == 0


def test_resolve_drug_marks_curated_vs_inferred():
    curated = resolve_drug("warfarin")
    assert curated.confident and "anticoagulant" in curated.classes

    inferred = resolve_drug("some unknown drug", drug_class_text="NSAID / analgesic")
    assert not inferred.confident
    assert "nsaid" in inferred.classes


def test_status_reports_coverage():
    status = interaction_status()
    assert status["rules"] == len(INTERACTION_RULES)
    assert status["classified_ingredients"] > 50
