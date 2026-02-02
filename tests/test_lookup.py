from app.services.drug_lookup import init_store, lookup_drug


def setup_module():
    # Ensure dataset is loaded once for tests
    init_store()


def assert_schema(r: dict):
    # Required keys (schema stability)
    for k in (
        "query",
        "normalized",
        "status",
        "confidence",
        "best_score",
        "best_match",
        "matches",
        "suggestions",
        "message",
        "match_type",
        "resolution_path",
    ):
        assert k in r, f"Missing key: {k}"

    # Types + ranges
    assert r["status"] in ("ok", "low_confidence", "no_match")
    assert isinstance(r["confidence"], (int, float))
    assert 0.0 <= float(r["confidence"]) <= 1.0

    assert isinstance(r["best_score"], (int, float))
    assert 0.0 <= float(r["best_score"]) <= 100.0

    assert isinstance(r["matches"], list)
    assert isinstance(r["suggestions"], list)
    assert isinstance(r["resolution_path"], list)


def test_alias_paracetamol():
    r = lookup_drug("paracetamol")
    assert_schema(r)
    assert r["status"] in ("ok", "low_confidence")
    assert r["normalized"] == "acetaminophen"
    assert r["best_match"] is not None
    assert r["best_score"] >= 80


def test_brand_panadol():
    r = lookup_drug("Panadol")
    assert_schema(r)
    assert r["normalized"] == "acetaminophen"
    assert r["best_match"] is not None
    assert r["best_score"] >= 80


def test_dosage_paracetamol_500mg():
    r = lookup_drug("paracetamol 500mg")
    assert_schema(r)
    assert r["normalized"] == "acetaminophen"
    assert r["best_match"] is not None


def test_typo_paracetmol():
    r = lookup_drug("paracetmol")
    assert_schema(r)
    # Acceptable outcomes:
    # - resolves via fuzzy/typo
    # - or no_match + suggestions
    assert r["status"] in ("ok", "low_confidence", "no_match")
    assert isinstance(r["suggestions"], list)


def test_gibberish_zzzzzz():
    r = lookup_drug("zzzzzz")
    assert_schema(r)
    assert r["status"] == "no_match"
    assert r["best_match"] is None
    assert r["matches"] == []
