from src.rag.evidence import (
    adjust_entropy_for_evidence,
    check_contradiction,
    get_evidence,
    is_supported_by_evidence,
)


def test_get_evidence_is_passthrough():
    assert get_evidence("Yale won the championship in 2018.") == "Yale won the championship in 2018."


def test_supported_claim_matching_evidence():
    response = "Reasoning here. Final answer: Yale"
    evidence = "Yale University won the 2018 NCAA lacrosse championship, their first title."
    assert is_supported_by_evidence(response, evidence)


def test_unsupported_claim_not_in_evidence():
    response = "Reasoning here. Final answer: Duke"
    evidence = "Yale University won the 2018 NCAA lacrosse championship, their first title."
    assert not is_supported_by_evidence(response, evidence)


def test_empty_claim_never_supported():
    response = ""
    evidence = "Yale University won the championship."
    assert not is_supported_by_evidence(response, evidence)


def test_check_contradiction_is_inverse_of_support():
    response = "Final answer: Yale"
    evidence = "Yale University won the 2018 championship."
    assert is_supported_by_evidence(response, evidence) == (not check_contradiction(response, evidence))


def test_check_contradiction_flags_unsupported_claim():
    response = "Final answer: Duke"
    evidence = "Yale University won the 2018 championship, defeating all other contenders."
    assert check_contradiction(response, evidence)


def test_overlap_threshold_is_configurable():
    response = "Final answer: Yale Bulldogs lacrosse team"
    evidence = "Yale won it in 2018."
    # low threshold -> supported; very high threshold -> not supported
    assert is_supported_by_evidence(response, evidence, overlap_threshold=0.1)
    assert not is_supported_by_evidence(response, evidence, overlap_threshold=0.95)


def test_adjust_entropy_inflates_when_flagged():
    assert adjust_entropy_for_evidence(0.5, flagged=True, inflation_factor=2.0) == 1.0


def test_adjust_entropy_unchanged_when_not_flagged():
    assert adjust_entropy_for_evidence(0.5, flagged=False, inflation_factor=2.0) == 0.5


def test_adjust_entropy_default_inflation_factor():
    result = adjust_entropy_for_evidence(0.3, flagged=True)
    assert result == 0.6  # default factor 2.0
