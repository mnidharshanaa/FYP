from src.metrics.text_metrics import exact_match, token_f1


def test_exact_match_true_on_identical_normalized_text():
    assert exact_match("The Yale Bulldogs", "yale bulldogs")


def test_exact_match_false_when_prediction_is_only_a_superset():
    # EM is stricter than grading.is_correct's containment fallback.
    assert not exact_match("The answer is Yale University.", "Yale")


def test_exact_match_checks_alternatives():
    assert exact_match("duke", "yale", alternatives=["duke", "cornell"])


def test_exact_match_false_for_wrong_answer():
    assert not exact_match("Duke", "Yale")


def test_token_f1_perfect_match_is_one():
    assert token_f1("Yale University", "Yale University") == 1.0


def test_token_f1_partial_overlap_between_zero_and_one():
    f1 = token_f1("Yale University team", "Yale University")
    assert 0.0 < f1 < 1.0


def test_token_f1_zero_for_no_overlap():
    assert token_f1("Duke", "Yale") == 0.0


def test_token_f1_zero_for_empty_prediction():
    assert token_f1("", "Yale") == 0.0


def test_token_f1_takes_max_over_alternatives():
    f1 = token_f1("Osama bin Laden", "Al-Qaeda", alternatives=["Osama bin Laden"])
    assert f1 == 1.0
