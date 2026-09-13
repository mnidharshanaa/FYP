from src.metrics.grading import compute_em, compute_f1, extract_final_answer, is_correct, normalize_answer


def test_normalize_lowercases_and_strips_punctuation():
    assert normalize_answer("Yale!") == "yale"


def test_normalize_removes_articles():
    assert normalize_answer("The Yale Bulldogs") == "yale bulldogs"


def test_normalize_collapses_whitespace():
    assert normalize_answer("  Yale   University ") == "yale university"


def test_is_correct_exact_match():
    assert is_correct("Yale", "Yale")


def test_is_correct_containment_in_full_sentence():
    assert is_correct("The answer is Yale University.", "Yale")


def test_is_correct_case_and_article_insensitive():
    assert is_correct("i believe it's the Yale team", "yale")


def test_is_correct_false_for_wrong_answer():
    assert not is_correct("The answer is Duke.", "Yale")


def test_is_correct_checks_alternatives():
    assert is_correct(
        "It was Osama bin Laden who caused it",
        gold_answer="Al-Qaeda caused the 9/11 attacks",
        alternatives=["Osama bin Laden caused the 9/11 attacks"],
    )


def test_is_correct_overlap_fallback_catches_paraphrase():
    # full-sentence gold answer, paraphrased rather than quoted verbatim —
    # containment alone would miss this (regression test for the bug this
    # fallback was added to fix)
    assert is_correct(
        "Bin Laden and Al-Qaeda were behind the 9/11 attacks, historians agree.",
        gold_answer="Al-Qaeda caused the 9/11 attacks",
    )


def test_is_correct_overlap_fallback_does_not_match_low_overlap():
    # shares only one low-signal token ("the") after normalization, should not match
    assert not is_correct(
        "The weather today is quite pleasant.",
        gold_answer="Al-Qaeda caused the 9/11 attacks",
    )


def test_is_correct_overlap_threshold_is_respected():
    # exactly at the boundary: 3/5 = 0.6 tokens overlap -> should match at default threshold
    assert is_correct("alpha beta gamma", gold_answer="alpha beta gamma delta epsilon")
    # well below threshold: 1/5 = 0.2 -> should not match
    assert not is_correct("alpha zzz zzz zzz zzz", gold_answer="alpha beta gamma delta epsilon")


def test_is_correct_boolq_yes_no():
    assert is_correct("Yes, that is correct.", "yes")
    assert not is_correct("No, that is not correct.", "yes")


def test_is_correct_empty_gold_never_matches():
    # guards against normalize_answer("") being an empty-string false positive
    assert not is_correct("anything at all", "")


def test_is_correct_empty_prediction_never_matches_any_gold():
    # regression test: an empty/unparseable prediction must never be judged
    # correct, regardless of gold answer — empty string is trivially a
    # substring of everything, so the containment check alone would
    # otherwise silently mark blank responses as "correct"
    assert not is_correct("", "Yale")
    assert not is_correct("   ", "Yale")
    assert not is_correct("", "Al-Qaeda caused the 9/11 attacks")


def test_extract_final_answer_with_marker():
    text = "Let me think about this. Final answer: Yale."
    assert extract_final_answer(text) == "Yale"


def test_extract_final_answer_case_insensitive_marker():
    text = "Reasoning here. FINAL ANSWER: Yale"
    assert extract_final_answer(text) == "Yale"


def test_extract_final_answer_falls_back_to_full_text_when_no_marker():
    text = "Yale won the championship."
    assert extract_final_answer(text) == "Yale won the championship."


def test_extract_final_answer_uses_last_marker_occurrence():
    text = "Final answer: Duke. Wait, let me reconsider. Final answer: Yale."
    assert extract_final_answer(text) == "Yale"


# ---------------------------------------------------------------------------
# EM / F1 (checklist's Module-3/DIG "EM"/"F1" columns)
# ---------------------------------------------------------------------------

def test_compute_em_exact_match_true():
    assert compute_em("Yale", "Yale") is True


def test_compute_em_false_on_partial_containment():
    # is_correct() would accept this (containment rule); EM must NOT —
    # that's the entire point of EM being stricter than is_correct.
    assert compute_em("I think it was Yale", "Yale") is False


def test_compute_em_checks_alternatives():
    assert compute_em("Yale University", "Yale", alternatives=["Yale University"]) is True


def test_compute_em_normalizes_before_comparing():
    assert compute_em("the Yale!", "yale") is True


def test_compute_f1_perfect_match_is_one():
    assert compute_f1("Yale Bulldogs", "Yale Bulldogs") == 1.0


def test_compute_f1_partial_overlap_between_zero_and_one():
    f1 = compute_f1("Yale Bulldogs team", "Yale Bulldogs")
    assert 0.0 < f1 < 1.0


def test_compute_f1_no_overlap_is_zero():
    assert compute_f1("Duke", "Yale") == 0.0


def test_compute_f1_empty_prediction_is_zero():
    assert compute_f1("", "Yale") == 0.0


def test_compute_f1_takes_best_across_alternatives():
    # closer to "Yale University" than to "Yale" alone -> should use the
    # alternative that gives the higher score, not just the first one
    f1 = compute_f1("Yale University", "Yale", alternatives=["Yale University"])
    assert f1 == 1.0
