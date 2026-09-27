from src.data.prompts import build_guided_correct_prompt, build_neutral_prompt


def test_neutral_prompt_contains_question():
    p = build_neutral_prompt("Who won the 2018 lacrosse championship?")
    assert "Who won the 2018 lacrosse championship?" in p
    assert "Final answer:" in p


def test_guided_prompt_contains_gold_answer_and_question():
    p = build_guided_correct_prompt(
        question="Who won?", gold_answer="Yale", correct_examples=["Yale won it."]
    )
    assert "Who won?" in p
    assert "Yale" in p
    assert "Yale won it." in p


def test_guided_prompt_handles_no_examples_gracefully():
    p = build_guided_correct_prompt(question="Who won?", gold_answer="Yale", correct_examples=[])
    assert "no prior correct examples" in p
    assert "Yale" in p


def test_guided_prompt_numbers_multiple_examples():
    p = build_guided_correct_prompt(
        question="Q", gold_answer="A", correct_examples=["ex one", "ex two"]
    )
    assert "Example correct response 1:" in p
    assert "Example correct response 2:" in p


def test_guided_prompt_caps_examples_at_max_examples():
    examples = [f"example number {i}" for i in range(10)]
    p = build_guided_correct_prompt(
        question="Q", gold_answer="A", correct_examples=examples, max_examples=3
    )
    assert "Example correct response 3:" in p
    assert "Example correct response 4:" not in p
    assert "example number 3" not in p  # 4th example (0-indexed) excluded


def test_guided_prompt_default_cap_is_3():
    examples = [f"example number {i}" for i in range(10)]
    p = build_guided_correct_prompt(question="Q", gold_answer="A", correct_examples=examples)
    assert "Example correct response 3:" in p
    assert "Example correct response 4:" not in p
