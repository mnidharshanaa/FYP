from src.llm.fake_client import FakeLLMClient
from src.reliability.propagation_debate import run_propagation_debate


def test_total_tokens_partial_accumulates_from_round1_fresh_generation():
    # n_rounds=1 -> the round-2+ loop never executes (range(2,2) is
    # empty), isolating this test to exactly the round-1 token-counting
    # path without needing to script the full multi-round call sequence.
    llm = FakeLLMClient(scripted_responses=[
        ("Final answer: Yale", [{"Yale": -0.1}, {"Yale": -0.2}, {"Yale": -0.3}]),   # agent 0: 3 tokens
        ("Final answer: Yale", [{"Yale": -0.1}, {"Yale": -0.2}]),                    # agent 1: 2 tokens
        ("Final answer: Duke", [{"Duke": -0.5}]),                                    # agent 2: 1 token
    ])
    result = run_propagation_debate(
        llm, "q1", "Who won?", gold_answer="Yale",
        n_agents=3, n_rounds=1, setup="standard",
        lam=0.5, tau_p=0.6, tau_u=0.5, n_resample=3, theta=0.9,
    )
    assert result.total_tokens_partial == 6  # 3 + 2 + 1
    assert result.n_rounds_run == 1


def test_total_tokens_partial_is_zero_when_no_logprobs_available():
    # response_fn mode never carries logprobs (see FakeLLMClient's own
    # docstring) -> the partial token count is honestly 0, not fabricated.
    llm = FakeLLMClient(response_fn=lambda p: "Final answer: Yale")
    result = run_propagation_debate(
        llm, "q1", "Who won?", gold_answer="Yale",
        n_agents=3, n_rounds=1, setup="standard",
        lam=0.5, tau_p=0.6, tau_u=0.5, n_resample=3, theta=0.9,
    )
    assert result.total_tokens_partial == 0
