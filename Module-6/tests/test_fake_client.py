from src.llm.client import GenerationResult, LLMClient
from src.llm.fake_client import FakeLLMClient


def test_fake_client_implements_llm_client_interface():
    assert issubclass(FakeLLMClient, LLMClient)


def test_scripted_responses_plain_strings():
    llm = FakeLLMClient(scripted_responses=["a", "b", "c"])
    results = llm.generate("prompt", n=3)
    assert [r.text for r in results] == ["a", "b", "c"]
    assert all(r.token_logprobs is None for r in results)


def test_scripted_responses_with_logprobs_tuple():
    logprobs = [{"yes": -0.1, "no": -3.0}]
    llm = FakeLLMClient(scripted_responses=[("yes", logprobs)])
    result = llm.generate("prompt", n=1)[0]
    assert result.text == "yes"
    assert result.token_logprobs == logprobs


def test_scripted_responses_mixed_plain_and_tuple():
    logprobs = [{"a": 0.0}]
    llm = FakeLLMClient(scripted_responses=["plain", ("with_lp", logprobs)])
    r1 = llm.generate("p", n=1)[0]
    r2 = llm.generate("p", n=1)[0]
    assert r1.text == "plain" and r1.token_logprobs is None
    assert r2.text == "with_lp" and r2.token_logprobs == logprobs


def test_generate_batch_supports_tuples_per_prompt():
    lp1, lp2 = [{"a": 0.0}], [{"b": -1.0}]
    llm = FakeLLMClient(scripted_responses=[("r1", lp1), ("r2", lp2)])
    results = llm.generate_batch(["p1", "p2"])
    assert results[0].text == "r1" and results[0].token_logprobs == lp1
    assert results[1].text == "r2" and results[1].token_logprobs == lp2


def test_forced_decode_pulls_logprobs_from_queue():
    lp = [{"yale": -0.05}]
    llm = FakeLLMClient(scripted_responses=[("ignored_text", lp)])
    result = llm.forced_decode(prompt="Q", target_text="Yale")
    assert result.text == "Yale"  # always target_text, regardless of queued text
    assert result.token_logprobs == lp


def test_forced_decode_plain_string_queue_item_yields_no_logprobs():
    llm = FakeLLMClient(scripted_responses=["plain string, no logprobs"])
    result = llm.forced_decode(prompt="Q", target_text="Yale")
    assert result.text == "Yale"
    assert result.token_logprobs is None


def test_forced_decode_and_generate_share_one_ordered_queue():
    lp = [{"a": 0.0}]
    llm = FakeLLMClient(scripted_responses=["gen_response", ("fd_ignored", lp)])
    gen_result = llm.generate("p", n=1)[0]
    fd_result = llm.forced_decode("p", target_text="target")
    assert gen_result.text == "gen_response"
    assert fd_result.token_logprobs == lp


def test_calls_are_recorded_for_all_methods():
    llm = FakeLLMClient(response_fn=lambda p: "x")
    llm.generate("p1", n=1)
    llm.generate_batch(["p2"])
    llm.forced_decode("p3", target_text="t")
    methods = [c["method"] for c in llm.calls]
    assert methods == ["generate", "generate_batch", "forced_decode"]


def test_exhausted_queue_raises_clear_error():
    llm = FakeLLMClient(scripted_responses=["only_one"])
    llm.generate("p", n=1)
    try:
        llm.generate("p", n=1)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "exhausted" in str(e)
