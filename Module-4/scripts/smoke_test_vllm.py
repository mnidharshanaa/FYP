"""
scripts/smoke_test_vllm.py

Run this on Kaggle IMMEDIATELY after installing vLLM, before running
01_build_pools.py or anything else that depends on VLLMClient. This is
the verification step flagged throughout src/llm/vllm_client.py's
docstrings — nothing in that file has been tested against real vLLM yet.

Usage:
    python scripts/smoke_test_vllm.py --model meta-llama/Llama-3.1-8B-Instruct

What it checks, and why each one matters:
  1. Model loads at all.
  2. generate(n=k) returns exactly k results (the contract every other
     module in this project assumes — see pool_generation.py's docstring).
  3. logprobs_topk actually returns per-token distributions in the shape
     GenerationResult expects.
  4. forced_decode's target-text token alignment is sane: forced-decoding
     a model's OWN just-generated greedy continuation of a short prompt
     should produce low entropy / high probability on those exact tokens.
     If this comes back looking like noise, the prompt/target boundary
     slicing in VLLMClient.forced_decode is misaligned and MUST be fixed
     before Module 5 (entropy/IG/IGR) can be trusted.
"""

from __future__ import annotations

import argparse
import sys

from src.llm.vllm_client import VLLMClient


def main(
    model_id: str,
    dtype: str = "float16",
    max_model_len: int = 2048,
    gpu_memory_utilization: float = 0.80,
    tensor_parallel_size: int = 1,
) -> None:
    print(f"=== Smoke test: {model_id} ===\n")

    print("[1/4] Loading model...")
    client = VLLMClient(
        model_id=model_id,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
    )
    print("  OK\n")

    print("[2/4] generate(n=5) returns exactly 5 results...")
    prompt = "Question: What is the capital of France?\nFinal answer:"
    results = client.generate(prompt, n=5, max_tokens=20, temperature=1.0)
    assert len(results) == 5, f"FAIL: expected 5 results, got {len(results)}"
    for i, r in enumerate(results):
        print(f"  [{i}] {r.text!r}")
    print("  OK\n")

    print("[3/4] logprobs_topk returns per-token distributions...")
    results_lp = client.generate(prompt, n=1, max_tokens=10, logprobs_topk=5)
    tlp = results_lp[0].token_logprobs
    assert tlp is not None, "FAIL: token_logprobs is None despite logprobs_topk=5"
    assert len(tlp) > 0, "FAIL: token_logprobs is empty"
    print(f"  First generated token's top-{len(tlp[0])} logprob dict:")
    print(f"    {tlp[0]}")
    print("  OK\n")

    print("[4/4] forced_decode alignment sanity check...")
    print("  Generating a short greedy continuation, then forced-decoding")
    print("  that EXACT text back and checking the model assigns it high")
    print("  probability (as it should, having just generated it itself).")
    greedy_prompt = "The sky is"
    greedy_results = client.generate(greedy_prompt, n=1, max_tokens=5, temperature=0.0)
    greedy_continuation = greedy_results[0].text
    print(f"  Greedy continuation: {greedy_continuation!r}")

    forced = client.forced_decode(greedy_prompt, greedy_continuation, logprobs_topk=5)
    print(f"  Forced-decode token_logprobs ({len(forced.token_logprobs)} positions):")
    for i, position in enumerate(forced.token_logprobs):
        print(f"    position {i}: {position}")

    print()
    print(
        "  MANUAL CHECK REQUIRED: does the top-1 token at each position above\n"
        "  match the corresponding token in the greedy continuation, with a\n"
        "  logprob close to 0 (i.e. probability close to 1)? If yes, the\n"
        "  prompt/target boundary alignment in forced_decode is correct.\n"
        "  If the tokens/logprobs look wrong or misaligned, STOP — fix\n"
        "  VLLMClient.forced_decode's tokenizer boundary logic before\n"
        "  trusting any entropy/IG/IGR result built on top of it."
    )
    print("\n=== Smoke test complete — review [4/4] manually before proceeding ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    args = parser.parse_args()
    try:
        main(
            model_id=args.model,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
        )
    except AssertionError as exc:
        print(f"\nSMOKE TEST FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
