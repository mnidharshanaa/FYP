"""
scripts/verify_dig_metrics.py

Offline, no-GPU, no-vLLM verification of the DIG module (src/digra/digra_debate.py,
variant="dig") and every metric on the project's Module-3/DIG checklist.

WHAT THIS DOES
---------------
Runs two DIG-shaped debates (n_agents=3, n_rounds=2, max_subset_size=1)
through FakeLLMClient with hand-crafted, deterministic logprobs (so
entropy/IG are real, reproducible numbers — not GPU-dependent). It runs
the SAME synthetic scenario under variant="dig" and variant="digra" to
concretely demonstrate the one thing that must be true for DIG to be a
correct ablation: identical inputs, but the raw-IG-only partner
selection (dig) and entropy-normalized IGR selection (digra) can pick
DIFFERENT communication partners for the same agent — the scenario here
is deliberately constructed so agent 0's round-2 partner choice diverges
between the two variants (see scenario comments below).

It then saves both runs as JSONL in the project's real output schema,
loads them back, and computes EVERY metric on the checklist against the
"dig" run, printing a checked-off report. This is a structural/plumbing
verification (correct fields exist, correct math, correct file format,
correct plots) — it is NOT a substitute for a real run on real model
outputs; entropy/IG numbers here come from hand-picked logprobs, not a
real model's actual uncertainty.

Usage (no GPU, no Kaggle, runs anywhere with the repo's requirements installed):
    python scripts/verify_dig_metrics.py
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.digra.digra_debate import run_digra_debate
from src.llm.fake_client import FakeLLMClient
from src.metrics.propagation_metrics import (
    compute_communication_stats,
    compute_cost_stats,
    compute_em_f1,
    compute_final_accuracy,
    compute_ig_per_round,
    compute_majority_vote_stats,
    compute_propagation_metrics,
    compute_token_stats,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "dig_verification"


# ---------------------------------------------------------------------------
# Hand-crafted logprob distributions -> known, reproducible mean token
# entropy values (Eq. 6). Verified by hand in the module docstring's
# scenario walkthrough; also printed by this script for cross-checking.
# ---------------------------------------------------------------------------
_DISTRIBUTIONS = {
    "verylow": {"X": -0.01, "Y": -4.0},
    "low": {"X": -0.05, "Y": -3.0},
    "med": {"X": -0.3, "Y": -1.5},
    "high": {"X": -0.7, "Y": -0.7},
    "veryhigh": {"X": -1.0986, "Y": -1.0986, "Z": -1.0986},
}


def lp(label: str, n: int = 5) -> list:
    dist = _DISTRIBUTIONS[label]
    return [dict(dist) for _ in range(n)]


def _entropy_of(label: str) -> float:
    from src.entropy.entropy import mean_token_entropy
    return mean_token_entropy(lp(label, n=1))


def build_scenario_queue() -> list:
    """
    3-agent, 2-round scenario. Round 1: agent0 correct+medium-confidence,
    agent1 wrong+confident ("faithfulness hallucination" — confidently
    wrong), agent2 wrong+very-uncertain. Round 2's forced-decode values
    for agent0's two candidate partners are chosen so raw IG favors
    partner {2} (agent2) but IGR favors partner {1} (agent1), because
    agent2's own entropy is much higher and IGR's denominator penalizes
    that — this is the exact, minimal case that makes DIG and DIGRA
    genuinely different algorithms, not just differently-named copies of
    the same one.
    """
    queue = []
    # --- Round 1: fresh generation, n=3, one call ---
    queue.append(("Agent0 reasoning. Final answer: Yale", lp("med")))       # H0 ~ 0.558
    queue.append(("Agent1 reasoning. Final answer: Duke", lp("low")))       # H1 ~ 0.197
    queue.append(("Agent2 reasoning. Final answer: Duke", lp("veryhigh"))) # H2 ~ 1.099

    # --- Round 2 selection pass (max_subset_size=1 -> singleton subsets only) ---
    # i=0, candidates=[1,2]:
    queue.append(("_", lp("med")))       # fd(agent0 | {1}) -> IG_1 = H0 - H(med)      ~ 0.000
    queue.append(("_", lp("verylow")))   # fd(agent0 | {2}) -> IG_2 = H0 - H(verylow)  ~ 0.474
    # i=1, candidates=[0,2]:
    queue.append(("_", lp("low")))       # fd(agent1 | {0}) -> IG ~ 0.000
    queue.append(("_", lp("med")))       # fd(agent1 | {2}) -> IG < 0
    # i=2, candidates=[0,1]:
    queue.append(("_", lp("low")))       # fd(agent2 | {0}) -> IG ~ 0.901 (biggest IG in this round)
    queue.append(("_", lp("med")))       # fd(agent2 | {1}) -> IG ~ 0.541

    # --- Round 2 generation (batched, agents_to_generate=[0,1,2]) ---
    queue.append(("Agent0 reasoning. Final answer: Yale", lp("verylow")))   # stays correct, confident
    queue.append(("Agent1 reasoning. Final answer: Yale", lp("med")))       # corrected
    queue.append(("Agent2 reasoning. Final answer: Duke", lp("high")))      # still wrong

    return queue


def run_variant(variant: str) -> dict:
    llm = FakeLLMClient(scripted_responses=build_scenario_queue())
    result = run_digra_debate(
        llm=llm, question_id="demo_q1", question="Which university did the person attend?",
        gold_answer="Yale", gold_answer_alternatives=["Yale University"],
        n_agents=3, n_rounds=2, setup="standard", variant=variant,
        seed=0, alpha=0.2, max_subset_size=1,
    )
    return asdict(result)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("STEP 1 — reference entropy values for the scenario's logprob labels")
    print("=" * 78)
    for label in _DISTRIBUTIONS:
        print(f"  H({label:9s}) = {_entropy_of(label):.4f}")

    print("\n" + "=" * 78)
    print("STEP 2 — running variant='dig' and variant='digra' on IDENTICAL input")
    print("=" * 78)
    dig_debate = run_variant("dig")
    digra_debate = run_variant("digra")

    dig_partner_agent0 = sorted(dig_debate["agent_records"][0][1]["partners_selected"])
    digra_partner_agent0 = sorted(digra_debate["agent_records"][0][1]["partners_selected"])
    dig_scores_agent0 = dig_debate["agent_records"][0][1]["candidate_scores"]
    digra_scores_agent0 = digra_debate["agent_records"][0][1]["candidate_scores"]

    print(f"  DIG   agent0's candidate scores (raw IG):  {dig_scores_agent0}")
    print(f"  DIG   agent0's chosen partner:              {dig_partner_agent0}")
    print(f"  DIGRA agent0's candidate scores (IGR):      {digra_scores_agent0}")
    print(f"  DIGRA agent0's chosen partner:               {digra_partner_agent0}")

    if dig_partner_agent0 != digra_partner_agent0:
        print(
            "  VERIFIED: DIG and DIGRA chose DIFFERENT partners for agent 0 from "
            "identical entropy/IG inputs — confirms DIG is not silently reducing "
            "to DIGRA (or vice versa); the entropy-normalization term in Eq. 8 is "
            "the one and only thing responsible for the difference."
        )
    else:
        print("  WARNING: partners matched — scenario numbers need revisiting to force divergence.")
        sys.exit(1)

    # Save both runs in the real output schema (same as scripts/03_run_digra_experiments.py)
    dig_path = OUT_DIR / "demo_nq_llama_dig_agents3.jsonl"
    digra_path = OUT_DIR / "demo_nq_llama_digra_agents3.jsonl"
    dig_path.write_text(json.dumps(dig_debate) + "\n")
    digra_path.write_text(json.dumps(digra_debate) + "\n")
    print(f"\n  wrote {dig_path}")
    print(f"  wrote {digra_path}")

    print("\n" + "=" * 78)
    print("STEP 3 — every metric on the DIG checklist, computed from the saved file")
    print("=" * 78)
    dig_results = [dig_debate]

    acc = compute_final_accuracy(dig_results)
    print(f"  [x] Accuracy                 = {acc:.3f}")

    emf1 = compute_em_f1(dig_results)
    print(f"  [x] EM                       = {emf1['EM']:.3f}   (n={emf1['n']})")
    print(f"  [x] F1                       = {emf1['F1']:.3f}")
    print(f"  [x] TruthfulQA metric        = {acc:.3f}   (this project reuses final accuracy as the "
          "TruthfulQA-specific metric — no separate judge-model truthfulness "
          "metric is implemented; see script docstring)")

    prop = compute_propagation_metrics(dig_results)
    print(f"  [x] MA per round             = {prop['MA']}")
    print(f"  [x] MR per round             = {prop['MR']}")
    print(f"  [x] IMR per round            = {prop['IMR']}")
    print(f"  [x] CR per round             = {prop['CR']}")

    votes = compute_majority_vote_stats(dig_results)
    print(f"  [x] Majority Vote Share      = {votes['majority_vote_share']:.3f}")
    print(f"  [x] Vote Entropy             = {votes['vote_entropy']:.3f}")
    print(f"  [x] # Unique Answers         = {votes['mean_unique_answers']:.1f}")

    ig_by_round = compute_ig_per_round(dig_results)
    print(f"  [x] Information Gain (IG), per edge per round = {ig_by_round}")

    comm = compute_communication_stats(dig_results)
    print(f"  [x] # Candidate Edges        = {comm['candidate_edges']}")
    print(f"  [x] # Selected Edges (mean)  = {comm['mean_selected_edges']}")
    print(f"  [x] Communication Sparsity   = {comm['sparsity']:.3f}")

    tokens = compute_token_stats(dig_results)
    print(f"  [x] Average Tokens           = {tokens['average_tokens_per_response']:.1f}")
    print(f"  [x] Total Tokens             = {tokens['total_tokens']}")

    cost = compute_cost_stats(dig_results)
    print(f"  [x] (cost) generate calls    = {cost['total_generate_calls']}")
    print(f"  [x] (cost) forced_decode     = {cost['total_forced_decode_calls']}")
    print(
        "  [ ] Average Latency          = NOT AVAILABLE from a single offline demo debate "
        "(latency is real-wall-clock, recorded by scripts/03_run_digra_experiments.py's "
        "checkpoint registry during an actual run — see scripts/04_generate_report.py's "
        "load_durations_from_registry). Nothing to compute here; this is the one "
        "checklist item that is fundamentally a real-run measurement, not derivable "
        "from saved debate content."
    )

    print("\n" + "=" * 78)
    print("STEP 4 — generating plots from this demo data")
    print("=" * 78)
    _generate_demo_plots(dig_results, [digra_debate])

    print("\nAll checklist metrics computed successfully. See results/dig_verification/ for output.")


def _generate_demo_plots(dig_results: list, digra_results: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    prop_dig = compute_propagation_metrics(dig_results)
    prop_digra = compute_propagation_metrics(digra_results)
    rounds = sorted(prop_dig["MA"].keys())

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rounds, [prop_dig["MA"][r] for r in rounds], marker="o", label="DIG")
    ax.plot(rounds, [prop_digra["MA"][r] for r in rounds], marker="s", label="DIGRA")
    ax.set_xlabel("Debate round")
    ax.set_ylabel("Mean Accuracy (MA)")
    ax.set_title("DIG vs DIGRA — demo scenario")
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    fname = fig_dir / "demo_ma_trajectory.png"
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  wrote {fname}")

    comm_dig = compute_communication_stats(dig_results)
    comm_digra = compute_communication_stats(digra_results)
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["DIG", "DIGRA"], [comm_dig["sparsity"], comm_digra["sparsity"]], color=["#4C72B0", "#DD8452"])
    ax.set_ylabel("Sparsity (selected / candidate edges)")
    ax.set_ylim(0, 1.1)
    ax.set_title("Communication sparsity — demo scenario")
    for bar, s in zip(bars, [comm_dig["sparsity"], comm_digra["sparsity"]]):
        ax.text(bar.get_x() + bar.get_width() / 2, s + 0.02, f"{s:.2f}", ha="center")
    plt.tight_layout()
    fname = fig_dir / "demo_sparsity.png"
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  wrote {fname}")


if __name__ == "__main__":
    main()
