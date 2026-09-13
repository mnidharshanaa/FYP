"""
scripts/verify_digra_metrics.py

Offline, no-GPU verification of variant="digra" against every metric on
the project's Module-4/DIGRA checklist. Sibling of
scripts/verify_dig_metrics.py — same scenario, same synthetic data, so
the two scripts' printed numbers are directly comparable (that's the
point: DIGRA should differ from DIG in exactly the ways Eq. 8's entropy
normalization predicts, nothing else).

Most of DIGRA's checklist is already covered by the SAME metric
functions used for DIG (compute_final_accuracy, compute_em_f1,
compute_propagation_metrics, compute_majority_vote_stats,
compute_communication_stats, compute_information_flow,
compute_token_stats) — DIGRA is the same debate loop, so nothing about
those computations is variant-specific. This script exists to (a)
exercise them against a digra run specifically, (b) print the
DIGRA-only checklist items DIG doesn't have (Entropy shown standalone,
IGR specifically, token savings vs. a baseline), and (c) print an
explicit DIG-vs-DIGRA diff so the entropy-normalization effect is
visible in one place.

Usage:
    python scripts/verify_digra_metrics.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.verify_dig_metrics import build_scenario_queue  # reuse the exact same scenario
from src.digra.digra_debate import run_digra_debate
from src.llm.fake_client import FakeLLMClient
from src.metrics.propagation_metrics import (
    compute_communication_stats,
    compute_em_f1,
    compute_final_accuracy,
    compute_ig_per_round,
    compute_information_flow,
    compute_majority_vote_stats,
    compute_propagation_metrics,
    compute_token_savings,
    compute_token_stats,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "dig_verification"


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

    digra_debate = run_variant("digra")
    dig_debate = run_variant("dig")  # same scenario, for the diff at the end
    digra_results = [digra_debate]

    path = OUT_DIR / "demo_nq_llama_digra_agents3.jsonl"
    path.write_text(json.dumps(digra_debate) + "\n")
    print(f"wrote {path}\n")

    print("=" * 78)
    print("Module-4/DIGRA checklist, computed from the saved file")
    print("=" * 78)

    acc = compute_final_accuracy(digra_results)
    print(f"  [x] Accuracy                 = {acc:.3f}")

    emf1 = compute_em_f1(digra_results)
    print(f"  [x] EM                       = {emf1['EM']:.3f}")
    print(f"  [x] F1                       = {emf1['F1']:.3f}")
    print(f"  [x] TruthfulQA metric        = {acc:.3f}  (reuses final accuracy — see script's DIG sibling for rationale)")

    prop = compute_propagation_metrics(digra_results)
    print(f"  [x] MA/MR/IMR/CR per round   = MA={prop['MA']}  MR={prop['MR']}  IMR={prop['IMR']}  CR={prop['CR']}")

    votes = compute_majority_vote_stats(digra_results)
    print(f"  [x] Majority Vote Share      = {votes['majority_vote_share']:.3f}")
    print(f"  [x] Vote Entropy             = {votes['vote_entropy']:.3f}")

    # "Entropy" standalone (the checklist lists it separately from IG/IGR) —
    # per-round mean of every agent's OWN entropy (AgentRoundRecord.entropy),
    # as opposed to the conditioned/candidate entropies inside IG/IGR.
    entropy_by_round: dict = {}
    for agent_hist in digra_debate["agent_records"]:
        for rec in agent_hist:
            entropy_by_round.setdefault(rec["round_idx"], []).append(rec["entropy"])
    mean_entropy_by_round = {r: sum(v) / len(v) for r, v in sorted(entropy_by_round.items())}
    print(f"  [x] Entropy per round (mean agent entropy) = {mean_entropy_by_round}")

    ig_by_round = compute_ig_per_round(digra_results)
    print(f"  [x] Information Gain Ratio (IGR) per edge per round = {ig_by_round}")
    print("      (same compute_ig_per_round() function as DIG's checklist — the values themselves "
          "ARE IGR here since variant='digra', vs. raw IG for variant='dig'; the function is "
          "variant-agnostic by design, see its docstring)")

    comm = compute_communication_stats(digra_results)
    print(f"  [x] Selected communication edges (mean) = {comm['mean_selected_edges']}")
    print(f"  [x] # Agents = {digra_debate['n_agents']}   # Rounds = {digra_debate['n_rounds_run']}   "
          f"# Candidate edges = {comm['candidate_edges']}")

    flow = compute_information_flow(digra_results)
    print(f"  [x] Correct->Correct flow    = {flow['correct_to_correct']}  (ratio {flow['correct_to_correct_ratio']})")
    print(f"  [x] Correct->Wrong flow      = {flow['correct_to_wrong']}  (ratio {flow['correct_to_wrong_ratio']})")
    print(f"  [x] Wrong->Correct flow      = {flow['wrong_to_correct']}  (ratio {flow['wrong_to_correct_ratio']})")
    print(f"  [x] Wrong->Wrong flow        = {flow['wrong_to_wrong']}  (ratio {flow['wrong_to_wrong_ratio']})")
    print(f"  [x] Information-flow ratios  = printed above (4 lines)")

    tokens = compute_token_stats(digra_results)
    print(f"  [x] Total Tokens             = {tokens['total_tokens']}")

    # Token savings needs a baseline (Standard MAD). We don't have real
    # MAD output in this repo yet (owned by the teammate) — using dig's
    # run from the SAME scenario here only to prove the function works
    # end-to-end; this is NOT a real savings number, and the print below
    # says so explicitly rather than implying it is.
    savings = compute_token_savings(baseline_results=[dig_debate], comparison_results=digra_results)
    print(
        f"  [~] Token savings            = {savings['savings_fraction']}  "
        "(computed against this demo's DIG run as a placeholder baseline, purely to verify "
        "compute_token_savings() runs correctly — NOT a real number; once Standard MAD's real "
        "output file exists, pass it as baseline_results instead)"
    )

    early_round = digra_debate["n_rounds_run"] if digra_debate["early_stopped"] else None
    print(f"  [x] Early stopping round     = {early_round}  (None means it ran the full {digra_debate['n_rounds_requested']} requested rounds)")

    print("\n" + "=" * 78)
    print("DIG vs DIGRA diff on this identical scenario")
    print("=" * 78)
    for i in range(3):
        dig_partner = sorted(dig_debate["agent_records"][i][1]["partners_selected"])
        digra_partner = sorted(digra_debate["agent_records"][i][1]["partners_selected"])
        marker = "DIFFERS" if dig_partner != digra_partner else "same"
        print(f"  agent{i}: DIG picked {dig_partner}, DIGRA picked {digra_partner}  [{marker}]")

    print("\nAll DIGRA checklist metrics computed successfully. See results/dig_verification/ for output.")


if __name__ == "__main__":
    main()
