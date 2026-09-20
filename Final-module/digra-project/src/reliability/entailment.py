"""
Base-2-style bidirectional entailment clustering — Decision A, frozen in
conversation. Two LLM calls per (agent, round), both using the SAME
generative debate model (no separate NLI model, per the frozen decision):

  1. Resample N independent completions from the exact prompt the agent
     used to produce its official round-r response (one `generate(n=N)`
     call — the identical "one call, N samples" pattern already used in
     src/baselines/cot_sc.py for CoT-SC).
  2. ONE additional call asks the same model to rate, for every ordered
     pair (i, j) of those N responses, how confident it is that i and j
     assert the SAME answer to the question — one NxN matrix of
     continuous 0-1 confidence scores. Forward(i,j) and reverse(i,j) are
     just the matrix read at (i,j) and (j,i); no separate reverse call.

The raw matrix is stored PERMANENTLY, unreduced (see EntailmentMatrix).
Clusters, rho, the representative response, and R are all DERIVED from
that stored matrix at a chosen theta (`build_clusters`,
`compute_agent_reliability`) and can be recomputed at any other theta
with zero further LLM calls — this is the entire reason cluster-
assignment-only output was rejected during design.

Naming: every score here is "llm_..._confidence", never "entailment_
probability" — per the frozen decision, these are model judgments under
one specific prompt, not calibrated probabilities. R=0.92 means "highly
semantically consistent under this clustering procedure," not "92%
likely true."
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional

from src.llm.client import LLMClient
from src.metrics.grading import extract_final_answer, is_correct
from src.reliability.schema import AgentReliabilityRecord

PARSER_VERSION = "v1"

CLUSTERING_PROMPT_TEMPLATE = """You will see {n} candidate answers to the same question, numbered 0 to {n_minus_1}.

Question: {question}

Candidate answers:
{numbered_responses}

For EVERY ordered pair (i, j) including i == j, rate your confidence from 0.0 to 1.0 that candidate i's claim and candidate j's claim assert the SAME answer to the question (paraphrases, different units, or different levels of detail for the same underlying answer all count as the same; a genuinely different answer does not, even if related).

A confidence of 1.0 means you are certain they assert the same answer. A confidence of 0.0 means you are certain they do not. Note that this judgment is NOT symmetric in general: your confidence for (i, j) may differ from (j, i).

Respond with ONLY a JSON object of this exact form, with no other text before or after it:
{{"matrix": [[<{n}-element row for response 0>], [<{n}-element row for response 1>], ...]}}

The matrix must be {n} rows by {n} columns. matrix[i][j] is your confidence for pair (i, j). Every value must be a number between 0.0 and 1.0."""


@dataclass
class EntailmentMatrix:
    n: int
    forward_matrix: list                  # NxN list of list of float, forward_matrix[i][j] in [0,1]
    raw_llm_output: str
    parser_status: str                    # "ok" | "failed_fallback"
    parser_version: str = PARSER_VERSION
    clustering_prompt: str = ""

    def bidirectional(self, i: int, j: int) -> float:
        """B(i,j) = min(F(i,j), F(j,i)) — thresholding B>=theta is exactly
        equivalent to requiring BOTH F(i,j)>=theta AND F(j,i)>=theta,
        matching Base 2's own "Entail(a,b)>theta AND Entail(b,a)>theta"
        clustering condition."""
        return min(self.forward_matrix[i][j], self.forward_matrix[j][i])


def _build_clustering_prompt(question: str, responses: list) -> str:
    numbered = "\n".join(f"{i}: {r}" for i, r in enumerate(responses))
    n = len(responses)
    return CLUSTERING_PROMPT_TEMPLATE.format(
        n=n, n_minus_1=n - 1, question=question, numbered_responses=numbered,
    )


def _fallback_matrix(n: int) -> list:
    """Conservative failure mode: identity only — every response entails
    only itself. This yields n singleton clusters (rho=1/n, the WORST
    possible reliability score), never a false consensus, when the
    parser can't trust the LLM's output. A parsing failure must never
    silently look like genuine agreement."""
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _parse_clustering_output(raw_text: str, n: int) -> tuple:
    """
    Returns (matrix, status) where status is "ok" or "failed_fallback".
    Tolerant of leading/trailing chatter around the JSON object (models
    reliably wrap output in commentary despite instructions not to);
    strict about the actual matrix shape and value range once found,
    since a malformed matrix is exactly the case this fallback exists for.
    """
    try:
        start = raw_text.index("{")
        end = raw_text.rindex("}") + 1
        parsed = json.loads(raw_text[start:end])
        matrix = parsed["matrix"]

        if len(matrix) != n:
            raise ValueError(f"expected {n} rows, got {len(matrix)}")
        for row in matrix:
            if len(row) != n:
                raise ValueError(f"expected {n} columns, got {len(row)}")

        clipped = [
            [max(0.0, min(1.0, float(v))) for v in row]
            for row in matrix
        ]
        return clipped, "ok"
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return _fallback_matrix(n), "failed_fallback"


def run_entailment_clustering(
    llm: LLMClient,
    question_id: str,
    round_idx: int,
    agent_id: int,
    prompt: str,
    question: str,
    n: int,
    seed: int = 0,
    max_tokens: int = 300,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    clustering_max_tokens: int = 2000,
) -> tuple:
    """
    Runs both LLM calls for one (agent, round): resample N responses from
    `prompt` (the agent's actual round-r context), then one clustering
    call over those N responses. Returns (responses: list[str],
    matrix: EntailmentMatrix, resample_latency_seconds: float,
    clustering_latency_seconds: float) — deliberately does NOT compute
    the AgentReliabilityRecord itself (that needs a theta and a gold
    answer; see `compute_agent_reliability`), so this function's output
    — the raw matrix — is exactly the thing the frozen design says must
    be stored permanently and never re-derived.
    """
    start = time.perf_counter()
    resample_results = llm.generate(
        prompt, n=n, seed=seed, max_tokens=max_tokens,
        temperature=temperature, top_p=top_p, top_k=top_k,
    )
    resample_latency = time.perf_counter() - start
    if len(resample_results) != n:
        raise RuntimeError(
            f"Expected {n} resamples for entailment clustering "
            f"(question_id={question_id!r}, agent_id={agent_id}, round={round_idx}), "
            f"got {len(resample_results)}."
        )
    responses = [r.text for r in resample_results]

    clustering_prompt = _build_clustering_prompt(question, responses)
    start = time.perf_counter()
    clustering_result = llm.generate(
        clustering_prompt, n=1, seed=seed, max_tokens=clustering_max_tokens,
        temperature=0.0, top_p=1.0, top_k=1,
    )[0]
    clustering_latency = time.perf_counter() - start

    matrix_values, status = _parse_clustering_output(clustering_result.text, n)
    matrix = EntailmentMatrix(
        n=n, forward_matrix=matrix_values, raw_llm_output=clustering_result.text,
        parser_status=status, clustering_prompt=clustering_prompt,
    )
    return responses, matrix, resample_latency, clustering_latency


def build_clusters(matrix: EntailmentMatrix, theta: float) -> list:
    """
    Connected components over the graph where an edge (i,j) exists iff
    matrix.bidirectional(i,j) >= theta — matching Base 2's Algorithm 1
    (transitive equivalence classes via graph connectivity). We have the
    FULL pairwise matrix from one call, so full graph clustering is
    cheap; the paper's own O(N log K) representative-comparison shortcut
    existed only to avoid making O(N^2) separate LLM calls, which this
    design already avoids by getting the whole matrix in one call.

    Returns a list of clusters, each a sorted list of response indices.
    """
    n = matrix.n
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if matrix.bidirectional(i, j) >= theta:
                union(i, j)

    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [sorted(members) for members in groups.values()]


def select_representative(cluster: list, matrix: EntailmentMatrix) -> int:
    """
    The cluster member with the highest average bidirectional score to
    the OTHER members of its own cluster (i.e. most "central"/consensus
    member) — a documented, principled choice the frozen spec doesn't
    pin down further. Ties broken by lowest index for determinism.
    """
    if len(cluster) == 1:
        return cluster[0]

    best_idx, best_avg = cluster[0], -1.0
    for i in cluster:
        others = [j for j in cluster if j != i]
        avg = sum(matrix.bidirectional(i, j) for j in others) / len(others)
        if avg > best_avg:
            best_idx, best_avg = i, avg
    return best_idx


def compute_agent_reliability(
    question_id: str,
    round_idx: int,
    agent_id: int,
    responses: list,
    matrix: EntailmentMatrix,
    theta: float,
    gold_answer: str,
    gold_answer_alternatives: Optional[list] = None,
    entailment_model: str = "same_llm",
) -> AgentReliabilityRecord:
    """
    Derives everything (clusters, rho, representative, R) from an
    already-computed EntailmentMatrix at the given theta — zero further
    LLM calls. Call this once per theta you want to evaluate; the
    expensive part (the matrix itself) is shared across every call.

    R is defined here as rho (largest_cluster_size / n) — the most
    direct reading of Base 2's own "largest-cluster-ratio -> accuracy"
    relationship (Section 8 of that paper). This is a documented design
    choice, not dictated verbatim by either base paper; a different
    weighting of the matrix could be layered on top of the same stored
    raw scores later without re-deriving anything.
    """
    gold_answer_alternatives = gold_answer_alternatives or []
    clusters = build_clusters(matrix, theta)
    largest = max(clusters, key=len)
    rho = len(largest) / matrix.n

    rep_idx = select_representative(largest, matrix)
    rep_response = responses[rep_idx]
    rep_answer = extract_final_answer(rep_response)
    rep_correct = is_correct(rep_answer, gold_answer, gold_answer_alternatives)

    forward_scores = [matrix.forward_matrix[i][j] for i in range(matrix.n) for j in range(matrix.n) if i != j]
    reverse_scores = [matrix.forward_matrix[j][i] for i in range(matrix.n) for j in range(matrix.n) if i != j]

    return AgentReliabilityRecord(
        question_id=question_id, round_idx=round_idx, agent_id=agent_id,
        n=matrix.n, theta=theta,
        forward_entailment_scores=forward_scores, reverse_entailment_scores=reverse_scores,
        num_clusters=len(clusters), largest_cluster_size=len(largest), rho=rho, r=rho,
        representative_response=rep_response, representative_answer=rep_answer,
        representative_correct=rep_correct, entailment_model=entailment_model,
        parser_status=matrix.parser_status,
    )
