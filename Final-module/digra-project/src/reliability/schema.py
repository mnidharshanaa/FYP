"""
The full logging schema for the final Propagation Control Module,
matching the frozen spec's four data levels (Section 30) plus run
metadata (Section 31) and the agent-level reliability record the spec's
follow-up message added explicitly. Every field here exists because the
spec named it — this module doesn't add or drop fields on its own
judgment, since the entire point (per the spec) is reconstructing WHY a
decision was made without ever rerunning the model.

Nothing in this module calls an LLM or touches a dataset — these are
plain dataclasses or ratifying every field the pipeline (once approved
and built, pending the two open questions in the conversation) will
populate. asdict() from these feeds append_jsonl exactly the way every
other result type in this project already does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Level 1 — raw response (Section 15). Standard MAD/DIGRA already log most
# of this in their own DebateResult/DigraDebateResult; this dataclass exists
# so the propagation module's own resampling (the N extra completions per
# agent/round used for entailment clustering — NOT the debate's single
# per-round response) has an equally complete record, since those extra
# samples are new data this module introduces and nothing else logs them.
# ---------------------------------------------------------------------------
@dataclass
class RawResponseRecord:
    question_id: str
    dataset: str
    model: str
    agent_id: int
    round_idx: int
    sample_idx: int              # which of the N resamples this is (0-indexed)
    response: str
    extracted_answer: str
    ground_truth: str
    is_correct: bool
    prompt: str
    seed: int
    temperature: float
    top_p: float
    max_tokens: int
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    latency_seconds: Optional[float] = None
    token_logprobs: Optional[list] = None


# ---------------------------------------------------------------------------
# Agent-level reliability (Base 2-style clustering; R is per-agent x
# per-round, per the confirmed decision — NOT per-edge).
# ---------------------------------------------------------------------------
@dataclass
class AgentReliabilityRecord:
    question_id: str
    round_idx: int
    agent_id: int
    n: int                                   # number of resampled responses used
    theta: float                             # threshold used to form clusters AT LOG TIME
    forward_entailment_scores: list          # raw per-pair scores (0-1), so theta is
    reverse_entailment_scores: list          # sweepable post-hoc without rerunning
    num_clusters: int
    largest_cluster_size: int
    rho: float                               # largest_cluster_size / n
    r: float                                 # the reliability score itself
    representative_response: str
    representative_answer: str
    representative_correct: bool
    entailment_model: str                    # e.g. "same_llm:llama" or a dedicated NLI model id
    parser_status: str = "ok"                # "ok" | "failed_fallback" — whether the clustering
                                              # call's JSON parsed cleanly (see entailment.py's
                                              # EntailmentMatrix). A real model's first run is
                                              # exactly where you need this visible per-record,
                                              # not just recoverable by re-deriving it.


# ---------------------------------------------------------------------------
# Level 2 — communication / edge-level (Section 17, 22).
# ---------------------------------------------------------------------------
@dataclass
class CommunicationRecord:
    question_id: str
    round_idx: int
    source_agent: int
    target_agent: int

    candidate_edge: bool                     # was this a possible partner this round
    selected_edge: bool                      # was it actually chosen for communication

    ig: Optional[float] = None
    entropy: Optional[float] = None
    igr: Optional[float] = None
    u: Optional[float] = None                # normalized utility

    r_source: Optional[float] = None         # = AgentReliabilityRecord.r for source_agent this round

    lam: Optional[float] = None
    ps: Optional[float] = None
    propagation_threshold: Optional[float] = None   # tau_p used for this decision
    utility_threshold: Optional[float] = None        # tau_u used for this decision
    decision: Optional[str] = None           # FINAL decision (post-RAG, if RAG ran)
    pre_rag_decision: Optional[str] = None   # decision BEFORE any RAG update — the only reliable
                                              # way to recover "did this edge trigger verify at all",
                                              # since `decision` above is overwritten with the
                                              # post-RAG outcome (propagate/suppress) once RAG runs
    quadrant: Optional[str] = None           # Section 10's four-quadrant label

    origin_correct: Optional[bool] = None    # source's correctness this round
    result_correct: Optional[bool] = None    # target's correctness AFTER this round
    information_flow: Optional[str] = None   # "C->C" | "C->W" | "W->C" | "W->W"


# ---------------------------------------------------------------------------
# Level 4 — RAG verification, only populated when decision == "verify"
# (Section 12-13, 30 Level 4).
# ---------------------------------------------------------------------------
@dataclass
class RagVerificationRecord:
    question_id: str
    round_idx: int
    source_agent: int
    target_agent: Optional[int] = None   # None: RAG is deduplicated per (source, round) — a
                                          # source's claim is verified once regardless of how
                                          # many receivers independently trigger "verify" on it

    retrieval_query: str = ""
    document_ids: list = field(default_factory=list)
    document_ranks: list = field(default_factory=list)
    retrieval_scores: list = field(default_factory=list)
    evidence_support_score: Optional[float] = None
    evidence_status: Optional[str] = None    # "ok" | "no_evidence_available" | "verification_parse_failed"
                                              # — distinguishes WHY evidence_support_score is None,
                                              # since those are three different, auditable situations

    r_before: Optional[float] = None
    r_after: Optional[float] = None
    ps_before: Optional[float] = None
    ps_after: Optional[float] = None
    decision_before: Optional[str] = None
    decision_after: Optional[str] = None

    rag_tokens: Optional[int] = None
    rag_latency_seconds: Optional[float] = None

    retriever_name: Optional[str] = None     # which EvidenceRetriever implementation was used


# ---------------------------------------------------------------------------
# Run metadata (Section 31). One record per experiment invocation.
# DIGRA's own alpha (IGR's internal balance param) and this module's lam
# are DELIBERATELY separate fields — the spec is explicit that conflating
# them is a real risk, not a hypothetical one.
# ---------------------------------------------------------------------------
@dataclass
class RunMetadata:
    run_id: str
    timestamp: str

    model: str
    dataset: str
    method: str

    n_agents: int
    n_rounds: int

    temperature: float
    top_p: float
    seed: int
    max_tokens: int

    digra_alpha: float           # DIGRA's own IGR balance parameter (paper default 0.2)
    lam: float                   # THIS module's U/R combination weight — never the same field as digra_alpha

    propagation_threshold: float  # tau_p
    utility_threshold: float      # tau_u

    n_resample: int               # N for entailment clustering
    theta: float                  # entailment clustering threshold used at log time
    entailment_model: str

    rag_model: Optional[str] = None
    retriever: Optional[str] = None
    retrieval_top_k: Optional[int] = None

    prompt_version: str = "v1"
    code_version: Optional[str] = None
    git_commit: Optional[str] = None
