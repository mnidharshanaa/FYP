"""
The Propagation Control debate loop — the final module (Section 14 of the
frozen spec), integrating everything built so far:

  - DIGRA's existing entropy/IG/IGR + partner-selection machinery decides
    WHO is a candidate communication partner (src.digra.partner_selection,
    reused UNCHANGED — same alpha, same forced-decode calls). Its
    `all_scores` dict is a free byproduct that already contains the
    per-candidate IGR needed for U, for every candidate, not just the
    selected subset.
  - src.reliability.entailment computes R per (agent, round): one
    resample + one clustering call, reused across every edge FROM that
    agent this round.
  - src.reliability.propagation_score combines U and R into PS and the
    propagate/verify/suppress decision.
  - src.rag.retriever + verification run conditional RAG for VERIFY edges,
    deduplicated per (source_agent, round) — a source's claim is a single
    fixed thing regardless of which receiver is asking, so it is only
    verified once per round even if multiple receivers independently
    reach "verify" for the same source (see `_get_or_run_rag` below).

Only edges whose FINAL decision (after any RAG update) is "propagate" are
included when building the receiving agent's next-round prompt — a
suppressed or unverifiable-and-insufficiently-reliable edge's text is
excluded entirely, not merely down-weighted. If every candidate edge to
an agent is excluded, that agent reconsiders based on its own prior
response alone (src.agents.prompts.build_debate_round_prompt already
handles an empty other_responses list).

Reuses src.digra.digra_debate.AgentRoundRecord for the per-agent-per-round
log so src.metrics.propagation_metrics's MA/MR/IMR/CR computation works
UNCHANGED on this module's results, exactly as it already does for
Standard MAD, DIG, and DIGRA — one shared aggregation code path across
every method in this project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.agents.prompts import build_debate_round_prompt, build_neutral_prompt
from src.agents.round_seeding import seed_round_one_from_pool
from src.digra.digra_debate import AgentRoundRecord
from src.digra.information_gain import compute_ig
from src.digra.partner_selection import select_best_partners
from src.entropy.entropy import mean_token_entropy
from src.llm.client import LLMClient
from src.metrics.grading import extract_final_answer, is_correct
from src.rag.verification import compute_evidence_support
from src.reliability.entailment import compute_agent_reliability, run_entailment_clustering
from src.reliability.information_flow import classify_information_flow
from src.reliability.propagation_score import normalize_utility, score_and_decide
from src.reliability.schema import AgentReliabilityRecord, CommunicationRecord, RagVerificationRecord
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

RAG_UNAVAILABLE_DEFAULT_DECISION = "suppress"  # documented, conservative choice — see module docstring


@dataclass
class PropagationDebateResult:
    question_id: str
    setup: str
    n_agents: int
    n_rounds_requested: int
    n_rounds_run: int
    early_stopped: bool
    gold_answer: str
    gold_answer_alternatives: list = field(default_factory=list)
    agent_records: list = field(default_factory=list)              # list[list[AgentRoundRecord]] — same
                                                                     # shape as DigraDebateResult, so
                                                                     # propagation_metrics.py works unchanged
    agent_reliability_records: list = field(default_factory=list)   # list[AgentReliabilityRecord]
    communication_records: list = field(default_factory=list)       # list[CommunicationRecord]
    rag_verification_records: list = field(default_factory=list)    # list[RagVerificationRecord]
    n_generate_calls: int = 0
    n_forced_decode_calls: int = 0
    n_rag_calls: int = 0
    total_tokens_partial: int = 0
    # ^ "partial" is not a hedge-word for show: this sums token_logprobs
    # length from round-1 generation, each round's own per-agent response,
    # and entropy forced-decode calls — the paths that already request
    # logprobs_topk. Entailment resampling/clustering calls (N extra
    # generations + 1 clustering call per agent per round) and RAG
    # verification calls do NOT currently request logprobs and are NOT
    # counted here. Given N resamples per agent per round, this is a
    # material undercount of true token cost, not a rounding error —
    # treat this field as a lower bound, not a total, until that gap is
    # closed (tracked as a known limitation, not silently hidden).


def _get_or_run_rag(
    llm: LLMClient,
    source_agent: int,
    round_idx: int,
    claim_text: str,
    question_id: str,
    retriever,
    top_k_evidence: int,
    r_before: float,
    ps_before: float,
    u_for_recompute: float,
    lam: float,
    tau_p: float,
    tau_u: float,
    cache: dict,
    call_counter: dict,
    rag_records: list,
) -> tuple:
    """
    Runs (or returns the cached result of) evidence verification for ONE
    (source_agent, round_idx) — deduplicated across every receiving edge
    that independently triggers "verify" for the same source this round,
    since the claim being verified is identical regardless of who's
    asking (see module docstring).

    Returns (r_after: float, decision_after: str) for THIS specific edge
    — r_after is shared across every edge from this source, but
    decision_after is recomputed per-edge from that shared r_after and
    the EDGE's OWN u_for_recompute, since U genuinely varies by receiver.

    If no evidence is available (no retriever, or the question isn't in
    the snapshot, or retrieval returns zero documents), r_after ==
    r_before (unchanged — never fabricated) and decision_after defaults
    to RAG_UNAVAILABLE_DEFAULT_DECISION, logged with status
    "no_evidence_available" so this is auditable and distinguishable from
    a genuinely completed verification.
    """
    cache_key = (source_agent, round_idx)
    if cache_key not in cache:
        evidence_status = "ok"
        documents = []
        if retriever is not None:
            try:
                if retriever.has(question_id):
                    documents = retriever.retrieve(question_id, top_k=top_k_evidence)
                else:
                    evidence_status = "no_evidence_available"
            except KeyError:
                evidence_status = "no_evidence_available"

        if not documents:
            evidence_status = "no_evidence_available"
            r_after = r_before
            support = None
            raw_output, prompt_used = "", ""
        else:
            result = compute_evidence_support(llm, claim_text, documents)
            call_counter["rag"] += 1
            support = result["score"]
            raw_output, prompt_used = result["raw_llm_output"], result["prompt"]
            if support is None:
                evidence_status = "verification_parse_failed"
                r_after = r_before
            else:
                # R moves toward the evidence-support score — a documented,
                # simple update rule: strong support pulls R up toward 1.0,
                # strong contradiction pulls it down toward 0.0, weak/no
                # evidence leaves it close to r_before.
                r_after = max(0.0, min(1.0, (r_before + support) / 2.0))

        cache[cache_key] = {
            "r_after": r_after, "evidence_status": evidence_status,
            "documents": documents, "support": support,
            "raw_output": raw_output, "prompt": prompt_used,
        }

        # Log the verification EVENT itself exactly once per (source,
        # round) — the actual retrieval/LLM call this represents only
        # happened once, right above, regardless of how many receiving
        # edges end up asking about this source this round. The
        # PER-EDGE decision outcome (which genuinely can differ by
        # receiver, since U varies by receiver even though R does not)
        # is NOT lost — it's recorded where it belongs, once per edge,
        # in CommunicationRecord, which the caller updates independently
        # for every edge regardless of whether this was a cache hit.
        # This record uses the triggering edge's own u/ps_before as a
        # representative snapshot of the event, not a claim that every
        # edge saw identical values.
        decision_after_for_log = (
            RAG_UNAVAILABLE_DEFAULT_DECISION if evidence_status != "ok"
            else score_and_decide(u_for_recompute, r_after, lam, tau_p, tau_u).decision
        )

        rag_records.append(RagVerificationRecord(
            question_id=question_id, round_idx=round_idx, source_agent=source_agent, target_agent=None,
            retrieval_query=claim_text, document_ids=[d.document_id for d in documents],
            document_ranks=[d.rank for d in documents],
            retrieval_scores=[d.retrieval_score for d in documents],
            evidence_support_score=support, evidence_status=evidence_status,
            r_before=r_before, r_after=r_after, ps_before=ps_before,
            ps_after=(ps_before if evidence_status != "ok"
                      else score_and_decide(u_for_recompute, r_after, lam, tau_p, tau_u).ps),
            decision_before="verify", decision_after=decision_after_for_log,
            rag_tokens=None, rag_latency_seconds=None,
            retriever_name=type(retriever).__name__ if retriever is not None else None,
        ))

    cached = cache[cache_key]
    r_after = cached["r_after"]

    if cached["evidence_status"] != "ok":
        decision_after = RAG_UNAVAILABLE_DEFAULT_DECISION
    else:
        decision_after = score_and_decide(u_for_recompute, r_after, lam, tau_p, tau_u).decision

    return r_after, decision_after


def run_propagation_debate(
    llm: LLMClient,
    question_id: str,
    question: str,
    gold_answer: str,
    n_agents: int,
    n_rounds: int,
    setup,
    lam: float,
    tau_p: float,
    tau_u: float,
    n_resample: int,
    theta: float,
    gold_answer_alternatives: Optional[list] = None,
    correct_pool: Optional[list] = None,
    incorrect_pool: Optional[list] = None,
    retriever=None,
    top_k_evidence: int = 3,
    seed: int = 0,
    alpha: float = 0.2,
    max_subset_size: Optional[int] = None,
    max_tokens: int = 300,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    logprobs_topk: int = 5,
    entailment_model_name: str = "same_llm",
) -> PropagationDebateResult:
    gold_answer_alternatives = gold_answer_alternatives or []
    call_counter = {"generate": 0, "forced_decode": 0, "rag": 0, "tokens": 0}

    # --- Round 1 (identical to DIGRA's round 1 — see src.digra.digra_debate) ---
    if setup == "standard":
        prompt = build_neutral_prompt(question)
        results = llm.generate(
            prompt, n=n_agents, seed=seed, max_tokens=max_tokens,
            temperature=temperature, top_p=top_p, top_k=top_k, logprobs_topk=logprobs_topk,
        )
        call_counter["generate"] += 1
        round1_texts = [r.text for r in results]
        round1_prompts = [prompt] * n_agents
        setup_label = "standard"
    else:
        n_incorrect, n_correct = setup
        if n_incorrect + n_correct != n_agents:
            raise ValueError(f"setup {setup} sums to {n_incorrect + n_correct}, expected n_agents={n_agents}")
        import random
        rng = random.Random(seed)
        round1_texts = seed_round_one_from_pool(n_incorrect, n_correct, correct_pool or [], incorrect_pool or [], rng)
        round1_prompts = [build_neutral_prompt(question)] * n_agents
        setup_label = f"{n_incorrect},{n_correct}"

    agent_records = [[] for _ in range(n_agents)]
    round1_fresh = (setup == "standard")
    for i in range(n_agents):
        text = round1_texts[i]
        answer = extract_final_answer(text)
        correct = is_correct(answer, gold_answer, gold_answer_alternatives)
        if round1_fresh:
            entropy = mean_token_entropy(results[i].token_logprobs)
            call_counter["tokens"] += len(results[i].token_logprobs or [])
        else:
            # Seeded (pool-derived) text carries no logprobs from a real
            # generation call — one forced-decode against the neutral
            # prompt establishes its entropy baseline, identical to
            # DIGRA's own round-1 handling (src.digra.digra_debate).
            fd_result = llm.forced_decode(round1_prompts[i], text, logprobs_topk=logprobs_topk)
            call_counter["forced_decode"] += 1
            call_counter["tokens"] += len(fd_result.token_logprobs or [])
            entropy = mean_token_entropy(fd_result.token_logprobs)
        agent_records[i].append(AgentRoundRecord(
            round_idx=1, response_text=text, extracted_answer=answer, is_correct=correct,
            entropy=entropy,
        ))

    prev_texts = list(round1_texts)
    prev_prompts = round1_prompts
    all_reliability_records: list = []
    all_communication_records: list = []
    all_rag_records: list = []
    n_rounds_run = 1
    early_stopped = False

    # --- Rounds 2+ : one communication/decision step per iteration,
    # producing round_idx's responses from prev_texts / prev_prompts. ---
    for round_idx in range(2, n_rounds + 1):
        # Step 1: R_{j, round_idx-1} for every agent j, from the prompt
        # that actually produced prev_texts[j] — one resample+cluster
        # call PER AGENT this round, shared across every edge from that agent.
        reliability_by_agent: dict = {}
        for j in range(n_agents):
            responses, matrix, _, _ = run_entailment_clustering(
                llm, question_id, round_idx - 1, j, prev_prompts[j], question, n_resample,
                seed=seed + round_idx * 1000 + j, max_tokens=max_tokens,
                temperature=temperature, top_p=top_p, top_k=top_k,
            )
            call_counter["generate"] += 2  # resample call + clustering call
            rec = compute_agent_reliability(
                question_id, round_idx - 1, j, responses, matrix, theta,
                gold_answer, gold_answer_alternatives, entailment_model=entailment_model_name,
            )
            reliability_by_agent[j] = rec
            all_reliability_records.append(rec)

        # Each agent's OWN unconditioned entropy for the round whose text
        # is prev_texts[j]. This is ALREADY computed and stored on that
        # round's AgentRoundRecord (from the generate() call that produced
        # it, in round 1's branch above or round r-1's generate() call
        # below) — reused directly here, matching this project's own
        # stated caching principle (src.digra.partner_selection's module
        # docstring): "H(R_i,t) comes directly from the logprobs already
        # returned by that agent's own generation call — no forced-decode
        # needed for it at all." An earlier draft of this function
        # recomputed it via a fresh forced_decode call every round, which
        # was a real, caught bug — doubling forced-decode cost for data
        # already sitting in agent_records.
        entropy_by_agent = {j: agent_records[j][-1].entropy for j in range(n_agents)}

        rag_cache: dict = {}
        new_texts = list(prev_texts)
        new_prompts = list(prev_prompts)

        for i in range(n_agents):
            candidates = [j for j in range(n_agents) if j != i]

            def ig_fn(subset, _i=i, _prev_texts=prev_texts):
                other_texts = [_prev_texts[j] for j in subset]
                prompt_with_context = build_debate_round_prompt(
                    question=question, own_previous_response="", other_responses=other_texts,
                )
                result = llm.forced_decode(prompt_with_context, _prev_texts[_i], logprobs_topk=logprobs_topk)
                call_counter["forced_decode"] += 1
                call_counter["tokens"] += len(result.token_logprobs or [])
                h_conditioned = mean_token_entropy(result.token_logprobs)
                return compute_ig(entropy_by_agent[_i], h_conditioned)

            selection = select_best_partners(
                candidate_ids=candidates, entropy_by_agent=entropy_by_agent,
                ig_fn=ig_fn, alpha=alpha, max_subset_size=max_subset_size,
            )
            igr_values = {j: selection.all_scores[frozenset({j})] for j in candidates}
            igr_min, igr_max = min(igr_values.values()), max(igr_values.values())

            allowed_partner_texts = []
            for j in candidates:
                u = normalize_utility(igr_values[j], igr_min, igr_max)
                r_source = reliability_by_agent[j].r
                decision_record = score_and_decide(u, r_source, lam, tau_p, tau_u)
                decision = decision_record.decision
                ps_final = decision_record.ps

                if decision == "verify":
                    r_after, decision_after = _get_or_run_rag(
                        llm, source_agent=j, round_idx=round_idx - 1,
                        claim_text=prev_texts[j], question_id=question_id, retriever=retriever,
                        top_k_evidence=top_k_evidence, r_before=r_source, ps_before=decision_record.ps,
                        u_for_recompute=u, lam=lam, tau_p=tau_p, tau_u=tau_u,
                        cache=rag_cache, call_counter=call_counter, rag_records=all_rag_records,
                    )
                    decision = decision_after
                    ps_final = score_and_decide(u, r_after, lam, tau_p, tau_u).ps

                origin_correct = agent_records[j][-1].is_correct
                communication_allowed = decision == "propagate"
                if communication_allowed:
                    allowed_partner_texts.append(prev_texts[j])

                all_communication_records.append(CommunicationRecord(
                    question_id=question_id, round_idx=round_idx - 1, source_agent=j, target_agent=i,
                    candidate_edge=True, selected_edge=j in selection.best_subset,
                    ig=None, entropy=entropy_by_agent[j], igr=igr_values[j], u=u,
                    r_source=r_source, lam=lam, ps=ps_final,
                    propagation_threshold=tau_p, utility_threshold=tau_u,
                    decision=decision, pre_rag_decision=decision_record.decision, quadrant=decision_record.quadrant,
                    origin_correct=origin_correct, result_correct=None, information_flow=None,
                ))

            prompt = build_debate_round_prompt(
                question=question, own_previous_response=prev_texts[i], other_responses=allowed_partner_texts,
            )
            result = llm.generate(
                prompt, n=1, seed=seed + round_idx * 1000 + i, max_tokens=max_tokens,
                temperature=temperature, top_p=top_p, top_k=top_k, logprobs_topk=logprobs_topk,
            )[0]
            call_counter["generate"] += 1

            text = result.text
            answer = extract_final_answer(text)
            correct = is_correct(answer, gold_answer, gold_answer_alternatives)
            agent_records[i].append(AgentRoundRecord(
                round_idx=round_idx, response_text=text, extracted_answer=answer,
                is_correct=correct, entropy=mean_token_entropy(result.token_logprobs),
            ))
            call_counter["tokens"] += len(result.token_logprobs or [])
            new_texts[i] = text
            new_prompts[i] = prompt

        # Backfill result_correct/information_flow now that round_idx's
        # outcomes are known for every agent.
        for rec in all_communication_records:
            if rec.round_idx == round_idx - 1 and rec.result_correct is None:
                rec.result_correct = agent_records[rec.target_agent][-1].is_correct
                rec.information_flow = classify_information_flow(rec.origin_correct, rec.result_correct)

        prev_texts = new_texts
        prev_prompts = new_prompts
        n_rounds_run = round_idx

        all_answers = [extract_final_answer(t) for t in prev_texts]
        if len(set(a.lower().strip() for a in all_answers)) == 1:
            early_stopped = True
            break

    return PropagationDebateResult(
        question_id=question_id, setup=setup_label, n_agents=n_agents,
        n_rounds_requested=n_rounds, n_rounds_run=n_rounds_run, early_stopped=early_stopped,
        gold_answer=gold_answer, gold_answer_alternatives=gold_answer_alternatives,
        agent_records=agent_records, agent_reliability_records=all_reliability_records,
        communication_records=all_communication_records, rag_verification_records=all_rag_records,
        n_generate_calls=call_counter["generate"], n_forced_decode_calls=call_counter["forced_decode"],
        n_rag_calls=call_counter["rag"], total_tokens_partial=call_counter["tokens"],
    )



