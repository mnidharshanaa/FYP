"""
Unified DIGRA debate loop — also home of the DIG ablation.

Runs one of four variants, controlled by `variant`:
  "dig"                — IG-only partner selection (Sec. III-B(v) of the base
                         paper's baselines): raw Information Gain (Eq. 7),
                         NO entropy normalization. A high-entropy (likely-
                         hallucinating) candidate isn't penalized just for
                         being uncertain — only for offering low information
                         gain. No RAG, no memory.
  "digra"              — entropy-normalized IGR (Eq. 8), exactly the base
                         paper's proposed method. This is DIG plus the
                         entropy-normalization term that penalizes
                         high-entropy partners in the denominator.
  "digra_rag"          — DIGRA + evidence-adjusted entropy (src/rag/evidence.py).
  "digra_rag_memory"   — DIGRA + RAG + persistent trust weighting (src/memory/trust.py).

All four share this exact same loop — variant only controls (a) whether
partner selection scores by raw IG or entropy-normalized IGR
(src/digra/partner_selection.py's `use_ratio`), and (b) which corrections
are active in the entropy fed into that scoring. This is deliberate: it's
what makes DIG-vs-DIGRA and DIGRA-vs-DIGRA+RAG genuine, fair ablations
(same code path, same seeds, same everything else) rather than
separately-written, potentially-inconsistent implementations that could
silently drift apart.

Round-1 seeding and prompt templates are reused from the Standard MAD
engine (src.agents.round_seeding, src.agents.prompts) — only the round-2+
communication logic differs (dynamic selection instead of fully-connected).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from src.agents.prompts import build_debate_round_prompt, build_neutral_prompt
from src.agents.round_seeding import seed_round_one_from_pool
from src.digra.information_gain import compute_ig
from src.digra.partner_selection import select_best_partners
from src.entropy.entropy import mean_token_entropy
from src.llm.client import LLMClient
from src.memory.trust import TrustTracker, compute_flag, cross_round_inconsistent
from src.metrics.grading import extract_final_answer, is_correct
from src.rag.evidence import adjust_entropy_for_evidence, check_contradiction, get_evidence
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

VALID_VARIANTS = {"dig", "digra", "digra_rag", "digra_rag_memory"}

# Variants that score partner subsets by raw IG (no entropy normalization).
# Everything not in this set uses IGR (Eq. 8) — the base paper's method.
_IG_ONLY_VARIANTS = {"dig"}


@dataclass
class AgentRoundRecord:
    round_idx: int
    response_text: str
    extracted_answer: str
    is_correct: bool
    entropy: float
    rag_flagged: Optional[bool] = None            # None if variant doesn't use RAG
    cross_round_inconsistent: Optional[bool] = None  # None for round 1, or if not using memory
    combined_flagged: Optional[bool] = None
    trust_after_round: Optional[float] = None      # None if variant doesn't use memory
    partners_selected: Optional[list] = None        # None for round 1 (seeded, no selection)
    igr_score: Optional[float] = None                # winning selection score: IGR (digra*
                                                       # variants) or raw IG (variant="dig")
                                                       # — see partner_selection.py's use_ratio
    candidate_scores: Optional[dict] = None          # ALL candidate subsets evaluated this
                                                       # selection event -> their score (IG for
                                                       # "dig", IGR otherwise), keyed by a sorted
                                                       # comma-joined agent-id string (JSON needs
                                                       # string keys; frozensets aren't valid
                                                       # JSON keys). Per-edge/per-round IG
                                                       # logging, not just the winner's average —
                                                       # None for round 1 (no selection happened).
    n_tokens: Optional[int] = None                    # generated-token count for this round's
                                                       # response: exact (len(token_logprobs))
                                                       # when logprobs were captured for this
                                                       # generation event, else a whitespace-split
                                                       # word-count approximation (documented,
                                                       # not silently passed off as exact) for
                                                       # pool-seeded round-1 text that was never
                                                       # itself generated by this run.


def _count_tokens(text: str, token_logprobs: Optional[list]) -> int:
    """Exact count when logprobs were captured this call (one dict per
    generated token); otherwise a documented whitespace-split
    approximation — see AgentRoundRecord.n_tokens docstring."""
    if token_logprobs:
        return len(token_logprobs)
    return len(text.split())


@dataclass
class DigraDebateResult:
    question_id: str
    variant: str
    setup: str
    n_agents: int
    n_rounds_requested: int
    n_rounds_run: int
    early_stopped: bool
    gold_answer: str
    gold_answer_alternatives: list = field(default_factory=list)
    agent_records: list = field(default_factory=list)   # list[list[AgentRoundRecord]]
    n_generate_calls: int = 0
    n_forced_decode_calls: int = 0
    total_tokens: int = 0    # sum of every AgentRoundRecord.n_tokens across all agents/rounds


def _majority_vote(answers: list) -> str:
    """Simple plurality vote over extracted answers, normalized for grouping."""
    from src.metrics.grading import normalize_answer
    counts: dict = {}
    for ans in answers:
        key = normalize_answer(ans)
        counts.setdefault(key, []).append(ans)
    best_key = max(counts, key=lambda k: len(counts[k]))
    return counts[best_key][0]  # return one representative original-form answer


def _compute_round1_entropy(
    llm: LLMClient,
    question: str,
    text: str,
    from_fresh_generation: bool,
    fresh_logprobs: Optional[list],
    logprobs_topk: int,
    call_counter: dict,
) -> float:
    """
    Round-1 entropy needs different handling depending on how the text
    was produced: a "standard" setup's fresh generation already carries
    logprobs; a seeded (pool-derived) text does not, and needs one
    forced_decode call against a neutral prompt to establish a baseline.
    """
    if from_fresh_generation:
        return mean_token_entropy(fresh_logprobs)

    neutral_prompt = build_neutral_prompt(question)
    result = llm.forced_decode(neutral_prompt, text, logprobs_topk=logprobs_topk)
    call_counter["forced_decode"] += 1
    return mean_token_entropy(result.token_logprobs)


def run_digra_debate(
    llm: LLMClient,
    question_id: str,
    question: str,
    gold_answer: str,
    n_agents: int,
    n_rounds: int,
    setup,
    variant: str = "digra",
    gold_answer_alternatives: Optional[list] = None,
    correct_pool: Optional[list] = None,
    incorrect_pool: Optional[list] = None,
    source_passage: Optional[str] = None,   # required for digra_rag / digra_rag_memory
    seed: int = 0,
    alpha: float = 0.2,
    max_subset_size: Optional[int] = None,
    entropy_inflation_factor: float = 2.0,
    trust_config: Optional[dict] = None,     # kwargs for TrustTracker
    max_tokens: int = 300,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    logprobs_topk: int = 5,
    unchanged_rounds_threshold: int = 2,
) -> DigraDebateResult:
    if variant not in VALID_VARIANTS:
        raise ValueError(f"variant must be one of {sorted(VALID_VARIANTS)}, got '{variant}'")

    use_rag = variant in ("digra_rag", "digra_rag_memory")
    use_memory = variant == "digra_rag_memory"
    use_ratio = variant not in _IG_ONLY_VARIANTS
    if use_rag and source_passage is None:
        raise ValueError(f"variant='{variant}' requires source_passage (evidence) to be provided")

    gold_answer_alternatives = gold_answer_alternatives or []
    call_counter = {"generate": 0, "forced_decode": 0}
    trust_tracker = TrustTracker(**(trust_config or {})) if use_memory else None

    # --- Round 1 ---
    if setup == "standard":
        prompt = build_neutral_prompt(question)
        results = llm.generate(
            prompt, n=n_agents, seed=seed, max_tokens=max_tokens,
            temperature=temperature, top_p=top_p, top_k=top_k, logprobs_topk=logprobs_topk,
        )
        call_counter["generate"] += 1
        round1_texts = [r.text for r in results]
        round1_logprobs = [r.token_logprobs for r in results]
        setup_label = "standard"
        fresh = True
    else:
        n_incorrect, n_correct = setup
        if n_incorrect + n_correct != n_agents:
            raise ValueError(f"setup {setup} sums to {n_incorrect + n_correct}, expected n_agents={n_agents}")
        rng = random.Random(seed)
        round1_texts = seed_round_one_from_pool(
            n_incorrect, n_correct, correct_pool or [], incorrect_pool or [], rng,
        )
        round1_logprobs = [None] * n_agents
        setup_label = f"{n_incorrect},{n_correct}"
        fresh = False

    agent_records = [[] for _ in range(n_agents)]
    entropy_by_agent = {}
    flagged_by_agent = {}

    for i in range(n_agents):
        text = round1_texts[i]
        entropy = _compute_round1_entropy(
            llm, question, text, fresh, round1_logprobs[i], logprobs_topk, call_counter,
        )
        answer = extract_final_answer(text)
        correct = is_correct(answer, gold_answer, gold_answer_alternatives)

        rag_flagged = None
        if use_rag:
            rag_flagged = check_contradiction(text, source_passage)
        flagged = compute_flag(rag_flagged or False, False) if (use_rag or use_memory) else None

        trust_after = None
        if use_memory:
            trust_after = trust_tracker.update(i, flagged=bool(flagged))

        entropy_by_agent[i] = entropy
        flagged_by_agent[i] = flagged or False

        agent_records[i].append(AgentRoundRecord(
            round_idx=1, response_text=text, extracted_answer=answer, is_correct=correct,
            entropy=entropy, rag_flagged=rag_flagged, cross_round_inconsistent=None,
            combined_flagged=flagged, trust_after_round=trust_after,
            partners_selected=None, igr_score=None, candidate_scores=None,
            n_tokens=_count_tokens(text, round1_logprobs[i]),
        ))

    # --- Rounds 2+ ---
    prev_texts = list(round1_texts)
    frozen = [False] * n_agents  # early-stopping: agent's answer unchanged for threshold rounds
    unchanged_streak = [0] * n_agents
    n_rounds_run = 1
    early_stopped = False

    for round_idx in range(2, n_rounds + 1):
        if all(frozen):
            early_stopped = True
            break

        # --- Selection pass: one agent at a time (forced_decode is
        # inherently per-agent-per-subset, can't be batched the way
        # generation can), but nothing here calls generate() yet — all
        # generation for this round happens in one batched call below. ---
        selected_partners = [None] * n_agents
        selected_igr = [None] * n_agents
        selected_candidate_scores = [None] * n_agents
        prompts_to_generate = []
        agents_to_generate = []

        for i in range(n_agents):
            if frozen[i]:
                continue

            candidates = [j for j in range(n_agents) if j != i]

            def ig_fn(subset, _i=i, _prev_texts=prev_texts):
                other_texts = [_prev_texts[j] for j in subset]
                prompt_with_context = build_debate_round_prompt(
                    question=question, own_previous_response="", other_responses=other_texts,
                )
                result = llm.forced_decode(prompt_with_context, _prev_texts[_i], logprobs_topk=logprobs_topk)
                call_counter["forced_decode"] += 1
                h_conditioned = mean_token_entropy(result.token_logprobs)
                return compute_ig(entropy_by_agent[_i], h_conditioned)

            effective_entropy = {}
            for j in candidates:
                h = entropy_by_agent[j]
                if use_rag:
                    h = adjust_entropy_for_evidence(h, flagged_by_agent[j], entropy_inflation_factor)
                if use_memory:
                    trust_j = trust_tracker.get(j)
                    # Fold trust in via the entropy term: lower trust ->
                    # treat as higher effective entropy, penalizing low-
                    # trust partners in IGR's denominator. Kept as a
                    # single combined adjustment (rather than a separate
                    # post-hoc IGR multiply) so partner_selection.py's
                    # interface stays unchanged across all 3 variants.
                    h = h / max(trust_j, 1e-6)
                effective_entropy[j] = h

            selection = select_best_partners(
                candidate_ids=candidates, entropy_by_agent=effective_entropy,
                ig_fn=ig_fn, alpha=alpha, max_subset_size=max_subset_size,
                use_ratio=use_ratio,
            )
            selected_partners[i] = selection.best_subset
            selected_igr[i] = selection.best_igr
            # Per-edge/per-round score logging (checklist requirement):
            # every candidate subset evaluated this event, not just the
            # winner. Keys must be JSON-safe strings (frozensets aren't
            # valid JSON object keys) — sorted, comma-joined agent ids,
            # e.g. frozenset({0, 2}) -> "0,2".
            selected_candidate_scores[i] = {
                ",".join(str(a) for a in sorted(subset)): score
                for subset, score in selection.all_scores.items()
            }

            partner_texts = [prev_texts[j] for j in selection.best_subset]
            prompt = build_debate_round_prompt(
                question=question, own_previous_response=prev_texts[i], other_responses=partner_texts,
            )
            prompts_to_generate.append(prompt)
            agents_to_generate.append(i)

        # --- Generation pass: ONE batched call for every non-frozen
        # agent's round-`round_idx` response, instead of one call per
        # agent (see module-level docstring / this section's header
        # comment for why this matters). ---
        new_texts = list(prev_texts)
        new_entropies = dict(entropy_by_agent)
        new_flags = dict(flagged_by_agent)

        if prompts_to_generate:
            batch_results = llm.generate_batch(
                prompts_to_generate, seed=seed + round_idx * 1000, max_tokens=max_tokens,
                temperature=temperature, top_p=top_p, top_k=top_k, logprobs_topk=logprobs_topk,
            )
            call_counter["generate"] += 1

            for idx, i in enumerate(agents_to_generate):
                result = batch_results[idx]
                text = result.text
                entropy = mean_token_entropy(result.token_logprobs)
                answer = extract_final_answer(text)
                correct = is_correct(answer, gold_answer, gold_answer_alternatives)

                rag_flagged = check_contradiction(text, source_passage) if use_rag else None
                inconsistent = cross_round_inconsistent(
                    text, prev_texts[i], gold_answer, gold_answer_alternatives,
                ) if (use_rag or use_memory) else None
                flagged = compute_flag(rag_flagged or False, inconsistent or False) \
                    if (use_rag or use_memory) else None

                trust_after = trust_tracker.update(i, flagged=bool(flagged)) if use_memory else None

                new_texts[i] = text
                new_entropies[i] = entropy
                new_flags[i] = flagged or False

                agent_records[i].append(AgentRoundRecord(
                    round_idx=round_idx, response_text=text, extracted_answer=answer, is_correct=correct,
                    entropy=entropy, rag_flagged=rag_flagged, cross_round_inconsistent=inconsistent,
                    combined_flagged=flagged, trust_after_round=trust_after,
                    partners_selected=sorted(selected_partners[i]), igr_score=selected_igr[i],
                    candidate_scores=selected_candidate_scores[i],
                    n_tokens=_count_tokens(text, result.token_logprobs),
                ))

                if answer == extract_final_answer(prev_texts[i]):
                    unchanged_streak[i] += 1
                else:
                    unchanged_streak[i] = 0
                if unchanged_streak[i] >= unchanged_rounds_threshold - 1:
                    frozen[i] = True

        for i in range(n_agents):
            if frozen[i] and i not in agents_to_generate:
                import dataclasses as _dc
                last = agent_records[i][-1]
                # New record for this round with the SAME content (the
                # agent's answer genuinely didn't change) but the correct
                # round_idx, and no selection info (none happened this
                # round) — NOT a copy of the old record's round_idx, which
                # was a real bug caught during manual testing: it produced
                # two records both labeled with the earlier round_idx and
                # none for the current round, corrupting any per-round
                # metric computed from agent_records later.
                carried = _dc.replace(
                    last, round_idx=round_idx, partners_selected=None, igr_score=None, candidate_scores=None,
                )
                agent_records[i].append(carried)

        prev_texts = new_texts
        entropy_by_agent = new_entropies
        flagged_by_agent = new_flags
        n_rounds_run = round_idx

        all_answers = [extract_final_answer(t) for t in prev_texts]
        if len(set(a.lower().strip() for a in all_answers)) == 1:
            early_stopped = True
            break

    total_tokens = sum(
        rec.n_tokens or 0
        for agent_hist in agent_records
        for rec in agent_hist
    )

    return DigraDebateResult(
        question_id=question_id, variant=variant, setup=setup_label,
        n_agents=n_agents, n_rounds_requested=n_rounds, n_rounds_run=n_rounds_run,
        early_stopped=early_stopped, gold_answer=gold_answer,
        gold_answer_alternatives=gold_answer_alternatives, agent_records=agent_records,
        n_generate_calls=call_counter["generate"], n_forced_decode_calls=call_counter["forced_decode"],
        total_tokens=total_tokens,
    )
