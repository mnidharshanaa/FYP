# DIGRA + RAG + Memory: Mitigating Hallucination Propagation in Multi-Agent Debate

Extends Zhang et al., *"Beware of the Woozle Effect"* (IEEE TASLP, 2026) — reproduces
their DIGRA framework, then adds an evidence-grounded correction (RAG) and a
persistent trust/memory mechanism to address the paper's own documented
"faithfulness hallucination" failure mode (Section III-C1).

## Project status

Modules are built and verified one at a time, in dependency order. A module is
only considered "done" once its unit tests pass — see `tests/`.

| # | Module | Status |
|---|--------|--------|
| 0 | Project scaffold, config, logging, checkpointing | ✅ done |
| 1 | Data layer — FARM loader (schema-verified against live repo) | ✅ done |
| 1b | Response pool builder (Appendix B-3) + LLM client abstraction | ✅ done (logic verified; VLLMClient verified on Kaggle T4x2) |
| 2 | Core debate engine — Standard MAD, fully-connected topology, batched round-2+ generation | ✅ done (orchestration fully tested; VLLMClient wiring untestable here) |
| 5a | Entropy, Information Gain, Information Gain Ratio | ✅ done, hand-verified |
| 5b | Partner selection (subset search maximizing IGR, entropy caching) | ✅ done |
| 6 | RAG evidence + contradiction module | ✅ done |
| 7 | Trust/memory module (RAG flag + cross-round consistency, both triggers) | ✅ done |
| 5c | Unified DIGRA debate loop (digra / digra_rag / digra_rag_memory, one shared code path) | ✅ done, 216 tests passing |
| 8a | DIGRA orchestration + runnable script (--model/--variant/--calibrate) | ✅ done, 224 tests passing |
| 3 | Propagation metrics (MA/MR/IMR/CR) + RAG precision/recall + cost instrumentation | ✅ done (pre-existing, src/metrics/propagation_metrics.py) |
| 1 (Module 1 / "4" baselines) | CoT / CoT-SC baseline — fresh sampling (NOT pool-seeded), self-consistency vote at every N in one generation pass, full checklist metrics (Accuracy/EM/F1/TruthfulQA-metric/Majority-Vote-Share/Vote-Entropy/#Unique-Answers/Wrong-Consensus-Rate/Invalid-Answer-Rate/Avg+Total-Tokens/Avg-Latency) | ✅ done, 279 tests passing (`src/baselines/cot_sc.py`, `src/baselines/orchestration.py`, `src/metrics/text_metrics.py`, `src/metrics/baseline_metrics.py`, `scripts/05_run_baselines.py`, `scripts/06_aggregate_baselines.py`) |
| 4 (remainder) | MAD variants for Table II (mad_sparse_half, mad_random) — Standard MAD fully-connected already covered by Module 2 | ⬜ next |
| 9 | Figure generation | partially done — see `scripts/04_generate_report.py` (DIGRA/Standard-MAD) and `scripts/06_aggregate_baselines.py` (CoT-SC accuracy-vs-N, paper's Fig. 5 equivalent) |

## Repository layout

```
configs/            YAML configs. base.yaml is the single source of truth for
                     every dataset path, model id, seed, and hyperparameter.
                     Ablations are override YAMLs merged on top (see
                     src/utils/config.py), never hardcoded in code.
src/
  data/              FARM dataset loading, correct/incorrect response pools
  agents/            Agent / Debate classes, generation wrapper
  entropy/           Mean token entropy, forced decoding
  digra/             IG, IGR, partner selection, early stopping, DIG ablation
  rag/                Retriever, contradiction detector, evidence-adjusted IGR
  memory/            Persistent trust score
  baselines/         CoT, CoT-SC (done — cot_sc.py + orchestration.py),
                     MAD standard (see agents/debate.py)/sparse/random (pending)
  metrics/           MA/MR/IMR/CR + timing/token cost instrumentation
  utils/             config, seeding, logging, checkpoint/resume
scripts/             Thin, numbered entry points (01_build_pools.py, etc.)
                     called from Kaggle notebooks — no logic lives here,
                     only orchestration of src/ calls.
notebooks/           Kaggle notebooks; import scripts/src, do not contain logic
tests/               One test file per src/ module; run before any GPU time
                     is spent using that module
results/
  raw/               Per-run CSV rows (one row per dataset/model/method/seed/round)
  aggregated/         Mean±std tables, paper-figure-equivalent data
  figures/           Generated plots, one script per paper-figure-equivalent
```

## Conventions

- **No hardcoded constants in `src/`.** Everything configurable lives in
  `configs/base.yaml`. If you find yourself typing a dataset path, seed, or
  hyperparameter directly into a `.py` file, it belongs in the config instead.
- **Every module ships with tests before it's used in a real experiment.**
  Entropy/IGR/metrics math especially — these are cheap to unit-test and
  expensive to debug after burning GPU hours on a wrong formula.
- **Logging, not print.** `from src.utils.logging_config import get_logger`.
- **Reproducibility.** The same `project.seeds` list (configs/base.yaml) is
  reused identically across every method/dataset/model combination, and
  DIGRA/RAG/Memory variants always start from the same seeded round-1 state
  as Standard MAD (mirrors Appendix B-1 of the DIGRA paper) — this is what
  makes the comparison fair, so never let a method get its own independent
  round-1 sampling.

## Kaggle session safety (read this before starting a long run)

A real Kaggle session was lost mid-pool-building because `/kaggle/working`
was never persisted anywhere outside the live kernel session, and Kaggle's
"Save Version" failed while the GPU cell was still actively running (a
known, common Kaggle platform issue — don't try to Save Version mid-run).
Everything computed in that session (~1 hour of NQ pool-building) was gone
on restart, with no way to recover it.

**Two-part mitigation, both required:**

1. **Clone into a genuinely empty directory.** A doubled path like
   `digra-project/digra-project/` (from running `git clone <url>` a second
   time inside an already-cloned folder) silently causes `results/` to end
   up somewhere other than where you expect, making backups/checks look
   like they're finding nothing even when data exists. Always verify with
   `find /kaggle/working -name results -type d` before assuming a path.

2. **Back up automatically, from inside the script — not a separate cell.**
   Both `scripts/01_build_pools.py` and `scripts/02_run_debates.py` now
   call `src/utils/backup.py`'s `PeriodicBackup` after every completed
   question (`configs/base.yaml`'s `backup.every_n_questions`), from
   *within* the same process — this sidesteps the single-kernel problem
   entirely, since there's no second cell to coordinate with.

   **Read this carefully — it's the part that caused the original data
   loss:** `backup.local_dest` (a copy to another folder under
   `/kaggle/working`) is **NOT** protection against session/kernel loss —
   it's the same ephemeral filesystem, and dies with the session just the
   same. It only guards against a different failure (e.g. `results/`
   getting corrupted mid-write). **Real protection requires the data to
   leave the session**, via one of:
   - Manually downloading the local-backup folder through Kaggle's file
     browser periodically (still requires you to remember to do it, but
     at least the copy itself is automatic and always up to date).
   - Setting `backup.kaggle_dataset_slug` to push to a Kaggle Dataset via
     the `kaggle` CLI, which persists independently of any kernel. Needs
     `pip install kaggle`, a Kaggle API token (`KAGGLE_USERNAME`/
     `KAGGLE_KEY` env vars, set from a Kaggle Secret the same way
     `HF_TOKEN` is), and the dataset created once via
     `kaggle datasets create -p <dir>` before this can push *versions* to
     it. Not independently verified in this dev environment (no Kaggle
     API access here) — confirm the first push manually before trusting
     it silently in a long unattended run.

`n_questions` was also cut from 150 to 30 per dataset after this incident,
specifically to bound how much is ever at risk in a single uninterrupted
run, and to preserve GPU budget for the actual DIGRA/RAG/Memory comparison
(Modules 5-7) rather than spending most of it on baseline pool-building.



1. Push this repo to GitHub (private is fine), or upload it as a Kaggle
   Dataset (Kaggle notebooks can attach both a code repo and a dataset).
   Note: `python scripts/00_fetch_farm.py` clones the FARM dataset repo
   directly (needs internet access, which Kaggle notebooks have by default
   unless you're in a no-internet competition environment) — no need to
   bundle the raw data files yourself.
2. **First real GPU run, in order:**
   a. `python scripts/00_fetch_farm.py` — stages the dataset, verifies line counts.
   b. `python scripts/smoke_test_vllm.py --model <hf_id> --tensor-parallel-size <N>` —
      verifies VLLMClient behaves as the LLMClient contract requires. Adjust
      `--dtype`/`--max-model-len`/`--gpu-memory-utilization`/`--tensor-parallel-size`
      to your GPU (T4 needs `--dtype float16`, native bfloat16 isn't supported).
      Read the `[4/4]` output manually before trusting anything built on top of it.
   c. `python scripts/01_build_pools.py --model llama` then
      `python scripts/01_build_pools.py --model mistral` — **one model per
      invocation, not both in one run.** Confirmed on real Kaggle hardware
      (dual T4, tensor_parallel_size=2): loading a second vLLM model into
      the same process after finishing the first crashes with a GPU OOM
      error (`Free memory on device cuda:1 (2.39/14.56 GiB)...`), because
      vLLM's tensor-parallel worker subprocesses don't reliably release
      GPU memory just from Python variable reassignment. Both scripts'
      `--model` flag exists specifically for this — see their module
      docstrings.
   d. Same for `python scripts/02_run_debates.py --model llama` /
      `--model mistral` — same underlying failure mode applies.
   d2. **Module 1 baseline (CoT / CoT-SC)** — does NOT require
      scripts/01_build_pools.py first (it samples fresh, never touches
      the pools — see `src/baselines/cot_sc.py`'s module docstring for
      why reusing pool-seeded "correct_texts" would bias this baseline):
      ```
      python scripts/05_run_baselines.py --model llama
      python scripts/05_run_baselines.py --model mistral
      ```
      Then, on your laptop (CPU only, no GPU/Kaggle needed) once you've
      downloaded `results/baselines/`:
      ```
      python scripts/06_aggregate_baselines.py --baselines-dir results/baselines --out-dir results
      ```
      writes `results/aggregated/baselines_summary.csv` (the full
      Accuracy/EM/F1/TruthfulQA-metric/Majority-Vote-Share/Vote-Entropy/
      #Unique-Answers/Wrong-Consensus-Rate/Invalid-Answer-Rate/
      Avg+Total-Tokens/Avg-Latency checklist, one row per dataset/model/N)
      and `results/figures/cotsc_accuracy_vs_n_<dataset>_<model>.png`
      (the paper's Fig. 5-style accuracy-vs-sampling-count plot) — both
      derived from ONE generation pass per question at
      `cot_sc.n_samples_max` (configs/base.yaml), sliced down to every N
      in `cot_sc.sc_ks`, so the whole N=1..10 sweep costs exactly as much
      GPU time as the N=10 run alone.
   e. **Never invoke with `python -m scripts.01_build_pools`** — beyond
      `scripts/` deliberately having no `__init__.py`, `01_build_pools`
      starts with a digit, which isn't a valid Python module identifier.
      Always use the direct path: `python scripts/01_build_pools.py`.

3. **Scale strategy.** `configs/base.yaml` intentionally runs a reduced
   first pass (`seeds: [0]`, `agent_counts: [3]`, but the full
   `n_questions` per dataset) — enough to validate the entire pipeline
   through Module 3's metrics without committing your full GPU quota
   up front. Once that's confirmed working, restore the full sweep with:
   ```
   python scripts/01_build_pools.py --overrides configs/full_scale.yaml
   python scripts/02_run_debates.py --overrides configs/full_scale.yaml
   ```
   This is always safe to run at any time — checkpoint resume means it
   only computes newly-added seed/agent-count combinations, never redoes
   or invalidates anything already completed.

4. **Performance note.** `run_debate`'s round 2+ loop batches all agents'
   prompts into a single `generate_batch()` call per round (see
   `src/llm/client.py`'s docstring), instead of one call per agent. This
   matters a lot: for a 5-agent debate, that's 5x fewer individual vLLM
   calls per round, and running at batch size >1 uses the GPU far more
   effectively than a Python loop calling `generate()` one agent at a
   time ever could. If you're resuming a run that started before this fix
   (i.e. any `results/` data generated by an earlier commit), it's still
   valid and compatible — only the *speed* changed, not the output format
   — no need to discard or redo anything already completed.

5. **Running DIGRA experiments** (`scripts/03_run_digra_experiments.py`)
   — this is the actual comparison: `digra` / `digra_rag` /
   `digra_rag_memory`, one shared code path (`src/digra/digra_debate.py`),
   run separately per variant so results are directly comparable.
   **Always calibrate first**, on a small scope, before committing GPU
   time to a full run:
   ```
   python scripts/03_run_digra_experiments.py --model llama --variant digra --calibrate
   ```
   This times a handful of real debates (default 3, `--calibrate-n` to
   change) and prints a projected total for your configured scope — the
   calibration debates are NOT wasted, they count toward the resumable
   registry. Once you have a real number and it fits your budget, run
   each variant for real:
   ```
   python scripts/03_run_digra_experiments.py --model llama --variant digra
   python scripts/03_run_digra_experiments.py --model llama --variant digra_rag
   python scripts/03_run_digra_experiments.py --model llama --variant digra_rag_memory
   ```
   Same `--model`-per-invocation rule as the other scripts (see script
   docstring for the real OOM failure this avoids). Requires
   `scripts/01_build_pools.py` to have completed for the model in use —
   the RAG variants also need the FARM `source` column, which
   `load_dataset()` already provides with no extra step.

2. In the Kaggle notebook: `!pip install -r requirements.txt`, then
   `import sys; sys.path.append("/kaggle/working/digra-project")`.
3. Point `project.output_root` (configs/base.yaml) at a path under
   `/kaggle/working/` for the session, and periodically save
   `results/` as a Kaggle Dataset version so it survives across sessions —
   the `RunRegistry` (src/utils/checkpoint.py) will pick up exactly where
   it left off on the next session as long as `results/checkpoints/` was
   restored from that saved dataset version.
4. Run `python -m pytest` at the start of every session before spending any
   GPU time, as a sanity check that nothing broke on re-attach.

## Running tests locally

```bash
pip install -r requirements.txt
python -m pytest -v
```
