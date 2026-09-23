# LLM integration — stage 1

Branch: `feature/llm-integration`, based on `integration/final` at `d2beb2f`.
This branch prepares optional LLM-assisted exploration and reviewed training
examples. **No live API call or model training has been performed.** The API key
will be supplied locally in stage 2. The official default remains offline.

## Local key configuration

A local `.env` has been created with a blank key. Open that file and fill in:

```dotenv
OPENAI_API_KEY=your-key-here
OPENAI_MODEL=gpt-4.1-mini-2025-04-14
```

Do not put the real key in `.env.example`, a prompt, source code, Git, or chat.
`.env` is Git-ignored; `.env.example` is a tracked blank template. In another
checkout, copy `.env.example` to `.env`. The explicitly invoked experiment CLI
loads this file. An existing shell environment value takes precedence.
`Agent()` does not load `.env`, and finding a key never enables LLM calls.

Install the optional dependencies in your virtualenv:

```bash
python -m pip install -r requirements-dev.txt -r requirements-llm.txt
python -m tools.llm_eval --check-config
```

The check prints only whether a key is configured and the selected model, never
the key. It makes zero API requests. In the current workspace, the verified
environment is `/tmp/beeline-final-venv`; activate it with
`source /tmp/beeline-final-venv/bin/activate`. The older default Anaconda pandas
installation is incompatible with its NumPy, so use the virtualenv.

## Modes and architecture

```text
public profile/history/catalog
    -> candidate_engine.generate_candidates
    -> optional LLM advice over aggregate candidates
    -> pilot_policy.run_adaptive_pilots
    -> portfolio_optimizer.build_campaigns
    -> final campaigns and last_trace
```

| Mode | API behavior | Effect on campaigns |
|---|---|---|
| `off` | No SDK import or API call | Existing deterministic policy |
| `shadow` | At most one request per act | Record recommendation; preserve existing policy |
| `assist` | At most one request per act | Prioritize at most two eligible pilot candidates |

`Agent(config=None, *, advisor=None)` accepts an injectable adviser for offline
testing. Set `config={"llm_mode": "shadow"}` or `"assist"` explicitly to enable
it programmatically. `OPENAI_API_KEY` must then be in the process environment.
An optional `llm_model` overrides `OPENAI_MODEL`; otherwise a fixed model
snapshot is used. The `.env` loader belongs only to `tools.llm_eval`.

`llm_advisor.py` builds a bounded input containing at most 40 candidate summaries,
relevant tariff prices/packages, channel economics, and residual resources. It
never sends customer IDs, individual rows, raw history, hidden effects, or keys.
The input does not include evaluator seed or score. The model proposes existing
candidate IDs and a short explanation. Its response cannot change historical
scores, pilot evidence, campaign filters, the optimizer objective, or limits.

The adapter uses the Responses API with a strict JSON schema, `store=False`,
800 output tokens by default, a 10-second SDK timeout and zero automatic
retries. Local validation rejects unknown IDs, duplicates, same-cell proposals,
extra fields, oversized explanations and excessive recommendations. Configuration
caps are 30 seconds, 2000 output tokens and two recommendations. These bound
individual requests, not a guaranteed dollar spend; token usage is recorded.

The explicit endpoint is OpenAI's API. NVIDIA support is not added in this stage.
A compatible fine-tuned OpenAI model can later replace `OPENAI_MODEL` without
changing the agent/optimizer interface; account access must be tested separately.

Missing keys/SDK, provider errors, refusals, incomplete responses and invalid
advice leave the existing policy in place. Provider exception messages are not
written to traces because they can contain request details. A concise error code
records the fallback. The live CLI exits nonzero on failed advice, even if the
agent completed through the fallback, so a failed API test is not called success.

Pilot ordering is the only decision seam: `exploration_priority_ids` names up to
two hypotheses. Real sample sizes, all spending, normalized estimates and final
campaign selection remain numerical and evidence-based. A suggestion can still
be skipped if the public pilot/resource constraints do not permit execution.

An additional bounded robustness fix handles readable historical files with
missing required columns: history becomes unavailable and catalog exploration
continues, with a diagnostic event. Invalid required environment schemas are not
automatically guessed or repaired.

## Offline preview — safe to run now

```bash
python -m tools.llm_eval --dry-run --seed 42 \
  --output-dir reports/llm_runs/preview-01
```

This uses an authored empty-advice fixture, **even if a real key is present**.
It runs the actual numerical pipeline and official evaluator, writes aggregate
context/trace and review templates, and makes zero model requests. It proves
wiring and fallback behavior, not LLM quality.

Outputs go to an ignored experiment directory. Choose a fresh output directory
for each run; existing experiments and reviewed examples are not overwritten.
Official `submission.csv`, benchmark and decision-trace artifacts remain separate.

## Stage 2 — after inserting the key

First verify configuration, then make one real shadow request:

```bash
python -m tools.llm_eval --check-config
python -m tools.llm_eval --mode shadow --runs 1 --seed 42 \
  --output-dir reports/llm_runs/live-shadow-01
```

Inspect `evaluation.json` and `trace_42.json`: adviser status, selected IDs,
reported token usage and any fallback reason. Once the response is valid, an
explicit assisted comparison is available:

```bash
python -m tools.llm_eval --mode assist --runs 10 --seed 0 \
  --output-dir reports/llm_runs/live-assist-01
```

Each seed is evaluated against a fresh `Agent()` baseline on the same seed.
There are at most ten API requests for this command. Do not interpret a good
single result as evidence of improvement; compare median, minimum, failures and
cost/latency across scenarios. Repeated calls may select different candidates.
No API call, dataset upload, fine-tuning job, or credit expenditure is automatic.

## Path to a trainable agent

The learned component would select exploration hypotheses. Pilot estimation and
constraint enforcement remain deterministic. This is a supervised-training
foundation, not a claim that the agent already updates model weights.

1. Collect `experiences.jsonl` from multiple audience/catalog/history scenarios.
   It contains the aggregate input, proposals and observed pilot/evaluator
   outcomes for expert review. Outcomes are not proof of causal benefit from a
   particular recommendation; overlapping pilots and campaigns complicate credit.
2. Copy `reviews.pending.jsonl` to `reviews.jsonl`. A reviewer supplies the best
   defensible candidate IDs, a reason and reviewer name, then sets `approved` to
   `true`. All generated templates start unapproved. The reviewer may correct a
   model suggestion or choose an empty list.
3. Export only approved choices:

```bash
python -m tools.export_training_data \
  --experiences reports/llm_runs/scenario-a/experiences.jsonl reports/llm_runs/scenario-b/experiences.jsonl \
  --reviews reports/llm_runs/scenario-a/reviews.jsonl reports/llm_runs/scenario-b/reviews.jsonl \
  --output-dir artifacts/training/version-01
```

The exporter creates `train.jsonl`, `validation.jsonl`, and `manifest.json`.
Examples have system/user/assistant messages matching the live prompt and JSON
answer format. Context hashes detect mismatches; conflicting approved labels
are rejected. Exact contexts are deduplicated, and identical candidate/catalog/
channel contexts with different seeds or budgets stay in the same split.
At least two independent groups and ten unique **training** examples after the
validation split are required. A ten-seed run on one fixed input does not meet
this requirement and must not be inflated into ten independent examples.

The grouping is based on supplied aggregate context; separately curate truly
held-out populations and counterexamples before claiming generalization.
A successful export is only format/label preparation, not proof of training
quality. No file is uploaded, and no fine-tuning job is started by this tool.
A later deliberate step can train a supported model, compare it with the base
model and the no-LLM policy, and configure the resulting `ft:...` model ID.

The current model supports structured outputs and fine-tuning according to
[its official model page](https://developers.openai.com/api/docs/models/gpt-4.1-mini).
The export format follows [supervised fine-tuning documentation](https://developers.openai.com/api/docs/guides/supervised-fine-tuning).
[Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
constrain response shape; Python still validates meaning and feasibility.

## Release boundaries

`python make_submission.py` still uses `Agent()` with LLM mode `off`, preserving
exact submission reproduction. Live assisted decisions are experimental and are
not automatically made the judging path. Tests use authored responses and an
in-memory mock HTTP transport; they never consume API credits. Integration/final
remains the prior verified release on its own branch.

## Stage-1 verification

- 125 offline tests passed, including an actual SDK call through a mock HTTP
  transport (no external request).
- Supplied-dataset shadow preview: zero API requests; net gain 3,206,241.49,
  identical to the numerical baseline.
- Default official ten-seed output is byte-identical to the integration release:
  10/10 positive, median 3,103,097, minimum 1,567,731 (rounded evaluator output).
- Two default submission generations preserve SHA-256
  `3cba2865d369ad86c93e110476a4436c42f25c3583cbb9265382844e8e848e5e`.
- Organizer checksums are unchanged. No real training dataset has been exported
  or uploaded, and no model has been trained.

Evidence: `reports/llm_preparation_tests.txt`,
`reports/llm_preparation_10_seeds.txt`,
`reports/llm_preparation_verification.json`.
