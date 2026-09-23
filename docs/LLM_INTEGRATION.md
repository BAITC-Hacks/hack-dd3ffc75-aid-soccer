# LLM integration: verified release

Branch `feature/llm-integration`, based on `integration/final` (`d2beb2f`).
The integration has been tested against the real OpenAI API. It can nominate
up to two exploration hypotheses; measured pilot effects and the numerical
portfolio optimizer remain authoritative. No model weights have been trained.

## Run the submitted agent

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
python local_eval.py --runs 10
python make_submission.py
```

`Agent()` automatically replays the bundled `artifacts/llm_policy.json`, containing
the **first successful live recommendation**, when its public input context and
prompt match exactly. This requires no API key, SDK, or network. The snapshot
contains no evaluator seed, hidden effects, customer IDs, or evaluator-selected
winners. The input hash covers candidate aggregates, relevant catalog, public
channel economics, available resources, and prompt version; the instruction hash
also prevents reuse after a prompt change.

Changed inputs, a damaged policy, or invalid recommendations fall back to the
numerical policy. A changed dataset therefore never receives a stale nomination.
If the policy file is absent, the default is `off`. Replay never calls the API,
even when a key is configured. Ship the artifact and all custom Python modules.

## Use the live model on a new dataset

Install the optional SDK and place the key only in local, ignored `.env`:

```bash
python -m pip install -r requirements-llm.txt
# In a fresh checkout: cp .env.example .env, then fill OPENAI_API_KEY locally.
python -m tools.llm_eval --check-config
python -m tools.llm_eval --mode assist --runs 10 --seed 0 \
  --output-dir reports/llm_runs/new-dataset-01
```

```dotenv
OPENAI_API_KEY=your-key-here
OPENAI_MODEL=gpt-4.1-mini-2025-04-14
```

The explicit CLI loads `.env`; `Agent()` does not. An existing shell environment
value takes precedence. Never put a key into tracked source, `.env.example`,
traces, or chat. The verified workspace virtualenv is `/tmp/beeline-final-venv`;
the default Anaconda installation has an unrelated pandas/NumPy ABI problem.

The command compares live decisions with explicit `llm_mode=off` on identical
seeds. Each act makes at most one API request. It saves traces, actual token
usage, comparisons, pending review templates, and the first valid `policy.json`
in a fresh ignored directory. Existing experiment directories are preserved.
Choose a new directory for each experiment. Inspect failures and the distribution
of results, then install the first successful live trace for the new context:

```bash
python -m tools.freeze_llm_policy \
  --trace reports/llm_runs/new-dataset-01/trace_0.json --replace
python local_eval.py --runs 10
python make_submission.py
```

Use the first successful trace if seed 0's request failed. Do not choose a model
response because it scored best on the mock evaluator. Freezing validates the
recommendation and records prompt/model/context provenance; it makes no API call.
Regenerate and recheck submission whenever the policy or data changes.

For a zero-network comparison of a saved policy:

```bash
python -m tools.llm_eval --mode assist --policy artifacts/llm_policy.json \
  --runs 10 --seed 0 --output-dir reports/llm_runs/replay-01
```

## Decision boundary and failure behavior

```text
public profile/history/catalog -> candidate engine
    -> bounded LLM nomination / exact-context replay
    -> feasible SMS exploration -> adaptive confirmation
    -> pilot uncertainty estimates -> constrained channel/portfolio choice
```

| Mode | Network | Decision effect |
|---|---|---|
| `off` | None | Numerical baseline |
| `shadow` | Up to one request | Record recommendation without applying it |
| `assist` | Up to one request, or injected offline adviser | Nominate up to two exploration candidates |
| `replay` | None | Apply validated, exact-context frozen nominations |

Programmatic usage: `Agent({"llm_mode": "assist"})`; set `OPENAI_API_KEY` in the
process environment. `llm_model` overrides `OPENAI_MODEL`. `llm_policy_path`
selects a policy in explicit `replay` mode. `Agent({"llm_mode": "off"})` disables
both live advice and the bundled policy.

The model receives at most 40 aggregate candidate records, relevant tariff
prices/packages, channel costs, and residual resources. No individual customer
rows, raw histories, IDs, hidden effects, seeds, or evaluation scores are sent.
It can select only existing eligible candidate IDs in distinct cells. Historical
before/after lift is explicitly described as noncausal, not conversion evidence.

If a nomination is already in the shortlist, its original order is preserved.
New nominations replace at most two other positions while retaining the order
of surviving candidates. This avoids reassigning pilot randomness merely because
the model agrees with existing choices. `llm_exploration.selection_changed`
records whether membership actually changed. Synthetic integration tests verify
that a novel nomination can change the executed pilot and final campaign.

The Responses API uses strict structured output, `store=False`, an 800-token
output cap, 10-second SDK timeout, and zero automatic retries. Local validation
also rejects unknown/duplicate IDs, same-cell proposals, extra fields, oversized
explanations, and excessive nominations. Timeouts, API failures, refusals,
incomplete output, missing keys/SDK, and invalid advice trigger fallback. Traces
contain fixed error codes, never provider exception bodies or credentials.
A failed live adviser makes the evaluation CLI exit nonzero, even if fallback
produced a valid portfolio. NVIDIA is not used by this implementation.

## Optional adaptive sample sizing

```bash
python -m tools.llm_eval --mode assist --policy artifacts/llm_policy.json \
  --adaptive-sizing --runs 10 --seed 0 \
  --output-dir reports/llm_runs/adaptive-sizing-01
```

Or set `adaptive_confirmation_n=True` in `Agent` config. This plans the next
confirmation batch using measured uncertainty and distance to zero-profit,
channel-switch, and competing-target boundaries. Clear decisions stop receiving
confirmations; ambiguous decisions receive bounded batches. Actual returned
sample counts still determine estimates; resource limits always take precedence.
This is a heuristic under adaptive selection, not a calibrated coverage claim.

In the measured ten-seed experiment, confirmation sizes included 40, 43, 46, 54,
68, 69, 78, 116, 166, and 200. All runs remained positive, but the minimum and
median declined, so this feature is **available but disabled by default**. The
release retains the already adaptive confirmation selection and early stopping.

## Actual results

See [release evidence](../reports/LLM_RELEASE.md) and
[machine-readable verification](../reports/llm_final_verification.json).

- 147 offline tests passed. Tests use authored responses or mock HTTP transport.
- 11 real API requests in this completion stage: one initial request and ten
  final live assisted runs. All eleven returned valid recommendations.
- In all final ten live runs, the model selected hypotheses already in the
  numerical shortlist. Gains equal the baseline: 10/10 positive, median
  3,103,097.10, minimum 1,567,730.55. **No LLM profit improvement is demonstrated.**
- The first ordering implementation degraded seed 42 to 1,257,358.74; preserving
  the original order when membership is unchanged fixed that unnecessary change.
- The optional adaptive-sizing experiment: median 3,002,247.39, minimum
  1,305,074.91, 10/10 positive. It is not the submitted default.
- Default offline replay passes official evaluation and reproduces the four-row
  submission byte-for-byte. Organizer files remain unchanged.

## Trainable component and next training step

The adviser is replaceable through `OPENAI_MODEL` / `llm_model`, including a
compatible fine-tuned model ID after training. The current integration records
aggregate contexts, decisions, and outcomes in `experiences.jsonl`. It creates
`reviews.pending.jsonl` with every approval initially false.

Collect genuinely different audience/catalog/history scenarios, have a reviewer
supply defensible candidate IDs and a reason, then set `approved=true` and a
reviewer name in a separate `reviews.jsonl`. Outcomes and model suggestions are
not automatically treated as correct training labels.

```bash
python -m tools.export_training_data \
  --experiences reports/llm_runs/scenario-a/experiences.jsonl reports/llm_runs/scenario-b/experiences.jsonl \
  --reviews reports/llm_runs/scenario-a/reviews.jsonl reports/llm_runs/scenario-b/reviews.jsonl \
  --output-dir artifacts/training/version-01
```

The exporter writes message-format `train.jsonl`, `validation.jsonl`, and a
manifest. It rejects conflicting labels and invalid IDs, deduplicates contexts,
and keeps related scenarios in the same split. It requires at least two groups
and ten unique training examples after the validation split. Ten random seeds
on this one fixed dataset are **one context**, insufficient for useful training.
No upload, fine-tuning job, or weight update is automatic or has been performed.
The frozen recommendation is an inference artifact, not a trained model.

The API format follows [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
The future training format follows [supervised fine-tuning](https://developers.openai.com/api/docs/guides/supervised-fine-tuning).
