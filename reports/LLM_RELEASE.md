# LLM integration release

Branch: `feature/llm-integration`, from verified `integration/final` commit
`d2beb2f`; preparation commit `9b3f651` remains in history.

## What is complete

- Real OpenAI Responses API integration with structured output and semantic
  validation. At most two candidate nominations can affect pilot selection;
  measured effects, limits, and final channel allocation remain numerical.
- A bundled, exact-context frozen recommendation makes default `Agent()` and
  official submission generation offline and deterministic. New contexts reject
  stale advice and fall back; explicit live evaluation can refresh the policy.
- Existing nominations preserve pilot order. Novel nominations replace at most
  two shortlist entries, verified with independent authored integration tests.
- Optional evidence-dependent confirmation sizes and early stopping. The new
  sizing option is disabled by default because its measured downside worsened.
- Reviewed training-data collection/export, scenario grouping, and a replaceable
  model ID. No model training or fine-tuning upload has occurred.
- Trace/report provenance, safe fallback codes, tests for shifted resources,
  catalog, history and audiences, unseen tariff labels, damaged policies, API
  failures, malformed responses, and actual sample/resource accounting.

## Actual results

The user explicitly authorized sending aggregate tariff/segment/history metrics
to OpenAI. Eleven real requests succeeded, using `gpt-4.1-mini-2025-04-14`:
76,791 input tokens and 1,400 output tokens (78,191 total). No NVIDIA API was used.
Per-response usage, choices and all comparison rows are preserved in
[llm_live_evidence.json](llm_live_evidence.json). These are token counts, not a
claim about billed dollars or remaining credits.

| Variant, official seeds 0–9 | Mean net gain | Median | Minimum | Positive |
|---|---:|---:|---:|---:|
| Numerical baseline | 2,898,763.77 | 3,103,097.10 | 1,567,730.55 | 10/10 |
| Final live LLM assist | 2,898,763.77 | 3,103,097.10 | 1,567,730.55 | 10/10 |
| Default frozen LLM replay | 2,898,763.77 | 3,103,097.10 | 1,567,730.55 | 10/10 |
| Replay + optional adaptive sizes | 2,840,051.32 | 3,002,247.39 | 1,305,074.91 | 10/10 |
| Unchanged starter | -480,836.18 | -357,947.80 | -1,019,431.06 | 0/10 |

**The final live LLM calls did not improve profit on this dataset.** Every call
nominated `tariff_4|MID|tariff_8` and `tariff_10|HIGH|tariff_20`, both already in
the numerical shortlist. The integration allows new nominations on other data;
these results do not establish generalization or superiority over the baseline.

The initial implementation reordered existing nominations. Its first live seed
42 result was 1,257,358.74 versus baseline 3,206,241.49; its frozen ten-seed median
was 2,819,905.73. Preserving the order when membership is unchanged removed this
unnecessary perturbation. The shipped artifact still contains the **first**
successful live response; no response was selected by mock score.

Optional sizing produced confirmation batches of 40–200, including 43, 46, 54,
68, 69, 78, 116, and 166. Its decline is reported rather than promoted as an
improvement. Default confirmation selection/count remain adaptive; maximum
confirmation size remains 200.

## Verification and handoff

- 147 offline tests passed: [test log](llm_final_tests.txt).
- Unchanged `local_eval.py`, including official ten seeds: [log](llm_final_10_seeds.txt).
- Full starter comparison and public plan validation: [benchmark](benchmark.json).
- Default seed 42: 3,206,241.49 net, 15 pilots, four final campaigns, 74,256 total
  cost, 4,848 contacts. No discarded campaigns or residual resource truncation.
- Two unchanged `make_submission.py` runs produced identical bytes. Both public
  validators accepted all four campaign dictionaries. SHA-256:
  `3cba2865d369ad86c93e110476a4436c42f25c3583cbb9265382844e8e848e5e`.
- All 15 organizer source/data checksums match the original integrity manifest.
- Clean export verification is recorded in [llm_final_verification.json](llm_final_verification.json).

The standalone [demo report](demo.html) includes model provenance, nominations,
whether exploration changed, pilot uncertainty, channel choices, and the final
portfolio. Run instructions and refreshing advice on new data are in
[LLM_INTEGRATION.md](../docs/LLM_INTEGRATION.md).

The runtime is ready for submission without a key. Live API use requires the
optional dependencies and local `.env`. Ship `artifacts/llm_policy.json` and every
custom module. `integration/final` remains available as the prior numerical
release. This branch has not been pushed.
