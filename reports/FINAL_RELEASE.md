# Final integration release

Branch: `integration/final`, based on the available published pilot-policy tip
`c40ed21`. Instructions: `PROMTS/04_FINAL_INTEGRATION.md`; frozen shared contract
unchanged. Runtime decisions were not retuned during integration.

## Included teammate history

| Source | Original commits included | Integration history |
|---|---|---|
| Pilot policy | `7f64d97`, `c40ed21` | Both inherited from branch base |
| Shared participant kit / initial prior prototype | `b0e5cf9` | `6ce4a80` |
| Contract prior engine | `d0a8d0b` | `6856700` |
| Portfolio optimizer | `c404377` | `9024414` |
| Portfolio report and documentation | `883a99f` | `1d2b316` |

All required commits were included, including the portfolio implementation
preceding its final documentation commit. Cherry-picks retain original SHAs in
commit messages. The ignore-file conflict was resolved by combining both sets
of rules. No force-push, reset, or replacement of teammate decision logic.

Remote fetch failed because HTTPS GitHub credentials were unavailable. These
exact commits were already present locally on the matching `origin/feature/*`
refs; no newer remote state is claimed. The integration release is committed
locally, not published.

## Integration changes

- Moved `portfolio_optimizer.py`, `reporting.py`, and their tests to the root
  layout used by `agent.py`, `candidate_engine.py`, and `pilot_policy.py`.
- Included 15 unchanged public organizer/data assets at the root, so official
  commands and the agent's relative history path work in a clean checkout.
  Existing organizer files and CSVs were not edited. Byte checks and SHA-256
  hashes are in `organizer_integrity.json`.
- Added `PROMTS/*.md` to the tracked release so instructions survive cloning.
- Extended the offline benchmark to validate real final filters, public
  sanitizer/validator acceptance, nonempty/non-self audiences, separate final
  counts, distinct cells, residual resources, runtime, and scorer truncation.
  The starter remains unchanged, including overlapping cells or an empty plan.
- Attached seed-42 evaluator measurement to the saved trace's optional
  `benchmark` metadata, matching the existing report renderer.
- Added integration regressions for the actual three-module pipeline, partial
  returned pilot samples, negative evidence and trace reset, invalid/truncated
  plans, unchanged starter behavior, and measured-result rendering.
- Updated README/demo/handoff documents and generated actual benchmark, trace,
  HTML, official evaluator logs, submission and verification artifacts.

## Actual checks and results

Environment: Python 3.12.3, pandas 3.0.3, NumPy 2.4.4, pytest 9.0.2, installed
from the existing pinned requirements in an isolated temporary virtualenv.

| Command/check | Actual result |
|---|---|
| `python -m pytest tests -q` | 93 passed |
| `python local_eval.py` | Seed 42: net 3,206,241.49; 15 pilots, 4 final campaigns |
| `python local_eval.py --runs 10` | Median 3,103,097.10; minimum 1,567,730.55; 10/10 positive |
| `python -m tools.benchmark --runs 10 --compare-starter --trace-seed 42` | 10 paired wins; no evaluation exceptions/discards |
| `python make_submission.py`, twice | Identical bytes, four valid campaigns |
| Public `sanitize_campaigns` / `validate_strategy` | All four final campaigns accepted |
| `python -m reporting --input reports/decision_trace.json --output reports/demo.html` | Actual trace rendered; measured net gain displayed |
| Clean staged-file export without `task/` | 93 tests passed; generated submission matches byte-for-byte |

Adaptive ten-seed mean: **2,898,763.77**; median: **3,103,097.10**; minimum:
**1,567,730.55**. Starter mean: **−480,836.18**; median: **−357,947.80**;
minimum: **−1,019,431.06**; **0/10** positive. Difference of medians:
**3,461,044.89**. Starter seed 3 returns no final campaigns; its actual pilot-only
score **−140,280.93** is retained, and its final-count noncompliance is flagged
separately in benchmark JSON/CSV. This is not counted as an evaluator exception.

Across adaptive seeds 0–9: 15 pilots and 3–6 final campaigns each; maximum total
spend **96,542/100,000**, maximum total contacts **8,201/15,000**; no campaign
exceeds the effective 5000-contact cap, and no budget/contact truncation is used.
Maximum recorded full evaluation runtime **0.554 seconds**, below 600 seconds.

Seed 42 submission: pilot contacts/cost **1,800 / 7,200**, final contacts/cost
**3,048 / 67,056**, totals **4,848 / 74,256**; remaining contacts/budget
**10,152 / 25,744**. All final audiences are disjoint; no claim is made about
pilot/final overlap, whose IDs are not exposed to the planner.

Submission SHA-256:
`3cba2865d369ad86c93e110476a4436c42f25c3583cbb9265382844e8e848e5e`.
The clean-export check also reproduced this hash using only staged assets.

Evidence: `tests.txt`, `clean_checkout_tests.txt`, `clean_checkout_verification.json`,
`local_eval_seed42.txt`, `local_eval_10_seeds.txt`, `benchmark.json`,
`benchmark.csv`, `benchmark_run.txt`, `submission_verification.json`,
`organizer_integrity.json`, `decision_trace.json`, `demo.html`.

## Remaining limitations

These are synthetic mock results, not hidden-evaluation or real-business
performance. History ranks hypotheses and does not establish causal uplift or
conversion. Risk bands after adaptive selection are heuristic. Beam search is
bounded, not a global-optimality guarantee. Pilot overlap is unknown to the
planner; the official evaluator computes actual deduplication. Free push can
still reduce revenue. Publication remains a separate team action.
