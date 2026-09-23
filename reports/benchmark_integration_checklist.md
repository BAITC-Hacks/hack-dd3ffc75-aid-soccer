# Pilot policy handoff and integration checklist

End-to-end evaluation is **pending** the candidate and portfolio owners' commits.
Neither module has been replaced or edited. Missing imports surface with an error
trace, rather than a silent fallback.

## Owned implementation

- `pilot_policy.py`: up to ten SMS exploration calls and five adaptive
  confirmations; five optional slots remain unused. Defaults reserve 1,000 final
  contacts and cap pilot spending at 3,000 contacts / 12,000 c.u.
- Actual returned sample sizes control channel-normalized means and SEs.
  Negative observations remain intact; history only ranks exploration.
- Confirmation scores combine served ARPU, SE reduction, decision ambiguity and
  contact/cost burden. Dominated options are not confirmed.
- `agent.py`: loads history relative to its file, coordinates the shared APIs,
  resets trace state, records residual resources, and checks final resource limits.
- `tools/benchmark.py`: black-box evaluation, swallowed-exception detection,
  separate final campaign counts, strict JSON/CSV/Markdown reports, and optional
  `--exploration-only` ablation.
- Tested dependencies: pandas 3.0.3, numpy 2.4.4, pytest 9.0.2.

## Independent validation

Command: `python -m pytest tests/test_pilot_policy.py tests/test_agent_integration.py -q`.
Result: **42 passed** (one non-failing Windows pytest cache-write warning).
Tests cover evidence aggregation, resource boundaries, invalid responses, reversed
history signals, catalog diversity, adaptive confirmation, deterministic replay,
dependency diagnostics, history recovery, trace reset and benchmark error detection.

A public-evaluator smoke test executes one 80-contact pilot on the supplied
dataset. This validates pilot mechanics only, not final campaigns. Existing
benchmark artifacts retain earlier missing-dependency diagnostics and starter
measurements; adaptive performance and improvement remain unavailable.

## Integration when commits arrive

1. Record both owner SHAs and inspect changed paths. Integrate their commits
   without rewriting their modules. Organizer assets/data are currently untracked
   and intentionally excluded from the pilot commit; the integration checkout must
   contain the original participant pack and shared contract.
2. Verify `generate_candidates(profile, history, tariffs, *, config=None, trace=None)`
   and `build_campaigns(env, observations, *, config=None, trace=None)`, required
   contract fields, normalized evidence, and absence of runtime evaluator imports.
3. Run `python -m pytest tests -q`; report precise failures to the relevant owner.
4. Run `python -X utf8 local_eval.py` and `python -X utf8 local_eval.py --runs 10`.
   UTF-8 avoids Windows console encoding failures in organizer diagnostics.
5. Run `python -m tools.benchmark --runs 10 --compare-starter --trace-seed 42`.
   Replace pending artifacts with completed measurements. Inspect exit status,
   per-run failures, final campaign counts and trace status. Report actual positive
   runs and absolute median improvement; do not tune to tariff IDs or mock seeds.
6. Run `python -X utf8 make_submission.py`. Use organizer `sanitize_campaigns` and
   `validate_strategy`; check 1–10 legal campaigns, nonempty audiences, valid
   targets/channels, residual resource limits, pilot caps and nonzero pilot count.
7. Regenerate seed-42 submission and compare identical CSV bytes. Never edit the
   CSV manually. Include all runtime modules and requirements in the release.
8. Only after both owners explicitly freeze their modules, read
   `prompts/04_FINAL_INTEGRATION.md` for bounded final integration fixes.

Full-agent evaluation, submission generation, organizer validation and seed-42
submission reproducibility remain pending. No release success is claimed.
