# Student 3 — Portfolio Optimizer, Analyst Report and README

Paste everything below into your coding agent. The repository must contain prompts/SHARED_CONTRACT.md. Use Astra 6 / high as selected by the team.

---

Implement the portfolio optimizer and a clear analyst-facing explanation for Adaptive Campaign Portfolio Agent. We have three hours. A tested, integrable optimizer must be ready in 45 minutes; documentation and a lightweight demo follow. Start implementing now, use reasonable reversible assumptions, and keep progress independent of teammates.

## Read and ownership

Read prompts/SHARED_CONTRACT.md, PARTICIPANT_GUIDE.md and scoring_core.py. The scoring file is public mechanics. Do not inspect mock effects to select tariffs and do not access hidden env internals.

Only create/edit:
- portfolio_optimizer.py
- tests/test_portfolio_optimizer.py
- reporting.py
- tests/test_reporting.py
- README.md
- docs/DEMO.md
- docs/ARCHITECTURE.md
- reports/demo.html

Work in feature/portfolio-optimizer in your own clone. Do not edit agent.py, candidate_engine.py, pilot_policy.py, dependency files, shared contract, datasets or organizer code. Do not create shared fixtures, global config modules or extra coding agents.

Use pandas, numpy and standard library. pytest is for tests. Reporter uses standard-library HTML generation and inline CSS/SVG; no web service, React, Streamlit, CDN or external assets are necessary. Portfolio correctness takes priority over UI.

## Required functions

~~~python
def build_campaigns(env, observations, *, config=None, trace=None) -> list[dict]:
    ...

def render_report(trace: dict, output_path: str) -> None:
    ...
~~~

Use the exact observation schema in the shared contract. Develop with tiny synthetic observations and FakeEnv; missing teammate files are not a blocker. Do not infer a different field name from the old outline.

## First working optimizer

1. Reconstruct each candidate's actual audience from env.customer_profile using current tariff + ARPU segment. Validate target membership, non-self transition and finite evidence.
2. Recompute served_size and served_arpu from numeric ID_NUMBER order with the scorer's 5000 cap. Do not select customers by ARPU or trust stale fixture aggregates blindly.
3. For measured observations, enumerate push, sms and digital_ads only when supported by env.channels.
4. Observed mean_lift/safe_lift are already normalized to multiplier 1. Multiply by the channel multiplier exactly once. Compute mean and conservative gross, cost and net using summed predicted ARPU.
5. Keep skip as a choice. Normal selected options should have positive conservative net. Do not clip a negative effect to zero to justify push.
6. Select at most one option per cell and at most ten campaigns, under remaining budget and contacts after pilots.
7. Return only the documented campaign keys. Stable campaign names and tie-breaking, no extra scores or explicit_ids.
8. Order by conservative net descending and validate total planned resources again.

Make a correct deterministic greedy baseline first; test it and hand off. Do not wait to finish the stronger search.

## Portfolio improvement after baseline works

Implement a bounded multiple-choice search over audience cells. Each cell has skip plus valid target/channel alternatives. A beam search (default width 64) is sufficient; no optimization dependency is needed.

- State records exact contacts, cost, campaign count, selected options and conservative objective.
- Prune dominated alternatives within a cell and impossible states.
- Preserve feasible resource tradeoffs and the skip path; do not retain only the same high-cost prefix repeatedly.
- At each cell expand feasible choices, deduplicate states and keep a deterministic bounded frontier.
- Use the greedy result as a baseline incumbent; the returned solution should not be worse under the same modeled objective.
- Do not advertise global optimality. This is a bounded heuristic.
- Include a constructed test where standalone-profit greedy loses to a cheaper combination. Compare beam to exhaustive enumeration on a tiny fixture.
- Prefer readable, tested logic to an elaborate dynamic program.

Use whole effective cells in this sprint. A final campaign cannot request 1200 arbitrary customers. Its actual reach is defined by filters and scorer caps. Don't rely on accidental budget truncation to rescue an infeasible plan. Larger-than-5000 audiences use the known ID prefix.

Pilots may overlap final audiences; actual pilot IDs are unavailable. Label the objective as a conservative final-campaign proxy and do not add raw pilot observed_lift_total to it. Total real mock performance is evaluated by the captain. Optional overlap approximation can wait until the complete baseline passes; no need to hold integration for it.

## Fallback that remains honest

The brief requires 1–10 final campaigns. If no normal option has positive conservative net:
- choose a valid push campaign with small nonempty reach and the best defensible risk/value tradeoff;
- prefer measured evidence and minimize potential aggregate loss;
- use valid data/call filters to narrow a large cell if appropriate;
- respect actual remaining contacts and maximum size;
- record emergency_fallback and the reason, lack of confidence, audience and estimated downside.

Do not describe zero contact cost as zero business risk. Do not fabricate a no-op self-transition or an empty audience to satisfy the row count. If no executable valid audience exists, return [] with infeasible_fallback; constraints cannot be solved by invented users.

Unknown/unmeasured candidates cannot enter ordinary exploitation as if they had safe_lift=0 with certainty. They may inform the explicit last-resort fallback only.

## Decision explanation

Append structured events with:
- evaluated channel economics;
- selected candidate and target/channel;
- reasons for meaningful rejections: nonpositive conservative net, same-cell alternative, budget, contacts, campaign cap, no measurement, invalid data;
- comparison with the closest relevant alternative;
- final resource totals;
- fallback or infeasibility.

For a search heuristic, don't fabricate a uniquely decisive rejection reason. Say "not selected by bounded portfolio search" when several tradeoffs determine the outcome. Show what is directly known, what is estimated and what is unknown.

## Tests required for the optimizer

Use independent synthetic data and env objects:
- correct scaling of mean and SE-derived safe lift from normalized units;
- channels change appropriately with audience ARPU and budget;
- a cheap multi-campaign combination beats one expensive action in the constructed example;
- one choice per cell, <=10 campaigns, resource fit with pilots already consumed;
- numeric ID prefix and per-campaign 5000 cap;
- zero-cost push does not trigger division by zero;
- negative safe lift is not silently declared profitable;
- same data gives deterministic output and input env/observations remain unchanged;
- no observations, invalid targets, all-negative effects, tight budgets and tiny reach produce honest feasible fallback or explicit infeasibility;
- accepted campaigns survive public sanitize_campaigns and validate_strategy without being discarded.

Run:
~~~text
python -m pytest tests/test_portfolio_optimizer.py -q
~~~

## Report and demo: small but useful

After the optimizer handoff, implement:
~~~text
python -m reporting --input reports/decision_trace.json --output reports/demo.html
~~~

Build the renderer against the trace schema now using a synthetic test fixture. The captain will produce the real trace later; do not write their reports/decision_trace.json.

The final self-contained HTML should show:
- pilot spend, planned final spend, total contacts and remaining resources;
- selected campaigns and why they were selected;
- pilot evidence with uncertainty bands, clearly labelled as heuristic estimates;
- selected/rejected candidate table with reason;
- fallback warnings if present;
- trace-based estimates separated from externally measured mock results.

Use clean typography, restrained yellow/black accents, readable tables and one useful uncertainty visualization. Optional client-side selected/rejected filter is enough interactivity. Avoid promotional metrics with no data. Never render Infinity as a meaningful ROI or call an estimate "guaranteed profit". Escape dynamic text and produce a readable empty-state report.

Reporter must not call Agent, pilots, API services or evaluator. It only renders a saved trace. Test key content, missing optional metrics and HTML escaping. Do not overtest CSS.

## README: worth 25 points

Write a concise, runnable README with:
1. Problem, analyst scenario and net-gain objective.
2. Exact setup/run commands and Python/dependencies actually tested, supplied by captain if not yet measured. Mark unverified claims clearly until confirmed.
3. File map, data locations and all required runtime modules.
4. Simple Mermaid architecture.
5. Candidate ranking, adaptive pilots, uncertainty normalization, portfolio choice and fallback.
6. Budget/contact/campaign constraints including pilot consumption and overlap.
7. Baseline comparison and 10-seed results from reports/benchmark.md once available. Before then label results pending; never insert invented numbers.
8. Submission generation/reproduction.
9. Limitations: different populations, selected switchers, no causal identification, heuristic uncertainty/search, mock vs hidden effects, unknown pilot overlap.
10. Production next steps: randomized control groups, logged treatment propensities, channel calibration, drift monitoring, approval in a real campaign platform and optional Analyst Copilot.

Document code that actually exists. Do not claim full contextual-bandit learning or LLM decision-making if it isn't implemented.

docs/ARCHITECTURE.md should connect design choices to the judging rubric: functionality 25, technical 25, reproducibility 25, applicability 15, originality 10. The participant Markdown guide omits the rubric table, but it was supplied in the case brief.

docs/DEMO.md: a 2–3 minute Russian demo script. Show one real case where evidence changed a decision, one rejected high-mean but uncertain candidate if the run contains it, the portfolio tradeoff, budget compliance and reproducible benchmark. If no such event occurred, choose an actual event or clearly identify a synthetic test; never manufacture a live story.

## Delivery and freeze

First give the captain an optimizer commit with focused test results, before finishing visuals. Later give a second commit for report/docs and bounded improvements. Stage only owned files, no main/force push and no secrets.

At freeze, provide final SHA and stop edits before captain's cross-module fixes. Final handoff: files, actual test results, report command, measured limitations and where the actual benchmark evidence is expected. Continue useful owned work while another module is unavailable.
