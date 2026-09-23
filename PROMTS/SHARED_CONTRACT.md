# Shared implementation contract — v1.0

Frozen for the three-hour sprint. This file is a specification, not an implementation task. All three modules use plain dictionaries and standard Python numeric types. Do not create another shared dependency or change this contract during parallel work.

## Scope and integrity

Implement the participant API, not an alternative scoring system. Organizer source and CSVs remain unchanged. Public API/scoring mechanics may be read. Agent runtime must never access hidden model data, closures, reflection-based internals, organizer oracle helpers, or mock effects. Offline evaluation may call the supplied evaluate_agent function. Test environments may contain synthetic truths authored by us, but those truths stay outside the object passed to the agent.

Only final campaign filters documented in the participant guide are allowed. Do not emit explicit_ids, n_customers, arbitrary query strings, limits, or confidence fields in campaign dictionaries. explicit_ids is an internal pilot bookkeeping mechanism, not our submission interface.

## Public functions

~~~python
generate_candidates(profile, history, tariffs, *, config=None, trace=None) -> list[dict]
run_adaptive_pilots(env, candidates, *, config=None, trace=None) -> list[dict]
build_campaigns(env, observations, *, config=None, trace=None) -> list[dict]
~~~

Required positional arguments match the original team plan. config is a plain dictionary; missing/None uses defaults, extra unrelated keys are ignored. trace is either None or a list to append JSON-compatible events to. Do not mutate input DataFrames, candidates, observations, config, or env counters. Only env.run_pilot legitimately consumes resources.

Modules have no circular imports: candidate_engine imports no team module; pilot_policy imports no team module; portfolio_optimizer imports no team module; agent.py coordinates them. reporting.py reads a trace, not runtime internals.

## Candidate dictionary

All these fields are required. Optional additional diagnostic fields must not become required by another module.

~~~python
{
    "candidate_id": "tariff_4|HIGH|tariff_9",
    "cell_id": "tariff_4|HIGH",
    "current_tariff": "tariff_4",
    "arpu_segment": "HIGH",
    "target_tariff": "tariff_9",
    "history_count": 80,
    "prior_lift": 0.12,
    "positive_rate": 0.61,
    "audience_size": 300,
    "audience_arpu": 2100000.0,
    "served_size": 300,
    "served_arpu": 2100000.0,
    "prior_score": 150000.0,
    "source": "history",
}
~~~

Numbers above illustrate types; they are not real data or target values.

- IDs use exact current tariff, segment and target strings separated by "|".
- history_count counts usable historical observations for the exact transition and historical pre-switch ARPU segment.
- prior_lift is a robust, shrunk historical relative before/after change among switchers. It is a ranking prior, NOT an estimated campaign response or causal effect.
- positive_rate is the fraction of historical switchers with positive change. It is NOT conversion probability. For no matching history, history_count=0, prior_lift=0.0, positive_rate=0.5 and source="catalog"; rank using a separately documented catalog heuristic.
- audience_size/arpu cover the complete cell. served_size/arpu cover the first min(5000, cell size) rows in numeric ID_NUMBER order, matching public scoring.
- source is "history" or "catalog".
- prior_score is a finite relative ranking score. Never present it as validated profit.
- Invalid/blank current tariffs or segments are excluded from campaign candidates, with counts reported. Do not turn NaN into the string "nan".
- Keep negative historical signals as negative. No final action should rely on historical score as if it were a pilot.
- Deterministic sorting: descending prior_score, then candidate_id. Suggested return limit 40; the pilot module chooses the exploration shortlist.
- Return [] when no valid cells/targets exist. An empty history is not an empty audience: generate catalog candidates if possible.

## Observation dictionary

Return one NEW dictionary per input candidate, preserving every candidate field, with:

~~~python
{
    # ...all candidate fields...
    "status": "measured",
    "pilot_count": 2,
    "pilot_n": 280,
    "pilot_cost": 1120.0,
    "mean_lift": 0.20,
    "se_lift": 0.0739,
    "safe_lift": 0.1054,
    "optimistic_lift": 0.2946,
    "pilots": [
        {"channel": "sms", "n_customers": 80,
         "cost": 320.0, "observed_lift_ratio": 0.12},
        {"channel": "sms", "n_customers": 200,
         "cost": 800.0, "observed_lift_ratio": 0.134}
    ],
}
~~~

This example is illustrative, not a golden arithmetic fixture.

status is "measured", "untested", or "error". With no valid measurement, all four lift fields are None, pilot_n=0, pilot_count=0; errors/charged invalid responses go in trace. A successful measurement followed by a failed call remains "measured".

All four lift fields use **channel-normalized units: hypothetical conversion multiplier 1.0**. For the SMS-first policy:

~~~text
N = sum(actual returned n_customers)
sms_mean = sum(actual_n * observed_lift_ratio) / N
mean_lift = sms_mean / sms_multiplier
se_lift = 0.804 / (sms_multiplier * sqrt(N))
safe_lift = mean_lift - risk_z * se_lift
optimistic_lift = mean_lift + risk_z * se_lift
~~~

The noise 0.804 is documented in environment.py. Use returned actual sample counts, not requested counts. Preserve negative observations. Do not average absolute observed_lift_total as if it were a ratio. Do not mix different cells or target tariffs. Do not multiply uncertainty by historical sample count.

For future mixed push/SMS/ads pilots, combine normalized observations by inverse variance: variance_i = (0.804 / (multiplier_i * sqrt(n_i)))**2. Current sprint uses SMS pilots only. No call pilots.

risk_z=1.28 is a configurable risk preference, not a simultaneous 90% guarantee after adaptive selection. Report these as uncertainty bands/conservative estimates; selected winners and repeated testing invalidate casual claims of calibrated coverage.

## Config defaults

~~~python
{
    "candidate_limit": 40,
    "shrinkage_strength": 50.0,
    "exploration_pilots": 10,
    "exploration_n": 80,
    "confirmation_pilots": 5,
    "confirmation_n": 200,
    "pilot_cap": 20,
    "pilot_contact_cap": 3000,
    "pilot_money_cap": 12000.0,
    "final_contact_reserve": 1000,
    "risk_z": 1.28,
    "beam_width": 64,
    "max_campaigns": 10,
    "max_per_campaign": 5000,
}
~~~

These are policy defaults, not data-derived truths or judging constants. Environment residual limits always take precedence. Ten 80-person pilots plus five 200-person confirmations cost at most 1800 contacts and 7200 c.u. under standard SMS pricing. Remaining pilot slots are optional, never a quota to exhaust.

## Actual env interface and channels

env.customer_profile is a DataFrame.
env.tariffs has tariff_plan_code and price_tariff.
env.channels maps channel name to {"cost_per_contact": ..., "conversion_multiplier": ...}.
env.remaining_budget, env.remaining_contacts, env.pilots_left reflect pilots already executed.
env.pilot_history exposes responses, but does NOT include audience filters or contacted IDs.

~~~python
result = env.run_pilot(
    target_tariff=candidate["target_tariff"],
    channel="sms",
    n_customers=n,
    filter_arpu_segment=candidate["arpu_segment"],
    filter_current_tariff=candidate["current_tariff"],
)
~~~

Result fields: pilot, target_tariff, channel, n_customers, cost, observed_lift_ratio, observed_lift_total, remaining_budget, remaining_contacts.

Record candidate association locally as you call. Never infer it by assuming pilot_history contains cell IDs.

Read costs and multipliers from env. For supplied values push=0/.50, sms=4/.65, digital_ads=22/.85, call=160/1.20, where each pair is cost/multiplier. With conversion probability in [0,1], the first three do not reach the upper clipping branch, so scaling normalized lift is justified by public scoring. Call may saturate; omit it in the sprint. This transfer is a property of this simulator, not a claim about real channel behavior.

## Portfolio economics and feasibility

For a cell and supported channel c:

~~~text
expected_gross = mean_lift * multiplier[c] * served_arpu
conservative_gross = safe_lift * multiplier[c] * served_arpu
contact_cost = served_size * cost_per_contact[c]
conservative_net = conservative_gross - contact_cost
~~~

Start from env.remaining_budget and env.remaining_contacts: pilot costs are already consumed. Final campaign planning must not decrement env attributes.

Use at most one target/channel per current_tariff + arpu_segment cell. Distinct valid cells do not overlap. Optimize the set, not each cell's expensive channel independently. Treat skip as an option. Use whole effective cells that fit remaining limits; don't pretend a final campaign can specify arbitrary n_customers. Large cells use the scorer's 5000-person numeric ID prefix. Initially do not deliberately exploit budget truncation; exact selected costs must fit.

Final campaign dictionary:

~~~python
{
    "campaign_name": "campaign_01_tariff_4_HIGH_tariff_9_sms",
    "filter_current_tariff": "tariff_4",
    "filter_arpu_segment": "HIGH",
    "target_tariff": "tariff_9",
    "channel": "sms"
}
~~~

Optional documented filter_data_segment and filter_call_segment may be used for a smaller emergency fallback. Do not add explanation metadata to campaign dictionaries; keep it in trace.

Campaign order: descending conservative net with stable ID/channel tie-breakers. Validate expected contact and monetary totals after ordering. At most 10 final campaigns, distinct cell IDs, valid non-self targets, and nonempty served audiences.

**Pilot overlap:** pilots and final campaigns can contact the same subscriber; contacts/cost still count, uplift is the best campaign once. Actual pilot IDs are not public. Never claim exact deduplication of pilot overlap. First version labels the final-only gross estimate as a proxy, does not add estimated pilot gross to it, and compares actual total performance through local_eval. A later documented random-overlap approximation is optional; do not assume pilot samples are disjoint or pretend their IDs are known.

**Emergency fallback:** when no normal action has positive conservative net, the brief still asks for 1–10 campaigns. Pick a small valid push audience, preferably measured, balancing best available effect and total potential loss. Narrow using documented data/call filters if needed. Never call this risk-free or assert positive profit. Emit emergency_fallback with evidence and lack of confidence. If no executable valid audience or contacts exist, return [] and explicitly record infeasible_fallback; do not invent a customer or corrupt limits.

## Trace and reporting

Each module appends events to the provided list:

~~~python
{
    "stage": "pilot",          # candidate | pilot | portfolio | agent
    "event": "confirmation",   # free but stable event name
    "candidate_id": "tariff_4|HIGH|tariff_9",  # or None
    "reason": "High-value cell; uncertainty can change selection.",
    "details": {"requested_n": 200, "actual_n": 200}
}
~~~

No customer IDs, raw secrets, NaN or Infinity. Use finite Python floats/ints or None. No timestamps in decision sorting or campaign names.

Agent.act resets self.last_trace for every run:

~~~python
{
    "schema_version": "1.0",
    "candidates": [],
    "observations": [],
    "campaigns": [],
    "events": [],
    "summary": {
        "pilot_count": 0,
        "pilot_contacts": 0,
        "pilot_cost": 0.0,
        "final_contacts": 0,
        "final_cost": 0.0,
        "remaining_budget_after_plan": 0.0,
        "remaining_contacts_after_plan": 0,
        "status": "ok",  # ok | fallback | infeasible | error
    }
}
~~~

Trace remains available on Agent; normal act does not require disk or network access. Only the benchmarking CLI writes reports/decision_trace.json. reporting.py exposes:

~~~python
render_report(trace: dict, output_path: str) -> None
~~~

It renders existing trace only. It never starts pilots or changes campaigns. Mock benchmark net gain is a separately labelled measured evaluator result, not an inferred future score.

## Test independence and reproducibility

Use pytest, tiny synthetic DataFrames, and private FakeEnv classes inside your own tests. A FakeEnv implements only public fields and run_pilot. It consumes resources realistically and returns actual sample size. Never put truth/oracle fields on the object passed to the agent.

No teammate module is required to test a pure module. Imports or temporary replacements needed for agent integration live in tests via monkeypatch; no stub production files.

Deterministic decisions for identical public observations and data, explicit tie-breaking, no Python hash-based order. Reset all state between act calls. Submission seed belongs to organizer runner (42); don't inspect or hardcode seed-dependent campaign choices.
