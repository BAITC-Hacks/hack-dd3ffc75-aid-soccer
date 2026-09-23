# Student 1 — Historical Prior and Candidate Diversity

Paste everything below into your coding agent. The repository must contain prompts/SHARED_CONTRACT.md. Use Astra 6 / high as selected by the team.

---

Implement your portion of Adaptive Campaign Portfolio Agent now. We have a three-hour hackathon sprint, and your first tested, integrable version is due within 45 minutes. Deliver code and focused tests; do not stop at a plan or request routine design approvals. Make reasonable reversible choices, explain material assumptions, and remain inside your ownership.

## Outcome

Build a trustworthy candidate engine that gives pilot exploration a diverse, useful shortlist despite population shift. Our marketing analyst needs profitable tariff recommendations, and history comes from completely different customers. Your module proposes experiments; it does not claim to know current-audience campaign effects.

Read in this order:
1. prompts/SHARED_CONTRACT.md.
2. PARTICIPANT_GUIDE.md and the CSV headers.
3. Relevant history/profile/catalog data needed to implement and verify your module.

Do not spend time researching frameworks or building infrastructure.

## Ownership

Only create/edit:
- candidate_engine.py
- tests/test_candidate_engine.py
- tools/profile_data.py
- reports/data_audit.json
- docs/DATA_FINDINGS.md

Work on feature/prior-engine in your own clone. Check existing changes before editing. Do not change agent.py, pilot_policy.py, portfolio_optimizer.py, README.md, dependencies, shared contract, organizer files, or any CSV. Do not create shared test fixtures or production stubs for other students. Do not spawn additional coding agents; the team already has three independent workstreams.

If an external module is absent, your work continues. Your function depends only on pandas, numpy and standard library. Unit tests are independent.

## Required API

~~~python
def generate_candidates(profile, history, tariffs, *, config=None, trace=None) -> list[dict]:
    ...
~~~

Use all required candidate keys and exact units in the shared contract. Return a list of new plain dictionaries. No disk I/O or environment access inside this function; data is passed in. Deterministic order by descending prior_score, then candidate_id. Missing history should activate catalog hypotheses rather than crash.

## First working implementation

1. Validate required columns. Work on copies or local Series; don't mutate input frames.
2. Numeric-coerce historical AVG_ARPU_PREV_3M and AVG_ARPU_NEXT_3M. Exclude nonfinite values and prev <= 0 from relative-lift estimation, recording counts. Handle missing/unknown tariff labels without stringifying nulls.
3. Derive historical ARPU segment from PRE-switch ARPU: LOW < 1000; MID 1000 through 5000 inclusive; HIGH > 5000. Never use post-switch revenue to derive context.
4. Compute raw relative lift (next-prev)/prev. Protect ranking against tiny-denominator outliers with a simple documented robust estimate, such as pooled quantile winsorization when sample size supports it. Retain raw mean, median and positive rate as diagnostics. Do not silently convert negative outcomes to zero or tune cutoffs to mock winners.
5. Aggregate by from-tariff, historical ARPU segment, target tariff. Apply configurable shrinkage n/(n+50) to the robust historical mean; expose history_count and prior_lift with their correct meaning.
6. Build valid audience cells from current_tariff + arpu_segment and predicted_arpu. Only known nonblank tariffs and valid segments; exclude unusable rows from candidates without modifying env's eventual profile. Audit invalid revenue/ID cases. Audience statistics use the actual target profile, not history.
7. For each cell, compute complete audience_size/audience_arpu and served_size/served_arpu from the first 5000 rows in numeric ID_NUMBER order. Count all valid served contacts even if their predicted ARPU is zero. Do not reorder by ARPU.
8. Generate known non-self target transitions, join history aggregates, and create stable IDs. All targets must come from the supplied tariffs DataFrame.
9. Produce a transparent prior_score combining robust shrunk lift, positive transition rate and served ARPU. Explain the formula in the function docstring. It is a ranking score; positive_rate is not a conversion probability and the score is not expected campaign profit.
10. Preserve signed prior_lift. If all historical candidates look weak, still expose the best plausible hypotheses for pilots; avoid an all-empty output when the audience and catalog permit alternatives.

## Strong version after the first tests pass

Improve hypothesis coverage before adding any model:

- Prevent the candidate_limit=40 list being swallowed by many transitions from one audience cell.
- Retain a small number of best historical transitions per valuable cell, with up to two target alternatives when useful. Fill a bounded pool across cells, then return it in deterministic prior_score order.
- Include a few catalog-based alternatives not observed in exact historical cells. Rank plausibility using actual price/package information, not tariff ID ordering or invented revenue lift. Source="catalog", zero history_count, neutral historical fields per contract, and separate diagnostics for the heuristic.
- Keep source scores comparable enough that neither catalog guesses nor huge historical outliers monopolize the pool. Document the simple calibration/rank blending if used.
- Include negative and low-support cases in tests so shrinkage cannot become a winner-selection bug.
- No neural network, hyperparameter grid, causal model, training pipeline, or learned conversion rate is needed.

Optional if useful and time allows: a two-level historical fallback from exact cell to transition pair. It must remain explicitly weaker evidence; don't overwrite history_count with pooled observations or imply pooled data is exact-cell evidence. Do not change the public contract.

## Audit deliverable

tools/profile_data.py should run with:
~~~text
python -m tools.profile_data
~~~

It should write reports/data_audit.json and support the concise facts in docs/DATA_FINDINGS.md: row counts, unique IDs, duplicate IDs, blank tariff/segment counts, known-target coverage, history/profile ID intersection, transition count, cell sizes, usable historical lift count and outlier diagnostics.

Already observed in this pack: 23,441 profile rows, 14,823 history rows, 21 tariffs, 168 directed transition pairs, no overlapping customer IDs, 95 missing current_tariff and 5 missing arpu_segment. Recompute; do not hardcode these as assertions required on any dataset. Note that missing-value categories can overlap. Other historical CSVs are optional context; do not join them onto target customers as if the populations matched.

## Focused acceptance tests

Use tiny constructed DataFrames and assert behavior, not the exact implementation:

- A supported positive transition beats an otherwise comparable two-observation spike after robust treatment/shrinkage.
- Empty history still yields valid catalog candidates for a nonempty valid audience.
- Empty/invalid audience returns [].
- Missing labels, nonfinite amounts, zero pre-ARPU and unknown target codes are handled explicitly.
- Self-transitions never appear; audiences are nonempty; every target is valid.
- Segment boundaries are correct and based on pre-switch ARPU.
- Supplied inputs remain unchanged; repeated calls and reordered equivalent rows yield deterministic candidates within numerical tolerance.
- A cell larger than 5000 has correct ID-prefix served counts and ARPU, not the richest 5000.
- A constructed multi-cell dataset demonstrates candidate diversity and catalog exploration slots without exceeding the limit.

Use pytest and run:
~~~text
python -m pytest tests/test_candidate_engine.py -q
python -m tools.profile_data
~~~

Do not repeatedly broaden testing after these pass unless you change behavior.

## Handoff

At the first working milestone, make the changes reviewable in a commit containing only owned files if local Git permissions allow. Publishing follows the team's Git workflow; do not push to main or alter anyone's work.

Return a concise handoff:
- branch and commit SHA, or exact changed files if not committed;
- implemented API and units;
- test command and actual result;
- one example candidate with values from an actual run;
- assumptions and known limitations.

Continue improving only your owned files until the agreed handoff/freeze. If the captain reports an interface problem, fix it without renaming contract keys. Do not claim tests ran if they did not. No invented performance numbers.
