"""SMS exploration and decision-focused confirmation using public pilot evidence."""

from copy import deepcopy
import math


DEFAULTS = dict(exploration_pilots=10, exploration_n=80, confirmation_pilots=5,
                confirmation_n=200, pilot_cap=20, pilot_contact_cap=3000,
                pilot_money_cap=12000.0, final_contact_reserve=1000, risk_z=1.28)


def _event(trace, event, candidate=None, reason="", **details):
    if trace is not None:
        trace.append(dict(stage="pilot", event=event,
                          candidate_id=candidate and candidate["candidate_id"],
                          reason=reason, details=details))


def _aggregate(record, multiplier, risk_z):
    samples = record["pilots"]
    n = sum(p["n_customers"] for p in samples)
    mean = sum(p["n_customers"] * p["observed_lift_ratio"] for p in samples) / n / multiplier
    se = 0.804 / (multiplier * math.sqrt(n))
    record.update(status="measured", pilot_count=len(samples), pilot_n=n,
                  pilot_cost=sum(p["cost"] for p in samples), mean_lift=mean,
                  se_lift=se, safe_lift=mean - risk_z * se,
                  optimistic_lift=mean + risk_z * se)


def _shortlist(records, limit):
    ranked = sorted((r for r in records if r["audience_size"] >= 10),
                    key=lambda r: (-r["prior_score"], r["candidate_id"]))
    primary, alternatives, seen = [], [], set()
    for r in ranked:
        if r["cell_id"] in seen:
            alternatives.append(r)
        else:
            primary.append(r)
        seen.add(r["cell_id"])
    # Only reserve slots for positive ranking signals or plausible catalog options.
    plausible = [r for r in alternatives if r["prior_score"] > 0]
    reserve = min(2, len(plausible), max(0, limit // 3))
    selected = primary[:max(0, limit - reserve)] + plausible[:reserve]
    selected_ids = {r["candidate_id"] for r in selected}
    selected += [r for r in ranked if r["candidate_id"] not in selected_ids][:max(0, limit - len(selected))]
    return selected[:limit]


def _priority(record, records, env, n, multiplier, price):
    """Value-of-information heuristic, not exact EVSI or calibrated coverage.

    stake (c.u. ARPU) * normalized SE reduction * ambiguity / burden (contacts).
    Burden = n * (1 + SMS cost / max(1, average served ARPU)); cost is thus
    converted to contact equivalents. Ambiguity is proximity in SE units to
    zero-profit push or a competing target. Dominated/nonpositive optimistic
    options and whole cells that cannot fit final reach have zero priority.
    """
    reach = min(int(record["served_size"]), int(env.remaining_contacts) - n, 5000)
    if reach < record["served_size"] or reach <= 0 or record["optimistic_lift"] <= 0:
        return None
    peers = [r for r in records if r["cell_id"] == record["cell_id"]
             and r["candidate_id"] != record["candidate_id"] and r["status"] == "measured"]
    if any(r["safe_lift"] > record["optimistic_lift"] for r in peers):
        return None
    se = record["se_lift"]
    distances = [abs(record["mean_lift"]) / se]
    distances += [abs(record["mean_lift"] - p["mean_lift"]) /
                  math.hypot(se, p["se_lift"]) for p in peers]
    ambiguity = 1 / (1 + min(distances) ** 2)
    reduction = se - 0.804 / (multiplier * math.sqrt(record["pilot_n"] + n))
    stake = max(0.0, float(record["served_arpu"]))
    burden = n * (1 + price / max(1.0, stake / reach))
    return dict(priority=stake * reduction * ambiguity / burden,
                value_at_stake=stake, se_reduction=reduction,
                ambiguity_weight=ambiguity, burden=burden, feasible_reach=reach)


def run_adaptive_pilots(env, candidates, *, config=None, trace=None):
    """Return one fresh observation per candidate; only run_pilot spends resources."""
    cfg = {**DEFAULTS, **(config or {})}
    records = [dict(deepcopy(c), status="untested", pilot_count=0, pilot_n=0,
                    pilot_cost=0.0, mean_lift=None, se_lift=None, safe_lift=None,
                    optimistic_lift=None, pilots=[]) for c in candidates]
    if not records:
        return records
    sms = env.channels.get("sms", {})
    price = float(sms.get("cost_per_contact", -1))
    multiplier = float(sms.get("conversion_multiplier", 0))
    if not math.isfinite(price) or price < 0 or not math.isfinite(multiplier) or multiplier <= 0:
        _event(trace, "invalid_channel", reason="SMS pricing or multiplier is invalid.")
        return records
    start_budget, start_contacts = env.remaining_budget, env.remaining_contacts
    attempts = 0
    reported_n, reported_cost = 0, 0.0
    disabled = set()

    def feasible(record, requested):
        spent_n = max(reported_n, start_contacts - env.remaining_contacts)
        spent_cost = max(reported_cost, start_budget - env.remaining_budget)
        if attempts >= min(20, cfg["pilot_cap"]) or env.pilots_left <= 0:
            return 0
        budget = min(env.remaining_budget, cfg["pilot_money_cap"] - spent_cost)
        contacts = min(env.remaining_contacts - cfg["final_contact_reserve"],
                       cfg["pilot_contact_cap"] - spent_n)
        n = min(200, requested, record["audience_size"], contacts)
        if budget < 0:
            return 0
        if price > 0:
            n = min(n, math.floor(budget / price))
        return max(0, int(n)) if n >= 10 else 0

    def run(record, requested, stage, components=None):
        nonlocal attempts, reported_n, reported_cost
        n = feasible(record, requested)
        if n < 10:
            _event(trace, "skipped", record, "Fewer than ten feasible pilot contacts.")
            return
        attempts += 1
        try:
            response = env.run_pilot(target_tariff=record["target_tariff"], channel="sms",
                                     n_customers=n, filter_arpu_segment=record["arpu_segment"],
                                     filter_current_tariff=record["current_tariff"])
        except (RuntimeError, ValueError) as exc:
            disabled.add(record["candidate_id"])
            if not record["pilots"]:
                record["status"] = "error"
            _event(trace, "call_error", record, str(exc), error_type=type(exc).__name__,
                   exhausted=feasible(record, requested) < 10,
                   remaining_contacts=int(env.remaining_contacts),
                   remaining_budget=float(env.remaining_budget), pilots_left=int(env.pilots_left))
            return
        try:
            actual = float(response["n_customers"])
            cost = float(response["cost"])
            # Charge any usable resource report even when the effect is invalid.
            if math.isfinite(actual) and actual >= 0:
                reported_n += actual
            if math.isfinite(cost) and cost >= 0:
                reported_cost += cost
            ratio = float(response["observed_lift_ratio"])
            if (not all(math.isfinite(x) for x in (actual, cost, ratio)) or
                    actual != int(actual) or not 0 < actual <= n or cost < 0 or
                    response.get("channel", "sms") != "sms" or
                    response.get("target_tariff", record["target_tariff"]) != record["target_tariff"]):
                raise ValueError("Invalid pilot sample, cost, association or effect")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            disabled.add(record["candidate_id"])
            if not record["pilots"]:
                record["status"] = "error"
            _event(trace, "invalid_response", record, str(exc), requested_n=n)
            return
        record["pilots"].append(dict(channel="sms", n_customers=int(actual), cost=cost,
                                     observed_lift_ratio=ratio))
        _aggregate(record, multiplier, cfg["risk_z"])
        _event(trace, stage, record, "Pilot-only evidence; historical ranking is not pooled.",
               requested_n=n, actual_n=int(actual), cost=cost, observed_lift_ratio=ratio,
               **(components or {}))

    for record in _shortlist(records, max(0, min(10, int(cfg["exploration_pilots"])))):
        run(record, cfg["exploration_n"], "exploration")
    for _ in range(max(0, min(5, int(cfg["confirmation_pilots"])))):
        choices = []
        for record in records:
            if record["status"] != "measured" or record["candidate_id"] in disabled:
                continue
            n = feasible(record, cfg["confirmation_n"])
            if n < 10:
                continue
            components = _priority(record, records, env, n, multiplier, price)
            if components and components["priority"] > 0:
                choices.append((record, n, components))
        if not choices:
            break
        record, n, components = min(choices, key=lambda x: (-x[2]["priority"], x[0]["candidate_id"]))
        run(record, n, "confirmation", components)
    _event(trace, "complete", reason="Optional pilot slots are left unused.", attempts=attempts,
           pilot_contacts=int(max(reported_n, start_contacts - env.remaining_contacts)),
           pilot_cost=float(max(reported_cost, start_budget - env.remaining_budget)))
    return records
