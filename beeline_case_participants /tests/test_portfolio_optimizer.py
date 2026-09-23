from __future__ import annotations

from copy import deepcopy
from itertools import product

import pandas as pd
import pandas.testing as pdt
import pytest

from portfolio_optimizer import build_campaigns
from scoring_core import sanitize_campaigns, validate_strategy


CHANNELS = {
    "push": {"cost_per_contact": 0, "conversion_multiplier": 0.50},
    "sms": {"cost_per_contact": 4, "conversion_multiplier": 0.65},
    "digital_ads": {"cost_per_contact": 22, "conversion_multiplier": 0.85},
    "call": {"cost_per_contact": 160, "conversion_multiplier": 1.20},
}


class FakeEnv:
    def __init__(self, profile, *, budget=100_000, contacts=15_000, channels=None):
        self.customer_profile = profile
        tariffs = sorted(set(profile["current_tariff"]) | {"t1", "t2", "t3", "t4"})
        self.tariffs = pd.DataFrame(
            {"tariff_plan_code": tariffs, "price_tariff": range(len(tariffs))}
        )
        self.channels = deepcopy(channels or CHANNELS)
        self.remaining_budget = budget
        self.remaining_contacts = contacts
        self.pilots_left = 7


def observation(current, segment, target, mean, safe, *, status="measured"):
    return {
        "candidate_id": f"{current}|{segment}|{target}",
        "cell_id": f"{current}|{segment}",
        "current_tariff": current,
        "arpu_segment": segment,
        "target_tariff": target,
        "status": status,
        "mean_lift": mean,
        "safe_lift": safe,
        "se_lift": None if mean is None else 0.05,
    }


def profile_for(*cells):
    rows = []
    next_id = 1
    for current, segment, count, arpu in cells:
        for _ in range(count):
            rows.append(
                {
                    "ID_NUMBER": next_id,
                    "current_tariff": current,
                    "arpu_segment": segment,
                    "data_segment": "LITE",
                    "call_segment": "MEDIUM",
                    "predicted_arpu": float(arpu),
                }
            )
            next_id += 1
    return pd.DataFrame(rows)


def event(trace, name):
    return [item for item in trace if item["event"] == name]


def test_normalized_lift_is_scaled_once_and_cost_is_subtracted():
    env = FakeEnv(profile_for(("t1", "HIGH", 2, 500)))
    trace = []
    campaigns = build_campaigns(env, [observation("t1", "HIGH", "t2", .2, .1)], trace=trace)

    economics = event(trace, "evaluated_channel")
    push = next(item for item in economics if item["details"]["channel"] == "push")
    sms = next(item for item in economics if item["details"]["channel"] == "sms")
    assert push["details"]["mean_gross"] == 100.0
    assert push["details"]["conservative_gross"] == 50.0
    assert sms["details"]["conservative_net"] == 57.0  # .1 * .65 * 1000 - 8
    assert campaigns


def test_channel_choice_changes_with_value_and_budget():
    low = FakeEnv(profile_for(("t1", "LOW", 10, 10)), budget=10_000)
    high = FakeEnv(profile_for(("t1", "HIGH", 10, 10_000)), budget=10_000)
    low_result = build_campaigns(low, [observation("t1", "LOW", "t2", .1, .1)])
    high_result = build_campaigns(high, [observation("t1", "HIGH", "t2", .1, .1)])
    assert low_result[0]["channel"] == "push"
    assert high_result[0]["channel"] == "digital_ads"

    constrained = FakeEnv(high.customer_profile.copy(), budget=100)
    constrained_result = build_campaigns(
        constrained, [observation("t1", "HIGH", "t2", .1, .1)]
    )
    assert constrained_result[0]["channel"] in {"push", "sms"}


def test_beam_finds_cheaper_combination_that_beats_standalone_greedy():
    profile = profile_for(
        ("t1", "HIGH", 20, 550),
        ("t2", "HIGH", 10, 600),
        ("t3", "HIGH", 10, 600),
    )
    env = FakeEnv(profile, budget=440, contacts=20)
    observations = [
        observation("t1", "HIGH", "t4", .2, .2),
        observation("t2", "HIGH", "t4", .2, .2),
        observation("t3", "HIGH", "t4", .2, .2),
    ]
    trace = []
    result = build_campaigns(env, observations, config={"beam_width": 64}, trace=trace)
    totals = event(trace, "resource_totals")[0]["details"]
    assert len(result) == 2
    assert totals["conservative_final_net_proxy"] == pytest.approx(1600.0)
    assert totals["greedy_baseline_objective"] == pytest.approx(1430.0)

    # Exhaustive enumeration is practical on this tiny fixture and acts as an
    # independent oracle for the bounded search test.
    by_candidate = {}
    for item in event(trace, "evaluated_channel"):
        detail = item["details"]
        if detail["conservative_net"] > 0:
            by_candidate.setdefault(item["candidate_id"], []).append(detail)
    exhaustive = 0.0
    groups = [[None, *options] for _, options in sorted(by_candidate.items())]
    for choices in product(*groups):
        chosen = [choice for choice in choices if choice is not None]
        if sum(choice["contacts"] for choice in chosen) > env.remaining_contacts:
            continue
        if sum(choice["cost"] for choice in chosen) > env.remaining_budget:
            continue
        exhaustive = max(exhaustive, sum(choice["conservative_net"] for choice in chosen))
    assert totals["conservative_final_net_proxy"] == pytest.approx(exhaustive)


def test_constraints_one_choice_per_cell_and_pilot_residual_resources():
    profile = profile_for(
        ("t1", "MID", 20, 1000),
        ("t2", "MID", 20, 1000),
        ("t3", "MID", 20, 1000),
    )
    env = FakeEnv(profile, budget=200, contacts=40)
    observations = [
        observation("t1", "MID", "t3", .2, .2),
        observation("t1", "MID", "t4", .19, .19),
        observation("t2", "MID", "t4", .2, .2),
        observation("t3", "MID", "t4", .2, .2),
    ]
    trace = []
    result = build_campaigns(env, observations, trace=trace)
    cells = {(row["filter_current_tariff"], row["filter_arpu_segment"]) for row in result}
    totals = event(trace, "resource_totals")[0]["details"]
    assert len(cells) == len(result) <= 10
    assert totals["final_contacts"] <= env.remaining_contacts
    assert totals["final_cost"] <= env.remaining_budget


def test_numeric_id_prefix_and_5000_cap_are_recomputed():
    ids = list(range(1, 5003))
    profile = pd.DataFrame(
        {
            "ID_NUMBER": [str(value) for value in reversed(ids)],
            "current_tariff": ["t1"] * len(ids),
            "arpu_segment": ["HIGH"] * len(ids),
            "predicted_arpu": [float(value) for value in reversed(ids)],
        }
    )
    env = FakeEnv(profile, contacts=6000)
    trace = []
    build_campaigns(env, [observation("t1", "HIGH", "t2", .1, .1)], trace=trace)
    push = next(
        item for item in event(trace, "evaluated_channel")
        if item["details"]["channel"] == "push"
    )
    assert push["details"]["contacts"] == 5000
    assert push["details"]["served_arpu"] == sum(range(1, 5001))


def test_zero_cost_push_and_negative_safe_lift_use_honest_fallback():
    env = FakeEnv(profile_for(("t1", "LOW", 3, 100)))
    trace = []
    result = build_campaigns(
        env, [observation("t1", "LOW", "t2", .02, -.10)], trace=trace
    )
    assert len(result) == 1
    assert result[0]["channel"] == "push"
    assert event(trace, "emergency_fallback")
    assert event(trace, "resource_totals")[0]["details"]["final_cost"] == 0.0
    evaluated = event(trace, "evaluated_channel")
    assert all(item["details"]["conservative_net"] < 0 for item in evaluated)


def test_deterministic_and_does_not_mutate_inputs():
    profile = profile_for(("t1", "HIGH", 5, 1000), ("t2", "MID", 4, 800))
    env = FakeEnv(profile)
    observations = [
        observation("t1", "HIGH", "t3", .2, .1),
        observation("t2", "MID", "t3", .2, .1),
    ]
    original_profile = profile.copy(deep=True)
    original_observations = deepcopy(observations)
    original_budget = env.remaining_budget

    first = build_campaigns(env, observations)
    second = build_campaigns(env, observations)
    assert first == second
    pdt.assert_frame_equal(profile, original_profile)
    assert observations == original_observations
    assert env.remaining_budget == original_budget


def test_invalid_unmeasured_empty_and_tight_reach_cases():
    profile = profile_for(("t1", "HIGH", 6, 1000))
    profile.loc[:1, "data_segment"] = "NON_USER"
    env = FakeEnv(profile, contacts=2)

    no_trace = []
    assert build_campaigns(env, [], trace=no_trace) == []
    assert event(no_trace, "infeasible_fallback")

    invalid_trace = []
    invalid = [observation("t1", "HIGH", "missing", .2, .1)]
    assert build_campaigns(env, invalid, trace=invalid_trace) == []
    assert event(invalid_trace, "infeasible_fallback")

    unmeasured = observation("t1", "HIGH", "t2", None, None, status="untested")
    fallback_trace = []
    fallback = build_campaigns(env, [unmeasured], trace=fallback_trace)
    assert fallback[0]["filter_data_segment"] == "NON_USER"
    assert event(fallback_trace, "emergency_fallback")[0]["details"]["confidence"] == "unknown"


def test_public_sanitizer_and_validator_accept_output():
    env = FakeEnv(profile_for(("t1", "HIGH", 5, 1000)))
    campaigns = build_campaigns(
        env, [observation("t1", "HIGH", "t2", .2, .1)]
    )
    sanitized = sanitize_campaigns(campaigns, env.tariffs)
    assert sanitized == campaigns
    validate_strategy(pd.DataFrame(sanitized), env.tariffs)
