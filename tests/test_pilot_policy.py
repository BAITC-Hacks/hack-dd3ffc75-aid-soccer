import copy
import json
import math

import pandas as pd
import pytest

from pilot_policy import run_adaptive_pilots


def candidate(cell="a", target="b", size=1000, score=100, arpu=1000000):
    return dict(candidate_id=f"{cell}|HIGH|{target}", cell_id=f"{cell}|HIGH",
                current_tariff=cell, arpu_segment="HIGH", target_tariff=target,
                history_count=80, prior_lift=0.9, positive_rate=0.8,
                audience_size=size, audience_arpu=arpu, served_size=min(size, 5000),
                served_arpu=arpu, prior_score=score, source="history")


def fake_env(response=None, *, budget=100000, contacts=15000, slots=20):
    # Response function lives in the closure, not on the agent's input object.
    class FakeEnv:
        channels = {"sms": dict(cost_per_contact=4, conversion_multiplier=0.65)}
        tariffs = pd.DataFrame({"tariff_plan_code": ["a", "b", "c", "d"]})

        def __init__(self):
            self.remaining_budget = budget
            self.remaining_contacts = contacts
            self.pilots_left = slots
            self.pilot_history = []

        def run_pilot(self, **kwargs):
            assert kwargs["channel"] == "sms"
            assert kwargs["filter_arpu_segment"] == "HIGH"
            requested = kwargs["n_customers"]
            assert 10 <= requested <= 200
            assert requested <= self.remaining_contacts
            assert requested * 4 <= self.remaining_budget
            assert self.pilots_left > 0
            if kwargs["target_tariff"] not in self.tariffs.tariff_plan_code.tolist():
                raise ValueError("Unknown target")
            value = response(kwargs, len(self.pilot_history)) if response else (requested, 0.1)
            if isinstance(value, Exception):
                raise value
            n, lift = value
            self.remaining_budget -= n * 4
            self.remaining_contacts -= n
            self.pilots_left -= 1
            result = dict(n_customers=n, cost=n * 4, observed_lift_ratio=lift)
            self.pilot_history.append(dict(kwargs, **result))
            return result
    return FakeEnv()


def test_weighted_actual_samples_and_normalization():
    env = fake_env(lambda k, i: (20, -0.02) if i == 0 else (120, 0.2))
    result = run_adaptive_pilots(env, [candidate()], config={"confirmation_pilots": 1})[0]
    mean = (20 * -0.02 + 120 * 0.2) / 140 / 0.65
    assert result["mean_lift"] == pytest.approx(mean)
    assert result["pilot_n"] == 140
    assert result["se_lift"] == pytest.approx(0.804 / (0.65 * math.sqrt(140)))
    assert result["safe_lift"] == pytest.approx(mean - 1.28 * result["se_lift"])
    assert result["pilots"][0]["observed_lift_ratio"] == -0.02


@pytest.mark.parametrize("kwargs", [{"budget": 0}, {"budget": 39}, {"contacts": 1009}, {"slots": 0}])
def test_exhausted_resources(kwargs):
    env = fake_env(**kwargs)
    result = run_adaptive_pilots(env, [candidate()])
    assert not env.pilot_history
    assert result[0]["se_lift"] is None


def test_tiny_and_empty():
    env = fake_env()
    assert run_adaptive_pilots(env, []) == []
    assert run_adaptive_pilots(env, [candidate(size=9)])[0]["status"] == "untested"
    assert not env.pilot_history


@pytest.mark.parametrize("failure", [RuntimeError("bad cell"), ValueError("bad target"), (80, float("nan")), (80, float("inf"))])
def test_errors_do_not_fabricate_evidence(failure):
    trace = []
    env = fake_env(lambda k, i: failure)
    record = run_adaptive_pilots(env, [candidate()], trace=trace)[0]
    assert record["status"] == "error"
    assert record["pilot_count"] == 0 and record["mean_lift"] is None
    json.dumps(trace, allow_nan=False)


def test_unknown_candidate():
    result = run_adaptive_pilots(fake_env(), [candidate(target="unknown")])
    assert result[0]["status"] == "error"


def test_failure_consumes_resources_and_is_reread():
    env = fake_env()
    calls = []
    def fail(**kwargs):
        calls.append(kwargs)
        env.remaining_contacts = 1000
        raise RuntimeError("Adapter failed after spending contacts")
    env.run_pilot = fail
    run_adaptive_pilots(env, [candidate(), candidate("c")])
    assert len(calls) == 1


@pytest.mark.parametrize("cfg", [{}, {"pilot_contact_cap": 100}, {"pilot_money_cap": 380},
                                  {"pilot_cap": 1}, {"final_contact_reserve": 14970}])
def test_caps(cfg):
    env = fake_env()
    run_adaptive_pilots(env, [candidate(str(i)) for i in range(20)], config=cfg)
    assert 15000 - env.remaining_contacts <= cfg.get("pilot_contact_cap", 3000)
    assert 100000 - env.remaining_budget <= cfg.get("pilot_money_cap", 12000)
    assert len(env.pilot_history) <= cfg.get("pilot_cap", 20)
    assert env.remaining_contacts >= cfg.get("final_contact_reserve", 1000)


def confirmation_path(first_lift):
    env = fake_env(lambda k, i: (k["n_customers"], first_lift if k["filter_current_tariff"] == "a" else 0.05))
    trace = []
    records = run_adaptive_pilots(env, [candidate(score=1000), candidate("c")],
                                 config={"confirmation_pilots": 1}, trace=trace)
    return records, [e["candidate_id"] for e in trace if e["event"] == "confirmation"]


def test_historical_reversal_changes_confirmation_path():
    negative, path_negative = confirmation_path(-0.9)
    _, path_ambiguous = confirmation_path(0.0)
    assert negative[0]["mean_lift"] < 0 and negative[0]["pilot_count"] == 1
    assert path_negative != path_ambiguous


def test_large_ambiguous_outranks_tiny_certain_winner():
    env = fake_env(lambda k, i: (min(20, k["n_customers"]), 0.8) if k["filter_current_tariff"] == "a" else (k["n_customers"], 0.05))
    records = run_adaptive_pilots(env, [candidate(size=20, arpu=20000), candidate("c")],
                                 config={"confirmation_pilots": 1})
    assert records[1]["pilot_count"] == 2
    assert records[0]["pilot_count"] == 1


def test_dominated_target_not_confirmed():
    env = fake_env(lambda k, i: (k["n_customers"], -0.5 if k["target_tariff"] == "b" else 0.5))
    records = run_adaptive_pilots(env, [candidate(), candidate(target="c", score=90)])
    assert records[0]["pilot_count"] == 1
    assert records[1]["pilot_count"] > 1


def test_all_negative_stops_early():
    env = fake_env(lambda k, i: (k["n_customers"], -0.8))
    records = run_adaptive_pilots(env, [candidate(), candidate("c")])
    assert all(r["pilot_count"] == 1 and r["mean_lift"] < 0 for r in records)


def test_repeatability_and_inputs_unchanged():
    candidates = [candidate(), candidate("c")]
    original = copy.deepcopy(candidates)
    one = run_adaptive_pilots(fake_env(), candidates)
    two = run_adaptive_pilots(fake_env(), candidates)
    assert one == two and candidates == original
    json.dumps(one, allow_nan=False)


def test_alternative_reserved():
    candidates = [candidate(str(i), score=100-i) for i in range(12)]
    candidates.append(candidate("0", target="c", score=20))
    result = run_adaptive_pilots(fake_env(), candidates, config={"confirmation_pilots": 0})
    assert result[-1]["status"] == "measured"
    assert sum(r["pilot_count"] for r in result) == 10


def test_failed_confirmation_preserves_measurement():
    env = fake_env(lambda k, i: (80, 0.1) if i == 0 else RuntimeError("bad repeat"))
    record = run_adaptive_pilots(env, [candidate()])[0]
    assert record["status"] == "measured"
    assert record["pilot_count"] == 1


def test_residual_budget_clamps_request():
    env = fake_env(budget=44)
    record = run_adaptive_pilots(env, [candidate()])[0]
    assert record["pilot_n"] == 11
    assert env.remaining_budget == 0


def test_catalog_exploration_and_weak_effects():
    alternative = dict(candidate("a", target="c", score=10), source="catalog", history_count=0)
    env = fake_env(lambda k, i: (k["n_customers"], 0.5 if k["target_tariff"] == "c" else -0.01))
    result = run_adaptive_pilots(env, [candidate(score=1000), alternative])
    assert result[1]["safe_lift"] > 0
    assert result[0]["pilot_count"] == 1  # Dominated despite historical ranking.


def test_catalog_only_shortlist_preserves_cell_diversity():
    candidates = [dict(candidate("a", target=t, score=100-i), source="catalog")
                  for i, t in enumerate(["b", "c", "d"])]
    candidates += [dict(candidate(cell, score=10), source="catalog") for cell in ["c", "d"]]
    result = run_adaptive_pilots(fake_env(), candidates,
                                 config={"exploration_pilots": 3, "confirmation_pilots": 0})
    measured = [r for r in result if r["status"] == "measured"]
    assert len({r["cell_id"] for r in measured}) >= 2


def test_missing_ratio_still_accounts_for_reported_resources():
    env = fake_env()
    calls = []
    def invalid(**kwargs):
        calls.append(kwargs)
        # Adapter reports charges before its public counters catch up.
        return {"n_customers": 80, "cost": 320}
    env.run_pilot = invalid
    result = run_adaptive_pilots(env, [candidate(), candidate("c")],
                                 config={"pilot_contact_cap": 80})
    assert len(calls) == 1
    assert result[0]["status"] == "error"
    assert result[1]["status"] == "untested"
