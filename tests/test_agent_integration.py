"""Wiring tests use local doubles until independently owned modules arrive."""
import json
import sys
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

import agent


def install_modules(monkeypatch, generate, portfolio):
    candidate_module = ModuleType("candidate_engine")
    candidate_module.generate_candidates = generate
    portfolio_module = ModuleType("portfolio_optimizer")
    portfolio_module.build_campaigns = portfolio
    monkeypatch.setitem(sys.modules, "candidate_engine", candidate_module)
    monkeypatch.setitem(sys.modules, "portfolio_optimizer", portfolio_module)


def environment():
    return SimpleNamespace(customer_profile=pd.DataFrame(dict(
        ID_NUMBER=[1, 2], current_tariff=["a", "a"], arpu_segment=["HIGH", "HIGH"],
        predicted_arpu=[1000, 1000])), tariffs=pd.DataFrame({"tariff_plan_code": ["a", "b"]}),
        channels={"push": {"cost_per_contact": 0, "conversion_multiplier": 0.5}},
        remaining_budget=100, remaining_contacts=100, pilots_left=20)


def campaign():
    return dict(campaign_name="test", filter_current_tariff="a", filter_arpu_segment="HIGH",
                target_tariff="b", channel="push")


def test_pipeline_config_trace_and_reset(monkeypatch):
    calls = []
    def generate(profile, history, tariffs, *, config, trace):
        calls.append("candidates")
        assert config["risk_z"] == 1.1
        return [{"candidate_id": "test"}]
    def pilots(env, candidates, *, config, trace):
        calls.append("pilots")
        assert candidates == [{"candidate_id": "test"}]
        env.remaining_budget -= 40
        env.remaining_contacts -= 10
        env.pilots_left -= 1
        return [{"candidate_id": "test", "status": "measured"}]
    def portfolio(env, observations, *, config, trace):
        calls.append("portfolio")
        assert observations[0]["status"] == "measured"
        return [campaign()]
    install_modules(monkeypatch, generate, portfolio)
    monkeypatch.setattr(agent, "run_adaptive_pilots", pilots)
    instance = agent.Agent({"risk_z": 1.1})
    assert instance.act(environment()) == [campaign()]
    old = instance.last_trace
    assert old["summary"]["pilot_contacts"] == 10
    assert old["summary"]["remaining_contacts_after_plan"] == 88
    instance.act(environment())
    assert old == instance.last_trace and old is not instance.last_trace
    assert calls == ["candidates", "pilots", "portfolio"] * 2
    json.dumps(instance.last_trace, allow_nan=False)


@pytest.mark.parametrize("failure", [OSError("missing"), pd.errors.ParserError("bad csv"),
                                    pd.errors.EmptyDataError("empty"), UnicodeError("encoding")])
def test_missing_history_is_recoverable(monkeypatch, failure):
    def read(*args, **kwargs):
        raise failure
    def generate(profile, history, tariffs, **kwargs):
        assert history.empty
        return []
    install_modules(monkeypatch, generate, lambda *a, **k: [])
    monkeypatch.setattr(agent.pd, "read_csv", read)
    instance = agent.Agent()
    assert instance.act(environment()) == []
    assert instance.last_trace["summary"]["status"] == "infeasible"
    assert instance.last_trace["events"][0]["event"] == "history_unavailable"


def test_programming_error_is_not_hidden(monkeypatch):
    def fail(*args, **kwargs):
        raise TypeError("Programming error")
    install_modules(monkeypatch, fail, lambda *a, **k: [])
    instance = agent.Agent()
    with pytest.raises(TypeError, match="Programming error"):
        instance.act(environment())
    assert instance.last_trace["summary"]["status"] == "error"


def test_fallback_status(monkeypatch):
    def portfolio(*args, trace, **kwargs):
        trace.append(dict(stage="portfolio", event="emergency_fallback", candidate_id=None,
                          reason="No positive conservative action", details={}))
        return [campaign()]
    install_modules(monkeypatch, lambda *a, **k: [], portfolio)
    instance = agent.Agent()
    instance.act(environment())
    assert instance.last_trace["summary"]["status"] == "fallback"


def test_bad_portfolio_rejected(monkeypatch):
    install_modules(monkeypatch, lambda *a, **k: [], lambda *a, **k: [campaign()] * 11)
    instance = agent.Agent()
    with pytest.raises(ValueError, match="Portfolio exceeds"):
        instance.act(environment())
    assert instance.last_trace["summary"]["status"] == "error"


def test_missing_dependency_is_diagnostic(monkeypatch):
    monkeypatch.setitem(sys.modules, "candidate_engine", None)
    instance = agent.Agent()
    with pytest.raises(ModuleNotFoundError):
        instance.act(environment())
    assert instance.last_trace["summary"]["status"] == "error"
    assert instance.last_trace["events"][0]["event"] == "missing_dependency"


def test_history_path_is_relative_to_agent(monkeypatch):
    paths = []
    def read(path):
        paths.append(path)
        return pd.DataFrame()
    install_modules(monkeypatch, lambda *a, **k: [], lambda *a, **k: [])
    monkeypatch.setattr(agent.pd, "read_csv", read)
    monkeypatch.chdir(agent.Path(agent.__file__).resolve().parent / "data")
    agent.Agent().act(environment())
    assert paths == [agent.Path(agent.__file__).resolve().parent / "data" / "change_tariff.csv"]


def test_portfolio_failure_retains_actual_pilot_spend(monkeypatch):
    def pilots(env, *args, **kwargs):
        env.remaining_budget -= 40
        env.remaining_contacts -= 10
        env.pilots_left -= 1
        return []
    def portfolio(*args, **kwargs):
        raise ValueError("Portfolio failure")
    install_modules(monkeypatch, lambda *a, **k: [], portfolio)
    monkeypatch.setattr(agent, "run_adaptive_pilots", pilots)
    instance = agent.Agent()
    with pytest.raises(ValueError, match="Portfolio failure"):
        instance.act(environment())
    assert instance.last_trace["summary"]["remaining_budget_after_plan"] == 60
    assert instance.last_trace["summary"]["remaining_contacts_after_plan"] == 90
    assert instance.last_trace["summary"]["status"] == "error"


def test_pilot_failure_retains_actual_spend(monkeypatch):
    def pilots(env, *args, **kwargs):
        env.remaining_budget -= 40
        env.remaining_contacts -= 10
        env.pilots_left -= 1
        raise TypeError("Unexpected adapter bug after charging")
    install_modules(monkeypatch, lambda *a, **k: [], lambda *a, **k: [])
    monkeypatch.setattr(agent, "run_adaptive_pilots", pilots)
    instance = agent.Agent()
    with pytest.raises(TypeError, match="Unexpected adapter bug"):
        instance.act(environment())
    assert instance.last_trace["summary"]["pilot_contacts"] == 10
    assert instance.last_trace["summary"]["pilot_cost"] == 40
    assert instance.last_trace["summary"]["pilot_count"] == 1
    assert instance.last_trace["summary"]["remaining_budget_after_plan"] == 60
    assert instance.last_trace["summary"]["status"] == "error"


def test_repeated_act_uses_fresh_real_pilot_observations(monkeypatch):
    c = dict(candidate_id="a|HIGH|b", cell_id="a|HIGH", current_tariff="a",
             arpu_segment="HIGH", target_tariff="b", history_count=100,
             prior_lift=0.9, positive_rate=0.8, audience_size=20, audience_arpu=20000.0,
             served_size=20, served_arpu=20000.0, prior_score=100.0, source="history")
    install_modules(monkeypatch, lambda *a, **k: [c], lambda *a, **k: [])
    monkeypatch.setattr(agent.pd, "read_csv", lambda *a, **k: pd.DataFrame())
    def make_env(lift):
        env = environment()
        env.remaining_contacts = 2000
        env.channels = {"sms": {"cost_per_contact": 4, "conversion_multiplier": 0.65}}
        env.customer_profile = pd.DataFrame(dict(ID_NUMBER=range(20), current_tariff=["a"] * 20,
                                                arpu_segment=["HIGH"] * 20, predicted_arpu=[1000] * 20))
        def run_pilot(*, target_tariff, channel, n_customers, filter_arpu_segment, filter_current_tariff):
            audience = env.customer_profile[(env.customer_profile.current_tariff == filter_current_tariff) &
                                             (env.customer_profile.arpu_segment == filter_arpu_segment)]
            assert target_tariff == "b" and channel == "sms"
            n = min(n_customers, len(audience))
            env.remaining_contacts -= n
            env.remaining_budget -= n * 4
            env.pilots_left -= 1
            return dict(n_customers=n, cost=n * 4, observed_lift_ratio=lift)
        env.run_pilot = run_pilot
        return env
    instance = agent.Agent({"confirmation_pilots": 0})
    instance.act(make_env(0.2))
    first = instance.last_trace
    instance.act(make_env(-0.4))
    assert first["observations"][0]["mean_lift"] == pytest.approx(0.2 / 0.65)
    assert instance.last_trace["observations"][0]["mean_lift"] == pytest.approx(-0.4 / 0.65)
    assert instance.last_trace["observations"][0]["pilot_n"] == 20
    assert instance.last_trace["summary"]["pilot_count"] == 1


def test_benchmark_detects_swallowed_exception(monkeypatch):
    from tools import benchmark
    def evaluator(instance, **kwargs):
        try:
            instance.act(environment())
        except ValueError:
            pass
        return {"net_arpu_gain": 123, "n_pilots": 1}
    class Broken:
        def act(self, env):
            raise ValueError("broken")
    monkeypatch.setattr(benchmark, "evaluate_agent", evaluator)
    row, _ = benchmark.run_one(Broken(), 0, "test")
    assert row["failure"] == "ValueError: broken"
    assert benchmark.summarize([row])["successful_runs"] == 0


def test_benchmark_final_count_excludes_pilots(monkeypatch):
    from tools import benchmark
    class Healthy:
        def act(self, env):
            return [campaign()]
    def evaluator(instance, **kwargs):
        instance.act(environment())
        return {"net_arpu_gain": 10, "n_campaigns": 16, "n_pilots": 15}
    monkeypatch.setattr(benchmark, "evaluate_agent", evaluator)
    row, _ = benchmark.run_one(Healthy(), 0, "test")
    assert row["final_campaign_count"] == 1
    assert row["pilots"] == 15
    assert row["failure"] is None


def test_provided_dataset_public_pilot_smoke():
    # Black-box evaluator supplies env. No mock model/oracle is accessed here.
    from local_eval import evaluate_agent
    from pilot_policy import run_adaptive_pilots
    class PilotProbe:
        records = None
        def act(self, env):
            cells = env.customer_profile.groupby(["current_tariff", "arpu_segment"], observed=True).size()
            current, segment = cells.idxmax()
            audience = env.customer_profile[(env.customer_profile.current_tariff == current) &
                                             (env.customer_profile.arpu_segment == segment)]
            served = audience.assign(_numeric_id=pd.to_numeric(audience.ID_NUMBER)).sort_values("_numeric_id").head(5000)
            target = next(t for t in env.tariffs.tariff_plan_code if t != current)
            candidates = [dict(candidate_id=f"{current}|{segment}|{target}", cell_id=f"{current}|{segment}",
                               current_tariff=current, arpu_segment=segment, target_tariff=target,
                               history_count=0, prior_lift=0.0, positive_rate=0.5,
                               audience_size=len(audience), audience_arpu=float(audience.predicted_arpu.sum()),
                               served_size=len(served), served_arpu=float(served.predicted_arpu.sum()),
                               prior_score=1.0, source="catalog")]
            self.records = run_adaptive_pilots(env, candidates, config={"confirmation_pilots": 0})
            return []  # This probe validates pilots only, not the production portfolio.
    probe = PilotProbe()
    result = evaluate_agent(probe, seed=42, verbose=False)
    assert result["n_pilots"] == 1
    assert probe.records[0]["pilot_n"] == 80


def test_benchmark_rejects_plan_that_scorer_would_truncate():
    from tools.benchmark import validate_plan
    env = environment()
    env.remaining_contacts = 1
    with pytest.raises(ValueError, match="residual resources"):
        validate_plan(env, [campaign()])


def test_benchmark_rejects_unsupported_campaign_keys():
    from tools.benchmark import validate_plan
    with pytest.raises(ValueError, match="unsupported keys"):
        validate_plan(environment(), [dict(campaign(), explicit_ids=[1])])


def test_benchmark_rejects_duplicate_cells_without_changing_starter():
    from tools.benchmark import validate_plan
    with pytest.raises(ValueError, match="distinct-cell"):
        validate_plan(environment(), [campaign(), campaign()])
    baseline = validate_plan(environment(), [campaign(), campaign()], strict=False)
    assert not baseline["distinct_final_cells"]
    assert baseline["final_contacts"] == 4


def test_real_three_module_pipeline_uses_actual_samples_and_fresh_evidence(monkeypatch):
    """Synthetic pilot outcomes live in the test closure, never on env."""
    monkeypatch.setattr(agent.pd, "read_csv", lambda *a, **k: pd.DataFrame())

    def make_env(ratio):
        env = environment()
        env.customer_profile = pd.DataFrame(dict(
            ID_NUMBER=range(30), current_tariff=["a"] * 30, arpu_segment=["MID"] * 30,
            predicted_arpu=[2000.0] * 30))
        env.tariffs = pd.DataFrame(dict(tariff_plan_code=["a", "b"], price_tariff=[1000, 1200]))
        env.channels = {"push": {"cost_per_contact": 0, "conversion_multiplier": 0.5},
                        "sms": {"cost_per_contact": 4, "conversion_multiplier": 0.65}}
        env.remaining_budget = 1000
        env.remaining_contacts = 100
        env.pilot_history = []

        def run_pilot(**request):
            assert request["target_tariff"] == "b" and request["channel"] == "sms"
            n = min(request["n_customers"], 12)
            env.remaining_budget -= n * 4
            env.remaining_contacts -= n
            env.pilots_left -= 1
            result = dict(n_customers=n, cost=n * 4, observed_lift_ratio=ratio)
            env.pilot_history.append(result)
            return result
        env.run_pilot = run_pilot
        return env

    instance = agent.Agent({"confirmation_pilots": 0, "final_contact_reserve": 0})
    positive = instance.act(make_env(0.8))
    first = instance.last_trace
    assert positive[0]["channel"] == "sms"
    assert first["summary"]["pilot_contacts"] == 12
    assert first["observations"][0]["mean_lift"] == pytest.approx(0.8 / 0.65)
    negative = instance.act(make_env(-0.8))
    assert negative[0]["channel"] == "push"
    assert instance.last_trace["summary"]["status"] == "fallback"
    assert instance.last_trace["observations"][0]["pilot_n"] == 12
    assert instance.last_trace["observations"][0]["mean_lift"] < 0
    assert first["summary"]["status"] == "ok"
    json.dumps(instance.last_trace, allow_nan=False)


def test_starter_empty_final_plan_remains_evaluable_and_adaptive_rejects_it():
    from tools.benchmark import validate_plan
    baseline = validate_plan(environment(), [], strict=False)
    assert baseline["final_contacts"] == 0
    assert baseline["final_cost"] == 0
    with pytest.raises(ValueError, match="1–10"):
        validate_plan(environment(), [])
