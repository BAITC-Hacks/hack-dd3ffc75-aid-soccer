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
