"""Authored local fixtures only; never external API calls."""
import json

import pandas as pd
import pytest

import agent
from llm_advisor import (AdviceUnavailable, FrozenAdvisor, freeze_advice,
                         get_advice)
from test_llm_advisor import FakeAdvisor, candidate, environment, make_pipeline_env


def policy_fixture(path):
    advisor = FakeAdvisor({"candidate_ids": ["a|MID|b"], "reason": "Authored test fixture"})
    advisor.source = "openai"
    advice = get_advice(environment(), [candidate()], mode="assist", advisor=advisor)
    path.write_text(json.dumps(freeze_advice(advice)))
    return advice


def test_frozen_replay_requires_no_key_sdk_or_network(tmp_path, monkeypatch):
    path = tmp_path / "policy.json"
    advice = policy_fixture(path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    frozen = FrozenAdvisor(path)
    assert frozen.recommend(advice["context"])["recommendation"] == advice["recommendation"]
    assert frozen.request_count == 0
    result = get_advice(environment(), [candidate()], mode="replay", config={"llm_policy_path": path})
    assert result["source"] == "frozen_openai" and result["status"] == "ok"
    assert result["applied_candidate_ids"] == ["a|MID|b"]


@pytest.mark.parametrize("mutation", ["budget", "catalog", "audience", "history"])
def test_shifted_public_context_rejects_stale_policy(tmp_path, mutation):
    path = tmp_path / "policy.json"
    policy_fixture(path)
    env, cs = environment(), [candidate()]
    if mutation == "budget":
        env.remaining_budget -= 1
    elif mutation == "catalog":
        env.tariffs.loc[0, "price_tariff"] += 1
    elif mutation == "audience":
        cs[0]["audience_size"] += 1
    else:
        cs[0]["history_count"] += 1
    result = get_advice(env, cs, mode="replay", config={"llm_policy_path": path})
    assert result["error_code"] == "policy_context_mismatch"
    assert result["applied_candidate_ids"] == []


@pytest.mark.parametrize("field,value", [("instructions_hash", "stale"), ("source", "preview"),
    ("prompt_version", "stale"), ("schema_version", "2"), ("model", None)])
def test_incompatible_policy_is_rejected(tmp_path, field, value):
    path = tmp_path / "policy.json"
    policy_fixture(path)
    policy = json.loads(path.read_text())
    policy[field] = value
    path.write_text(json.dumps(policy))
    with pytest.raises(AdviceUnavailable, match="invalid_policy"):
        FrozenAdvisor(path)


def test_frozen_invalid_candidate_is_not_trusted(tmp_path):
    path = tmp_path / "policy.json"
    policy_fixture(path)
    policy = json.loads(path.read_text())
    policy["recommendation"]["candidate_ids"] = ["unknown"]
    path.write_text(json.dumps(policy))
    result = get_advice(environment(), [candidate()], mode="replay", config={"llm_policy_path": path})
    assert result["error_code"] == "unknown_candidate"


def test_preview_cannot_be_frozen_as_live_policy(tmp_path):
    advice = policy_fixture(tmp_path / "policy.json")
    advice["source"] = "injected"
    with pytest.raises(AdviceUnavailable, match="invalid_policy"):
        freeze_advice(advice)


def test_default_agent_replays_model_decision_and_falls_back_on_changed_data(tmp_path, monkeypatch):
    monkeypatch.setattr("agent.pd.read_csv", lambda *a, **k: pd.DataFrame())
    path = tmp_path / "policy.json"
    monkeypatch.setattr(agent, "DEFAULT_POLICY_PATH", path)
    cfg = dict(confirmation_pilots=0, final_contact_reserve=0, exploration_pilots=1)
    authored = FakeAdvisor({"candidate_ids": ["a|MID|c"], "reason": "Authored alternative"})
    authored.source = "openai"
    live_fixture = agent.Agent({**cfg, "llm_mode": "assist"}, advisor=authored)
    expected = live_fixture.act(make_pipeline_env())
    path.write_text(json.dumps(freeze_advice(live_fixture.last_trace["llm"])))
    instance = agent.Agent(cfg)
    assert instance.act(make_pipeline_env()) == expected
    assert instance.last_trace["llm"]["source"] == "frozen_openai"
    assert expected[0]["target_tariff"] == "c"
    changed = make_pipeline_env()
    changed.remaining_budget += 1
    actual = instance.act(changed)
    off = agent.Agent({**cfg, "llm_mode": "off"})
    changed_off = make_pipeline_env()
    changed_off.remaining_budget += 1
    assert actual == off.act(changed_off)
    assert instance.last_trace["llm"]["error_code"] == "policy_context_mismatch"


def test_missing_and_broken_policy_fall_back(tmp_path):
    path = tmp_path / "missing.json"
    for content in (None, "{broken"):
        if content:
            path.write_text(content)
        result = get_advice(environment(), [candidate()], mode="replay", config={"llm_policy_path": path})
        assert result["status"] == "fallback" and result["error_code"] == "invalid_policy"


def test_replay_cannot_call_an_injected_live_adapter(tmp_path):
    path = tmp_path / 'policy.json'
    policy_fixture(path)
    must_not_call = FakeAdvisor(error=AssertionError('No live adapter in replay'))
    result = get_advice(environment(), [candidate()], mode='replay', advisor=must_not_call,
                        config={'llm_policy_path': path})
    assert result['status'] == 'ok' and must_not_call.contexts == []


def test_unseen_tariff_labels_and_empty_history_use_catalog_fallback(tmp_path, monkeypatch):
    path = tmp_path / 'policy.json'
    policy_fixture(path)
    monkeypatch.setattr(agent, 'DEFAULT_POLICY_PATH', path)
    monkeypatch.setattr('agent.pd.read_csv', lambda *a, **k: pd.DataFrame())
    env = make_pipeline_env()
    env.tariffs['tariff_plan_code'] = ['new-a', 'new-b', 'new-c']
    env.customer_profile['current_tariff'] = 'new-a'
    instance = agent.Agent({'confirmation_pilots': 0, 'final_contact_reserve': 0})
    campaigns = instance.act(env)
    assert campaigns and all(c['target_tariff'] in {'new-b', 'new-c'} for c in campaigns)
    assert instance.last_trace['llm']['error_code'] == 'policy_context_mismatch'
    assert all(c['source'] == 'catalog' for c in instance.last_trace['candidates'])


def test_freeze_cli_requires_live_trace_and_explicit_replace(tmp_path):
    from tools.freeze_llm_policy import main
    advice = policy_fixture(tmp_path / 'fixture.json')
    trace, output = tmp_path / 'trace.json', tmp_path / 'installed.json'
    trace.write_text(json.dumps({'llm': advice}))
    assert main(['--trace', str(trace), '--output', str(output)]) == 0
    original = output.read_bytes()
    with pytest.raises(SystemExit):
        main(['--trace', str(trace), '--output', str(output)])
    assert output.read_bytes() == original
    advice['source'] = 'injected'
    trace.write_text(json.dumps({'llm': advice}))
    with pytest.raises(SystemExit):
        main(['--trace', str(trace), '--output', str(output), '--replace'])
    assert output.read_bytes() == original
