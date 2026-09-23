"""All model responses are authored fixtures; these tests never use a live API."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from agent import Agent
from llm_advisor import (AdviceUnavailable, OpenAIAdvisor, build_context, context_hash,
                         get_advice, recommendation_schema, validate_recommendation)
from pilot_policy import _shortlist


def candidate(current="a", target="b", score=100.0):
    return dict(candidate_id=f"{current}|MID|{target}", cell_id=f"{current}|MID",
                current_tariff=current, arpu_segment="MID", target_tariff=target,
                history_count=0, prior_lift=0.0, positive_rate=0.5, source="catalog",
                audience_size=30, served_size=30, served_arpu=60000.0,
                audience_arpu=60000.0, prior_score=score, customer_secret="DO_NOT_SEND")


def environment():
    return SimpleNamespace(
        tariffs=pd.DataFrame(dict(tariff_plan_code=["a", "b", "c"], price_tariff=[1000, 1200, 1400],
                                 private_note=["DO_NOT_SEND"] * 3)),
        customer_profile=pd.DataFrame(dict(ID_NUMBER=range(30), current_tariff=["a"] * 30,
            arpu_segment=["MID"] * 30, predicted_arpu=[2000.0] * 30)),
        channels={"push": dict(cost_per_contact=0, conversion_multiplier=.5),
                  "sms": dict(cost_per_contact=4, conversion_multiplier=.65)},
        remaining_budget=1000.0, remaining_contacts=100, pilots_left=20, pilot_history=[])


class FakeAdvisor:
    model = "test-model"

    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.contexts = []

    def recommend(self, context):
        self.contexts.append(deepcopy(context))
        if self.error:
            raise self.error
        return {"recommendation": self.payload or {"candidate_ids": [], "reason": "No proposal"},
                "usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}}


def test_context_is_aggregate_whitelist_bounded_and_deterministic():
    env = environment()
    before = deepcopy(env.tariffs)
    cs = [candidate("a", "b"), candidate("b", "c")]
    context = build_context(env, cs)
    text = json.dumps(context, allow_nan=False)
    assert "DO_NOT_SEND" not in text and "ID_NUMBER" not in text
    assert context_hash(context) == context_hash(build_context(env, list(reversed(cs))))
    assert len(build_context(env, [candidate(f"x{i}", "b") for i in range(100)])["candidates"]) == 40
    pd.testing.assert_frame_equal(env.tariffs, before)
    assert recommendation_schema(context)["additionalProperties"] is False


@pytest.mark.parametrize("payload", [
    {"candidate_ids": ["missing"], "reason": "bad"},
    {"candidate_ids": ["a|MID|b"] * 2, "reason": "bad"},
    {"candidate_ids": ["a|MID|b", "a|MID|c"], "reason": "same cell"},
    {"candidate_ids": [], "reason": ""},
    {"candidate_ids": [], "reason": "x" * 1001},
    {"candidate_ids": "a|MID|b", "reason": "wrong type"},
    {"candidate_ids": [], "reason": "bad", "prior_lift": 999},
])
def test_invalid_advice_falls_back_without_changing_candidates(payload):
    cs = [candidate(), candidate(target="c")]
    before = deepcopy(cs)
    result = get_advice(environment(), cs, mode="assist", advisor=FakeAdvisor(payload))
    assert result["status"] == "fallback"
    assert result["applied_candidate_ids"] == []
    assert result["recommendation"] is None
    assert cs == before


def test_shadow_records_but_assist_can_apply_valid_advice():
    payload = {"candidate_ids": ["a|MID|b"], "reason": "Test a plausible unseen transition"}
    for mode, expected in [("shadow", []), ("assist", payload["candidate_ids"])]:
        result = get_advice(environment(), [candidate()], mode=mode, advisor=FakeAdvisor(payload))
        assert result["status"] == "ok"
        assert result["applied_candidate_ids"] == expected
        assert result["usage"]["total_tokens"] == 30
        json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("error", [TimeoutError("SECRET_KEY must not appear"),
                                    RuntimeError("SECRET_KEY provider body"),
                                    AdviceUnavailable("SECRET_KEY")])
def test_exceptions_are_redacted_and_preserve_fallback(error):
    result = get_advice(environment(), [candidate()], mode="assist", advisor=FakeAdvisor(error=error))
    assert result["status"] == "fallback"
    assert "SECRET_KEY" not in json.dumps(result)
    assert not result["applied_candidate_ids"]


def test_missing_key_and_no_candidates_make_no_request(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = get_advice(environment(), [candidate()], mode="assist")
    assert result["error_code"] == "missing_api_key"
    advisor = FakeAdvisor()
    result = get_advice(environment(), [], mode="assist", advisor=advisor)
    assert result["status"] == "skipped" and advisor.contexts == []


def test_shortlist_honors_two_ids_without_changing_scores():
    cs = [candidate(f"t{i}", score=100-i) for i in range(12)]
    before = deepcopy(cs)
    preferred = [cs[11]["candidate_id"], cs[10]["candidate_id"], cs[9]["candidate_id"]]
    selected = _shortlist(cs, 5, preferred)
    assert [c["candidate_id"] for c in selected[-2:]] == preferred[:2]
    assert len(selected) == 5 and cs == before
    assert _shortlist(cs, 0, preferred) == []
    assert _shortlist(cs, 5, ["missing"]) == _shortlist(cs, 5)


def make_pipeline_env():
    env = environment()
    def run_pilot(**kwargs):
        n = min(kwargs["n_customers"], 12)
        env.remaining_contacts -= n
        env.remaining_budget -= 4 * n
        env.pilots_left -= 1
        env.pilot_history.append(kwargs)
        return dict(n_customers=n, cost=n * 4, observed_lift_ratio=.8)
    env.run_pilot = run_pilot
    return env


def test_agent_off_never_calls_advisor_and_shadow_preserves_plan(monkeypatch):
    monkeypatch.setattr("agent.pd.read_csv", lambda *a, **k: pd.DataFrame())
    config = dict(llm_mode="off", confirmation_pilots=0, final_contact_reserve=0, exploration_pilots=1)
    failing = FakeAdvisor(error=AssertionError("must not call in off mode"))
    baseline = Agent(config, advisor=failing)
    expected = baseline.act(make_pipeline_env())
    assert failing.contexts == [] and "llm" not in baseline.last_trace
    advisor = FakeAdvisor({"candidate_ids": ["a|MID|c"], "reason": "Alternative"})
    shadow = Agent({**config, "llm_mode": "shadow"}, advisor=advisor)
    assert shadow.act(make_pipeline_env()) == expected
    assist = Agent({**config, "llm_mode": "assist"}, advisor=advisor)
    campaigns = assist.act(make_pipeline_env())
    assert campaigns[0]["target_tariff"] == "c"
    assert assist.last_trace["observations"][0]["prior_lift"] == 0
    assert config == dict(llm_mode="off", confirmation_pilots=0, final_contact_reserve=0, exploration_pilots=1)
    first = assist.last_trace
    advisor.payload = {"candidate_ids": ["unknown"], "reason": "Invalid second response"}
    assert assist.act(make_pipeline_env()) == expected
    assert assist.last_trace["llm"]["status"] == "fallback"
    assert first["llm"]["status"] == "ok"


def test_malformed_optional_history_becomes_catalog_hypotheses(monkeypatch):
    monkeypatch.setattr("agent.pd.read_csv", lambda *a, **k: pd.DataFrame({"wrong_column": [123]}))
    instance = Agent({"confirmation_pilots": 0, "final_contact_reserve": 0})
    assert instance.act(make_pipeline_env())
    assert all(c["source"] == "catalog" for c in instance.last_trace["candidates"])
    assert instance.last_trace["events"][0]["event"] == "history_unavailable"


@pytest.mark.parametrize("status,output,error", [
    ("incomplete", "{}", "incomplete_response"),
    ("completed", "", "empty_or_oversized_response"),
    ("completed", "not json", "invalid_json"),
])
def test_adapter_handles_incomplete_refused_and_invalid_json(status, output, error):
    response = SimpleNamespace(status=status, output_text=output)
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: response))
    result = get_advice(environment(), [candidate()], mode="assist", advisor=OpenAIAdvisor(client=client))
    assert result["error_code"] == error


def test_real_sdk_request_shape_with_mock_http_transport():
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    captured = []
    def handle(request):
        captured.append(json.loads(request.content))
        assert request.url.path == "/v1/responses"
        return httpx.Response(200, json={
            "id": "resp_fixture", "object": "response", "created_at": 1, "status": "completed",
            "model": "gpt-4.1-mini-2025-04-14", "error": None, "incomplete_details": None,
            "output": [{"type": "message", "id": "msg_fixture", "role": "assistant",
                        "status": "completed", "content": [{"type": "output_text", "annotations": [],
                        "text": json.dumps({"candidate_ids": ["a|MID|b"], "reason": "Fixture"})}]}],
            "usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}})
    with openai.OpenAI(api_key="unit-test-placeholder", max_retries=0,
                       http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
        result = get_advice(environment(), [candidate()], mode="assist", advisor=OpenAIAdvisor(client=client))
    assert result["status"] == "ok"
    assert len(captured) == 1
    assert captured[0]["store"] is False
    assert captured[0]["text"]["format"]["strict"] is True
    assert captured[0]["max_output_tokens"] == 800
    assert "unit-test-placeholder" not in json.dumps(captured)


def test_existing_nominations_preserve_pilot_order():
    cs = [candidate(f"t{i}", score=100-i) for i in range(12)]
    assert _shortlist(cs, 10, [cs[4]["candidate_id"], cs[0]["candidate_id"]]) == _shortlist(cs, 10)


def test_new_nominations_change_membership_and_keep_existing_nominee():
    cs = [candidate(f"t{i}", score=100-i) for i in range(12)]
    ids = [cs[4]["candidate_id"], cs[11]["candidate_id"]]
    selected = _shortlist(cs, 5, ids)
    assert all(value in [c["candidate_id"] for c in selected] for value in ids)
    assert selected != _shortlist(cs, 5)
