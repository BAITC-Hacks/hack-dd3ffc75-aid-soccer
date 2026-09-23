import json

import pytest

from tools import llm_eval


def test_config_check_never_prints_key_or_calls_api(monkeypatch, capsys):
    monkeypatch.setattr(llm_eval, "_load_environment", lambda root: True)
    monkeypatch.setenv("OPENAI_API_KEY", "private-unit-test-value")
    monkeypatch.setattr(llm_eval, "run_one", lambda *a: pytest.fail("Config check must not evaluate"))
    assert llm_eval.main(["--check-config"]) == 0
    output = capsys.readouterr().out
    assert "private-unit-test-value" not in output
    assert json.loads(output)["api_key_configured"] is True


def test_missing_key_stops_live_evaluation(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_eval, "_load_environment", lambda root: True)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(llm_eval, "run_one", lambda *a: pytest.fail("Must not evaluate without credentials"))
    with pytest.raises(SystemExit) as error:
        llm_eval.main(["--output-dir", str(tmp_path / "live")])
    assert error.value.code == 2
    assert not (tmp_path / "live").exists()


def test_dry_run_cannot_construct_live_advisor_even_with_key(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_eval, "_load_environment", lambda root: True)
    monkeypatch.setenv("OPENAI_API_KEY", "private-unit-test-value")
    monkeypatch.setattr(llm_eval, "OpenAIAdvisor", lambda **kw: pytest.fail("No live adapter in preview"))
    def run(instance, seed, name):
        return dict(agent=name, seed=seed, net_arpu_gain=1.0, failure=None), {
            "llm": {"status": "ok", "context": None}}
    monkeypatch.setattr(llm_eval, "run_one", run)
    assert llm_eval.main(["--dry-run", "--output-dir", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "evaluation.json").read_text())
    assert report["dry_run"] and report["api_requests"] == 0
    assert "private-unit-test-value" not in json.dumps(report)


def test_existing_experiments_are_not_overwritten(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_eval, "_load_environment", lambda root: True)
    path = tmp_path / "reviews.jsonl"
    path.write_text("reviewed data")
    monkeypatch.setattr(llm_eval, "run_one", lambda *a: pytest.fail("Must stop before evaluation"))
    with pytest.raises(SystemExit):
        llm_eval.main(["--dry-run", "--output-dir", str(tmp_path)])
    assert path.read_text() == "reviewed data"
