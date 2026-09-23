from pathlib import Path

from reporting import render_report


def sample_trace():
    return {
        "schema_version": "1.0",
        "observations": [
            {
                "candidate_id": "t1|HIGH|t2",
                "status": "measured",
                "safe_lift": -0.01,
                "mean_lift": 0.08,
                "optimistic_lift": 0.17,
            }
        ],
        "campaigns": [
            {
                "campaign_name": "campaign_01",
                "filter_current_tariff": "t1",
                "filter_arpu_segment": "HIGH",
                "target_tariff": "t2",
                "channel": "sms",
            }
        ],
        "events": [
            {
                "stage": "portfolio",
                "event": "selected",
                "candidate_id": "t1|HIGH|t2",
                "reason": "Best joint tradeoff",
                "details": {"channel": "sms", "conservative_net": 123.5},
            },
            {
                "stage": "portfolio",
                "event": "rejected",
                "candidate_id": "<script>alert(1)</script>",
                "reason": "uncertain & expensive",
                "details": {"channel": "digital_ads", "conservative_net": -10},
            },
            {
                "stage": "portfolio",
                "event": "resource_totals",
                "candidate_id": None,
                "reason": "proxy",
                "details": {
                    "final_contacts": 20,
                    "final_cost": 80.0,
                    "remaining_budget_after_plan": 900.0,
                    "remaining_contacts_after_plan": 500,
                },
            },
        ],
        "summary": {"pilot_cost": 40.0, "pilot_contacts": 10},
    }


def test_report_contains_key_sections_and_escapes_dynamic_text(tmp_path: Path):
    destination = tmp_path / "report.html"
    render_report(sample_trace(), str(destination))
    html = destination.read_text(encoding="utf-8")

    assert "Selected campaigns" in html
    assert "Pilot evidence and uncertainty" in html
    assert "Best joint tradeoff" in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<script>alert(1)</script>" not in html
    assert "Infinity" not in html


def test_report_handles_missing_optional_data_and_empty_state(tmp_path: Path):
    destination = tmp_path / "empty.html"
    render_report({"summary": {"final_cost": float("inf")}}, str(destination))
    html = destination.read_text(encoding="utf-8")

    assert "No final campaigns in this trace" in html
    assert "No measured pilot uncertainty bands" in html
    assert "No externally measured mock result" in html
    assert "Infinity" not in html
