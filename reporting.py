"""Render a saved decision trace as a self-contained analyst report."""

from __future__ import annotations

import argparse
from html import escape
import json
import math
from pathlib import Path
from typing import Any


def _text(value: Any) -> str:
    if value is None:
        return "—"
    return escape(str(value), quote=True)


def _number(value: Any, digits: int = 0) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    return f"{number:,.{digits}f}".replace(",", " ")


def _events(trace: dict) -> list[dict]:
    events = trace.get("events", [])
    return events if isinstance(events, list) else []


def _find_resource_totals(events: list[dict]) -> dict:
    for item in reversed(events):
        if isinstance(item, dict) and item.get("event") == "resource_totals":
            details = item.get("details")
            return details if isinstance(details, dict) else {}
    return {}


def _summary_cards(trace: dict, events: list[dict]) -> str:
    summary = trace.get("summary") if isinstance(trace.get("summary"), dict) else {}
    totals = _find_resource_totals(events)
    values = [
        ("Pilot spend", summary.get("pilot_cost"), "currency"),
        ("Planned final spend", summary.get("final_cost", totals.get("final_cost")), "currency"),
        ("Total contacts", (summary.get("pilot_contacts", 0) or 0)
         + (summary.get("final_contacts", totals.get("final_contacts", 0)) or 0), "integer"),
        ("Budget after plan", summary.get("remaining_budget_after_plan",
                                         totals.get("remaining_budget_after_plan")), "currency"),
        ("Contacts after plan", summary.get("remaining_contacts_after_plan",
                                            totals.get("remaining_contacts_after_plan")), "integer"),
    ]
    cards = []
    for label, value, kind in values:
        rendered = _number(value, 2 if kind == "currency" else 0)
        cards.append(
            f'<article class="card"><span>{_text(label)}</span><strong>{rendered}</strong></article>'
        )
    return "".join(cards)


def _campaign_rows(trace: dict, events: list[dict]) -> str:
    campaigns = trace.get("campaigns", [])
    campaigns = campaigns if isinstance(campaigns, list) else []
    selected = {
        (item.get("candidate_id"), (item.get("details") or {}).get("channel")): item
        for item in events
        if isinstance(item, dict) and item.get("event") == "selected"
        and isinstance(item.get("details"), dict)
    }
    rows = []
    for campaign in campaigns:
        if not isinstance(campaign, dict):
            continue
        candidate = "|".join(
            str(campaign.get(key, ""))
            for key in ("filter_current_tariff", "filter_arpu_segment", "target_tariff")
        )
        detail = selected.get((candidate, campaign.get("channel")))
        reason = detail.get("reason") if detail else "Selected final campaign."
        rows.append(
            "<tr>"
            f"<td>{_text(campaign.get('campaign_name'))}</td>"
            f"<td>{_text(campaign.get('filter_current_tariff'))} / "
            f"{_text(campaign.get('filter_arpu_segment'))}</td>"
            f"<td>{_text(campaign.get('target_tariff'))}</td>"
            f"<td>{_text(campaign.get('channel'))}</td>"
            f"<td>{_text(reason)}</td>"
            "</tr>"
        )
    if not rows:
        return '<tr><td colspan="5" class="empty">No final campaigns in this trace.</td></tr>'
    return "".join(rows)


def _uncertainty_chart(trace: dict) -> str:
    observations = trace.get("observations", [])
    measured = [
        item for item in observations
        if isinstance(item, dict) and item.get("status") == "measured"
        and all(_is_finite(item.get(key)) for key in ("safe_lift", "mean_lift", "optimistic_lift"))
    ]
    if not measured:
        return '<p class="empty">No measured pilot uncertainty bands available.</p>'
    measured = measured[:12]
    low = min(float(item["safe_lift"]) for item in measured)
    high = max(float(item["optimistic_lift"]) for item in measured)
    if high <= low:
        high = low + 1.0

    width, left, right, row_height = 920, 220, 30, 38
    plot_width = width - left - right

    def x(value: float) -> float:
        return left + (value - low) * plot_width / (high - low)

    svg = [
        f'<svg viewBox="0 0 {width} {55 + row_height * len(measured)}" '
        'role="img" aria-label="Pilot uncertainty intervals">',
        f'<line x1="{x(0):.1f}" y1="25" x2="{x(0):.1f}" '
        f'y2="{45 + row_height * len(measured)}" class="zero"/>',
    ]
    for index, item in enumerate(measured):
        y = 42 + index * row_height
        label = str(item.get("candidate_id", f"candidate {index + 1}"))
        safe = float(item["safe_lift"])
        mean = float(item["mean_lift"])
        optimistic = float(item["optimistic_lift"])
        svg.extend(
            [
                f'<text x="8" y="{y + 4}" class="svg-label">{_text(label[:32])}</text>',
                f'<line x1="{x(safe):.1f}" y1="{y}" x2="{x(optimistic):.1f}" '
                f'y2="{y}" class="band"/>',
                f'<circle cx="{x(mean):.1f}" cy="{y}" r="5" class="mean"/>',
            ]
        )
    svg.append("</svg>")
    return "".join(svg)


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _decision_rows(events: list[dict]) -> str:
    rows = []
    for item in events:
        if not isinstance(item, dict) or item.get("event") not in {"selected", "rejected"}:
            continue
        details = item.get("details") if isinstance(item.get("details"), dict) else {}
        decision = item.get("event")
        rows.append(
            f'<tr data-decision="{_text(decision)}">'
            f"<td><span class=\"pill {decision}\">{_text(decision)}</span></td>"
            f"<td>{_text(item.get('candidate_id'))}</td>"
            f"<td>{_text(details.get('channel'))}</td>"
            f"<td>{_text(item.get('reason'))}</td>"
            f"<td>{_number(details.get('conservative_net'), 2)}</td>"
            "</tr>"
        )
    if not rows:
        return '<tr><td colspan="5" class="empty">No selection events available.</td></tr>'
    return "".join(rows)


def _warnings(events: list[dict]) -> str:
    warning_events = [
        item for item in events
        if isinstance(item, dict)
        and item.get("event") in {"emergency_fallback", "infeasible_fallback"}
    ]
    return "".join(
        f'<aside class="warning"><strong>{_text(item.get("event"))}</strong> '
        f'{_text(item.get("reason"))}</aside>'
        for item in warning_events
    )


def _measured_result(trace: dict) -> str:
    benchmark = trace.get("benchmark")
    if not isinstance(benchmark, dict) or not _is_finite(benchmark.get("net_gain")):
        return '<p class="empty">No externally measured mock result is attached to this trace.</p>'
    return (
        '<p class="measured"><strong>Externally measured mock net gain:</strong> '
        f'{_number(benchmark.get("net_gain"), 2)}. '
        'This evaluator result is separate from the planning estimate.</p>'
    )


def render_report(trace: dict, output_path: str) -> None:
    """Render *trace* without running the agent, pilots, or evaluator."""
    if not isinstance(trace, dict):
        raise TypeError("trace must be a dictionary")
    events = _events(trace)
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Adaptive Campaign Portfolio Report</title>
<style>
:root{{--ink:#171717;--muted:#666;--paper:#f7f5ef;--card:#fff;--accent:#ffd400;--line:#dedbd2;}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1120px;margin:auto;padding:40px 24px 72px}} header{{border-left:10px solid var(--accent);padding-left:20px}}
h1{{font-size:clamp(30px,5vw,54px);line-height:1.02;margin:0 0 12px}} h2{{margin:40px 0 14px;font-size:23px}}
.sub{{color:var(--muted);max-width:760px}} .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:26px 0}}
.card{{background:var(--card);border:1px solid var(--line);padding:18px}} .card span{{display:block;color:var(--muted);font-size:13px}}
.card strong{{display:block;font-size:25px;margin-top:6px}} .panel{{background:var(--card);border:1px solid var(--line);padding:18px;overflow:auto}}
table{{width:100%;border-collapse:collapse;min-width:720px}} th,td{{padding:11px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}
th{{font-size:12px;text-transform:uppercase;letter-spacing:.06em}} .empty{{color:var(--muted);font-style:italic}}
.pill{{display:inline-block;border-radius:99px;padding:3px 9px;font-size:12px}} .selected{{background:#e7f5e8}} .rejected{{background:#f4eceb}}
.warning{{margin:12px 0;padding:14px 18px;background:#fff4c4;border-left:5px solid var(--accent)}}
svg{{width:100%;min-width:660px}} .band{{stroke:#555;stroke-width:4;stroke-linecap:round}} .mean{{fill:var(--accent);stroke:#111;stroke-width:1.5}}
.zero{{stroke:#c8c4b9;stroke-dasharray:4}} .svg-label{{font-size:11px;fill:#444}} .toolbar{{margin-bottom:12px}}
button{{border:1px solid #222;background:#fff;padding:7px 12px;margin-right:6px;cursor:pointer}} button:hover{{background:var(--accent)}}
.note{{border-top:1px solid var(--line);margin-top:42px;padding-top:18px;color:var(--muted)}}
</style>
</head>
<body><main>
<header><h1>Campaign portfolio</h1><p class="sub">Analyst-facing decision trace. Uncertainty bands and conservative net values are heuristic planning estimates, not guaranteed profit.</p></header>
{_warnings(events)}
<section class="cards">{_summary_cards(trace, events)}</section>
<h2>Selected campaigns</h2><div class="panel"><table><thead><tr><th>Name</th><th>Audience</th><th>Target</th><th>Channel</th><th>Why</th></tr></thead><tbody>{_campaign_rows(trace, events)}</tbody></table></div>
<h2>Pilot evidence and uncertainty</h2><div class="panel">{_uncertainty_chart(trace)}<p class="sub">Line: safe-to-optimistic heuristic interval. Dot: normalized mean lift. Adaptive selection means this is not a simultaneous calibrated guarantee.</p></div>
<h2>Selected and rejected alternatives</h2><div class="panel"><div class="toolbar"><button data-filter="all">All</button><button data-filter="selected">Selected</button><button data-filter="rejected">Rejected</button></div><table><thead><tr><th>Decision</th><th>Candidate</th><th>Channel</th><th>Reason</th><th>Conservative net</th></tr></thead><tbody id="decisions">{_decision_rows(events)}</tbody></table></div>
<h2>Externally measured result</h2><div class="panel">{_measured_result(trace)}</div>
<p class="note">Final-campaign economics are a trace-based proxy. Pilot contacts may overlap final audiences; exact pilot IDs are not available to the planner. Real mock performance must come from the external evaluator.</p>
</main><script>
document.querySelectorAll('[data-filter]').forEach(button=>button.addEventListener('click',()=>{{
 const wanted=button.dataset.filter;
 document.querySelectorAll('#decisions tr[data-decision]').forEach(row=>{{row.hidden=wanted!=='all'&&row.dataset.decision!==wanted;}});
}}));
</script></body></html>"""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Saved decision trace JSON")
    parser.add_argument("--output", required=True, help="Destination HTML file")
    args = parser.parse_args()
    with Path(args.input).open(encoding="utf-8") as handle:
        trace = json.load(handle)
    render_report(trace, args.output)


if __name__ == "__main__":
    main()
