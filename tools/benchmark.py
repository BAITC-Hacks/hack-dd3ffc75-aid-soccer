"""Black-box evaluator benchmark. Run from the repository root with -m tools.benchmark."""
import argparse
import contextlib
import csv
import io
import json
import math
from pathlib import Path
import statistics
import time

import pandas as pd

from local_eval import evaluate_agent
from scoring_core import (MAX_CAMPAIGNS, MAX_CUSTOMERS_PER_CAMPAIGN,
                          MAX_TOTAL_CONTACTS, TOTAL_BUDGET, apply_filters,
                          sanitize_campaigns, validate_strategy)


def strict_value(value):
    """Unavailable/nonfinite display values are null, never NaN/Infinity."""
    if isinstance(value, dict):
        return {str(k): strict_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_value(v) for v in value]
    if hasattr(value, "item"):
        return strict_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def validate_plan(env, campaigns, *, strict=True):
    """Validate public campaign filters and whole audiences after pilot spending.

    This is offline verification, not part of agent decision making. Starter
    plans are checked for valid syntax but retain their original overlaps and
    scorer truncation behavior so the baseline is evaluated unchanged.
    """
    if not isinstance(campaigns, list) or not (1 if strict else 0) <= len(campaigns) <= MAX_CAMPAIGNS:
        raise ValueError("Expected 1–10 final campaign dictionaries")
    allowed = {"campaign_name", "target_tariff", "channel", "filter_current_tariff",
               "filter_arpu_segment", "filter_data_segment", "filter_call_segment"}
    if any(not isinstance(c, dict) or set(c) - allowed for c in campaigns):
        raise ValueError("Final campaigns contain unsupported keys")
    if sanitize_campaigns(campaigns, env.tariffs) != campaigns:
        raise ValueError("Public sanitizer discarded a final campaign")
    if campaigns:
        validate_strategy(pd.DataFrame(campaigns), env.tariffs)
    contacts, cost, max_size = 0, 0.0, 0
    cells, customer_ids = set(), set()
    distinct_cells, disjoint = True, True
    for campaign in campaigns:
        audience = apply_filters(env.customer_profile, pd.Series(campaign))
        audience = audience.sort_values("ID_NUMBER").head(MAX_CUSTOMERS_PER_CAMPAIGN)
        if audience.empty or audience.current_tariff.eq(campaign["target_tariff"]).any():
            raise ValueError("Final campaign has an empty or self-transition audience")
        cell = (campaign.get("filter_current_tariff"), campaign.get("filter_arpu_segment"))
        ids = set(audience.ID_NUMBER)
        distinct_cells = distinct_cells and cell not in cells
        disjoint = disjoint and not bool(customer_ids & ids)
        cells.add(cell)
        customer_ids.update(ids)
        contacts += len(audience)
        max_size = max(max_size, len(audience))
        cost += len(audience) * env.channels[campaign["channel"]]["cost_per_contact"]
    fits = contacts <= env.remaining_contacts and cost <= env.remaining_budget
    if strict and (not fits or not distinct_cells or not disjoint):
        raise ValueError("Final plan violates residual resources or distinct-cell constraints")
    return dict(final_contacts=int(contacts), final_cost=float(cost),
                max_final_campaign_contacts=max_size, distinct_final_cells=distinct_cells,
                disjoint_final_audiences=disjoint, fits_residual_resources=fits,
                remaining_budget_after_plan=float(env.remaining_budget - cost),
                remaining_contacts_after_plan=int(env.remaining_contacts - contacts))


class CaptureAgent:
    def __init__(self, instance, strict=True):
        self.instance = instance
        self.campaigns = None
        self.failure = None
        self.strict = strict
        self.validation = {}

    def act(self, env):
        try:
            self.campaigns = self.instance.act(env)
            self.validation = validate_plan(env, self.campaigns, strict=self.strict)
            return self.campaigns
        except Exception as exc:
            # Offline instrumentation only: evaluator catches this re-raised error.
            self.failure = f"{type(exc).__name__}: {exc}"
            raise


def run_one(instance, seed, name):
    captured = CaptureAgent(instance, strict=name != "starter")
    output = io.StringIO()
    start = time.perf_counter()
    result = None
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        try:
            result = evaluate_agent(captured, seed=seed, verbose=False)
        except Exception as exc:
            captured.failure = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - start
    trace = getattr(instance, "last_trace", {})
    diagnostics = output.getvalue().strip()
    status = trace.get("summary", {}).get("status")
    failure = captured.failure
    if not failure and (result is None or "[!]" in diagnostics or status in ("error", "infeasible")):
        failure = diagnostics or f"Unsuccessful result/trace status: {status}"
    result = result or {}
    if not failure:
        issues = []
        if result.get("n_pilots") is not None and not 1 <= result["n_pilots"] <= 20:
            issues.append("Pilot count must be 1–20")
        if result.get("total_cost", 0) > TOTAL_BUDGET or result.get("total_contacts", 0) > MAX_TOTAL_CONTACTS:
            issues.append("Evaluator resource limits exceeded")
        if elapsed >= 600:
            issues.append("Evaluation exceeded 600 seconds")
        if name != "starter" and any(d.get("capped_at_reach_budget") or d.get("capped_at_money_budget")
                                     for d in result.get("campaigns_detail", [])):
            issues.append("Unexpected scorer resource truncation")
        failure = "; ".join(issues) or None
    row = dict(agent=name, seed=seed, net_arpu_gain=result.get("net_arpu_gain"),
               cost=result.get("total_cost"), contacts=result.get("total_contacts"),
               pilots=result.get("n_pilots"),
               final_campaign_count=len(captured.campaigns) if captured.campaigns is not None else None,
               meets_final_count_requirement=(isinstance(captured.campaigns, list)
                                               and 1 <= len(captured.campaigns) <= 10),
               elapsed_seconds=elapsed, failure=failure, diagnostics=diagnostics or None,
               **captured.validation)
    return strict_value(row), strict_value(trace)


def summarize(rows):
    values = [r["net_arpu_gain"] for r in rows if not r["failure"] and r["net_arpu_gain"] is not None]
    return dict(runs=len(rows), successful_runs=len(values), failures=sum(bool(r["failure"]) for r in rows),
                mean=statistics.mean(values) if values else None,
                median=statistics.median(values) if values else None,
                minimum=min(values) if values else None,
                standard_deviation=statistics.pstdev(values) if values else None,
                positive_runs=sum(v > 0 for v in values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--compare-starter", action="store_true")
    parser.add_argument("--trace-seed", type=int, default=42)
    parser.add_argument("--exploration-only", action="store_true")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    from agent import Agent
    from agent_template import Agent as Starter
    variants = [("adaptive", Agent)]
    if args.compare_starter:
        variants.append(("starter", Starter))
    if args.exploration_only:
        variants.append(("exploration_only", lambda: Agent({"confirmation_pilots": 0})))
    rows = []
    for seed in range(args.runs):
        for name, factory in variants:
            row, _ = run_one(factory(), seed, name)
            rows.append(row)
            print(f"{name} seed={seed} net={row['net_arpu_gain']} failure={row['failure']}")
    summary = {name: summarize([r for r in rows if r["agent"] == name]) for name, _ in variants}
    differences = []
    if args.compare_starter:
        for seed in range(args.runs):
            ours = next(r for r in rows if r["seed"] == seed and r["agent"] == "adaptive")
            base = next(r for r in rows if r["seed"] == seed and r["agent"] == "starter")
            difference = None if ours["failure"] or base["failure"] else ours["net_arpu_gain"] - base["net_arpu_gain"]
            differences.append(dict(seed=seed, absolute_improvement=difference))
    trace_row, trace = run_one(Agent(), args.trace_seed, "adaptive")
    trace["benchmark"] = dict(source="official local_eval.evaluate_agent", seed=args.trace_seed,
                              net_gain=trace_row["net_arpu_gain"] if not trace_row["failure"] else None,
                              failure=trace_row["failure"])
    report = dict(summary=summary, runs=rows, per_seed_differences=differences,
                  trace_seed=args.trace_seed, trace_evaluation=trace_row,
                  notes=["Null means unavailable or nonfinite; failed runs are excluded from summary statistics.",
                         "Mock evaluator results do not predict judging rank; pilot gains are not added separately.",
                         "Final-only portfolio estimates cannot exactly deduplicate unknown pilot overlap.",
                         "Starter is evaluated unchanged, including pilot-only scores when no final campaigns are returned; see meets_final_count_requirement."])
    if args.compare_starter:
        for metric in ("mean", "median"):
            a, b = summary["adaptive"][metric], summary["starter"][metric]
            report[metric + "_absolute_improvement"] = a - b if a is not None and b is not None else None
    root = Path(__file__).resolve().parents[1] / "reports"
    root.mkdir(exist_ok=True)
    (root / "benchmark.json").write_text(json.dumps(strict_value(report), indent=2, allow_nan=False), encoding="utf-8")
    (root / "decision_trace.json").write_text(json.dumps(trace, indent=2, allow_nan=False), encoding="utf-8")
    with (root / "benchmark.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(dict.fromkeys(k for row in rows for k in row)), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Mock benchmark", "", *report["notes"], "", "```json", json.dumps(summary, indent=2), "```", "",
             "## Absolute per-seed improvements", "", "```json", json.dumps(differences, indent=2), "```", "",
             f"Mean absolute improvement: {report.get('mean_absolute_improvement')}",
             f"Median absolute improvement: {report.get('median_absolute_improvement')}", "",
             "## Failures", ""]
    lines += [f"- {r['agent']} seed {r['seed']}: {r['failure']}" for r in rows if r["failure"]]
    (root / "benchmark.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return int(any(r["failure"] for r in rows) or bool(trace_row["failure"]))


if __name__ == "__main__":
    raise SystemExit(main())
