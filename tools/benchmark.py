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

from local_eval import evaluate_agent


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


class CaptureAgent:
    def __init__(self, instance):
        self.instance = instance
        self.campaigns = None
        self.failure = None

    def act(self, env):
        try:
            self.campaigns = self.instance.act(env)
            return self.campaigns
        except Exception as exc:
            # Offline instrumentation only: evaluator catches this re-raised error.
            self.failure = f"{type(exc).__name__}: {exc}"
            raise


def run_one(instance, seed, name):
    captured = CaptureAgent(instance)
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
    row = dict(agent=name, seed=seed, net_arpu_gain=result.get("net_arpu_gain"),
               cost=result.get("total_cost"), contacts=result.get("total_contacts"),
               pilots=result.get("n_pilots"),
               final_campaign_count=len(captured.campaigns) if captured.campaigns is not None else None,
               elapsed_seconds=elapsed, failure=failure, diagnostics=diagnostics or None)
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
    report = dict(summary=summary, runs=rows, per_seed_differences=differences,
                  trace_seed=args.trace_seed, trace_evaluation=trace_row,
                  notes=["Null means unavailable or nonfinite; failed runs are excluded from summary statistics.",
                         "Mock evaluator results do not predict judging rank; pilot gains are not added separately.",
                         "Final-only portfolio estimates cannot exactly deduplicate unknown pilot overlap."])
    if args.compare_starter:
        for metric in ("mean", "median"):
            a, b = summary["adaptive"][metric], summary["starter"][metric]
            report[metric + "_absolute_improvement"] = a - b if a is not None and b is not None else None
    root = Path(__file__).resolve().parents[1] / "reports"
    root.mkdir(exist_ok=True)
    (root / "benchmark.json").write_text(json.dumps(strict_value(report), indent=2, allow_nan=False), encoding="utf-8")
    (root / "decision_trace.json").write_text(json.dumps(trace, indent=2, allow_nan=False), encoding="utf-8")
    with (root / "benchmark.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Mock benchmark", "", *report["notes"], "", "```json", json.dumps(summary, indent=2), "```", "",
             "## Absolute per-seed improvements", "", "```json", json.dumps(differences, indent=2), "```", "",
             f"Mean absolute improvement: {report.get('mean_absolute_improvement')}",
             f"Median absolute improvement: {report.get('median_absolute_improvement')}", "",
             "## Failures", ""]
    lines += [f"- {r['agent']} seed {r['seed']}: {r['failure']}" for r in rows if r["failure"]]
    (root / "benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return int(any(r["failure"] for r in rows) or bool(trace_row["failure"]))


if __name__ == "__main__":
    raise SystemExit(main())
