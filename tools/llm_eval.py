"""Explicit LLM experiment runner; --dry-run never calls any API."""
import argparse
import json
import os
from pathlib import Path

from agent import Agent
from llm_advisor import DEFAULT_MODEL, OpenAIAdvisor, scenario_group
from tools.benchmark import run_one, summarize


class PreviewAdvisor:
    model = "offline-preview"
    request_count = 0

    def recommend(self, context):
        return {"recommendation": {"candidate_ids": [],
                                    "reason": "Offline preview: no model was called."}, "usage": {}}


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def _load_environment(root):
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    # Only this explicitly invoked CLI reads .env. Agent.act does not read it.
    load_dotenv(root / ".env", override=False)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("shadow", "assist"), default="shadow")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.runs <= 10:
        parser.error("--runs must be 1–10; each live run makes at most one API request")
    root = Path(__file__).resolve().parents[1]
    dotenv_available = _load_environment(root)
    model = args.model or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL
    configured = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    if args.check_config:
        print(json.dumps(dict(api_key_configured=configured, model=model,
                              dotenv_available=dotenv_available, api_calls=0)))
        return 0 if configured else 1
    if not args.dry_run and not configured:
        parser.error("Set OPENAI_API_KEY locally in .env or the environment; no API request made")
    if not args.dry_run:
        try:
            import openai  # noqa: F401
        except ImportError:
            parser.error("Install requirements-llm.txt before live evaluation")
    output = args.output_dir or root / "reports" / "llm_runs" / ("preview" if args.dry_run else args.mode)
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory is not empty; choose a new --output-dir to preserve reviewed examples")
    output.mkdir(parents=True, exist_ok=True)
    advisor = PreviewAdvisor() if args.dry_run else OpenAIAdvisor(model=model)
    rows, experiences, reviews = [], {}, {}
    # Organizer runners expect cwd-relative public CSV paths.
    previous_cwd = Path.cwd()
    try:
        os.chdir(root)
        for seed in range(args.seed, args.seed + args.runs):
            baseline, _ = run_one(Agent(), seed, "baseline")
            row, trace = run_one(Agent({"llm_mode": args.mode}, advisor=advisor), seed, args.mode)
            advice = trace.get("llm", {})
            row["llm_status"] = advice.get("status")
            row["llm_error_code"] = advice.get("error_code")
            rows.extend([baseline, row])
            _write_json(output / f"trace_{seed}.json", trace)
            context = advice.get("context")
            key = advice.get("context_hash")
            if context and key:
                experience = experiences.setdefault(key, dict(
                    schema_version="1.0", context_hash=key, scenario_group=scenario_group(context),
                    context=context, model=advice.get("model"),
                    proposal=advice.get("recommendation"), evaluations=[]))
                # Outcomes support human review; they never become automatic labels.
                experience["evaluations"].append(dict(seed=seed, mode=args.mode,
                    llm_status=advice.get("status"), failure=row["failure"],
                    net_gain=row["net_arpu_gain"], baseline_net_gain=baseline["net_arpu_gain"],
                    recommendation=advice.get("recommendation"),
                    applied_candidate_ids=advice.get("applied_candidate_ids", []),
                    pilot_evidence=[{field: observation.get(field) for field in
                                    ("candidate_id", "pilot_n", "mean_lift", "se_lift", "safe_lift")}
                                    for observation in trace.get("observations", [])
                                    if observation.get("status") == "measured"]))
                reviews.setdefault(key, dict(context_hash=key, approved=False, reviewer="",
                    candidate_ids=[], reason=""))
            print(f"seed={seed} mode={args.mode} net={row['net_arpu_gain']} "
                  f"llm_status={row['llm_status']} error={row['llm_error_code']}")
    finally:
        os.chdir(previous_cwd)
    summary = {name: summarize([row for row in rows if row["agent"] == name])
               for name in ("baseline", args.mode)}
    report = dict(mode=args.mode, dry_run=args.dry_run, requested_model=model,
                  api_requests=advisor.request_count, summary=summary, runs=rows,
                  notes=["Preview runs do not use an LLM and do not demonstrate LLM quality.",
                         "Live advice is not guaranteed deterministic; the official submission defaults to LLM off.",
                         "Model suggestions and evaluator net gain are not supervised training labels.",
                         "Repeated seeds on identical context are one training example, not independent scenarios."])
    _write_json(output / "evaluation.json", report)
    for name, records in (("experiences.jsonl", experiences), ("reviews.pending.jsonl", reviews)):
        with (output / name).open("w", encoding="utf-8") as handle:
            for key in sorted(records):
                handle.write(json.dumps(records[key], ensure_ascii=False, allow_nan=False) + "\n")
    failures = any(row["failure"] for row in rows)
    advice_failures = any(row.get("llm_status") != "ok" for row in rows if row["agent"] == args.mode)
    print(f"Saved {output}; API requests: {advisor.request_count}")
    return int(failures or (advice_failures and not args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
