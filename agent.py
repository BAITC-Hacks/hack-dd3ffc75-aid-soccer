"""Coordinate candidate generation, public pilots, and portfolio planning."""

from pathlib import Path
import pandas as pd

from pilot_policy import run_adaptive_pilots

DEFAULT_POLICY_PATH = Path(__file__).resolve().parent / "artifacts" / "llm_policy.json"


class Agent:
    def __init__(self, config=None, *, advisor=None):
        self.config = dict(config or {})
        # Bundled policy replay is offline and tied to an exact public context.
        # Explicit off/assist/shadow settings always override this release default.
        if "llm_mode" not in self.config and DEFAULT_POLICY_PATH.is_file():
            self.config.update(llm_mode="replay", llm_policy_path=str(DEFAULT_POLICY_PATH))
        self.advisor = advisor
        self.last_trace = {}

    def act(self, env):
        self.last_trace = dict(schema_version="1.0", candidates=[], observations=[],
                               campaigns=[], events=[], summary=dict(
                                   pilot_count=0, pilot_contacts=0, pilot_cost=0.0,
                                   final_contacts=0, final_cost=0.0,
                                   remaining_budget_after_plan=float(env.remaining_budget),
                                   remaining_contacts_after_plan=int(env.remaining_contacts),
                                   status="error"))
        trace = self.last_trace
        events = trace["events"]
        # Missing team modules are an integration failure, never a silent fallback.
        try:
            from candidate_engine import generate_candidates
            from portfolio_optimizer import build_campaigns
        except ModuleNotFoundError as exc:
            events.append(dict(stage="agent", event="missing_dependency", candidate_id=None,
                               reason=str(exc), details={"module": exc.name}))
            raise

        try:
            history = pd.read_csv(Path(__file__).resolve().parent / "data" / "change_tariff.csv")
        except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            history = pd.DataFrame()
            events.append(dict(stage="agent", event="history_unavailable", candidate_id=None,
                               reason=str(exc), details={}))
        required_history = {"tariff_plan_code_from", "tariff_plan_code_to",
                            "AVG_ARPU_PREV_3M", "AVG_ARPU_NEXT_3M"}
        missing_history = sorted(required_history - set(history.columns))
        if not history.empty and missing_history:
            events.append(dict(stage="agent", event="history_unavailable", candidate_id=None,
                               reason="Optional history schema is incomplete; using catalog hypotheses.",
                               details={"missing_columns": missing_history}))
            history = pd.DataFrame()
        start_budget, start_contacts, start_slots = env.remaining_budget, env.remaining_contacts, env.pilots_left
        trace["candidates"] = generate_candidates(env.customer_profile, history, env.tariffs,
                                                    config=self.config, trace=events)
        # API use is explicit configuration, never triggered by finding a key.
        pilot_config = dict(self.config)
        mode = self.config.get("llm_mode", "off")
        if mode not in {"off", "shadow", "assist", "replay"}:
            raise ValueError("llm_mode must be off, shadow, assist, or replay")
        if mode != "off":
            from llm_advisor import get_advice
            advice = get_advice(env, trace["candidates"], mode=mode,
                                advisor=self.advisor, config=self.config)
            trace["llm"] = advice
            events.append(dict(stage="agent", event="llm_advice", candidate_id=None,
                               reason="Optional exploration advice; pilot evidence and resource checks remain authoritative.",
                               details={key: advice[key] for key in
                                        ("mode", "status", "context_hash", "applied_candidate_ids", "error_code")}))
            if advice["applied_candidate_ids"]:
                pilot_config["exploration_priority_ids"] = advice["applied_candidate_ids"]
        summary = trace["summary"]
        try:
            trace["observations"] = run_adaptive_pilots(env, trace["candidates"], config=pilot_config, trace=events)
        finally:
            # Preserve actual spending even when a programming/adapter error propagates.
            summary.update(pilot_count=int(start_slots - env.pilots_left),
                           pilot_contacts=int(start_contacts - env.remaining_contacts),
                           pilot_cost=float(start_budget - env.remaining_budget),
                           remaining_budget_after_plan=float(env.remaining_budget),
                           remaining_contacts_after_plan=int(env.remaining_contacts))
        campaigns = build_campaigns(env, trace["observations"], config=self.config, trace=events)
        trace["campaigns"] = campaigns
        contacts, cost = 0, 0.0
        for campaign in campaigns:
            audience = env.customer_profile
            for column in ("arpu_segment", "data_segment", "call_segment", "current_tariff"):
                value = campaign.get("filter_" + column)
                if value is not None:
                    wanted = str(value).split(";") if column == "current_tariff" else [value]
                    audience = audience[audience[column].isin([v.strip() if isinstance(v, str) else v for v in wanted])]
            n = min(5000, len(audience))
            contacts += n
            cost += n * env.channels[campaign["channel"]]["cost_per_contact"]
        status = "ok" if campaigns else "infeasible"
        if campaigns and any(e["event"] == "emergency_fallback" for e in events):
            status = "fallback"
        summary.update(final_contacts=int(contacts), final_cost=float(cost),
                       remaining_budget_after_plan=float(env.remaining_budget - cost),
                       remaining_contacts_after_plan=int(env.remaining_contacts - contacts), status=status)
        if contacts > env.remaining_contacts or cost > env.remaining_budget or len(campaigns) > 10:
            summary["status"] = "error"
            raise ValueError("Portfolio exceeds residual resource limits")
        return campaigns
