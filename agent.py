"""Coordinate candidate generation, public pilots, and portfolio planning."""

from pathlib import Path
import pandas as pd

from pilot_policy import run_adaptive_pilots


class Agent:
    def __init__(self, config=None):
        self.config = dict(config or {})
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
        start_budget, start_contacts, start_slots = env.remaining_budget, env.remaining_contacts, env.pilots_left
        trace["candidates"] = generate_candidates(env.customer_profile, history, env.tariffs,
                                                    config=self.config, trace=events)
        summary = trace["summary"]
        try:
            trace["observations"] = run_adaptive_pilots(env, trace["candidates"], config=self.config, trace=events)
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
