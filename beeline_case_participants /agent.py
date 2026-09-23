"""Adaptive tariff-campaign agent.

The agent uses historical data only to decide what to explore.  Final campaign
decisions are driven primarily by observed pilot outcomes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from candidate_engine import build_candidates


MAX_FINAL_CAMPAIGNS = 10
MAX_CUSTOMERS_PER_CAMPAIGN = 5000
PILOT_SIZE = 200
PILOT_LIMIT = 18
PILOT_NOISE_PER_CUSTOMER = 0.804


class Agent:
    def _load_candidates(self, env) -> pd.DataFrame:
        base_dir = Path(__file__).resolve().parent
        history = pd.read_csv(base_dir / "data" / "change_tariff.csv")
        return build_candidates(
            history,
            env.customer_profile,
            env.tariffs,
            min_history=5,
            shrinkage=20,
            max_candidates=60,
        )

    @staticmethod
    def _pilot_plan(candidates: pd.DataFrame, limit: int) -> pd.DataFrame:
        """Keep broad cell coverage, then add alternative targets."""
        if candidates.empty or limit <= 0:
            return candidates.head(0)

        cell_columns = ["current_tariff", "arpu_segment"]
        primary = candidates.drop_duplicates(cell_columns, keep="first")
        selected_indexes = list(primary.head(limit).index)

        if len(selected_indexes) < limit:
            alternatives = candidates.loc[~candidates.index.isin(selected_indexes)]
            selected_indexes.extend(
                alternatives.head(limit - len(selected_indexes)).index.tolist()
            )
        return candidates.loc[selected_indexes].head(limit).copy()

    @staticmethod
    def _channel_option(
        candidate: pd.Series,
        channel: str,
        env,
        remaining_budget: float,
        remaining_contacts: int,
    ) -> dict | None:
        channel_info = env.channels[channel]
        cost = float(channel_info["cost_per_contact"])
        multiplier = float(channel_info["conversion_multiplier"])

        contacts = min(
            int(candidate["audience_size"]),
            MAX_CUSTOMERS_PER_CAMPAIGN,
            int(remaining_contacts),
        )
        if cost > 0:
            contacts = min(contacts, int(remaining_budget // cost))
        if contacts <= 0:
            return None

        gross_per_contact = (
            float(candidate["estimated_base_ratio"])
            * multiplier
            * float(candidate["mean_predicted_arpu"])
        )
        net_per_contact = gross_per_contact - cost
        return {
            "channel": channel,
            "contacts": contacts,
            "cost": contacts * cost,
            "expected_net": contacts * net_per_contact,
            "net_per_contact": net_per_contact,
        }

    def act(self, env) -> list[dict]:
        candidates = self._load_candidates(env)
        if candidates.empty:
            return []

        pilot_limit = min(PILOT_LIMIT, int(env.pilots_left), len(candidates))
        pilot_candidates = self._pilot_plan(candidates, pilot_limit)
        observations = []

        for _, candidate in pilot_candidates.iterrows():
            if env.pilots_left <= 0 or env.remaining_contacts < 10:
                break

            requested_size = min(PILOT_SIZE, int(candidate["audience_size"]))
            if requested_size < 10:
                continue

            try:
                result = env.run_pilot(
                    target_tariff=candidate["target_tariff"],
                    channel="push",
                    n_customers=requested_size,
                    filter_arpu_segment=candidate["arpu_segment"],
                    filter_current_tariff=candidate["current_tariff"],
                )
            except (RuntimeError, ValueError):
                continue

            prior_push_ratio = (
                float(candidate["conservative_lift"])
                * float(candidate["transition_share"])
                * float(env.channels["push"]["conversion_multiplier"])
            )
            actual_size = max(int(result["n_customers"]), 1)
            standard_error = PILOT_NOISE_PER_CUSTOMER / np.sqrt(actual_size)

            # Pilots dominate the posterior, while the historical prior prevents
            # one unlucky noisy observation from completely reversing a strong
            # and well-supported hypothesis.
            posterior_push_ratio = (
                0.8 * float(result["observed_lift_ratio"])
                + 0.2 * prior_push_ratio
            )
            conservative_push_ratio = posterior_push_ratio - 0.25 * standard_error
            push_multiplier = float(env.channels["push"]["conversion_multiplier"])

            observation = candidate.to_dict()
            observation.update(
                {
                    "pilot_lift_ratio": float(result["observed_lift_ratio"]),
                    "pilot_size": actual_size,
                    "posterior_push_ratio": posterior_push_ratio,
                    "estimated_base_ratio": conservative_push_ratio / push_multiplier,
                }
            )
            observations.append(observation)

        if not observations:
            return []

        observed = pd.DataFrame(observations)
        observed = observed[observed["estimated_base_ratio"] > 0].copy()
        if observed.empty:
            return []

        # Only one final campaign per current-tariff/ARPU cell.  Those filters
        # create disjoint final audiences and avoid paying twice for a customer.
        observed["estimated_push_value"] = (
            observed["estimated_base_ratio"]
            * float(env.channels["push"]["conversion_multiplier"])
            * observed["audience_predicted_arpu"]
        )
        observed = observed.sort_values(
            ["estimated_push_value", "candidate_score"], ascending=False
        ).drop_duplicates(["current_tariff", "arpu_segment"], keep="first")

        remaining = observed.to_dict("records")
        campaigns: list[dict] = []
        known_channels = list(env.channels)
        planning_budget = float(env.remaining_budget)
        planning_contacts = int(env.remaining_contacts)

        # Greedy resource allocation: at each step choose the campaign/channel
        # pair with the highest expected total net value under current limits.
        while remaining and len(campaigns) < MAX_FINAL_CAMPAIGNS:
            best = None
            for candidate in remaining:
                row = pd.Series(candidate)
                for channel in known_channels:
                    option = self._channel_option(
                        row,
                        channel,
                        env,
                        planning_budget,
                        planning_contacts,
                    )
                    if option is None or option["expected_net"] <= 0:
                        continue
                    choice = {"candidate": candidate, "option": option}
                    if best is None or option["expected_net"] > best["option"]["expected_net"]:
                        best = choice

            if best is None:
                break

            candidate = best["candidate"]
            option = best["option"]
            campaigns.append(
                {
                    "campaign_name": (
                        f"main_{candidate['current_tariff']}_"
                        f"{candidate['arpu_segment']}_{candidate['target_tariff']}"
                    ),
                    "filter_arpu_segment": candidate["arpu_segment"],
                    "filter_current_tariff": candidate["current_tariff"],
                    "target_tariff": candidate["target_tariff"],
                    "channel": option["channel"],
                }
            )

            planning_contacts -= int(option["contacts"])
            planning_budget -= float(option["cost"])
            remaining.remove(candidate)

        return campaigns
