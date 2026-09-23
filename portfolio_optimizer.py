"""Deterministic portfolio planning for final tariff campaigns.

The optimizer consumes public pilot observations and environment attributes only.
It deliberately does not inspect mock effects or mutate the environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Iterable

import pandas as pd


DEFAULT_CONFIG = {
    "beam_width": 64,
    "max_campaigns": 10,
    "max_per_campaign": 5000,
}
NORMAL_CHANNELS = ("push", "sms", "digital_ads")
CAMPAIGN_KEYS = {
    "campaign_name",
    "filter_current_tariff",
    "filter_arpu_segment",
    "filter_data_segment",
    "filter_call_segment",
    "target_tariff",
    "channel",
}


@dataclass(frozen=True)
class Option:
    candidate_id: str
    cell_id: str
    current_tariff: str
    arpu_segment: str
    target_tariff: str
    channel: str
    contacts: int
    cost: float
    mean_gross: float
    conservative_gross: float
    conservative_net: float
    mean_lift: float
    safe_lift: float
    served_arpu: float

    @property
    def tie_key(self) -> tuple[str, str]:
        return (self.candidate_id, self.channel)


@dataclass(frozen=True)
class State:
    selected: tuple[Option, ...] = ()
    contacts: int = 0
    cost: float = 0.0
    objective: float = 0.0

    @property
    def tie_key(self) -> tuple[tuple[str, str], ...]:
        return tuple(option.tie_key for option in self.selected)


def _append(trace: list[dict] | None, event: str, candidate_id: str | None,
            reason: str, details: dict[str, Any] | None = None) -> None:
    if trace is None:
        return
    trace.append(
        {
            "stage": "portfolio",
            "event": event,
            "candidate_id": candidate_id,
            "reason": reason,
            "details": details or {},
        }
    )


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _safe_text(value: Any) -> str | None:
    if value is None:
        return None
    missing = pd.isna(value)
    if isinstance(missing, bool) and missing:
        return None
    if not isinstance(value, (str, int, float)):
        return None
    text = str(value).strip()
    return text or None


def _numeric_id_order(frame: pd.DataFrame) -> pd.DataFrame | None:
    numeric = pd.to_numeric(frame["ID_NUMBER"], errors="coerce")
    if numeric.isna().any():
        return None
    ordered = frame.assign(_numeric_id=numeric).sort_values(
        ["_numeric_id"], kind="mergesort"
    )
    return ordered.drop(columns="_numeric_id")


def _valid_profile(env: Any, trace: list[dict] | None) -> pd.DataFrame | None:
    profile = getattr(env, "customer_profile", None)
    required = {"ID_NUMBER", "current_tariff", "arpu_segment", "predicted_arpu"}
    if not isinstance(profile, pd.DataFrame) or not required.issubset(profile.columns):
        _append(trace, "infeasible_fallback", None,
                "Customer profile is missing required public columns.",
                {"required_columns": sorted(required)})
        return None
    return profile.copy(deep=True)


def _known_tariffs(env: Any) -> set[str]:
    tariffs = getattr(env, "tariffs", None)
    if not isinstance(tariffs, pd.DataFrame) or "tariff_plan_code" not in tariffs:
        return set()
    return {str(value) for value in tariffs["tariff_plan_code"].dropna()}


def _audience(profile: pd.DataFrame, current_tariff: str, arpu_segment: str,
              max_per_campaign: int) -> tuple[pd.DataFrame | None, str | None]:
    frame = profile[
        (profile["current_tariff"] == current_tariff)
        & (profile["arpu_segment"] == arpu_segment)
    ].copy()
    if frame.empty:
        return frame, None
    ordered = _numeric_id_order(frame)
    if ordered is None:
        return None, "ID_NUMBER must be numeric for scorer-compatible ordering."
    predicted = pd.to_numeric(ordered["predicted_arpu"], errors="coerce")
    if predicted.isna().any() or not predicted.map(math.isfinite).all():
        return None, "Audience has non-finite predicted_arpu values."
    ordered["predicted_arpu"] = predicted.astype(float)
    return ordered.iloc[:max_per_campaign].copy(), None


def _option_from_observation(
    env: Any,
    observation: dict,
    channel: str,
    audience: pd.DataFrame,
) -> Option | None:
    channel_info = getattr(env, "channels", {}).get(channel)
    if not isinstance(channel_info, dict):
        return None
    cost_per_contact = _finite_number(channel_info.get("cost_per_contact"))
    multiplier = _finite_number(channel_info.get("conversion_multiplier"))
    mean_lift = _finite_number(observation.get("mean_lift"))
    safe_lift = _finite_number(observation.get("safe_lift"))
    if None in (cost_per_contact, multiplier, mean_lift, safe_lift):
        return None
    if cost_per_contact < 0 or multiplier < 0:
        return None

    contacts = len(audience)
    served_arpu = float(audience["predicted_arpu"].sum())
    cost = contacts * cost_per_contact
    mean_gross = mean_lift * multiplier * served_arpu
    conservative_gross = safe_lift * multiplier * served_arpu
    current = str(observation["current_tariff"])
    segment = str(observation["arpu_segment"])
    target = str(observation["target_tariff"])
    candidate_id = str(
        observation.get("candidate_id") or f"{current}|{segment}|{target}"
    )
    cell_id = str(observation.get("cell_id") or f"{current}|{segment}")
    return Option(
        candidate_id=candidate_id,
        cell_id=cell_id,
        current_tariff=current,
        arpu_segment=segment,
        target_tariff=target,
        channel=channel,
        contacts=contacts,
        cost=float(cost),
        mean_gross=float(mean_gross),
        conservative_gross=float(conservative_gross),
        conservative_net=float(conservative_gross - cost),
        mean_lift=mean_lift,
        safe_lift=safe_lift,
        served_arpu=served_arpu,
    )


def _dominates(left: Option, right: Option) -> bool:
    return (
        left.contacts <= right.contacts
        and left.cost <= right.cost + 1e-9
        and left.conservative_net >= right.conservative_net - 1e-9
        and (
            left.contacts < right.contacts
            or left.cost < right.cost - 1e-9
            or left.conservative_net > right.conservative_net + 1e-9
        )
    )


def _prune_cell(options: Iterable[Option]) -> list[Option]:
    ordered = sorted(
        options,
        key=lambda option: (
            -option.conservative_net,
            option.cost,
            option.contacts,
            option.tie_key,
        ),
    )
    return [
        option
        for option in ordered
        if not any(_dominates(other, option) for other in ordered if other is not option)
    ]


def _greedy(cells: list[tuple[str, list[Option]]], budget: float, contacts: int,
            max_campaigns: int) -> State:
    all_options = [option for _, options in cells for option in options]
    all_options.sort(key=lambda item: (-item.conservative_net, item.cost, item.tie_key))
    state = State()
    used_cells: set[str] = set()
    for option in all_options:
        if option.cell_id in used_cells or len(state.selected) >= max_campaigns:
            continue
        if state.cost + option.cost > budget + 1e-9:
            continue
        if state.contacts + option.contacts > contacts:
            continue
        state = State(
            selected=state.selected + (option,),
            contacts=state.contacts + option.contacts,
            cost=state.cost + option.cost,
            objective=state.objective + option.conservative_net,
        )
        used_cells.add(option.cell_id)
    return state


def _state_sort_key(state: State) -> tuple:
    return (-state.objective, state.cost, state.contacts, len(state.selected), state.tie_key)


def _bounded_frontier(states: list[State], width: int, budget: float,
                      contacts: int) -> list[State]:
    deduplicated: dict[tuple[int, int, int], State] = {}
    for state in states:
        key = (state.contacts, int(round(state.cost * 100)), len(state.selected))
        previous = deduplicated.get(key)
        if previous is None or _state_sort_key(state) < _state_sort_key(previous):
            deduplicated[key] = state
    states = sorted(deduplicated.values(), key=_state_sort_key)
    if len(states) <= width:
        return states

    # Keep the best representative from different resource regions before
    # filling by objective. This avoids a frontier made only of costly prefixes.
    diverse: dict[tuple[int, int, int], State] = {}
    for state in states:
        budget_bucket = int(4 * state.cost / max(budget, 1.0))
        contact_bucket = int(4 * state.contacts / max(contacts, 1))
        key = (min(budget_bucket, 3), min(contact_bucket, 3), len(state.selected))
        diverse.setdefault(key, state)
    kept = sorted(diverse.values(), key=_state_sort_key)[:width]
    kept_ids = {id(state) for state in kept}
    for state in states:
        if len(kept) >= width:
            break
        if id(state) not in kept_ids:
            kept.append(state)
    return sorted(kept, key=_state_sort_key)


def _beam_search(cells: list[tuple[str, list[Option]]], budget: float, contacts: int,
                 max_campaigns: int, beam_width: int) -> State:
    frontier = [State()]
    for _, options in cells:
        expanded: list[State] = []
        for state in frontier:
            expanded.append(state)  # explicit skip option
            if len(state.selected) >= max_campaigns:
                continue
            for option in options:
                if state.cost + option.cost > budget + 1e-9:
                    continue
                if state.contacts + option.contacts > contacts:
                    continue
                expanded.append(
                    State(
                        selected=state.selected + (option,),
                        contacts=state.contacts + option.contacts,
                        cost=state.cost + option.cost,
                        objective=state.objective + option.conservative_net,
                    )
                )
        frontier = _bounded_frontier(
            expanded, max(1, beam_width), budget, contacts
        )
    return min(frontier, key=_state_sort_key)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or "unknown"


def _campaign(option: Option, index: int) -> dict:
    return {
        "campaign_name": (
            f"campaign_{index:02d}_{_slug(option.current_tariff)}_"
            f"{_slug(option.arpu_segment)}_{_slug(option.target_tariff)}_"
            f"{_slug(option.channel)}"
        ),
        "filter_current_tariff": option.current_tariff,
        "filter_arpu_segment": option.arpu_segment,
        "target_tariff": option.target_tariff,
        "channel": option.channel,
    }


def _fallback_audiences(profile: pd.DataFrame, observation: dict, limit: int
                        ) -> list[tuple[pd.DataFrame, dict[str, str]]]:
    current = str(observation["current_tariff"])
    arpu = str(observation["arpu_segment"])
    base = profile[
        (profile["current_tariff"] == current)
        & (profile["arpu_segment"] == arpu)
    ].copy()
    variants: list[tuple[pd.DataFrame, dict[str, str]]] = []
    filters = {
        "filter_current_tariff": current,
        "filter_arpu_segment": arpu,
    }
    variants.append((base, filters))
    for column, key in (
        ("data_segment", "filter_data_segment"),
        ("call_segment", "filter_call_segment"),
    ):
        if column not in base:
            continue
        for value in sorted(base[column].dropna().astype(str).unique()):
            subset = base[base[column].astype(str) == value].copy()
            variants.append((subset, {**filters, key: value}))
    if "data_segment" in base and "call_segment" in base:
        pairs = base[["data_segment", "call_segment"]].dropna().astype(str)
        for data_value, call_value in sorted(map(tuple, pairs.drop_duplicates().to_numpy())):
            subset = base[
                (base["data_segment"].astype(str) == data_value)
                & (base["call_segment"].astype(str) == call_value)
            ].copy()
            variants.append(
                (subset, {**filters, "filter_data_segment": data_value,
                          "filter_call_segment": call_value})
            )
    usable: list[tuple[pd.DataFrame, dict[str, str]]] = []
    for frame, chosen_filters in variants:
        ordered = _numeric_id_order(frame) if not frame.empty else frame
        if ordered is not None and 0 < len(ordered) <= limit:
            predicted = pd.to_numeric(ordered["predicted_arpu"], errors="coerce")
            if predicted.notna().all() and predicted.map(math.isfinite).all():
                ordered = ordered.copy()
                ordered["predicted_arpu"] = predicted.astype(float)
                usable.append((ordered, chosen_filters))
    return usable


def _emergency_fallback(env: Any, profile: pd.DataFrame, observations: list[dict],
                        known_tariffs: set[str], contacts: int,
                        max_per_campaign: int, trace: list[dict] | None) -> list[dict]:
    if contacts <= 0 or "push" not in getattr(env, "channels", {}):
        _append(trace, "infeasible_fallback", None,
                "No remaining contacts or no supported push channel.",
                {"remaining_contacts": contacts})
        return []
    multiplier = _finite_number(env.channels["push"].get("conversion_multiplier"))
    if multiplier is None or multiplier < 0:
        _append(trace, "infeasible_fallback", None,
                "Push channel has invalid public economics.")
        return []

    choices: list[tuple] = []
    limit = min(contacts, max_per_campaign)
    for observation in observations:
        current = _safe_text(observation.get("current_tariff"))
        segment = _safe_text(observation.get("arpu_segment"))
        target = _safe_text(observation.get("target_tariff"))
        if not current or not segment or not target or target not in known_tariffs:
            continue
        if target == current:
            continue
        measured = observation.get("status") == "measured"
        safe_lift = _finite_number(observation.get("safe_lift"))
        mean_lift = _finite_number(observation.get("mean_lift"))
        for audience, filters in _fallback_audiences(profile, observation, limit):
            served_arpu = float(audience["predicted_arpu"].sum())
            evidence = safe_lift if safe_lift is not None else mean_lift
            downside = (evidence * multiplier * served_arpu
                        if evidence is not None else float("-inf"))
            candidate_id = str(
                observation.get("candidate_id") or f"{current}|{segment}|{target}"
            )
            # Prefer measured evidence, then least estimated loss, then smaller reach.
            filter_signature = tuple(sorted(filters.items()))
            choices.append((not measured, -downside, len(audience), candidate_id,
                            target, filter_signature, filters, downside, evidence))
    if not choices:
        _append(trace, "infeasible_fallback", None,
                "No executable non-self audience fits the remaining contact limit.",
                {"remaining_contacts": contacts})
        return []

    choice = min(choices)
    _, _, audience_size, candidate_id, target, _, filters, downside, evidence = choice
    campaign = {
        "campaign_name": f"campaign_01_emergency_{_slug(candidate_id)}_push",
        **filters,
        "target_tariff": target,
        "channel": "push",
    }
    _append(
        trace,
        "emergency_fallback",
        candidate_id,
        "No normal option had positive conservative net; selected the smallest "
        "defensible executable push audience. Business risk remains.",
        {
            "contacts": audience_size,
            "filters": dict(filters),
            "estimated_downside_or_value": None if not math.isfinite(downside) else float(downside),
            "evidence_lift": evidence,
            "confidence": "measured but conservative estimate" if evidence is not None else "unknown",
        },
    )
    _append(
        trace,
        "resource_totals",
        None,
        "Emergency push fallback uses zero communication spend but still carries business risk.",
        {
            "campaign_count": 1,
            "final_contacts": audience_size,
            "final_cost": 0.0,
            "conservative_final_net_proxy": None if not math.isfinite(downside) else float(downside),
            "remaining_budget_after_plan": float(max(0.0, getattr(env, "remaining_budget", 0.0))),
            "remaining_contacts_after_plan": int(contacts - audience_size),
            "search": "emergency fallback",
        },
    )
    return [campaign]


def build_campaigns(env: Any, observations: list[dict], *, config: dict | None = None,
                    trace: list[dict] | None = None) -> list[dict]:
    """Build a feasible final campaign portfolio from normalized pilot evidence."""
    settings = dict(DEFAULT_CONFIG)
    if config:
        for key in settings:
            if key in config:
                settings[key] = config[key]
    max_campaigns = max(0, min(10, int(settings["max_campaigns"])))
    max_per_campaign = max(1, min(5000, int(settings["max_per_campaign"])))
    beam_width = max(1, int(settings["beam_width"]))
    budget = max(0.0, float(getattr(env, "remaining_budget", 0.0)))
    contacts = max(0, int(getattr(env, "remaining_contacts", 0)))
    original_observations = list(observations or [])

    profile = _valid_profile(env, trace)
    known_tariffs = _known_tariffs(env)
    if profile is None or not known_tariffs or max_campaigns == 0:
        _append(trace, "infeasible_fallback", None,
                "Missing valid profile/tariff data or campaign capacity.")
        return []

    options_by_cell: dict[str, list[Option]] = {}
    measured_candidates: set[str] = set()
    for raw in original_observations:
        if not isinstance(raw, dict):
            _append(trace, "rejected", None, "Observation is not a dictionary.",
                    {"category": "invalid data"})
            continue
        current = _safe_text(raw.get("current_tariff"))
        segment = _safe_text(raw.get("arpu_segment"))
        target = _safe_text(raw.get("target_tariff"))
        candidate_id = str(raw.get("candidate_id") or "unknown")
        if not current or not segment or not target or target not in known_tariffs:
            _append(trace, "rejected", candidate_id,
                    "Candidate has invalid current/segment/target data.",
                    {"category": "invalid data"})
            continue
        if target == current:
            _append(trace, "rejected", candidate_id,
                    "Self-transition is not executable.",
                    {"category": "invalid data"})
            continue
        if raw.get("status") != "measured":
            _append(trace, "rejected", candidate_id,
                    "Candidate has no valid pilot measurement for normal exploitation.",
                    {"category": "no measurement"})
            continue
        if _finite_number(raw.get("mean_lift")) is None or _finite_number(raw.get("safe_lift")) is None:
            _append(trace, "rejected", candidate_id,
                    "Measured candidate contains non-finite evidence.",
                    {"category": "invalid data"})
            continue
        audience, error = _audience(profile, current, segment, max_per_campaign)
        if audience is None or audience.empty:
            _append(trace, "rejected", candidate_id,
                    error or "Candidate audience is empty.",
                    {"category": "invalid data"})
            continue
        measured_candidates.add(candidate_id)
        # The actual cell is defined by executable filters, not by diagnostic IDs.
        cell_id = f"{current}|{segment}"
        normalized = dict(raw)
        normalized["cell_id"] = cell_id
        for channel in NORMAL_CHANNELS:
            option = _option_from_observation(env, normalized, channel, audience)
            if option is None:
                continue
            _append(
                trace,
                "evaluated_channel",
                option.candidate_id,
                "Computed channel economics from normalized pilot evidence.",
                {
                    "channel": channel,
                    "contacts": option.contacts,
                    "served_arpu": option.served_arpu,
                    "mean_gross": option.mean_gross,
                    "conservative_gross": option.conservative_gross,
                    "cost": option.cost,
                    "conservative_net": option.conservative_net,
                },
            )
            if option.conservative_net <= 0:
                _append(trace, "rejected", option.candidate_id,
                        "Conservative net is nonpositive.",
                        {"category": "nonpositive conservative net", "channel": channel,
                         "conservative_net": option.conservative_net})
                continue
            if option.contacts > contacts:
                _append(trace, "rejected", option.candidate_id,
                        "Whole effective audience exceeds remaining contacts.",
                        {"category": "contacts", "channel": channel})
                continue
            if option.cost > budget + 1e-9:
                _append(trace, "rejected", option.candidate_id,
                        "Whole effective audience exceeds remaining budget.",
                        {"category": "budget", "channel": channel})
                continue
            options_by_cell.setdefault(cell_id, []).append(option)

    cells = [
        (cell_id, _prune_cell(options))
        for cell_id, options in sorted(options_by_cell.items())
        if options
    ]
    if not cells:
        return _emergency_fallback(
            env, profile, original_observations, known_tariffs, contacts,
            max_per_campaign, trace
        )

    greedy = _greedy(cells, budget, contacts, max_campaigns)
    beam = _beam_search(cells, budget, contacts, max_campaigns, beam_width)
    chosen = beam if _state_sort_key(beam) <= _state_sort_key(greedy) else greedy
    selected_keys = {option.tie_key for option in chosen.selected}

    for _, options in cells:
        for option in options:
            if option.tie_key not in selected_keys:
                selected_in_cell = any(
                    chosen_option.cell_id == option.cell_id
                    for chosen_option in chosen.selected
                )
                at_cap = len(chosen.selected) >= max_campaigns
                category = (
                    "same-cell alternative" if selected_in_cell
                    else "campaign cap" if at_cap
                    else "not selected by bounded portfolio search"
                )
                reason = (
                    "Another target/channel alternative was selected for this cell."
                    if selected_in_cell else
                    "Portfolio reached the campaign cap; bounded search selected a different joint tradeoff."
                    if at_cap else
                    "Not selected by bounded portfolio search; resource tradeoffs were evaluated jointly."
                )
                _append(trace, "rejected", option.candidate_id, reason,
                        {"category": category,
                         "channel": option.channel,
                         "conservative_net": option.conservative_net})

    ordered = sorted(
        chosen.selected,
        key=lambda option: (-option.conservative_net, option.tie_key),
    )
    campaigns = [_campaign(option, index) for index, option in enumerate(ordered, 1)]
    total_cost = sum(option.cost for option in ordered)
    total_contacts = sum(option.contacts for option in ordered)
    if (len(campaigns) > max_campaigns or total_cost > budget + 1e-9
            or total_contacts > contacts
            or len({option.cell_id for option in ordered}) != len(ordered)):
        _append(trace, "infeasible_fallback", None,
                "Internal validation rejected the planned portfolio.")
        return []

    for option in ordered:
        alternatives = [
            other for _, group in cells for other in group
            if other.cell_id == option.cell_id and other.tie_key != option.tie_key
        ]
        closest = min(
            alternatives,
            key=lambda other: abs(other.conservative_net - option.conservative_net),
            default=None,
        )
        _append(trace, "selected", option.candidate_id,
                "Selected by bounded portfolio search under joint resource limits.",
                {
                    "channel": option.channel,
                    "contacts": option.contacts,
                    "cost": option.cost,
                    "conservative_net": option.conservative_net,
                    "closest_alternative": None if closest is None else {
                        "candidate_id": closest.candidate_id,
                        "channel": closest.channel,
                        "conservative_net": closest.conservative_net,
                    },
                })
    _append(
        trace,
        "resource_totals",
        None,
        "Final-only estimates are a conservative proxy; pilot overlap is unknown.",
        {
            "campaign_count": len(campaigns),
            "final_contacts": total_contacts,
            "final_cost": float(total_cost),
            "conservative_final_net_proxy": float(chosen.objective),
            "remaining_budget_after_plan": float(budget - total_cost),
            "remaining_contacts_after_plan": int(contacts - total_contacts),
            "greedy_baseline_objective": float(greedy.objective),
            "search": "bounded beam heuristic",
        },
    )
    assert all(set(campaign).issubset(CAMPAIGN_KEYS) for campaign in campaigns)
    return campaigns
