"""Budget-aware, disjoint final campaign allocation."""

from .candidates import Candidate
from .experiments import PILOT_CHANNEL

MAX_CAMPAIGNS = 10

def _plan(env, tested: list[Candidate]) -> list[dict]:
    """Pick disjoint cells, then spend money on profitable channel upgrades."""
    sms_multiplier = env.channels[PILOT_CHANNEL]["conversion_multiplier"]
    options = []
    for candidate in tested:
        if not candidate.observations:
            continue
        mean, sd = candidate.posterior
        conservative = mean - 0.75 * sd
        if conservative <= 0:
            continue
        push = env.channels["push"]
        push_lift = conservative * push["conversion_multiplier"] / sms_multiplier
        push_gain = candidate.n * candidate.arpu * push_lift
        if push_gain > 0:
            options.append((push_gain / candidate.n, push_gain, candidate))
    options.sort(key=lambda item: item[0], reverse=True)

    selected: list[Candidate] = []
    used_cells = set()
    contacts_left = int(env.remaining_contacts)
    for _, _, candidate in options:
        if len(selected) >= MAX_CAMPAIGNS:
            break
        if candidate.key in used_cells or candidate.n > contacts_left:
            continue
        selected.append(candidate)
        used_cells.add(candidate.key)
        contacts_left -= candidate.n

    # Free push gives a feasible base plan. Upgrade one chosen cell at a time
    # using incremental conservative gain per additional currency unit.
    channels = ["push"] * len(selected)
    budget_left = float(env.remaining_budget)
    while True:
        best = None
        for index, candidate in enumerate(selected):
            mean, sd = candidate.posterior
            safe = max(0.0, mean - 0.75 * sd)
            current = channels[index]
            for alternative in env.channels:
                old_cost = env.channels[current]["cost_per_contact"]
                new_cost = env.channels[alternative]["cost_per_contact"]
                extra_cost = candidate.n * (new_cost - old_cost)
                if extra_cost <= 0 or extra_cost > budget_left:
                    continue
                old_mult = env.channels[current]["conversion_multiplier"]
                new_mult = env.channels[alternative]["conversion_multiplier"]
                extra_lift = candidate.n * candidate.arpu * safe * (
                    new_mult - old_mult) / sms_multiplier
                extra_net = extra_lift - extra_cost
                if extra_net <= 0:
                    continue
                efficiency = extra_net / extra_cost
                if best is None or efficiency > best[0]:
                    best = (efficiency, extra_cost, index, alternative)
        if best is None:
            break
        _, extra_cost, index, alternative = best
        channels[index] = alternative
        budget_left -= extra_cost

    campaigns = []
    for index, (candidate, channel) in enumerate(zip(selected, channels), 1):
        campaigns.append({
            "campaign_name": f"pilot_plan_{index}_{candidate.current}_{candidate.target}",
            "filter_arpu_segment": candidate.segment,
            "filter_current_tariff": candidate.current,
            "target_tariff": candidate.target,
            "channel": channel,
        })
    return campaigns
