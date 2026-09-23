"""Documented pilot API adapter."""

from .candidates import Candidate

PILOT_CHANNEL = "sms"

def _pilot(env, candidate: Candidate, requested: int) -> bool:
    if env.pilots_left <= 0 or env.remaining_contacts < 10:
        return False
    price = env.channels[PILOT_CHANNEL]["cost_per_contact"]
    affordable = int(env.remaining_budget // price) if price else env.remaining_contacts
    n = min(requested, 200, candidate.n, env.remaining_contacts, affordable)
    if n < 10:
        return False
    try:
        result = env.run_pilot(
            target_tariff=candidate.target, channel=PILOT_CHANNEL,
            n_customers=n, filter_arpu_segment=candidate.segment,
            filter_current_tariff=candidate.current,
        )
    except (RuntimeError, ValueError):
        return False
    candidate.observations.append((float(result["observed_lift_ratio"]),
                                   int(result["n_customers"])))
    return True
