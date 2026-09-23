"""Historical candidate generation and uncertain lift estimates."""

from dataclasses import dataclass, field
from pathlib import Path
from math import sqrt
import pandas as pd

NOISE_STD = 0.804
MAX_PER_CAMPAIGN = 5000

@dataclass
class Candidate:
    current: str
    segment: str
    target: str
    n: int
    arpu: float
    prior: float
    prior_n: float = 12.0  # small because historical and judging populations differ
    observations: list[tuple[float, int]] = field(default_factory=list)

    @property
    def posterior(self) -> tuple[float, float]:
        observed_n = sum(n for _, n in self.observations)
        total_n = self.prior_n + observed_n
        mean = (self.prior_n * self.prior +
                sum(ratio * n for ratio, n in self.observations)) / total_n
        return mean, NOISE_STD / sqrt(total_n)

    @property
    def key(self) -> tuple[str, str]:
        return self.current, self.segment


def _historical_priors(tariffs: pd.DataFrame) -> dict[tuple[str, str, str], float]:
    """Rough SMS lift prior from the *participant* history, never the mock model."""
    path = Path(__file__).resolve().parent.parent / "data" / "change_tariff.csv"
    if not path.exists():
        return {}
    history = pd.read_csv(path, usecols=[
        "AVG_ARPU_PREV_3M", "AVG_ARPU_NEXT_3M",
        "tariff_plan_code_from", "tariff_plan_code_to",
    ])
    history = history[history["AVG_ARPU_PREV_3M"] >= 100].copy()
    if history.empty:
        return {}
    history["segment"] = pd.cut(
        history["AVG_ARPU_PREV_3M"],
        bins=[-float("inf"), 1000, 5000, float("inf")],
        labels=["LOW", "MID", "HIGH"],
    )
    history["change"] = (
        (history["AVG_ARPU_NEXT_3M"] - history["AVG_ARPU_PREV_3M"])
        / history["AVG_ARPU_PREV_3M"]
    ).clip(-1, 3)
    counts = history.groupby(["tariff_plan_code_from", "segment"], observed=True).size()
    grouped = history.groupby(
        ["tariff_plan_code_from", "segment", "tariff_plan_code_to"], observed=True
    )["change"].agg(["mean", "size"])
    prior = {}
    for (current, segment, target), row in grouped.iterrows():
        # A smoothed observed transition share is merely a weak conversion prior.
        share = row["size"] / (counts[current, segment] + 20)
        prior[current, str(segment), target] = float(row["mean"] * share * 0.65)
    return prior


def _candidates(env) -> list[Candidate]:
    profile = env.customer_profile
    grouped = profile.groupby(["current_tariff", "arpu_segment"], observed=True).agg(
        n=("ID_NUMBER", "size"), arpu=("predicted_arpu", "mean")
    )
    prices = env.tariffs.set_index("tariff_plan_code")["price_tariff"].to_dict()
    tariff_names = list(prices)
    priors = _historical_priors(env.tariffs)
    median_price = max(float(env.tariffs["price_tariff"].median()), 1.0)
    result = []
    for (current, segment), row in grouped.iterrows():
        n = int(row["n"])
        if n < 80 or n > MAX_PER_CAMPAIGN:
            continue
        for target in tariff_names:
            if target == current:
                continue
            key = (current, str(segment), target)
            # For an unseen transition, tariff price supplies a deliberately
            # weak directional prior, not a simulated true effect.
            price_hint = max(-0.04, min(0.08,
                0.4 * (prices[target] - prices[current]) / median_price * 0.10 * 0.65))
            prior = priors.get(key, price_hint)
            result.append(Candidate(current, str(segment), target, n,
                                    float(row["arpu"]), prior))
    return result


def _prospects(candidates: list[Candidate], limit: int) -> list[Candidate]:
    """Explore distinct cells first, then alternate targets in promising cells."""
    by_cell: dict[tuple[str, str], list[Candidate]] = {}
    for candidate in candidates:
        by_cell.setdefault(candidate.key, []).append(candidate)
    for options in by_cell.values():
        options.sort(key=lambda c: c.prior * c.n * c.arpu, reverse=True)
    first = [options[0] for options in by_cell.values()]
    first.sort(key=lambda c: c.prior * c.n * c.arpu, reverse=True)
    selected = first[:min(limit, 11)]
    extras = [c for options in by_cell.values() for c in options[1:2]
              if options[0] in selected]
    extras.sort(key=lambda c: c.prior * c.n * c.arpu, reverse=True)
    return (selected + extras)[:limit]
