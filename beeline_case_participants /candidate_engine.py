"""Build data-driven tariff transition candidates for pilot campaigns.

The module deliberately does not run pilots and does not select channels.  Its
only responsibility is to rank hypotheses that ``Agent.act`` can validate via
``env.run_pilot``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ARPU_BINS = [-np.inf, 1000, 5000, np.inf]
ARPU_LABELS = ["LOW", "MID", "HIGH"]

HISTORY_COLUMNS = {
    "AVG_ARPU_PREV_3M",
    "AVG_ARPU_NEXT_3M",
    "ID_NUMBER",
    "tariff_plan_code_from",
    "tariff_plan_code_to",
}
PROFILE_COLUMNS = {
    "ID_NUMBER",
    "current_tariff",
    "arpu_segment",
    "predicted_arpu",
}


def _require_columns(frame: pd.DataFrame, required: set[str], frame_name: str) -> None:
    """Raise an actionable error when an input file has an unexpected schema."""
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def build_candidates(
    change_history: pd.DataFrame,
    customer_profile: pd.DataFrame,
    tariffs: pd.DataFrame | None = None,
    *,
    min_history: int = 5,
    shrinkage: float = 20.0,
    max_candidates: int | None = 50,
) -> pd.DataFrame:
    """Return ranked tariff-transition hypotheses for later pilot testing.

    ``historical_lift`` is the winsorised mean relative ARPU change among
    observed switchers.  Because small groups are noisy, ``conservative_lift``
    shrinks that estimate towards zero.  ``transition_share`` is used only as a
    proxy for how common a transition was; the real conversion effect must be
    learned from pilots.

    The score is a prioritisation heuristic, not a promised financial result:

        conservative lift * transition share * addressable predicted ARPU
    """
    if min_history < 1:
        raise ValueError("min_history must be at least 1")
    if shrinkage < 0:
        raise ValueError("shrinkage cannot be negative")
    if max_candidates is not None and max_candidates < 1:
        raise ValueError("max_candidates must be positive or None")

    _require_columns(change_history, HISTORY_COLUMNS, "change_history")
    _require_columns(customer_profile, PROFILE_COLUMNS, "customer_profile")

    history = change_history.copy()
    history = history[
        history["AVG_ARPU_PREV_3M"].notna()
        & history["AVG_ARPU_NEXT_3M"].notna()
        & (history["AVG_ARPU_PREV_3M"] >= 100)
    ].copy()
    history = history[
        history["tariff_plan_code_from"] != history["tariff_plan_code_to"]
    ].copy()

    history["arpu_segment"] = pd.cut(
        history["AVG_ARPU_PREV_3M"],
        bins=ARPU_BINS,
        labels=ARPU_LABELS,
    ).astype("object")
    history["lift"] = (
        (history["AVG_ARPU_NEXT_3M"] - history["AVG_ARPU_PREV_3M"])
        / history["AVG_ARPU_PREV_3M"]
    ).clip(-1.0, 3.0)
    history["is_positive"] = history["lift"] > 0

    keys = ["tariff_plan_code_from", "tariff_plan_code_to", "arpu_segment"]
    grouped = (
        history.groupby(keys, observed=True)
        .agg(
            history_count=("ID_NUMBER", "size"),
            historical_lift=("lift", "mean"),
            median_lift=("lift", "median"),
            positive_rate=("is_positive", "mean"),
        )
        .reset_index()
    )

    outgoing = (
        history.groupby(["tariff_plan_code_from", "arpu_segment"], observed=True)
        .size()
        .rename("outgoing_history_count")
        .reset_index()
    )
    grouped = grouped.merge(
        outgoing,
        on=["tariff_plan_code_from", "arpu_segment"],
        how="left",
        validate="many_to_one",
    )
    grouped["transition_share"] = (
        grouped["history_count"] / grouped["outgoing_history_count"]
    )

    profile = customer_profile.copy()
    profile = profile[
        profile["current_tariff"].notna()
        & profile["arpu_segment"].isin(ARPU_LABELS)
        & profile["predicted_arpu"].notna()
    ].copy()
    audience = (
        profile.groupby(["current_tariff", "arpu_segment"], observed=True)
        .agg(
            audience_size=("ID_NUMBER", "size"),
            audience_predicted_arpu=("predicted_arpu", "sum"),
            mean_predicted_arpu=("predicted_arpu", "mean"),
        )
        .reset_index()
    )

    candidates = grouped.merge(
        audience,
        left_on=["tariff_plan_code_from", "arpu_segment"],
        right_on=["current_tariff", "arpu_segment"],
        how="inner",
        validate="many_to_one",
    )

    if tariffs is not None:
        if "tariff_plan_code" not in tariffs.columns:
            raise ValueError("tariffs is missing required column: tariff_plan_code")
        valid_tariffs = set(tariffs["tariff_plan_code"].dropna())
        candidates = candidates[
            candidates["tariff_plan_code_from"].isin(valid_tariffs)
            & candidates["tariff_plan_code_to"].isin(valid_tariffs)
        ].copy()

    candidates = candidates[candidates["history_count"] >= min_history].copy()
    candidates["reliability"] = candidates["history_count"] / (
        candidates["history_count"] + shrinkage
    )

    # Blending mean and median reduces the influence of the remaining outliers.
    candidates["robust_lift"] = (
        0.7 * candidates["historical_lift"] + 0.3 * candidates["median_lift"]
    )
    candidates["conservative_lift"] = (
        candidates["robust_lift"] * candidates["reliability"]
    )
    candidates["candidate_score"] = (
        candidates["conservative_lift"]
        * candidates["transition_share"]
        * candidates["audience_predicted_arpu"]
    )

    candidates = candidates[
        (candidates["conservative_lift"] > 0)
        & (candidates["positive_rate"] >= 0.5)
        & (candidates["audience_size"] > 0)
    ].copy()

    # The audience join keeps its own ``current_tariff`` column because the
    # history uses a different key name.  Drop that duplicate before exposing
    # the canonical public column.
    candidates = candidates.drop(columns=["current_tariff"])
    candidates = candidates.rename(
        columns={
            "tariff_plan_code_from": "current_tariff",
            "tariff_plan_code_to": "target_tariff",
        }
    )

    output_columns = [
        "current_tariff",
        "target_tariff",
        "arpu_segment",
        "audience_size",
        "audience_predicted_arpu",
        "mean_predicted_arpu",
        "history_count",
        "outgoing_history_count",
        "historical_lift",
        "median_lift",
        "positive_rate",
        "transition_share",
        "reliability",
        "conservative_lift",
        "candidate_score",
    ]
    candidates = candidates.sort_values(
        ["candidate_score", "history_count"], ascending=[False, False]
    ).reset_index(drop=True)

    if max_candidates is not None:
        candidates = candidates.head(max_candidates)
    return candidates[output_columns]


def load_default_candidates(
    base_dir: str | Path = ".", **kwargs: object
) -> pd.DataFrame:
    """Load starter-kit CSV files and call :func:`build_candidates`."""
    base = Path(base_dir)
    history = pd.read_csv(base / "data" / "change_tariff.csv")
    profile = pd.read_csv(base / "customer_profile.csv")
    tariffs = pd.read_csv(base / "data" / "dict_tariff.csv")
    return build_candidates(history, profile, tariffs, **kwargs)


if __name__ == "__main__":
    result = load_default_candidates()
    pd.set_option("display.max_columns", None)
    print(result.head(20).to_string(index=False))
