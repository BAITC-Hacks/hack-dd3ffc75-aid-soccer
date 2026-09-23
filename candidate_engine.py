"""Historical ranking priors and diverse hypotheses; no pilot or environment access."""

from collections import Counter
import math

import numpy as np
import pandas as pd


_PROFILE_COLUMNS = {"ID_NUMBER", "current_tariff", "arpu_segment", "predicted_arpu"}
_HISTORY_COLUMNS = {
    "tariff_plan_code_from", "tariff_plan_code_to",
    "AVG_ARPU_PREV_3M", "AVG_ARPU_NEXT_3M",
}
_PACKAGES = (
    "Data_in_PKG", "Min_another_operator_in_PKG",
    "Min_another_operator_and_city_in_PKG",
)


def _require_columns(frame, required, name):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {', '.join(missing)}")


def _blank(series):
    return series.isna() | series.map(lambda x: isinstance(x, str) and not x.strip())


def _label_valid(value):
    # Exact labels must survive the public filter syntax and stable ID separator.
    return (isinstance(value, str) and bool(value) and value == value.strip()
            and "|" not in value and ";" not in value)


def _numbers(series):
    return pd.to_numeric(series, errors="coerce").astype(float)


def _sum(values):
    result = math.fsum(float(v) for v in values)
    if not math.isfinite(result):
        raise ValueError("ARPU total is not finite")
    return result


def _median(values):
    ordered = sorted(float(v) for v in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    # Avoid overflow when two individually finite raw lifts are enormous.
    return ordered[middle - 1] / 2.0 + ordered[middle] / 2.0


def _catalog(tariffs):
    _require_columns(tariffs, {"tariff_plan_code", "price_tariff"}, "tariffs")
    columns = ["tariff_plan_code", "price_tariff"] + [
        name for name in _PACKAGES if name in tariffs.columns
    ]
    frame = tariffs[columns].copy().reset_index(drop=True)
    valid = frame.tariff_plan_code.map(_label_valid).astype(bool)
    details = {"rows": len(frame), "invalid_code_rows": int((~valid).sum())}
    frame = frame.loc[valid].copy()
    for name in columns[1:]:
        values = _numbers(frame[name])
        frame[name] = values.where(np.isfinite(values) & (values >= 0))
    details["invalid_price_rows"] = int(frame.price_tariff.isna().sum())
    frame = frame.drop_duplicates()
    if frame.tariff_plan_code.duplicated().any():
        raise ValueError("tariffs has conflicting duplicate tariff_plan_code rows")
    catalog = {}
    for row in frame.sort_values("tariff_plan_code").to_dict("records"):
        code = row.pop("tariff_plan_code")
        catalog[code] = {k: None if pd.isna(v) else float(v) for k, v in row.items()}
    details["known_tariffs"] = len(catalog)
    return catalog, details


def _audience(profile, known):
    if profile.empty:
        profile = profile.reindex(columns=sorted(_PROFILE_COLUMNS))
    _require_columns(profile, _PROFILE_COLUMNS, "profile")
    frame = profile[sorted(_PROFILE_COLUMNS)].copy().reset_index(drop=True)
    ids = pd.to_numeric(frame.ID_NUMBER, errors="coerce")
    revenue = _numbers(frame.predicted_arpu)
    valid_id = (np.isfinite(ids) & (ids >= 0) & (ids == np.floor(ids))).fillna(False)
    valid_revenue = np.isfinite(revenue) & (revenue >= 0)
    valid_tariff = frame.current_tariff.isin(known)
    valid_segment = frame.arpu_segment.isin(["LOW", "MID", "HIGH"])
    valid = valid_id & valid_revenue & valid_tariff & valid_segment
    details = {
        "rows": len(frame), "usable_rows": int(valid.sum()),
        "excluded_rows": int((~valid).sum()),
        "blank_current_tariff_rows": int(_blank(frame.current_tariff).sum()),
        "invalid_current_tariff_rows": int((~valid_tariff).sum()),
        "blank_arpu_segment_rows": int(_blank(frame.arpu_segment).sum()),
        "invalid_arpu_segment_rows": int((~valid_segment).sum()),
        "invalid_id_rows": int((~valid_id).sum()),
        "duplicate_numeric_id_rows": int(ids[valid_id].duplicated().sum()),
        "invalid_revenue_rows": int((~valid_revenue).sum()),
        "zero_revenue_rows": int((revenue == 0).sum()),
    }
    frame["ID_NUMBER"] = ids
    frame["predicted_arpu"] = revenue
    cells = []
    for (current, segment), group in frame.loc[valid].groupby(
        ["current_tariff", "arpu_segment"], sort=True, observed=True
    ):
        # Revenue is only a deterministic tie-break for duplicate IDs, never
        # the primary audience selection criterion. No rows are deduplicated.
        group = group.sort_values(["ID_NUMBER", "predicted_arpu"])
        served = group.head(5000)
        cells.append({
            "cell_id": f"{current}|{segment}", "current_tariff": current,
            "arpu_segment": segment, "audience_size": len(group),
            "audience_arpu": _sum(group.predicted_arpu), "served_size": len(served),
            "served_arpu": _sum(served.predicted_arpu),
        })
    details["cells"] = cells
    return cells, details


def _historical_priors(history, known, strength):
    if history is None or (isinstance(history, pd.DataFrame) and history.empty):
        history = pd.DataFrame(columns=sorted(_HISTORY_COLUMNS))
    _require_columns(history, _HISTORY_COLUMNS, "history")
    frame = history[sorted(_HISTORY_COLUMNS)].copy().reset_index(drop=True)
    prev = _numbers(frame.AVG_ARPU_PREV_3M)
    following = _numbers(frame.AVG_ARPU_NEXT_3M)
    finite = np.isfinite(prev) & np.isfinite(following)
    valid_labels = (frame.tariff_plan_code_from.isin(known)
                    & frame.tariff_plan_code_to.isin(known))
    self_transition = frame.tariff_plan_code_from.eq(frame.tariff_plan_code_to).fillna(False)
    eligible = finite & (prev > 0) & valid_labels & ~self_transition
    raw = pd.Series(np.nan, index=frame.index, dtype=float)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        raw.loc[eligible] = (following[eligible] - prev[eligible]) / prev[eligible]
    usable = eligible & np.isfinite(raw)
    frame = frame.loc[usable].copy()
    frame["arpu_segment"] = np.where(
        prev[usable] < 1000, "LOW", np.where(prev[usable] <= 5000, "MID", "HIGH")
    )
    frame["raw_lift"] = raw[usable]
    values = np.sort(raw[usable].to_numpy())
    # Bound before computing quantiles: a tiny denominator cannot dictate a
    # giant robust cutoff, including in datasets too small for quantiles.
    bounded = np.clip(values, -1.0, 1.0)
    lower, upper = -1.0, 1.0
    if len(values) >= 20:
        q_low, q_high = np.quantile(bounded, [0.01, 0.99])
        # Keep a rare opposite-sign observation even when pooled quantiles
        # would erase its sign; use the hard bound on that side instead.
        lower = float(q_low) if q_low < 0 else (-1.0 if values[0] < 0 else 0.0)
        upper = float(q_high) if q_high > 0 else (1.0 if values[-1] > 0 else 0.0)
    frame["robust_lift"] = frame.raw_lift.clip(lower, upper)
    aggregates = {}
    for key, group in frame.groupby(
        ["tariff_plan_code_from", "arpu_segment", "tariff_plan_code_to"],
        sort=True, observed=True,
    ):
        n = len(group)
        robust_mean = math.fsum(sorted(group.robust_lift)) / n
        aggregates[key] = {
            "history_count": n, "prior_lift": float(robust_mean * (n / (n + strength))),
            "positive_rate": float((group.raw_lift > 0).sum() / n),
            "raw_mean_lift": math.fsum(sorted(group.raw_lift / n)),
            "raw_median_lift": _median(group.raw_lift),
            "robust_mean_lift": float(robust_mean),
        }
    details = {
        "rows": len(history), "usable_lift_rows": len(frame),
        "excluded_rows": int(len(history) - len(frame)),
        "nonfinite_arpu_rows": int((~finite).sum()),
        "nonpositive_pre_arpu_rows": int((np.isfinite(prev) & (prev <= 0)).sum()),
        "invalid_tariff_rows": int((~valid_labels).sum()),
        "self_transition_rows": int(self_transition.sum()),
        "nonfinite_relative_lift_rows": int((eligible & ~np.isfinite(raw)).sum()),
        "exact_history_cells": len(aggregates),
        "winsor_lower": lower, "winsor_upper": upper,
        "clipped_lift_rows": int(((values < lower) | (values > upper)).sum()),
        "abs_raw_lift_above_one_rows": int((np.abs(values) > 1).sum()),
        "raw_lift_min": float(values[0]) if len(values) else None,
        "raw_lift_max": float(values[-1]) if len(values) else None,
        "raw_lift_mean": math.fsum(values / len(values)) if len(values) else None,
        "raw_lift_median": _median(values) if len(values) else None,
    }
    return aggregates, details


def _catalog_plausibility(current, target):
    old_price, new_price = current["price_tariff"], target["price_tariff"]
    if old_price is None or new_price is None:
        return {"catalog_plausibility": 0.0, "catalog_price_similarity": None,
                "catalog_package_retention": None}
    price_similarity = math.exp(-abs(math.log1p(new_price) - math.log1p(old_price)))
    retention = []
    for name in _PACKAGES:
        old, new = current.get(name), target.get(name)
        if old is not None and new is not None and old > 0:
            retention.append(min(new / old, 1.0))
    package_retention = math.fsum(retention) / len(retention) if retention else None
    plausibility = price_similarity * (
        0.5 + 0.5 * package_retention if package_retention is not None else 1.0
    )
    return {"catalog_plausibility": plausibility,
            "catalog_price_similarity": price_similarity,
            "catalog_package_retention": package_retention}


def _diverse_pool(candidates, limit):
    ordered = sorted(candidates, key=lambda c: (-c["prior_score"], c["candidate_id"]))
    selected, seen, per_cell = [], set(), Counter()

    def take(source, count, cell_cap):
        for item in ordered:
            if count <= 0 or len(selected) >= limit:
                break
            if (item["candidate_id"] not in seen and per_cell[item["cell_id"]] < cell_cap
                    and (source is None or item["source"] == source)):
                selected.append(item)
                seen.add(item["candidate_id"])
                per_cell[item["cell_id"]] += 1
                count -= 1

    # Reserve 20% (when limit >= 2) for unobserved exact-cell transitions.
    reserve = min(math.ceil(limit * 0.2), limit - 1) if limit >= 2 else 0
    take("history", limit - reserve, 1)
    before = len(selected)
    take("catalog", reserve, 1)
    take("catalog", reserve - (len(selected) - before), 2)
    take(None, limit - len(selected), 1)
    take(None, limit - len(selected), 2)
    return sorted(selected, key=lambda c: (-c["prior_score"], c["candidate_id"]))


def generate_candidates(profile, history, tariffs, *, config=None, trace=None) -> list[dict]:
    """Propose deterministic experiments using public DataFrames only.

    History: relative (after-before)/before, with pre-switch LOW <1000,
    MID 1000..5000, HIGH >5000. Clip lifts to [-1, 1]; with >=20 usable
    observations also use pooled 1%/99% quantiles of bounded lifts, extending
    the interval to contain zero. If a quantile would erase an observed sign,
    retain the hard bound on that side. Shrink the mean by n/(n+strength).
    Historical score = served_arpu * prior_lift * (0.5 + 0.5*positive_rate).
    This signed ranking score is neither campaign profit nor conversion.

    Catalog score = served_arpu * 0.025 * plausibility. The 0.025 constant
    calibrates ranking ONLY, never prior_lift. Plausibility rewards similar
    log(1+price) and retention of existing nonzero packages; missing package
    columns use price alone, missing prices score zero. Catalog historical
    fields stay neutral. Reserve ~20% of the pool for catalog exploration;
    take one historical target per cell before second alternatives, with at
    most two targets per cell. Negative historical hypotheses remain eligible.

    Config: candidate_limit=40 (nonnegative integer), shrinkage_strength=50
    (finite nonnegative number); unrelated keys are ignored. Served ARPU is
    in supplied currency units, lift is a ratio, counts are rows/contacts.
    Invalid labels/IDs/revenue are excluded locally and counted in trace;
    zero revenue is valid. Input frames are never changed. Malformed nonempty
    schemas raise ValueError; empty history/None enables catalog exploration.
    """
    settings = {} if config is None else config
    limit = settings.get("candidate_limit", 40)
    strength = settings.get("shrinkage_strength", 50.0)
    if isinstance(limit, bool) or not isinstance(limit, (int, np.integer)) or limit < 0:
        raise ValueError("candidate_limit must be a nonnegative integer")
    try:
        strength = float(strength)
    except (TypeError, ValueError) as exc:
        raise ValueError("shrinkage_strength must be finite and nonnegative") from exc
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("shrinkage_strength must be finite and nonnegative")
    catalog, catalog_audit = _catalog(tariffs)
    cells, audience_audit = _audience(profile, catalog)
    priors, history_audit = _historical_priors(history, catalog, strength)
    candidates = []
    for cell in cells:
        for target in catalog:
            if target == cell["current_tariff"]:
                continue
            prior = priors.get((cell["current_tariff"], cell["arpu_segment"], target))
            candidate = dict(cell, candidate_id=f'{cell["cell_id"]}|{target}',
                             target_tariff=target)
            if prior is not None:
                candidate.update(prior, source="history")
                score = prior["prior_lift"] * (0.5 + 0.5 * prior["positive_rate"])
            else:
                candidate.update(history_count=0, prior_lift=0.0, positive_rate=0.5,
                                 source="catalog", raw_mean_lift=None,
                                 raw_median_lift=None, robust_mean_lift=None)
                heuristic = _catalog_plausibility(catalog[cell["current_tariff"]], catalog[target])
                candidate.update(heuristic)
                score = 0.025 * heuristic["catalog_plausibility"]
            candidate["prior_score"] = float(cell["served_arpu"] * score)
            candidates.append(candidate)
    result = _diverse_pool(candidates, int(limit))
    if trace is not None:
        for name, details in [("catalog_audit", catalog_audit),
                              ("audience_audit", audience_audit),
                              ("history_audit", history_audit)]:
            trace.append({"stage": "candidate", "event": name, "candidate_id": None,
                          "reason": "Validate public data; exclusion counts can overlap.",
                          "details": details})
        trace.append({"stage": "candidate", "event": "pool_selected", "candidate_id": None,
                      "reason": "Diverse ranking hypotheses require validation by pilots.",
                      "details": {"eligible_candidates": len(candidates), "selected": len(result),
                                  "selected_cells": len({c["cell_id"] for c in result}),
                                  "selected_catalog": sum(c["source"] == "catalog" for c in result),
                                  "candidate_limit": int(limit), "shrinkage_strength": strength}})
    return result
