"""Recompute the public-data audit: python -m tools.profile_data."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from candidate_engine import generate_candidates


def _blank(series):
    return series.isna() | series.map(lambda x: isinstance(x, str) and not x.strip())


def _ids(frame):
    numeric = pd.to_numeric(frame.ID_NUMBER, errors="coerce")
    valid = (np.isfinite(numeric) & (numeric >= 0) & (numeric == np.floor(numeric))).fillna(False)
    return numeric[valid]


def _identity_stats(frame):
    ids = _ids(frame)
    return {"rows": len(frame), "unique_ids": int(ids.nunique()),
            "duplicate_id_rows": int(ids.duplicated().sum()),
            "invalid_id_rows": len(frame) - len(ids)}


def build_audit(profile, history, tariffs):
    """Return aggregate diagnostics without exporting any customer IDs."""
    trace = []
    candidates = generate_candidates(profile, history, tariffs, trace=trace)
    events = {e["event"]: e["details"] for e in trace}
    known = set(tariffs.tariff_plan_code.dropna())
    from_known = history.tariff_plan_code_from.isin(known)
    to_known = history.tariff_plan_code_to.isin(known)
    pairs = history.loc[from_known & to_known,
                        ["tariff_plan_code_from", "tariff_plan_code_to"]].drop_duplicates()
    return {
        "schema_version": "1.0",
        "profile": {**_identity_stats(profile), **events["audience_audit"]},
        "history": {
            **_identity_stats(history), **events["history_audit"],
            "blank_from_tariff_rows": int(_blank(history.tariff_plan_code_from).sum()),
            "blank_to_tariff_rows": int(_blank(history.tariff_plan_code_to).sum()),
            "unknown_target_rows": int((~to_known & ~_blank(history.tariff_plan_code_to)).sum()),
            "known_target_rows": int(to_known.sum()),
            "known_target_fraction": float(to_known.mean()) if len(history) else None,
            "directed_transition_pairs": len(pairs),
            "nonself_directed_transition_pairs": int(
                pairs.tariff_plan_code_from.ne(pairs.tariff_plan_code_to).sum()),
        },
        "catalog": events["catalog_audit"],
        "history_profile_id_intersection": len(set(_ids(profile)) & set(_ids(history))),
        "candidate_pool": {**events["pool_selected"],
                           "example_candidate": candidates[0] if candidates else None},
        "notes": [
            "Synthetic hackathon data, not actual Beeline business evidence.",
            "Exclusion counters overlap; usable_rows/excluded_rows count their union.",
            "Duplicate counts mean rows after the first occurrence of a valid numeric ID.",
            "History_count counts usable switches, not contacts or independent pilot samples.",
            "History ranks hypotheses; positive_rate is not a conversion probability.",
            "Invalid profile rows are excluded locally; campaign filters cannot remove such rows in env.",
            "Duplicate profile IDs are counted as rows; scorer deduplication may change realized revenue.",
        ],
    }


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=root / "task" / "beeline_case_participants (1)")
    parser.add_argument("--output", type=Path, default=root / "reports" / "data_audit.json")
    args = parser.parse_args(argv)
    # Prefer the requested pack. A clean checkout of this existing feature
    # branch also has a legacy copy of the public participant data.
    data_dir = args.data_dir
    if not data_dir.exists() and data_dir == root / "task" / "beeline_case_participants (1)":
        data_dir = root / "beeline_case_participants "
    profile = pd.read_csv(data_dir / "customer_profile.csv")
    history = pd.read_csv(data_dir / "data" / "change_tariff.csv")
    tariffs = pd.read_csv(data_dir / "data" / "dict_tariff.csv")
    audit = build_audit(profile, history, tariffs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Profile: {len(profile)} rows; history: {len(history)} rows; tariffs: {len(tariffs)}")
    print(f"Usable lifts: {audit['history']['usable_lift_rows']}; "
          f"selected candidates: {audit['candidate_pool']['selected']}")


if __name__ == "__main__":
    main()
