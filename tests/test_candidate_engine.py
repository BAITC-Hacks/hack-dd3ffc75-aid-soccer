"""Independent behavioral tests for the student-prior contract."""
from collections import Counter
from copy import deepcopy
import json
import math

import numpy as np
import pandas as pd
import pytest

from candidate_engine import generate_candidates


def catalog(codes=("a", "b", "c", "d")):
    return pd.DataFrame({"tariff_plan_code": list(codes),
                         "price_tariff": [1000 + 100 * i for i in range(len(codes))],
                         "Data_in_PKG": [1000 + 200 * i for i in range(len(codes))]})


def audience(rows=None):
    return pd.DataFrame(rows if rows is not None else [(1, "a", "MID", 2000.0)],
                        columns=["ID_NUMBER", "current_tariff", "arpu_segment", "predicted_arpu"])


def history(rows=()):
    return pd.DataFrame(rows, columns=["tariff_plan_code_from", "tariff_plan_code_to",
                                       "AVG_ARPU_PREV_3M", "AVG_ARPU_NEXT_3M"])


def by_target(result):
    return {c["target_tariff"]: c for c in result}


def event(trace, name):
    return next(e["details"] for e in trace if e["event"] == name)


def test_supported_positive_beats_two_observation_spike():
    past = history([("a", "b", 2000, 2400)] * 100 + [("a", "c", 2000, 2e9)] * 2)
    candidates = by_target(generate_candidates(audience(), past, catalog(("a", "b", "c"))))
    assert candidates["b"]["prior_score"] > candidates["c"]["prior_score"]
    assert candidates["b"]["history_count"] == 100
    assert candidates["c"]["history_count"] == 2
    assert 0 < candidates["c"]["prior_lift"] < 0.04
    assert candidates["c"]["raw_mean_lift"] > 10000


@pytest.mark.parametrize("past", [None, pd.DataFrame(), history()])
def test_empty_history_has_neutral_catalog_hypotheses(past):
    result = generate_candidates(audience(), past, catalog())
    assert result
    for item in result:
        assert item["source"] == "catalog"
        assert (item["history_count"], item["prior_lift"], item["positive_rate"]) == (0, 0.0, 0.5)
        assert 0 <= item["catalog_plausibility"] <= 1


@pytest.mark.parametrize("profile", [pd.DataFrame(), audience([]),
    audience([(1, None, "MID", 2000), (2, "a", "oops", 10), (3, "unknown", "MID", 20)])])
def test_no_valid_audience_returns_empty(profile):
    assert generate_candidates(profile, history(), catalog()) == []


def test_invalid_rows_are_audited_without_losing_zero_revenue():
    profile = audience([(1, "a", "MID", "2000"), (2, "a", "MID", 0),
                        (3, None, "MID", 10), (4, "a", " ", 10),
                        (5, "a", "MID", np.inf), (6, "a", "MID", -1),
                        ("bad", "a", "MID", 10), (7.5, "a", "MID", 10),
                        (8, "unknown", "MID", 10)])
    past = history([("a", "b", "2000", "2200"), ("a", "b", 0, 3),
                    ("a", "b", np.inf, 4), ("a", "b", 2, np.nan),
                    ("a", "missing", 2, 3), (None, "b", 2, 3),
                    ("a", "a", 2, 3), ("a", "b", 1e-320, 1e308)])
    trace = []
    result = generate_candidates(profile, past, catalog(), trace=trace)
    measured = next(c for c in result if c["source"] == "history")
    assert measured["history_count"] == 1
    assert measured["audience_size"] == measured["served_size"] == 2
    assert measured["served_arpu"] == 2000
    pa, ha = event(trace, "audience_audit"), event(trace, "history_audit")
    assert pa["excluded_rows"] == 7
    assert pa["invalid_revenue_rows"] == pa["invalid_id_rows"] == 2
    assert ha["usable_lift_rows"] == 1
    assert ha["nonfinite_relative_lift_rows"] == 1
    assert ha["nonpositive_pre_arpu_rows"] == 1
    assert ha["invalid_tariff_rows"] == 2
    json.dumps([result, trace], allow_nan=False)


def test_pre_switch_segment_boundaries():
    profile = audience([(1, "a", "LOW", 50), (2, "a", "MID", 1000), (3, "a", "HIGH", 6000)])
    past = history([("a", "b", 999, 9000), ("a", "b", 1000, 9000),
                    ("a", "b", 5000, 1), ("a", "b", 5001, 1)])
    result = generate_candidates(profile, past, catalog(("a", "b")))
    assert {c["arpu_segment"]: c["history_count"] for c in result} == {"LOW": 1, "MID": 2, "HIGH": 1}
    assert next(c for c in result if c["arpu_segment"] == "HIGH")["prior_lift"] < 0


def test_negative_zero_signals_and_configurable_shrinkage():
    past = history([("a", "b", 2000, 1000), ("a", "c", 2000, 2000)])
    result = by_target(generate_candidates(audience(), past, catalog(("a", "b", "c"))))
    assert result["b"]["prior_lift"] < 0
    assert result["b"]["prior_score"] < 0
    assert result["b"]["positive_rate"] == 0
    assert result["c"]["prior_lift"] == 0
    unshrunk = by_target(generate_candidates(audience(), past, catalog(("a", "b", "c")),
                                            config={"shrinkage_strength": 0}))
    assert unshrunk["b"]["prior_lift"] == -0.5
    assert abs(result["b"]["prior_lift"]) < abs(unshrunk["b"]["prior_lift"])


@pytest.mark.parametrize("majority,minority", [(2400, 1800), (1800, 2400)])
def test_pooled_winsorization_preserves_rare_opposite_sign(majority, minority):
    past = history([("a", "b", 2000, majority)] * 200 + [("a", "c", 2000, minority)])
    result = by_target(generate_candidates(audience(), past, catalog(("a", "b", "c"))))
    assert math.copysign(1, result["c"]["prior_lift"]) == math.copysign(1, minority - 2000)
    assert result["c"]["prior_lift"] != 0


def test_large_cell_uses_numeric_id_prefix_not_richest_contacts():
    profile = audience([(str(i), "a", "MID", 0 if i == 1 else 1 if i <= 5000 else 1e6)
                        for i in range(5010, 0, -1)])
    result = generate_candidates(profile, None, catalog(("a", "b")))[0]
    assert result["audience_size"] == 5010
    assert result["served_size"] == 5000
    assert result["served_arpu"] == 4999
    assert result["audience_arpu"] == 10004999


def test_immutable_inputs_and_reordered_rows_are_deterministic():
    frames = [audience([(3, "a", "MID", 0), (1, "a", "MID", 1000), (2, "b", "LOW", 50)]),
              history([("a", "b", 2000, 2800), ("a", "c", 2500, 2600),
                       ("b", "a", 200, 300), ("a", "b", 1000, 900)]), catalog()]
    saved = [f.copy(deep=True) for f in frames]
    config = {"candidate_limit": 6, "shrinkage_strength": 20, "unrelated": [1]}
    saved_config = deepcopy(config)
    trace = [{"existing": True}]
    first = generate_candidates(*frames, config=config, trace=trace)
    second = generate_candidates(*[f.sample(frac=1, random_state=7) for f in frames], config=config)
    assert first == second == generate_candidates(*frames, config=config)
    for original, snapshot in zip(frames, saved):
        pd.testing.assert_frame_equal(original, snapshot)
    assert config == saved_config and trace[0] == {"existing": True}
    assert first == sorted(first, key=lambda c: (-c["prior_score"], c["candidate_id"]))
    json.dumps([first, trace], allow_nan=False)


def test_diversity_and_catalog_reserve():
    codes = tuple("abcdefghi")
    profile = audience([(i, code, "MID", 1e7 if code == "a" else 2000)
                        for i, code in enumerate(codes[:6])])
    past = history([(code, target, 2000, 2600) for code in codes[:6]
                    for target in codes[1:5] if target != code] * 50)
    result = generate_candidates(profile, past, catalog(codes), config={"candidate_limit": 8})
    assert len(result) == 8
    assert len({c["cell_id"] for c in result}) == 6
    assert max(Counter(c["cell_id"] for c in result).values()) <= 2
    assert sum(c["source"] == "catalog" for c in result) >= 2
    for item in result:
        assert item["target_tariff"] in codes
        assert item["target_tariff"] != item["current_tariff"]
        assert item["audience_size"] > 0
        assert item["candidate_id"] == f'{item["cell_id"]}|{item["target_tariff"]}'


def test_catalog_uses_price_and_package_not_code_order():
    tariffs = pd.DataFrame({"tariff_plan_code": ["start", "z_good", "a_bad"],
                           "price_tariff": [2000.0, 2100.0, 2100.0], "Data_in_PKG": [1000, 1000, 0]})
    result = generate_candidates(audience([(1, "start", "MID", 2000)]), None, tariffs)
    assert result[0]["target_tariff"] == "z_good"
    tariffs.loc[2, "price_tariff"] = np.inf
    trace = []
    result = generate_candidates(audience([(1, "start", "MID", 2000)]), None, tariffs, trace=trace)
    assert event(trace, "catalog_audit")["invalid_price_rows"] == 1
    assert by_target(result)["a_bad"]["catalog_plausibility"] == 0
    json.dumps(result, allow_nan=False)


def test_no_targets_and_zero_limit():
    assert generate_candidates(audience(), None, catalog(("a",))) == []
    assert generate_candidates(audience(), None, catalog(())) == []
    assert generate_candidates(audience(), None, catalog(), config={"candidate_limit": 0}) == []


@pytest.mark.parametrize("config", [{"candidate_limit": -1}, {"candidate_limit": 1.5},
    {"candidate_limit": True}, {"shrinkage_strength": -1}, {"shrinkage_strength": np.inf}])
def test_invalid_config_raises(config):
    with pytest.raises(ValueError):
        generate_candidates(audience(), None, catalog(), config=config)


@pytest.mark.parametrize("which,column", [(0, "predicted_arpu"), (1, "AVG_ARPU_PREV_3M"),
                                         (2, "price_tariff")])
def test_required_columns_fail_clearly(which, column):
    frames = [audience(), history([("a", "b", 2000, 2200)]), catalog()]
    frames[which] = frames[which].drop(columns=column)
    with pytest.raises(ValueError, match=column):
        generate_candidates(*frames)


def test_duplicate_catalog_conflicts_and_blank_codes():
    tariffs = pd.concat([catalog(), pd.DataFrame({"tariff_plan_code": [None, " "],
                                                "price_tariff": [10, 20]})], ignore_index=True)
    trace = []
    assert generate_candidates(audience(), None, tariffs, trace=trace)
    assert event(trace, "catalog_audit")["invalid_code_rows"] == 2
    duplicate = pd.concat([catalog(), pd.DataFrame({"tariff_plan_code": ["a"], "price_tariff": [3]})])
    with pytest.raises(ValueError, match="conflicting duplicate"):
        generate_candidates(audience(), None, duplicate)


def test_nullable_missing_id_is_counted_as_invalid():
    profile = audience([(1, "a", "MID", 100), (2, "a", "MID", 100)])
    profile["ID_NUMBER"] = pd.Series([1, pd.NA], dtype="Int64")
    trace = []
    result = generate_candidates(profile, None, catalog(), trace=trace)
    assert result[0]["audience_size"] == 1
    assert event(trace, "audience_audit")["invalid_id_rows"] == 1
    assert event(trace, "audience_audit")["excluded_rows"] == 1


def test_extreme_finite_raw_lifts_keep_diagnostics_json_safe():
    past = history([("a", "b", 1, 1e308)] * 2)
    trace = []
    result = generate_candidates(audience([(1, "a", "LOW", 100)]), past,
                                 catalog(("a", "b")), trace=trace)
    assert result[0]["raw_median_lift"] == 1e308
    assert 0 < result[0]["prior_lift"] < 1
    json.dumps([result, trace], allow_nan=False)
