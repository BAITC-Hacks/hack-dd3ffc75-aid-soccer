import unittest

import pandas as pd

from candidate_engine import build_candidates


class CandidateEngineTests(unittest.TestCase):
    def test_profitable_supported_transition_ranks_first(self):
        history = pd.DataFrame(
            {
                "AVG_ARPU_PREV_3M": [2000, 2100, 2200, 2000, 2100, 2200],
                "AVG_ARPU_NEXT_3M": [2600, 2700, 2800, 1800, 1900, 2000],
                "ID_NUMBER": range(1, 7),
                "tariff_plan_code_from": ["tariff_1"] * 6,
                "tariff_plan_code_to": ["tariff_2"] * 3 + ["tariff_3"] * 3,
            }
        )
        profile = pd.DataFrame(
            {
                "ID_NUMBER": [101, 102],
                "current_tariff": ["tariff_1", "tariff_1"],
                "arpu_segment": ["MID", "MID"],
                "predicted_arpu": [2000.0, 2500.0],
            }
        )
        tariffs = pd.DataFrame(
            {"tariff_plan_code": ["tariff_1", "tariff_2", "tariff_3"]}
        )

        result = build_candidates(
            history,
            profile,
            tariffs,
            min_history=2,
            shrinkage=2,
            max_candidates=None,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["current_tariff"], "tariff_1")
        self.assertEqual(result.iloc[0]["target_tariff"], "tariff_2")
        self.assertGreater(result.iloc[0]["candidate_score"], 0)

    def test_missing_profile_columns_raise_clear_error(self):
        with self.assertRaisesRegex(ValueError, "customer_profile"):
            build_candidates(pd.DataFrame(columns=sorted({
                "AVG_ARPU_PREV_3M",
                "AVG_ARPU_NEXT_3M",
                "ID_NUMBER",
                "tariff_plan_code_from",
                "tariff_plan_code_to",
            })), pd.DataFrame())


if __name__ == "__main__":
    unittest.main()
