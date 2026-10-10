from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from core.calendar_samples import (
    build_calendar_split,
    build_sample_plan,
    prepare_calendar_frame,
)
from core.protocol_hash import PROTOCOL_VERSION, protocol_hash


class CalendarSamplePlanTest(unittest.TestCase):
    def setUp(self) -> None:
        dates = pd.to_datetime(
            ["2021-01-01", "2021-01-02", "2021-01-04", "2021-01-05", "2021-01-06"]
        )
        self.x = pd.DataFrame({"x": np.arange(len(dates), dtype=float)})
        self.y = pd.Series(np.arange(len(dates), dtype=float) * 10, name="target")
        self.time = dates

    def test_gap_breaks_history_and_unavailable_target_is_dropped(self) -> None:
        plan = build_sample_plan(
            self.x,
            self.y,
            self.time,
            feature_names=["x"],
            target_name="target",
            window_size=1,
            past_lag_days=1,
            forecast_horizon_days=1,
        )

        retained = plan.retained_records()
        self.assertEqual(len(retained), 1)
        record = retained[0]
        self.assertEqual(record.forecast_origin.date().isoformat(), "2021-01-05")
        self.assertEqual(record.target_time.date().isoformat(), "2021-01-06")
        self.assertEqual(record.feature_dates, ["2021-01-04"])
        self.assertEqual(record.drop_reason, None)

        dropped_by_reason = plan.drop_reason_counts()
        self.assertEqual(dropped_by_reason["history_gap"], 2)
        self.assertEqual(dropped_by_reason["target_unavailable"], 2)
        self.assertNotIn("feature_missing", dropped_by_reason)

    def test_duplicate_calendar_dates_keep_first_row_deterministically(self) -> None:
        frame = prepare_calendar_frame(
            pd.DataFrame({"x": [1.0, 2.0, 3.0]}),
            pd.Series([10.0, 20.0, 30.0], name="target"),
            pd.Series(["2021-01-01", "2021-01-01", "2021-01-02"]),
        )
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame["x"].tolist(), [1.0, 3.0])
        self.assertEqual(frame["_hp_target"].tolist(), [10.0, 30.0])


class CalendarSplitTest(unittest.TestCase):
    def test_split_boundary_is_calendar_and_nonempty(self) -> None:
        dates = pd.date_range("2021-01-01", periods=10, freq="D")
        train_mask, split_date = build_calendar_split(dates, test_rate=0.25)

        self.assertEqual(split_date.date().isoformat(), "2021-01-08")
        train_dates = dates[train_mask]
        test_dates = dates[~train_mask]
        self.assertTrue((train_dates <= split_date).all())
        self.assertTrue((test_dates > split_date).all())
        self.assertGreater(len(train_dates), 0)
        self.assertGreater(len(test_dates), 0)


class ProtocolHashTest(unittest.TestCase):
    def test_protocol_hash_changes_with_protocol_parameters(self) -> None:
        base = {
            "model": "ridge",
            "alpha": 1.0,
            "features": ["By", "Bz", "Shadow"],
            "forecast_horizon_days": 3,
        }
        changed_alpha = {**base, "alpha": 2.0}
        changed_horizon = {**base, "forecast_horizon_days": 5}
        changed_features = {**base, "features": ["By"]}

        self.assertEqual(protocol_hash(base), protocol_hash(dict(base)))
        self.assertNotEqual(protocol_hash(base), protocol_hash(changed_alpha))
        self.assertNotEqual(protocol_hash(base), protocol_hash(changed_horizon))
        self.assertNotEqual(protocol_hash(base), protocol_hash(changed_features))

    def test_protocol_version_is_defined(self) -> None:
        self.assertTrue(PROTOCOL_VERSION.startswith("research-engine/"))


if __name__ == "__main__":
    unittest.main()
