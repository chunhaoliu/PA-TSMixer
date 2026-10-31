"""Boundary cases in loss interpolation, age alignment and receiver-log parsing."""
import sys, unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from link_calibration import Distribution, add_windows, parse_timestamps, track_slots
from real_trace import apply_masks, assignment, variable_age_slices


class LinkProtocol(unittest.TestCase):
    def test_interpolation_and_forecast_age(self):
        history = np.repeat((np.arange(60, dtype=float) ** 2)[None, :, None], 3, axis=2)
        mask = np.ones((1, 60), dtype=np.uint8)
        mask[0, :2] = 0
        mask[0, 10:13] = 0
        filled = apply_masks(history, mask)
        np.testing.assert_array_equal(filled[0, :2], np.full((2, 3), 4.))
        np.testing.assert_array_equal(filled[0, 10:13, 0], [103., 125., 147.])
        np.testing.assert_array_equal(filled[0, 59], history[0, 59])
        for k in (0, 3, 30):
            pred, truth = variable_age_slices(history, history + 1, np.array([k]))
            np.testing.assert_array_equal(pred, history[:, k:k + 30])
            np.testing.assert_array_equal(truth, history[:, k:k + 30] + 1)

    def test_replay_conditions_age_and_stable_decile_ties(self):
        # All masks tie on loss count; the first ceil(11/10) windows must be selected.
        pool = np.ones((4, 60), dtype=np.uint8)
        ages = np.array([0, 3, 30, 31], dtype=np.int16)
        _, sampled_age, subsets, stats = assignment(pool, ages, 11, "RLA")
        self.assertTrue((sampled_age <= 30).all())
        self.assertEqual(stats["excluded_fraction"], .25)
        self.assertEqual(stats["seed"], 2027)
        np.testing.assert_array_equal(np.flatnonzero(subsets["worst_decile"]), [0, 1])
        np.testing.assert_array_equal(subsets["k_ge_3"], sampled_age >= 3)

    def test_raw_window_right_endpoint_is_excluded_from_age(self):
        track = np.arange(61, dtype=float)
        loss, burst, age = Distribution((50,)), Distribution((50,)), Distribution((50,))
        self.assertEqual(add_windows(track_slots(track), loss, burst, age), 0)
        self.assertEqual(age.summarize()["percentiles"]["p50"], 1.)
        self.assertEqual(loss.summarize()["percentiles"]["p50"], 0.)
        self.assertEqual(burst.summarize()["percentiles"]["p50"], 0.)

    def test_both_raw_timestamp_formats_keep_fractional_seconds(self):
        frames = {
            "trajair": pd.DataFrame({"Date": ["01/01/2022", "01/01/2022"],
                                     "Time": ["00:00:00.5", "24:00:00"]}),
            "tartanaviation": pd.DataFrame({"Date": ["['2022', '01', '01']"] * 2,
                                            "Time": ["['00', '00', '00.5']", "['24', '00', '00']"]}),
        }
        for dataset, frame in frames.items():
            values, valid = parse_timestamps(frame, dataset)
            np.testing.assert_array_equal(valid, [True, False])
            self.assertEqual(values[0], 1640995200.5)
            self.assertTrue(np.isnan(values[1]))


if __name__ == "__main__":
    unittest.main()
