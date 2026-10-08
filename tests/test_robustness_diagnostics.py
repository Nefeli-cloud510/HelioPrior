import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from core.evaluation_unified import evaluate_experiment
from core.unified_schema import (
    ExecutionSummary,
    ExperimentResult,
    ExperimentRun,
    MetricDelta,
)


class RobustnessDiagnosticsTest(unittest.TestCase):
    def test_missing_artifacts_reports_unknown_not_stable(self) -> None:
        result = _result_with_artifacts(None, None)
        evaluation = evaluate_experiment(result)
        robustness = evaluation.robustness

        self.assertFalse(robustness.diagnostics_ran)
        self.assertFalse(robustness.overall.stable)
        self.assertEqual(robustness.overall.status, "unknown")
        self.assertFalse(robustness.bootstrap.stable)
        self.assertFalse(robustness.time_slice.stable)

    def test_real_artifacts_run_and_persist_bootstrap_and_time_slice(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prediction_dir = root / "results" / "round_01" / "E01" / "predictions"
            prediction_dir.mkdir(parents=True)
            n = 120
            times = pd.date_range("2023-01-01", periods=n, freq="D")
            y = 400 + 30 * np.sin(2 * np.pi * np.arange(n) / 7.0)
            rng = np.random.default_rng(7)
            noise = rng.normal(size=n)
            pd.DataFrame(
                {
                    "time": times.astype(str),
                    "y_true": y,
                    "y_pred": y + 8 * noise,
                }
            ).to_csv(prediction_dir / "E01_baseline.csv", index=False)
            pd.DataFrame(
                {
                    "time": times.astype(str),
                    "y_true": y,
                    "y_pred": y + 4 * noise,
                }
            ).to_csv(prediction_dir / "E01_treatment.csv", index=False)

            result = _result_with_artifacts(
                "results/round_01/E01/predictions/E01_baseline.csv",
                "results/round_01/E01/predictions/E01_treatment.csv",
            )
            evaluation = evaluate_experiment(result, project_root=root)
            robustness = evaluation.robustness

            self.assertTrue(robustness.diagnostics_ran)
            self.assertEqual(robustness.overall.status, "stable")
            self.assertTrue(robustness.overall.stable)
            self.assertIsNotNone(robustness.bootstrap.gain_rmse_ci)
            self.assertIsNotNone(robustness.bootstrap.gain_pearson_r_ci)
            self.assertGreater(robustness.bootstrap.n_draws_used, 0)
            self.assertEqual(len(robustness.time_slice.splits), 4)
            self.assertTrue(robustness.time_slice.stable)
            self.assertTrue(robustness.time_slice.splits[0].delta_rmse > 0)


def _result_with_artifacts(
    baseline_artifact: str | None,
    treatment_artifact: str | None,
) -> ExperimentResult:
    return ExperimentResult(
        experiment_id="E01",
        round_id=1,
        status="completed",
        runs=[
            ExperimentRun(
                run_id="baseline",
                name="baseline",
                prediction_artifact=baseline_artifact,
            ),
            ExperimentRun(
                run_id="treatment",
                name="treatment",
                prediction_artifact=treatment_artifact,
            ),
        ],
        comparison=MetricDelta(rmse=-4.0, pearson_r=0.1),
        execution=ExecutionSummary(status="completed", duration_seconds=1.0),
    )


if __name__ == "__main__":
    unittest.main()
