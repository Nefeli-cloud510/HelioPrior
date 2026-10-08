from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from helioprior.quality_audit import make_sample_weights, policy_train_mask
from helioprior.task_schema import ForecastTaskConfig


def _make_predictor(task: ForecastTaskConfig):
    if task.model_family == "elasticnet":
        return ElasticNet(
            alpha=task.alpha,
            l1_ratio=task.l1_ratio,
            random_state=task.random_state,
            max_iter=3000,
        )
    return Ridge(
        alpha=task.alpha,
        fit_intercept=task.fit_intercept,
        solver=task.ridge_solver,
    )


@dataclass
class ArmResult:
    arm_name: str
    horizon: int
    policy: str
    features: list[str]
    dates: list[str]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_persistence: np.ndarray
    metrics: dict[str, float]
    trained_origins: int
    eval_origins: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_name": self.arm_name,
            "horizon": self.horizon,
            "policy": self.policy,
            "n_features": len(self.features),
            "metrics": self.metrics,
            "trained_origins": self.trained_origins,
            "eval_origins": self.eval_origins,
        }


@dataclass
class PolicyComparison:
    horizon: int
    policy: str
    baseline: ArmResult
    treatment: ArmResult
    delta_metrics: dict[str, float]
    bootstrap_ci: dict[str, float] = field(default_factory=dict)
    folds: list[dict] = field(default_factory=list)
    stability: dict[str, Any] = field(default_factory=dict)
    strata: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "policy": self.policy,
            "baseline": self.baseline.to_dict(),
            "treatment": self.treatment.to_dict(),
            "delta": self.delta_metrics,
            "bootstrap_ci": self.bootstrap_ci,
            "stability": self.stability,
            "strata": self.strata,
        }


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    n = len(y_true)
    if n == 0:
        return {"rmse": np.nan, "mae": np.nan, "pearson_r": np.nan, "r2": np.nan}
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {
        "rmse": _round(rmse),
        "mae": _round(mae),
        "pearson_r": _round(pearson),
        "r2": _round(r2),
    }


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def build_origin_table(
    frame: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    horizon: int,
    history_window: int,
    max_lag_days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str], list[int]]:
    n = len(frame)
    if n <= history_window + horizon + 8:
        raise ValueError("data too short for the requested history window")

    depth = max(1, min(history_window, max_lag_days))
    feature_vals = frame[feature_cols].apply(pd.to_numeric, errors="coerce")
    feature_vals = feature_vals.interpolate(limit_direction="both").bfill().ffill()
    feature_vals = feature_vals.fillna(feature_vals.median())
    values = feature_vals.to_numpy(dtype=float)
    target = pd.to_numeric(frame[target_col], errors="coerce").to_numpy(dtype=float)
    dates = frame["date"].tolist()

    recent_names = [f"{col}@lag{lag}" for lag in range(1, depth + 1) for col in feature_cols]
    summary_names = []
    for op in ("mean", "min", "max", "std"):
        summary_names.extend([f"{col}_{op}" for col in feature_cols])
    feature_names = recent_names + summary_names

    rows: list[np.ndarray] = []
    y: list[float] = []
    persistence: list[float] = []
    origin_dates: list[str] = []
    origin_indexes: list[int] = []

    for t in range(history_window, n - horizon):
        y_true = target[t + horizon]
        if np.isnan(y_true):
            continue
        window = values[t - history_window + 1 : t + 1, :]
        recent = window[-depth:, :].reshape(-1)
        summaries = []
        for op_index in range(4):
            op_values = (
                np.nanmean(window, axis=0)
                if op_index == 0
                else np.nanmin(window, axis=0)
                if op_index == 1
                else np.nanmax(window, axis=0)
                if op_index == 2
                else np.nanstd(window, axis=0)
            )
            summaries.append(np.nan_to_num(op_values))
        vector = np.concatenate([recent, *summaries]).astype(float)
        rows.append(vector)
        y.append(float(y_true))
        persistence.append(float(target[t]))
        origin_dates.append(pd.Timestamp(dates[t]).strftime("%Y-%m-%d"))
        origin_indexes.append(t)

    return (
        np.asarray(rows, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(persistence, dtype=float),
        np.asarray(origin_dates, dtype=object),
        feature_names,
        feature_cols,
        origin_indexes,
    )


def _walk_forward_fit_predict(
    X: np.ndarray,
    y: np.ndarray,
    horizon: int,
    train_mask: pd.Series,
    split_index: int,
    weights: np.ndarray | None,
    task: ForecastTaskConfig,
    test_mask: pd.Series | None = None,
) -> tuple[np.ndarray, int]:
    n = len(y)
    if test_mask is None:
        test_mask = pd.Series(True, index=np.arange(n))
    test_indexes = np.where((np.arange(n) >= split_index) & test_mask.to_numpy())[0]
    predictions = np.full(n, np.nan)
    trained = 0

    for test_seq in test_indexes:
        eligible = np.arange(n) <= test_seq - horizon
        train_idx = np.where(eligible & train_mask.to_numpy())[0]
        if len(train_idx) < max(12, 2 * len(task.forecast_horizons)):
            continue
        xtr, ytr = X[train_idx], y[train_idx]
        scaler = StandardScaler().fit(xtr)
        xtr_s = scaler.transform(xtr)
        sample_weight = None
        if weights is not None:
            sample_weight = weights[train_idx]
        model = _make_predictor(task)
        model.fit(xtr_s, ytr, sample_weight=sample_weight)
        xte = scaler.transform(X[test_seq : test_seq + 1])
        predictions[test_seq] = float(model.predict(xte)[0])
        trained += 1
    return predictions, trained


def run_arm(
    frame: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    horizon: int,
    policy: str,
    task: ForecastTaskConfig,
    arm_name: str,
) -> ArmResult:
    X, y, persistence, origin_dates, _, _, _ = build_origin_table(
        frame,
        target_col,
        feature_cols,
        horizon,
        task.history_window_days,
        task.max_lag_days,
    )
    n = len(y)
    if n == 0:
        raise RuntimeError(f"no origins for {arm_name} h={horizon}")

    train_mask = policy_train_mask(
        frame.iloc[[_origin_date_to_index(frame, d) for d in origin_dates]],
        policy,
        task.quality_filter.ts_threshold,
    )
    split_index = max(20, int(n * task.train_fraction))
    weights = None
    if policy in ("weight", "filter_or_weight"):
        weights = make_sample_weights(
            frame.iloc[[_origin_date_to_index(frame, d) for d in origin_dates]],
            task.quality_filter.ts_threshold,
        )

    if task.quality_filter.apply_to_evaluation and policy in (
        "filter",
        "drop_anomalous",
    ):
        test_mask = train_mask
    else:
        test_mask = None

    predictions, trained = _walk_forward_fit_predict(
        X,
        y,
        horizon,
        train_mask,
        split_index,
        weights,
        task,
        test_mask,
    )
    eval_mask = (~np.isnan(predictions)) & (~np.isnan(y))
    if eval_mask.sum() == 0:
        return ArmResult(
            arm_name=arm_name,
            horizon=horizon,
            policy=policy,
            features=feature_cols,
            dates=[],
            y_true=np.array([]),
            y_pred=np.array([]),
            y_persistence=np.array([]),
            metrics={"rmse": np.nan, "mae": np.nan, "pearson_r": np.nan, "r2": np.nan},
            trained_origins=0,
            eval_origins=0,
        )
    return ArmResult(
        arm_name=arm_name,
        horizon=horizon,
        policy=policy,
        features=feature_cols,
        dates=[str(d) for d in origin_dates[eval_mask]],
        y_true=y[eval_mask],
        y_pred=predictions[eval_mask],
        y_persistence=persistence[eval_mask],
        metrics=_metrics(y[eval_mask], predictions[eval_mask]),
        trained_origins=trained,
        eval_origins=int(eval_mask.sum()),
    )


def _origin_date_to_index(frame: pd.DataFrame, date_str: str) -> int:
    mask = frame["date"].astype(str) == str(date_str)
    indexes = np.where(mask.to_numpy())[0]
    if len(indexes) == 0:
        raise KeyError(f"origin date not found: {date_str}")
    return int(indexes[0])


def compare_policy(
    frame: pd.DataFrame,
    target_col: str,
    base_features: list[str],
    treatment_features: list[str],
    horizon: int,
    policy: str,
    task: ForecastTaskConfig,
) -> PolicyComparison:
    baseline = run_arm(frame, target_col, base_features, horizon, policy, task, "baseline")
    treatment = run_arm(frame, target_col, treatment_features, horizon, policy, task, "treatment")
    delta = _delta_metrics(baseline, treatment)
    bootstrap_ci = _bootstrap_delta_skill(baseline, treatment, task.n_bootstrap, task.random_state)
    folds = _fold_stability(baseline, treatment)
    stable, consistency = _stability_summary(folds, bootstrap_ci)
    strata = _quality_strata(baseline, treatment, frame, task.quality_filter.ts_threshold)
    return PolicyComparison(
        horizon=horizon,
        policy=policy,
        baseline=baseline,
        treatment=treatment,
        delta_metrics=delta,
        bootstrap_ci=bootstrap_ci,
        folds=folds,
        stability={"stable": stable, "consistency_ratio": consistency},
        strata=strata,
    )


def _delta_metrics(baseline: ArmResult, treatment: ArmResult) -> dict[str, float]:
    b_rmse = baseline.metrics["rmse"]
    t_rmse = treatment.metrics["rmse"]
    b_skill = baseline.metrics["mae"]  # placeholder not used
    del b_skill
    return {
        "delta_rmse": _round(t_rmse - b_rmse, 4) if np.isfinite(t_rmse) and np.isfinite(b_rmse) else np.nan,
        "delta_mae": _round(treatment.metrics["mae"] - baseline.metrics["mae"], 4),
        "delta_pearson_r": _round(treatment.metrics["pearson_r"] - baseline.metrics["pearson_r"], 4),
        "relative_skill": _round(1.0 - t_rmse / b_rmse, 5) if b_rmse else np.nan,
    }


def _bootstrap_delta_skill(
    baseline: ArmResult,
    treatment: ArmResult,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    diff = (
        (baseline.y_pred - baseline.y_true) ** 2
        - (treatment.y_pred - treatment.y_true) ** 2
    )
    rng = np.random.default_rng(seed)
    if len(diff) < 8:
        return {"lower": np.nan, "upper": np.nan, "median": np.nan}
    stats = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(diff), size=len(diff))
        stats.append(np.mean(diff[idx]) / max(np.mean(
            (baseline.y_persistence[idx] - baseline.y_true[idx]) ** 2
        ), 1e-9))
    arr = np.asarray(stats)
    return {
        "lower": _round(np.percentile(arr, 5), 5),
        "upper": _round(np.percentile(arr, 95), 5),
        "median": _round(np.median(arr), 5),
    }


def _fold_stability(baseline: ArmResult, treatment: ArmResult) -> list[dict]:
    n = len(baseline.y_true)
    if n < 9:
        return []
    folds: list[dict] = []
    edges = np.linspace(0, n, 4, dtype=int)
    for i in range(3):
        idx = np.arange(edges[i], edges[i + 1])
        if len(idx) == 0:
            continue
        b_rmse = float(np.sqrt(np.mean((baseline.y_true[idx] - baseline.y_pred[idx]) ** 2)))
        t_rmse = float(np.sqrt(np.mean((treatment.y_true[idx] - treatment.y_pred[idx]) ** 2)))
        folds.append(
            {
                "fold": i + 1,
                "n": int(len(idx)),
                "date_start": baseline.dates[int(idx[0])],
                "date_end": baseline.dates[int(idx[-1])],
                "baseline_rmse": _round(b_rmse, 3),
                "treatment_rmse": _round(t_rmse, 3),
                "delta_skill": _round(1.0 - t_rmse / b_rmse, 5) if b_rmse else 0.0,
            }
        )
    return folds


def _stability_summary(
    folds: list[dict],
    bootstrap_ci: dict[str, float],
) -> tuple[bool, float]:
    if not folds:
        return False, 0.0
    positive = sum(1 for fold in folds if fold["delta_skill"] > 0)
    consistency = positive / len(folds)
    lower = bootstrap_ci.get("lower")
    upper = bootstrap_ci.get("upper")
    interval_same_side = (
        np.isfinite(lower)
        and np.isfinite(upper)
        and ((lower > 0 and upper > 0) or (lower < 0 and upper < 0))
    )
    stable = interval_same_side and consistency >= 0.5
    return stable, _round(consistency, 3)


def _quality_strata(
    baseline: ArmResult,
    treatment: ArmResult,
    frame: pd.DataFrame,
    ts_threshold: float,
) -> dict[str, dict]:
    if not baseline.dates:
        return {}
    date_to_ts = dict(zip(frame["date"].astype(str), frame["ts"].fillna(0.0)))
    ts_values = np.array([date_to_ts.get(str(date), 0.0) for date in baseline.dates])
    result: dict[str, dict] = {}
    for label, mask in (
        ("high_ts", ts_values >= ts_threshold),
        ("low_ts", ts_values < ts_threshold),
    ):
        if int(mask.sum()) < 2:
            result[label] = {"n": int(mask.sum()), "delta_skill": np.nan}
            continue
        b_rmse = float(np.sqrt(np.mean((baseline.y_true[mask] - baseline.y_pred[mask]) ** 2)))
        t_rmse = float(np.sqrt(np.mean((treatment.y_true[mask] - treatment.y_pred[mask]) ** 2)))
        result[label] = {
            "n": int(mask.sum()),
            "baseline_rmse": _round(b_rmse, 3),
            "treatment_rmse": _round(t_rmse, 3),
            "delta_skill": _round(1.0 - t_rmse / b_rmse, 5) if b_rmse else np.nan,
            "pearson_delta": _round(
                _corr(treatment.y_true[mask], treatment.y_pred[mask])
                - _corr(baseline.y_true[mask], baseline.y_pred[mask]),
                4,
            ),
        }
    return result


def run_policy_grid(
    frame: pd.DataFrame,
    target_col: str,
    base_features: list[str],
    treatment_features: list[str],
    task: ForecastTaskConfig,
) -> dict[str, dict[str, PolicyComparison]]:
    policies = ["all", "filter", "weight", "drop_anomalous"]
    if task.quality_filter.mode == "filter_or_weight":
        policies = ["all", "filter", "weight", "drop_anomalous", "filter_or_weight"]
    grid: dict[str, dict[str, PolicyComparison]] = {}
    for horizon in task.forecast_horizons:
        grid[str(horizon)] = {}
        for policy in policies:
            try:
                comparison = compare_policy(
                    frame,
                    target_col,
                    base_features,
                    treatment_features,
                    horizon,
                    policy,
                    task,
                )
            except Exception as exc:  # keep one failed policy from killing the batch
                comparison = PolicyComparison(
                    horizon=horizon,
                    policy=policy,
                    baseline=ArmResult(
                        arm_name="baseline", horizon=horizon, policy=policy,
                        features=base_features, dates=[], y_true=np.array([]),
                        y_pred=np.array([]), y_persistence=np.array([]),
                        metrics={}, trained_origins=0, eval_origins=0,
                    ),
                    treatment=ArmResult(
                        arm_name="treatment", horizon=horizon, policy=policy,
                        features=treatment_features, dates=[], y_true=np.array([]),
                        y_pred=np.array([]), y_persistence=np.array([]),
                        metrics={}, trained_origins=0, eval_origins=0,
                    ),
                    delta_metrics={"error": str(exc)},
                    bootstrap_ci={},
                    stability={"stable": False, "consistency_ratio": 0.0},
                )
            grid[str(horizon)][policy] = comparison
    return grid


def correlation_profile(frame: pd.DataFrame, max_lag: int = 3) -> dict[str, float]:
    shadow = pd.to_numeric(frame["shadow_y"], errors="coerce").to_numpy(dtype=float)
    by = pd.to_numeric(frame["by_gse"], errors="coerce").to_numpy(dtype=float)
    profile: dict[str, float] = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = shadow[:-lag] if lag else shadow, by[lag:] if lag else by
            if lag:
                a, b = shadow[:-lag], by[lag:]
        else:
            shift = -lag
            a, b = shadow[shift:], by[:-shift]
        profile[f"lag{lag:+d}"] = _round(_corr(a, b), 4)
    profile["contemporaneous"] = profile["lag+0"]
    return profile


def run_mechanism_probes(
    frame: pd.DataFrame,
    target_col: str,
    task: ForecastTaskConfig,
) -> dict[str, Any]:
    """Run the By-mediation and solar-proxy probes at the medium horizon."""
    horizon = 3 if 3 in task.forecast_horizons else task.forecast_horizons[0]
    by_free_base = select_subset(frame, include_by=False, include_shadow=False)
    by_free_treat = select_subset(frame, include_by=False, include_shadow=True)
    proxy_free_base = select_subset(frame, include_by=True, include_shadow=False, include_solar_proxy=False)
    proxy_free_treat = select_subset(frame, include_by=True, include_shadow=True, include_solar_proxy=False)
    base = select_subset(frame, include_by=True, include_shadow=False)
    treat = select_subset(frame, include_by=True, include_shadow=True)

    contributions = {}
    for label, (left, right) in {
        "by_free": (by_free_base, by_free_treat),
        "with_by": (base, treat),
        "proxy_free": (proxy_free_base, proxy_free_treat),
        "with_proxy": (base, treat),
    }.items():
        comparison = compare_policy(frame, target_col, left, right, horizon, "all", task)
        contributions[label] = comparison.to_dict()

    return {
        "horizon": horizon,
        "lead_lag_correlation": correlation_profile(frame),
        "cross_correlations": {
            "shadow_by": _round(_corr(frame["shadow_y"].to_numpy(float), frame["by_gse"].to_numpy(float)), 4),
            "shadow_photosphere": _round(
                _corr(frame["shadow_y"].to_numpy(float), frame["photosphere"].to_numpy(float)), 4
            ),
            "shadow_surface_magnetic": _round(
                _corr(frame["shadow_y"].to_numpy(float), frame["surface_magnetic"].to_numpy(float)), 4
            ),
        },
        "contribution_delta": {
            "shadow_without_by": contributions["by_free"]["delta"].get("relative_skill"),
            "shadow_with_by": contributions["with_by"]["delta"].get("relative_skill"),
            "shadow_without_solar_proxy": contributions["proxy_free"]["delta"].get("relative_skill"),
            "shadow_with_solar_proxy": contributions["with_proxy"]["delta"].get("relative_skill"),
        },
        "details": contributions,
    }


def select_subset(
    frame: pd.DataFrame,
    *,
    include_by: bool,
    include_shadow: bool,
    include_solar_proxy: bool = True,
) -> list[str]:
    from helioprior.data_loader import select_features

    return select_features(
        frame,
        include_shadow=include_shadow,
        include_by=include_by,
        include_solar_proxy=include_solar_proxy,
        include_autoregressive=True,
        quality_features=False,
    )


def to_jsonable(grid: dict[str, dict[str, PolicyComparison]]) -> dict:
    return {
        horizon: {policy: comparison.to_dict() for policy, comparison in policies.items()}
        for horizon, policies in grid.items()
    }
