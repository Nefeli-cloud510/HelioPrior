from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from core.unified_schema import (
    BootstrapReport,
    ExperimentResult,
    RobustnessOverall,
    RobustnessReport,
    TimeSliceFold,
    TimeSliceReport,
)

_BLOCK_SIZES_DAYS = (7, 14)
_N_BOOTSTRAP_DRAWS = 2000
_BOOTSTRAP_SEED = 20260826
_N_TIME_SLICES = 4
_MIN_ROWS_FOR_DIAGNOSTICS = 8
_MIN_ROWS_FOR_STABLE = 60
_MIN_VALID_DRAWS = 200


def build_robustness_report(
    result: ExperimentResult,
    project_root: Path | str | None = None,
) -> RobustnessReport:
    """Build a diagnostics-backed robustness report.

    The bootstrap and time-slice helpers are adapted from
    astro_rearrange/scripts/17_ab_bootstrap_v3.py so the same block
    resampling logic can be reused without modifying that project.
    """
    if result.status != "completed" or len(result.runs) < 2:
        return _unknown_report("实验状态不完整，未执行 bootstrap/time-slice 稳健性诊断。")

    frame = _load_prediction_pair(result, project_root)
    if frame is None or len(frame) < _MIN_ROWS_FOR_DIAGNOSTICS:
        return _unknown_report("未找到可用的成对预测工件，无法执行 bootstrap/time-slice 稳健性诊断。")

    observed_base = _metrics(frame["y_true"], frame["pred_base"])
    observed_treat = _metrics(frame["y_true"], frame["pred_treatment"])
    observed = {
        "gain_rmse": observed_base["rmse"] - observed_treat["rmse"],
        "gain_mae": observed_base["mae"] - observed_treat["mae"],
        "gain_pearson_r": observed_treat["pearson_r"] - observed_base["pearson_r"],
    }

    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    all_draws: list[dict[str, float | int]] = []
    for block_size in _BLOCK_SIZES_DAYS:
        draws = _block_bootstrap_draws(frame, block_size, rng, _N_BOOTSTRAP_DRAWS)
        for draw in draws:
            draw["block_size_calendar_days"] = block_size
            all_draws.append(draw)

    n_draws_used = len(all_draws)
    if n_draws_used < _MIN_VALID_DRAWS:
        return _unknown_report(
            "bootstrap 有效重采样次数不足（{n_draws_used}），无法给出稳定结论。".format(
                n_draws_used=n_draws_used
            )
        )

    bootstrap = _build_bootstrap_report(all_draws, n_draws_used)
    time_slice = _build_time_slice_report(frame)
    diagnostics_ran = True

    mixed_signal = _is_mixed_signal(result)
    boot_stable = bool(bootstrap.stable)
    slice_stable = bool(time_slice.stable)
    insufficient_for_stable = len(frame) < _MIN_ROWS_FOR_STABLE
    if insufficient_for_stable:
        return RobustnessReport(
            bootstrap=bootstrap,
            time_slice=time_slice,
            diagnostics_ran=True,
            overall=RobustnessOverall(
                stable=False,
                status="unknown",
                concern=(
                    f"样本量仅 {len(frame)} 行，bootstrap/time-slice 结果已落盘，"
                    "但不足以给出稳定结论。"
                ),
            ),
            recommendation="补充更长验证时段后重跑稳健性诊断，再据此更新假设支持度。",
        )
    stable = (not mixed_signal) and boot_stable and slice_stable
    if stable:
        status: str = "stable"
        concern: str | None = None
        recommendation = "主实验、bootstrap 与 time-slice 诊断方向一致，可以进入下一轮假设解释。"
    else:
        status = "unstable"
        reasons = []
        if mixed_signal:
            reasons.append("主实验指标呈混合信号")
        if not boot_stable:
            reasons.append("bootstrap 置信区间未稳定排除零增益")
        if not slice_stable:
            reasons.append("time-slice 各时段方向不一致")
        concern = "；".join(reasons) + "。"
        recommendation = "稳健性证据不足，建议补充样本或调整实验条件后重新验证，再更新假设支持度。"

    return RobustnessReport(
        bootstrap=bootstrap,
        time_slice=time_slice,
        diagnostics_ran=diagnostics_ran,
        overall=RobustnessOverall(stable=stable, status=status, concern=concern),
        recommendation=recommendation,
    )


def _unknown_report(reason: str) -> RobustnessReport:
    return RobustnessReport(
        bootstrap=BootstrapReport(stable=False, n_iterations=0),
        time_slice=TimeSliceReport(stable=False, split_mode="none"),
        diagnostics_ran=False,
        overall=RobustnessOverall(
            stable=False,
            status="unknown",
            concern=reason,
        ),
        recommendation="先补充可执行的 bootstrap 与 time-slice 稳健性诊断，再据此给出稳定结论。",
    )


def _is_mixed_signal(result: ExperimentResult) -> bool:
    rmse_delta = result.comparison.rmse
    pearson_delta = result.comparison.pearson_r
    if rmse_delta is None or pearson_delta is None:
        return False
    return (rmse_delta < 0 and pearson_delta < 0) or (
        rmse_delta > 0 and pearson_delta > 0
    )


def _load_prediction_pair(
    result: ExperimentResult,
    project_root: Path | str | None,
) -> pd.DataFrame | None:
    baseline_artifact = _resolve_artifact(
        result.runs[0].prediction_artifact,
        project_root,
    )
    treatment_artifact = _resolve_artifact(
        result.runs[1].prediction_artifact,
        project_root,
    )
    if baseline_artifact is None or treatment_artifact is None:
        return None
    try:
        baseline = pd.read_csv(baseline_artifact, usecols=["time", "y_true", "y_pred"])
        treatment = pd.read_csv(treatment_artifact, usecols=["time", "y_true", "y_pred"])
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"无法读取预测工件用于稳健性诊断: {exc}") from exc

    baseline["time"] = pd.to_datetime(baseline["time"], errors="coerce")
    treatment["time"] = pd.to_datetime(treatment["time"], errors="coerce")
    merged = baseline.merge(
        treatment,
        on="time",
        suffixes=("_base", "_treat"),
        how="inner",
    ).sort_values("time").reset_index(drop=True)
    merged = merged.dropna(subset=["time", "y_true_base", "y_pred_base", "y_pred_treat"])
    if merged.empty:
        return None
    return merged.rename(
        columns={
            "y_true_base": "y_true",
            "y_pred_base": "pred_base",
            "y_pred_treat": "pred_treatment",
        }
    )


def _resolve_artifact(
    artifact: str | None,
    project_root: Path | str | None,
) -> Path | None:
    if not artifact:
        return None
    path = Path(artifact)
    if path.is_absolute() and path.is_file():
        return path
    candidates: list[Path] = []
    if project_root is not None:
        candidates.append(Path(project_root) / artifact)
    candidates.extend(
        [
            Path.cwd() / artifact,
            Path(__file__).resolve().parent.parent / artifact,
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _as_float(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y = y_true[mask]
    p = y_pred[mask]
    if len(y) < 2 or np.ptp(y) == 0 or np.ptp(p) == 0:
        return float("nan")
    return float(np.corrcoef(y, p)[0, 1])


def _metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    y = _as_float(y_true)
    p = _as_float(y_pred)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if len(y) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "pearson_r": float("nan")}
    return {
        "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
        "mae": float(np.mean(np.abs(p - y))),
        "pearson_r": _pearson_r(y, p),
    }


def _weighted_metrics(
    y: np.ndarray,
    p: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    n = float(weights.sum())
    if n < 2:
        return {"rmse": float("nan"), "mae": float("nan"), "pearson_r": float("nan")}
    rmse = float(np.sqrt(np.sum(weights * (p - y) ** 2) / n))
    mae = float(np.sum(weights * np.abs(p - y)) / n)
    ybar = float(np.sum(weights * y) / n)
    pbar = float(np.sum(weights * p) / n)
    vy = float(np.sum(weights * (y - ybar) ** 2) / n)
    vp = float(np.sum(weights * (p - pbar) ** 2) / n)
    pearson = (
        float(np.sum(weights * (y - ybar) * (p - pbar)) / n / np.sqrt(vy * vp))
        if vy > 0 and vp > 0
        else float("nan")
    )
    return {"rmse": rmse, "mae": mae, "pearson_r": pearson}


def _block_bootstrap_draws(
    df: pd.DataFrame,
    block_size: int,
    rng: np.random.Generator,
    n_draws: int,
) -> list[dict[str, float | int]]:
    time = df["time"]
    start = time.min()
    end = time.max()
    span_days = int((end - start).total_seconds() // 86400) + 1
    block_size = max(1, min(int(block_size), max(1, span_days // 2)))
    n_total_blocks = int(math.ceil(span_days / block_size))
    block_id = ((time - start).dt.days // block_size).astype("int64").to_numpy()
    nonempty = np.unique(block_id).astype(int)
    if len(nonempty) < 2:
        return []

    y = _as_float(df["y_true"])
    p0 = _as_float(df["pred_base"])
    p1 = _as_float(df["pred_treatment"])
    draws: list[dict[str, float | int]] = []
    for _ in range(n_draws):
        chosen = rng.integers(0, len(nonempty), size=len(nonempty))
        counts = np.bincount(nonempty[chosen], minlength=n_total_blocks).astype(float)
        weights = counts[block_id]
        n_weighted = int(weights.sum())
        if n_weighted < 2:
            continue
        m0 = _weighted_metrics(y, p0, weights)
        m1 = _weighted_metrics(y, p1, weights)
        r0 = m0["pearson_r"]
        r1 = m1["pearson_r"]
        draws.append(
            {
                "n": n_weighted,
                "gain_rmse": m0["rmse"] - m1["rmse"],
                "gain_mae": m0["mae"] - m1["mae"],
                "gain_pearson_r": r1 - r0 if np.isfinite(r0) and np.isfinite(r1) else float("nan"),
            }
        )
    return draws


def _ci(values: list[float]) -> list[float] | None:
    clean = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(clean) == 0:
        return None
    return [float(v) for v in np.percentile(clean, [2.5, 97.5])]


def _gain_cis_consistent(
    rmse_ci: list[float] | None,
    pearson_ci: list[float] | None,
) -> bool:
    if rmse_ci is None or pearson_ci is None:
        return False
    rmse_positive = rmse_ci[0] > 0 and rmse_ci[1] > 0
    rmse_negative = rmse_ci[0] < 0 and rmse_ci[1] < 0
    pearson_positive = pearson_ci[0] > 0 and pearson_ci[1] > 0
    pearson_negative = pearson_ci[0] < 0 and pearson_ci[1] < 0
    return (rmse_positive and pearson_positive) or (
        rmse_negative and pearson_negative
    )


def _build_bootstrap_report(
    draws: list[dict[str, float | int]],
    n_draws_used: int,
) -> BootstrapReport:
    gain_rmse = [float(d["gain_rmse"]) for d in draws]
    gain_mae = [float(d["gain_mae"]) for d in draws]
    gain_pearson = [float(d["gain_pearson_r"]) for d in draws]
    rmse_ci = _ci(gain_rmse)
    mae_ci = _ci(gain_mae)
    pearson_ci = _ci(gain_pearson)
    stable = _gain_cis_consistent(rmse_ci, pearson_ci)
    return BootstrapReport(
        pearson_r_ci=pearson_ci,
        rmse_ci=rmse_ci,
        skill_ci=None,
        gain_rmse_ci=rmse_ci,
        gain_mae_ci=mae_ci,
        gain_pearson_r_ci=pearson_ci,
        stable=stable,
        n_iterations=_N_BOOTSTRAP_DRAWS * len(_BLOCK_SIZES_DAYS),
        n_draws_used=n_draws_used,
        block_sizes_calendar_days=list(_BLOCK_SIZES_DAYS),
    )


def _build_time_slice_report(frame: pd.DataFrame) -> TimeSliceReport:
    chunks = np.array_split(np.arange(len(frame), dtype=int), _N_TIME_SLICES)
    folds: list[TimeSliceFold] = []
    for fold_index, indices in enumerate(chunks, start=1):
        sub = frame.iloc[indices]
        base = _metrics(sub["y_true"], sub["pred_base"])
        treat = _metrics(sub["y_true"], sub["pred_treatment"])
        folds.append(
            TimeSliceFold(
                fold=fold_index,
                pearson_r=treat["pearson_r"],
                rmse=treat["rmse"],
                delta_rmse=base["rmse"] - treat["rmse"],
                delta_pearson_r=treat["pearson_r"] - base["pearson_r"],
            )
        )

    pearson_values = [f.pearson_r for f in folds if f.pearson_r is not None]
    std_pearson_r = float(np.std(pearson_values)) if len(pearson_values) >= 2 else None
    stable = _time_slice_stable(folds)
    return TimeSliceReport(
        splits=folds,
        std_pearson_r=std_pearson_r,
        split_mode=f"{_N_TIME_SLICES}_consecutive_blocks",
        stable=stable,
    )


def _time_slice_stable(folds: list[TimeSliceFold]) -> bool:
    if len(folds) < 2:
        return False
    valid = [
        f
        for f in folds
        if f.delta_rmse is not None
        and f.delta_pearson_r is not None
        and np.isfinite(f.delta_rmse)
        and np.isfinite(f.delta_pearson_r)
    ]
    if len(valid) != len(folds):
        return False
    rmse_signs = {1.0 if f.delta_rmse > 0 else -1.0 if f.delta_rmse < 0 else 0.0 for f in valid}
    pearson_signs = {
        1.0 if f.delta_pearson_r > 0 else -1.0 if f.delta_pearson_r < 0 else 0.0
        for f in valid
    }
    return (
        len(rmse_signs) == 1
        and len(pearson_signs) == 1
        and rmse_signs == pearson_signs
        and 0.0 not in rmse_signs
    )
