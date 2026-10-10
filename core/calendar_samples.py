from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


# forecast_origin (YYYY-MM-DD), target_time (YYYY-MM-DD), horizon_days
SampleKey = tuple[str, str, int]


def prepare_calendar_frame(
    X_raw: pd.DataFrame,
    y_raw: pd.Series,
    time_raw: pd.Series,
) -> pd.DataFrame:
    """Normalize a merged daily table to one row per calendar date.

    ``build_sliding_window`` and ``build_sample_plan`` use this frame so the
    forecast horizon and the history window are calendar differences, never
    record-count offsets. If multiple rows share one date, the first row in
    the sorted frame is kept for determinism.
    """

    if not isinstance(X_raw, pd.DataFrame) or X_raw.shape[1] == 0:
        raise ValueError("X_raw 必须是非空 DataFrame")

    times = pd.to_datetime(pd.Series(time_raw, index=X_raw.index), errors="coerce").dt.normalize()
    if times.isna().any():
        bad_count = int(times.isna().sum())
        raise ValueError(f"时间列存在 {bad_count} 个无法解析/空白的日期，不能构造日历样本")

    frame = pd.DataFrame(
        {
            "_hp_time": times.to_numpy(),
            "_hp_target": np.asarray(y_raw, dtype=float),
            "_hp_index": np.arange(len(X_raw)),
        },
        index=X_raw.index,
    )
    for column in X_raw.columns:
        frame[column] = np.asarray(X_raw[column], dtype=float)
    return (
        frame
        .sort_values(["_hp_time", "_hp_index"], kind="mergesort")
        .drop_duplicates("_hp_time", keep="first")
        .reset_index(drop=True)
    )


@dataclass(frozen=True)
class SampleRecord:
    forecast_origin: pd.Timestamp
    target_time: pd.Timestamp
    horizon_days: int
    feature_dates: list[str]
    feature_names: list[str]
    sample_condition: str = "complete_case"
    drop_reason: str | None = None

    def compact_key(self) -> SampleKey:
        return (
            self.forecast_origin.date().isoformat(),
            self.target_time.date().isoformat(),
            int(self.horizon_days),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "forecast_origin": self.forecast_origin.date().isoformat(),
            "target_time": self.target_time.date().isoformat(),
            "horizon_days": self.horizon_days,
            "feature_dates": list(self.feature_dates),
            "feature_names": list(self.feature_names),
            "sample_condition": self.sample_condition,
            "drop_reason": self.drop_reason,
        }


@dataclass
class SamplePlan:
    """Calendar-aligned sample plan with retention and exclusion bookkeeping."""

    records: list[SampleRecord]
    feature_names: list[str]
    target_name: str
    window_size: int
    past_lag_days: int
    forecast_horizon_days: int
    sample_condition: str = "complete_case"
    generated_at: str = ""
    date_start: str | None = None
    date_end: str | None = None

    def __post_init__(self) -> None:
        if not self.generated_at:
            self.generated_at = datetime.now().isoformat()

    @property
    def candidate_count(self) -> int:
        return len(self.records)

    @property
    def retained_count(self) -> int:
        return sum(1 for record in self.records if record.drop_reason is None)

    @property
    def dropped_count(self) -> int:
        return self.candidate_count - self.retained_count

    @property
    def coverage_ratio(self) -> float:
        if self.candidate_count <= 0:
            return 0.0
        return round(self.retained_count / self.candidate_count, 6)

    def drop_reason_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            reason = record.drop_reason or "retained"
            counts[reason] = counts.get(reason, 0) + 1
        return counts

    def compact_keys(self) -> set[SampleKey]:
        return {record.compact_key() for record in self.records if record.drop_reason is None}

    def retained_records(self) -> list[SampleRecord]:
        return [record for record in self.records if record.drop_reason is None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_name": self.target_name,
            "feature_names": list(self.feature_names),
            "window_size": self.window_size,
            "past_lag_days": self.past_lag_days,
            "forecast_horizon_days": self.forecast_horizon_days,
            "sample_condition": self.sample_condition,
            "generated_at": self.generated_at,
            "date_start": self.date_start,
            "date_end": self.date_end,
            "candidate_count": self.candidate_count,
            "retained_count": self.retained_count,
            "dropped_count": self.dropped_count,
            "coverage_ratio": self.coverage_ratio,
            "drop_reasons": self.drop_reason_counts(),
            "records": [record.to_dict() for record in self.records],
        }


def build_sample_plan(
    X_raw: pd.DataFrame,
    y_raw: pd.Series,
    time_raw: pd.Series,
    *,
    feature_names: list[str],
    target_name: str,
    window_size: int,
    past_lag_days: int,
    forecast_horizon_days: int,
) -> SamplePlan:
    """Build a complete-case sample plan on the real calendar axis.

    A record exists for every available forecast origin that has at least its
    declared history window. It is retained only when all required feature
    columns are present on every history date and the target date is present.
    The target date is strictly ``forecast_origin + forecast_horizon_days``.
    """

    frame = prepare_calendar_frame(X_raw, y_raw, time_raw)
    frame = frame.sort_values("_hp_time").reset_index(drop=True)
    if len(frame) == 0:
        raise ValueError("没有可用于构造日历样本的数据")

    feature_names = list(dict.fromkeys(str(item) for item in feature_names))
    for column in feature_names:
        if column not in frame.columns:
            raise KeyError(f"日历样本计划缺少特征列: {column}")

    window_size = int(window_size)
    past_lag_days = int(past_lag_days)
    forecast_horizon_days = int(forecast_horizon_days)
    if window_size < 1 or past_lag_days < 1:
        raise ValueError("window_size 与 past_lag_days 必须 >= 1")

    dates = frame["_hp_time"].to_numpy()
    date_to_index = {
        pd.Timestamp(value).normalize(): index
        for index, value in enumerate(dates)
    }
    feature_ok = frame[feature_names].notna().to_numpy(dtype=bool)

    records: list[SampleRecord] = []
    for origin_index in range(len(dates)):
        origin = pd.Timestamp(dates[origin_index]).normalize()
        feature_dates = [
            origin - pd.Timedelta(days=lag)
            for lag in range(past_lag_days, 0, -1)
        ]
        history_complete = all(
            pd.Timestamp(value).normalize() in date_to_index
            for value in feature_dates
        )
        if not history_complete:
            records.append(
                SampleRecord(
                    forecast_origin=origin,
                    target_time=origin + pd.Timedelta(days=forecast_horizon_days),
                    horizon_days=forecast_horizon_days,
                    feature_dates=[value.date().isoformat() for value in feature_dates],
                    feature_names=feature_names,
                    drop_reason="history_gap",
                )
            )
            continue

        feature_rows = [date_to_index[pd.Timestamp(value).normalize()] for value in feature_dates]
        missing_feature_rows = [
            row_index
            for row_index in feature_rows
            if not bool(feature_ok[row_index].all())
        ]
        if missing_feature_rows:
            records.append(
                SampleRecord(
                    forecast_origin=origin,
                    target_time=origin + pd.Timedelta(days=forecast_horizon_days),
                    horizon_days=forecast_horizon_days,
                    feature_dates=[value.date().isoformat() for value in feature_dates],
                    feature_names=feature_names,
                    drop_reason="feature_missing",
                )
            )
            continue

        target_time = origin + pd.Timedelta(days=forecast_horizon_days)
        if pd.Timestamp(target_time).normalize() not in date_to_index:
            records.append(
                SampleRecord(
                    forecast_origin=origin,
                    target_time=target_time,
                    horizon_days=forecast_horizon_days,
                    feature_dates=[value.date().isoformat() for value in feature_dates],
                    feature_names=feature_names,
                    drop_reason="target_unavailable",
                )
            )
            continue

        records.append(
            SampleRecord(
                forecast_origin=origin,
                target_time=target_time,
                horizon_days=forecast_horizon_days,
                feature_dates=[value.date().isoformat() for value in feature_dates],
                feature_names=feature_names,
                sample_condition="complete_case",
            )
        )

    dates_series = pd.Series(dates).dt.normalize()
    return SamplePlan(
        records=records,
        feature_names=feature_names,
        target_name=str(target_name),
        window_size=window_size,
        past_lag_days=past_lag_days,
        forecast_horizon_days=forecast_horizon_days,
        sample_condition="complete_case",
        date_start=dates_series.min().date().isoformat() if len(dates_series) else None,
        date_end=dates_series.max().date().isoformat() if len(dates_series) else None,
    )


def build_calendar_split(
    time_all: pd.Series | np.ndarray | list,
    test_rate: float,
) -> tuple[np.ndarray, pd.Timestamp]:
    """Split samples by their target calendar date.

    Both experiment arms are split with exactly the same boundary. ``split_date``
    is the last training target date; train masks use ``<= split_date`` and test
    masks are its complement.
    """

    if test_rate <= 0.0 or test_rate >= 1.0:
        raise ValueError("test_rate 必须在 (0, 1) 区间")
    dates = pd.Series(pd.to_datetime(pd.Series(time_all), errors="coerce")).dt.normalize()
    if dates.isna().any():
        raise ValueError("split 输入存在无法解析的目标日期")
    unique_dates = np.unique(dates.to_numpy()).tolist()
    if len(unique_dates) < 2:
        raise ValueError("训练/测试按日历切分时至少需要两个不同的目标日期")

    n_train = max(1, min(len(unique_dates) - 1, int(round(len(unique_dates) * (1.0 - test_rate)))))
    split_date = pd.Timestamp(unique_dates[n_train - 1]).normalize()
    train_mask = np.asarray(dates.dt.normalize() <= split_date, dtype=bool)
    return train_mask, split_date
