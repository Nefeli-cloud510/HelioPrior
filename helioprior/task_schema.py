from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


QualityMode = Literal[
    "all",
    "filter",
    "weight",
    "filter_or_weight",
    "drop_anomalous",
    "stratified",
]


class QualityFilterConfig(BaseModel):
    ts_threshold: float = Field(default=9.0, ge=0, le=50)
    mode: QualityMode = "filter_or_weight"
    apply_to_evaluation: bool = Field(
        default=True,
        description="True 时过滤/剔除同时作用于训练与评价样本；weight 始终只影响训练。",
    )


class ForecastTaskConfig(BaseModel):
    forecast_target: str = "solar_wind_speed"
    history_window_days: int = Field(default=14, ge=3, le=120)
    max_lag_days: int = Field(default=7, ge=1, le=30)
    forecast_horizons: list[int] = Field(default_factory=lambda: [1, 3, 5])
    quality_filter: QualityFilterConfig = Field(default_factory=QualityFilterConfig)
    evaluation_mode: Literal["rolling_backtest"] = "rolling_backtest"
    scenario: Literal["all", "non_icme"] = "non_icme"
    model_family: Literal["ridge", "elasticnet"] = "ridge"
    alpha: float = Field(default=1.0, ge=0.0, le=2000.0)
    l1_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    alpha_grid: list[float] = Field(
        default_factory=lambda: [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        description="Ridge alpha 候选网格；runner 在训练集内按时序折选参，滚动回测使用 task.alpha。",
    )
    ridge_solver: str = "svd"
    fit_intercept: bool = True
    random_state: int = 42
    n_bootstrap: int = Field(default=200, ge=50, le=1000)
    train_fraction: float = Field(default=0.70, ge=0.4, le=0.9)


class DataSourceConfig(BaseModel):
    omni: Path
    lhaaso: Path
    pfss: Path | None = None
    time_column: str = "TIME"
    target_column: str = "SW Plasma Speed, km/s"


class LabRunRequest(BaseModel):
    run_name: str | None = None
    task: ForecastTaskConfig = Field(default_factory=ForecastTaskConfig)
    note: str | None = None


class BatchSummary(BaseModel):
    run_id: str
    created_at: str = Field(default_factory=utc_now_iso)
    task: ForecastTaskConfig
    quality: dict
    metrics: dict
    hypotheses: list[dict]
    report: dict
    mechanisms: dict
    artifacts: dict = Field(default_factory=dict)
    legacy_evidence: list[dict] = Field(default_factory=list)
