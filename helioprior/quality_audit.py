from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class QualityAudit:
    summary: dict
    daily: list[dict]
    policies: dict[str, dict]
    anomalies: list[dict]
    strata: dict[str, dict]

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "daily": self.daily,
            "policies": self.policies,
            "anomalies": self.anomalies,
            "strata": self.strata,
        }


def _robust_outliers(values: pd.Series, threshold: float = 4.0) -> pd.Series:
    median = values.median()
    mad = (values - median).abs().median() or 1.0
    z = ((values - median) / (1.4826 * mad)).abs()
    return z > threshold


def run_quality_audit(frame: pd.DataFrame, ts_threshold: float = 9.0) -> QualityAudit:
    ts = frame["ts"].astype(float)
    shadow = frame["shadow_y"].astype(float)
    shadow_err = frame["shadow_y_err"].astype(float)
    n = len(frame)

    quantiles = ts.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
    summary = {
        "n_days": n,
        "date_start": frame["date"].min().strftime("%Y-%m-%d"),
        "date_end": frame["date"].max().strftime("%Y-%m-%d"),
        "ts_mean": float(ts.mean()),
        "ts_median": float(ts.median()),
        "ts_std": float(ts.std(ddof=0)),
        "ts_quantiles": {f"q{int(k * 100)}": float(v) for k, v in quantiles.items()},
        "ts_lt_threshold": int((ts < ts_threshold).sum()),
        "ts_ge_threshold": int((ts >= ts_threshold).sum()),
        "ts_min": float(ts.min()),
        "ts_max": float(ts.max()),
        "shadow_missing": int(shadow.isna().sum()),
        "shadow_err_missing": int(shadow_err.isna().sum()),
        "shadow_abs_mean": float(shadow.abs().mean()),
    }

    daily_records = [
        {
            "date": row.date.strftime("%Y-%m-%d"),
            "ts": None if pd.isna(row.ts) else float(row.ts),
            "shadow_y": None if pd.isna(row.shadow_y) else float(row.shadow_y),
            "shadow_y_err": None if pd.isna(row.shadow_y_err) else float(row.shadow_y_err),
            "vsw": None if pd.isna(row.vsw) else float(row.vsw),
            "by_gse": None if pd.isna(row.by_gse) else float(row.by_gse),
        }
        for row in frame.itertuples(index=False)
    ]

    anomaly_mask = (
        _robust_outliers(shadow)
        | (shadow_err > 0.45)
        | (frame["ts"] < 5.0)
    )
    anomalies = [
        {
            "date": row.date.strftime("%Y-%m-%d"),
            "reason": [],
            "shadow_y": float(row.shadow_y),
            "shadow_y_err": float(row.shadow_y_err),
            "ts": float(row.ts),
        }
        for row in frame[anomaly_mask].itertuples(index=False)
    ]
    for record, (_, raw) in zip(anomalies, frame[anomaly_mask].iterrows()):
        if float(raw["ts"]) < 5.0:
            record["reason"].append("ts<5")
        if float(raw["shadow_y_err"]) > 0.45:
            record["reason"].append("fit_error>0.45")
        if bool(_robust_outliers(pd.Series([raw["shadow_y"]]), threshold=4.0)[0]):
            record["reason"].append("label_outlier")
        if not record["reason"]:
            record["reason"].append("flagged")

    high = frame[ts >= ts_threshold]
    low = frame[ts < ts_threshold]
    strata = {
        "high_ts": _subset_stats(high, "ts>=threshold"),
        "low_ts": _subset_stats(low, "ts<threshold"),
    }

    policies = {
        "all": {"label": "全样本", "n": n, "kept_n": n, "note": "不剔除任何样本"},
        "filter": {
            "label": "TS≥阈值过滤",
            "n": n,
            "kept_n": len(high),
            "note": f"仅保留 TS≥{ts_threshold} 的样本",
        },
        "weight": {
            "label": "TS加权",
            "n": n,
            "kept_n": n,
            "note": "训练时按 TS 归一化权重，全部样本参与",
        },
        "drop_anomalous": {
            "label": "剔除异常定位样本",
            "n": n,
            "kept_n": int((~anomaly_mask).sum()),
            "note": "剔除鲁棒离群、拟合误差过大或 TS<5 的定位样本",
        },
    }
    policies["filter_or_weight"] = {
        "label": "过滤或加权",
        "n": n,
        "kept_n": len(high),
        "note": "数据充足时过滤，数据不足时按 TS 加权",
    }

    return QualityAudit(
        summary=summary,
        daily=daily_records,
        policies=policies,
        anomalies=anomalies,
        strata=strata,
    )


def _subset_stats(frame: pd.DataFrame, label: str) -> dict:
    if len(frame) == 0:
        return {"label": label, "n": 0}
    return {
        "label": label,
        "n": len(frame),
        "vsw_mean": float(frame["vsw"].mean()),
        "vsw_std": float(frame["vsw"].std(ddof=0)),
        "shadow_y_mean": float(frame["shadow_y"].mean()),
        "shadow_y_std": float(frame["shadow_y"].std(ddof=0)),
        "by_gse_mean": float(frame["by_gse"].mean()),
        "ts_mean": float(frame["ts"].mean()),
    }


def make_sample_weights(frame: pd.DataFrame, ts_threshold: float = 9.0) -> np.ndarray:
    ts = frame["ts"].fillna(0.0).clip(lower=0.1).to_numpy(dtype=float)
    return np.clip(ts / max(ts_threshold, 1.0), 0.25, 3.0)


def policy_train_mask(
    frame: pd.DataFrame,
    mode: str,
    ts_threshold: float = 9.0,
) -> pd.Series:
    ts = frame["ts"].fillna(0.0)
    if mode in ("filter", "filter_or_weight"):
        return ts >= ts_threshold
    if mode == "drop_anomalous":
        bad = (
            _robust_outliers(frame["shadow_y"])
            | (frame["shadow_y_err"] > 0.45)
            | (frame["ts"] < 5.0)
        )
        return ~bad
    return pd.Series(True, index=frame.index)
