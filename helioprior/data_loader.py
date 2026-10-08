from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from helioprior.task_schema import DataSourceConfig


LHAASO_COLUMN_MAP = {
    "_TS拟合优度": "ts",
    "p0振幅": "shadow_amplitude",
    "p0误差": "shadow_amplitude_err",
    "p1_x中心值": "shadow_x",
    "p1_x中心值误差": "shadow_x_err",
    "p2_y中心值": "shadow_y",
    "p2_y中心值误差": "shadow_y_err",
    "p3 sigma_宽度": "shadow_sigma",
    "p3 sigma_宽度误差": "shadow_sigma_err",
    "_llf 最优卡方值": "shadow_llf",
}

OMNI_COLUMN_MAP = {
    "Scalar B, nT": "b_total",
    "BY, nT (GSE)": "by_gse",
    "BZ, nT (GSE)": "bz_gse",
    "BY, nT (GSM)": "by_gsm",
    "BZ, nT (GSM)": "bz_gsm",
    "Kp*10 index": "kp10",
    "Dst-index, nT": "dst",
    "ap_index, nT": "ap",
    "SW Plasma Speed, km/s": "vsw",
    "SW Plasma Temperature, K": "sw_temp",
    "SW Proton Density, N/cm^3": "proton_density",
}

PFSS_COLUMN_MAP = {
    "photosphere": "photosphere",
    "Surface_Magnetic field": "surface_magnetic",
}


@dataclass
class LoadedCorpus:
    frame: pd.DataFrame
    sources: list[str]
    coverage: dict[str, dict]

    @property
    def n_days(self) -> int:
        return int(len(self.frame))

    @property
    def date_range(self) -> tuple[str, str]:
        first = self.frame["date"].min().strftime("%Y-%m-%d")
        last = self.frame["date"].max().strftime("%Y-%m-%d")
        return first, last


def _read_excel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing data file: {path}")
    return pd.read_excel(path)


def _rename_columns(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    present = {old: new for old, new in mapping.items() if old in frame.columns}
    missing = [old for old in mapping if old not in frame.columns]
    result = frame.rename(columns=present).copy()
    for old in missing:
        result[mapping[old]] = np.nan
    return result


def _normalize_time(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series).dt.normalize()
    text = series.astype(str).str.strip()
    yyyymmdd = text.str.extract(r"(\d{8})", expand=False)
    return pd.to_datetime(yyyymmdd, format="%Y%m%d", errors="coerce").dt.normalize()


def load_corpus(cfg: DataSourceConfig) -> LoadedCorpus:
    omni = _read_excel(cfg.omni)
    lhaaso = _read_excel(cfg.lhaaso)
    frames: list[pd.DataFrame] = []
    sources: list[str] = []

    omni_norm = _rename_columns(omni, OMNI_COLUMN_MAP)
    omni_norm["date"] = _normalize_time(omni_norm[cfg.time_column])
    omni_norm = omni_norm.drop(columns=[cfg.time_column], errors="ignore")
    frames.append(omni_norm)
    sources.append("OMNI")

    lhaaso_norm = _rename_columns(lhaaso, LHAASO_COLUMN_MAP)
    lhaaso_norm["date"] = _normalize_time(lhaaso_norm[cfg.time_column])
    lhaaso_norm = lhaaso_norm.drop(columns=[cfg.time_column], errors="ignore")
    frames.append(lhaaso_norm)
    sources.append("LHAASO")

    if cfg.pfss is not None:
        pfss = _read_excel(cfg.pfss)
        pfss_norm = _rename_columns(pfss, PFSS_COLUMN_MAP)
        pfss_norm["date"] = _normalize_time(pfss_norm[cfg.time_column])
        pfss_norm = pfss_norm.drop(columns=[cfg.time_column], errors="ignore")
        frames.append(pfss_norm)
        sources.append("PFSS")

    merged = frames[0]
    for frame in frames[1:]:
        merged = pd.merge(merged, frame, on="date", how="inner")
    merged = (
        merged.drop_duplicates(subset="date")
        .sort_values("date")
        .reset_index(drop=True)
    )

    coverage: dict[str, dict] = {}
    for label, frame in zip(sources, frames):
        observed = int(frame["date"].nunique())
        coverage[label] = {
            "observed_days": observed,
            "merged_days": int(merged["date"].isin(set(frame["date"])).sum()),
            "missing_after_merge": observed
            - int(merged["date"].isin(set(frame["date"])).sum()),
        }

    return LoadedCorpus(frame=merged, sources=sources, coverage=coverage)


def select_features(
    frame: pd.DataFrame,
    *,
    include_shadow: bool,
    include_by: bool = True,
    include_solar_proxy: bool = True,
    include_autoregressive: bool = True,
    quality_features: bool = False,
) -> list[str]:
    base = ["vsw", "b_total", "bz_gse", "kp10", "dst", "ap", "sw_temp", "proton_density"]
    if include_by:
        base.extend(["by_gse"])
    if include_solar_proxy:
        base.extend(["photosphere", "surface_magnetic"])
    if include_shadow:
        base.append("shadow_y")
        if quality_features:
            base.extend(["shadow_y_err", "shadow_x", "ts"])
    if not include_autoregressive:
        base = [col for col in base if col != "vsw"]
    return [col for col in dict.fromkeys(base) if col in frame.columns]
