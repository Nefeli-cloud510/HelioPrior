from __future__ import annotations

import csv
import io
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from core.protocol_hash import protocol_hash
from core.research_schema import (
    EvidenceCheck,
    EvidenceItem,
    EvidencePacket,
    MetricValue,
)


COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "forecast_origin": ("forecast_origin", "origin", "origin_time", "forecast_time", "发布时刻", "预测发布时刻"),
    "target_time": ("target_time", "target", "target_date", "目标时刻", "预测目标时刻"),
    "horizon_days": ("horizon_days", "horizon", "提前量"),
    "sample_id": ("sample_id", "sample", "样本编号"),
    "arm": ("arm", "group", "arm_id", "方案", "分组"),
    "y_true": ("y_true", "actual", "true", "真实值", "y"),
    "y_pred": ("y_pred", "predicted", "pred", "预测值", "yhat"),
    "split": ("split", "fold", "partition", "数据划分", "划分"),
    "run_id": ("run_id", "run", "运行编号"),
    "protocol_hash": ("protocol_hash", "协议哈希", "protocol_version"),
    "ts_quality": ("ts_quality", "ts", "quality", "TS", "日影质量"),
}

REQUIRED_METRIC_COLUMNS = ("y_true", "y_pred")
ARM_BASE_NAMES = {"base", "baseline", "control", "对照组", "基准", "基准组", "基线"}
ARM_AUG_NAMES = {"aug", "augmented", "enhanced", "增强", "增强组", "日影增强", "日影"}


@dataclass
class ParsedSource:
    kind: str
    rows: list[dict[str, Any]]
    imported_metrics: dict[str, Any] | None = None
    raw_columns: list[str] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)


@dataclass
class ParsedPredictions:
    rows: list[dict[str, Any]]
    columns: list[str]
    arm_values: list[str]
    missing_columns: list[str]
    non_numeric_count: int
    parse_errors: list[str] = field(default_factory=list)


def _normalize_header(value: str) -> str:
    return re.sub(r"\s+", "", str(value)).lower()


def _target_alias_variants(value: str) -> set[str]:
    norm = _normalize_header(value)
    variants = {norm}
    compact = re.sub(r"[^a-z0-9]", "", norm)
    variants.add(compact)
    for prefix in ("imf", "plasma", "solarwind", "sw"):
        if compact.startswith(prefix):
            variants.add(compact[len(prefix):])
    if compact.endswith("gse"):
        variants.add(compact[:-3])
    if compact.endswith("gsm"):
        variants.add(compact[:-3])
    if compact.endswith("nt"):
        variants.add(compact[:-2])
    return {item for item in variants if item}


def resolve_column(columns: list[str], canonical: str) -> str | None:
    aliases = COLUMN_ALIASES.get(canonical, (canonical,))
    normalized = {_normalize_header(column): column for column in columns}
    for alias in aliases:
        found = normalized.get(_normalize_header(alias))
        if found:
            return found
    return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"nan", "null", "none", "na"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_source_content(kind: str, content: str, file_name: str) -> ParsedSource:
    kind = str(kind or "prediction_csv").lower()
    errors: list[str] = []
    if kind == "prediction_csv" or file_name.lower().endswith(".csv"):
        kind = "prediction_csv"
        try:
            reader = csv.DictReader(io.StringIO(content))
            rows = [dict(row) for row in reader]
            if not rows:
                errors.append("CSV 没有任何数据行。")
            columns = list(reader.fieldnames or [])
            return ParsedSource(kind=kind, rows=rows, raw_columns=columns, parse_errors=errors)
        except csv.Error as exc:
            errors.append(f"CSV 解析失败：{exc}")
            return ParsedSource(kind=kind, rows=[], parse_errors=errors)

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        errors.append(f"JSON 解析失败：{exc}")
        return ParsedSource(kind=kind, rows=[], parse_errors=errors)

    if isinstance(payload, list):
        rows = [row for row in payload if isinstance(row, dict)]
        if len(rows) != len(payload):
            errors.append("JSON 数组中存在非对象记录，已忽略。")
        columns = sorted({key for row in rows for key in row})
        return ParsedSource(kind=kind, rows=rows, raw_columns=columns, parse_errors=errors)

    if not isinstance(payload, dict):
        errors.append("JSON 顶层必须是对象或对象数组。")
        return ParsedSource(kind=kind, rows=[], parse_errors=errors)

    rows_payload = payload.get("rows")
    if isinstance(rows_payload, list):
        rows = [row for row in rows_payload if isinstance(row, dict)]
        if len(rows) != len(rows_payload):
            errors.append("rows 中存在非对象记录，已忽略。")
        columns = sorted({key for row in rows for key in row})
        return ParsedSource(kind=kind, rows=rows, raw_columns=columns, parse_errors=errors)

    if kind in {"protocol_json", "evidence_json", "report", "other"}:
        return ParsedSource(
            kind=kind,
            rows=[],
            raw_columns=sorted(str(key) for key in payload),
            parse_errors=errors,
        )

    imported_metrics = payload.get("metrics")
    if isinstance(imported_metrics, dict):
        errors.append("未提供逐行预测记录，指标按导入值登记，不重新计算。")
        return ParsedSource(
            kind=kind,
            rows=[],
            imported_metrics=imported_metrics,
            raw_columns=[],
            parse_errors=errors,
        )

    errors.append("JSON 需要提供 rows 数组或 metrics 对象。")
    return ParsedSource(kind=kind, rows=[], parse_errors=errors)


def normalize_prediction_rows(source: ParsedSource, *, target_column: str | None = None) -> ParsedPredictions:
    if source.kind != "prediction_csv" and source.rows:
        pass
    columns = list(source.raw_columns)
    for row in source.rows:
        for key in row:
            if key not in columns:
                columns.append(str(key))

    resolved: dict[str, str | None] = {}
    for canonical in COLUMN_ALIASES:
        resolved[canonical] = resolve_column(columns, canonical)
    if target_column:
        resolved["y_true"] = target_column

    missing: list[str] = []
    for canonical in REQUIRED_METRIC_COLUMNS:
        if not resolved[canonical]:
            missing.append(canonical)

    rows: list[dict[str, Any]] = []
    non_numeric = 0
    for index, raw in enumerate(source.rows):
        row: dict[str, Any] = {"_row_index": index}
        for canonical, column in resolved.items():
            if column:
                row[canonical] = raw.get(column)
        for legacy in ("time", "date", "日期", "时间"):
            if legacy in raw and "forecast_origin" not in row:
                row["forecast_origin"] = raw.get(legacy)
                row["target_time"] = raw.get(legacy)
        y_true, y_pred = _to_float(row.get("y_true")), _to_float(row.get("y_pred"))
        if y_true is None or y_pred is None:
            non_numeric += 1
        row["y_true"] = y_true
        row["y_pred"] = y_pred
        row["arm"] = str(row.get("arm") or row.get("run_id") or "single").strip()
        rows.append(row)

    arm_values = sorted({str(row.get("arm") or "single") for row in rows})
    return ParsedPredictions(
        rows=rows,
        columns=columns,
        arm_values=arm_values,
        missing_columns=missing,
        non_numeric_count=non_numeric,
        parse_errors=list(source.parse_errors),
    )


def _pair_key(row: dict[str, Any]) -> tuple[str, str, str]:
    origin = str(row.get("forecast_origin") or f"raw:{row.get('_row_index')}").strip()
    target = str(row.get("target_time") or f"raw:{row.get('_row_index')}").strip()
    horizon = str(row.get("horizon_days") or "unknown").strip()
    return (origin, target, horizon)


def _has_time_columns(parsed: ParsedPredictions) -> bool:
    return any(row.get("forecast_origin") is not None and row.get("target_time") is not None for row in parsed.rows)


def _time_delta_days(origin: Any, target: Any) -> float | None:
    if origin is None or target is None:
        return None
    try:
        if isinstance(origin, datetime):
            origin_dt = origin
        else:
            origin_dt = datetime.fromisoformat(str(origin).replace("Z", "+00:00"))
        if isinstance(target, datetime):
            target_dt = target
        else:
            target_dt = datetime.fromisoformat(str(target).replace("Z", "+00:00"))
        delta = target_dt - origin_dt
        return abs(delta.total_seconds() / 86400.0)
    except (TypeError, ValueError):
        return None


def _arm_label(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_").lower() or "arm"


def _resolve_arms(parsed: ParsedPredictions, *, base_arm: str | None, aug_arm: str | None) -> tuple[str, str] | str:
    arm_values = parsed.arm_values
    if base_arm or aug_arm:
        if not base_arm or not aug_arm:
            return "指定组别时须同时提供 base_arm 与 aug_arm。"
        if base_arm not in arm_values or aug_arm not in arm_values:
            return "指定的 base_arm 或 aug_arm 不在预测文件中。"
        return (str(base_arm), str(aug_arm))
    if len(arm_values) > 2:
        return "预测文件包含超过两个方案组，须明确指定 base_arm 与 aug_arm。"
    if len(arm_values) == 2:
        base = next((value for value in arm_values if value.lower() in ARM_BASE_NAMES), None)
        augs = [value for value in arm_values if value.lower() in ARM_AUG_NAMES]
        aug = augs[0] if len(augs) == 1 else None
        if base and aug:
            return (base, aug)
        base = arm_values[0]
        aug = arm_values[1]
        return (base, aug)
    return "only_one_arm"


def build_checks(
    parsed: ParsedPredictions,
    *,
    horizon_days: int | None,
    protocol_hash_value: str | None,
    target_id: str,
    feature_columns: list[str] | None,
    arms: tuple[str, str] | str,
) -> list[EvidenceCheck]:
    checks: list[EvidenceCheck] = []

    has_time = _has_time_columns(parsed)
    time_check = EvidenceCheck(
        code="TIME_TARGET",
        status="unknown",
        detail="缺少发布时刻或目标时刻，未提供足够字段验证日历提前量。",
        blocking=True,
        reason="missing_time_columns",
    )
    if has_time:
        mismatches = 0
        checked = 0
        for row in parsed.rows:
            if row.get("forecast_origin") is None or row.get("target_time") is None:
                continue
            checked += 1
            delta = _time_delta_days(row.get("forecast_origin"), row.get("target_time"))
            if delta is None:
                continue
            if horizon_days is not None and abs(delta - float(horizon_days)) > 1e-6:
                mismatches += 1
        if checked == 0:
            time_check.status = "unknown"
            time_check.detail = "存在时间列，但没有可解析的发布/目标时刻。"
        elif mismatches == 0:
            time_check.status = "pass"
            time_check.detail = f"可解析的 {checked} 条记录满足提前量 {horizon_days} 天。"
        else:
            time_check.status = "fail"
            time_check.detail = f"{mismatches} 条记录的 target_time 与 forecast_origin 的日历差不符合 {horizon_days} 天。"
            time_check.reason = "calendar_offset_mismatch"
    checks.append(time_check)

    feature_check = EvidenceCheck(
        code="FEATURE_AVAILABLE",
        status="not_applicable",
        detail="本证据包未提供特征可用时间表，无法核验特征发布时间。",
        blocking=False,
        reason="no_feature_availability",
    )
    checks.append(feature_check)

    target_alias = EvidenceCheck(
        code="TARGET_ALIAS",
        status="unknown",
        detail="未提供特征列清单，无法检查目标或其同时刻别名是否混入特征。",
        blocking=True,
        reason="missing_feature_columns",
    )
    if feature_columns is not None:
        target_variants = _target_alias_variants(target_id)
        leak = [
            column
            for column in feature_columns
            if bool(_target_alias_variants(column) & target_variants)
        ]
        if leak:
            target_alias.status = "fail"
            target_alias.detail = f"目标字段或其别名出现在特征列中：{', '.join(leak)}"
            target_alias.reason = "target_leak_into_features"
        else:
            target_alias.status = "pass"
            target_alias.detail = "已提供特征列清单，未发现明显的目标字段同名泄漏。"
    checks.append(target_alias)

    split_check = EvidenceCheck(
        code="SPLIT_BOUNDARY",
        status="unknown",
        detail="预测文件缺少 split 列，无法独立核验训练/评价边界。",
        blocking=True,
        reason="missing_split_column",
    )
    split_values = {str(row.get("split") or "").strip().lower() for row in parsed.rows if str(row.get("split") or "").strip()}
    if split_values:
        split_check.status = "pass"
        split_check.detail = f"预测文件声明了数据划分：{', '.join(sorted(split_values))}。"
    checks.append(split_check)

    fit_scope = EvidenceCheck(
        code="FIT_SCOPE",
        status="unknown",
        detail="仅凭预测 CSV 无法从外部核验预处理与选参是否只在训练折内进行。",
        blocking=False,
        reason="external_scope_unknown",
    )
    checks.append(fit_scope)

    paired = EvidenceCheck(
        code="PAIRED_SAMPLES",
        status="not_applicable",
        detail="预测文件没有可比较的两组方案。",
        blocking=True,
        reason="single_arm_or_unknown_arms",
    )
    target_equal = EvidenceCheck(
        code="TARGET_EQUAL",
        status="not_applicable",
        detail="未形成共同样本对，无法比较真实目标。",
        blocking=True,
        reason="no_paired_samples",
    )
    if isinstance(arms, tuple):
        if not has_time:
            paired.status = "unknown"
            paired.detail = "缺少可解析的 forecast_origin/target_time，无法验证两组样本键一一对应。"
            paired.reason = "missing_time_for_pairing"
            target_equal.status = "unknown"
            target_equal.detail = "缺少时间键，无法比较两组真实目标。"
            target_equal.reason = "missing_time_for_pairing"
        else:
            base_keys: dict[str, float] = {}
            duplicate_base: list[str] = []
            for row in parsed.rows:
                if str(row.get("arm")) == arms[0]:
                    key = _pair_key(row)
                    if key in base_keys:
                        duplicate_base.append(key)
                    else:
                        base_keys[key] = row.get("y_true")
            aug_keys: dict[str, float] = {}
            duplicate_aug: list[str] = []
            for row in parsed.rows:
                if str(row.get("arm")) == arms[1]:
                    key = _pair_key(row)
                    if key in aug_keys:
                        duplicate_aug.append(key)
                    else:
                        aug_keys[key] = row.get("y_true")
            if duplicate_base or duplicate_aug:
                paired.status = "fail"
                paired.detail = f"预测键不唯一（base 重复 {len(duplicate_base)} 条，aug 重复 {len(duplicate_aug)} 条）。"
                paired.reason = "duplicate_prediction_keys"
            elif base_keys.keys() != aug_keys.keys():
                paired.status = "fail"
                paired.detail = (
                    f"两组预测键不一致：base {len(base_keys)} 条，aug {len(aug_keys)} 条，"
                    f"共同 {len(set(base_keys) & set(aug_keys))} 条。"
                )
                paired.reason = "sample_keys_mismatch"
            else:
                paired.status = "pass"
                paired.detail = f"两组在共同样本集上形成 {len(base_keys)} 条一一对应预测对。"
            if paired.status == "pass":
                mismatches = [
                    key
                    for key in base_keys
                    if not np.isclose(float(base_keys[key] or np.nan), float(aug_keys[key] or np.nan), atol=1e-6)
                ]
                if mismatches:
                    target_equal.status = "fail"
                    target_equal.detail = f"{len(mismatches)} 个共同样本对的 y_true 不一致。"
                    target_equal.reason = "target_value_mismatch"
                else:
                    target_equal.status = "pass"
                    target_equal.detail = f"{len(base_keys)} 个共同样本对的 y_true 在数值容差内一致。"
            else:
                target_equal.status = "unknown"
                target_equal.detail = "两组样本键未通过一致性检查，不能直接比较真实目标。"
                target_equal.reason = paired.reason
    checks.append(paired)
    checks.append(target_equal)

    version_check = EvidenceCheck(
        code="DATA_VERSION",
        status="pass" if protocol_hash_value else "unknown",
        detail=(
            f"协议哈希已登记：{protocol_hash_value[:16]}…"
            if protocol_hash_value
            else "未提供 protocol_hash，数据与协议版本无法追溯。"
        ),
        blocking=False,
        reason=None if protocol_hash_value else "missing_protocol_hash",
    )
    checks.append(version_check)

    quality = EvidenceCheck(
        code="QUALITY_COVERAGE",
        status="not_applicable",
        detail="预测文件中没有日影质量（ts_quality/TS）列，质量覆盖率未登记。",
        blocking=False,
        reason="missing_quality_column",
    )
    if any(row.get("ts_quality") is not None for row in parsed.rows):
        quality.status = "pass"
        quality.detail = "预测文件已提供日影质量列，可作为后续敏感性分析输入。"
    checks.append(quality)

    holdout = EvidenceCheck(
        code="HOLDOUT_REUSE",
        status="unknown",
        detail="无法从单个导入产物判断该评价集是否被反复用于方案选择。",
        blocking=False,
        reason="external_unknown",
    )
    checks.append(holdout)
    return checks


def _compute_arm_metrics(
    rows: list[dict[str, Any]],
    *,
    arm: str,
    unit: str | None,
) -> dict[str, MetricValue]:
    truths = [row.get("y_true") for row in rows if row.get("arm") == arm]
    preds = [row.get("y_pred") for row in rows if row.get("arm") == arm]
    pairs = list(zip(truths, preds))
    valid = [(float(t), float(p)) for t, p in pairs if t is not None and p is not None and not math.isnan(float(t)) and not math.isnan(float(p))]
    prefix = _arm_label(arm)
    if len(valid) < 2:
        return {
            f"rmse.{prefix}": MetricValue(value=None, unit=unit, reason="有效样本少于 2 条。", metric_ref=f"rmse.{prefix}"),
            f"mae.{prefix}": MetricValue(value=None, unit=unit, reason="有效样本少于 2 条。", metric_ref=f"mae.{prefix}"),
            f"r.{prefix}": MetricValue(value=None, reason="有效样本少于 2 条。", metric_ref=f"r.{prefix}"),
        }
    truth_array = np.asarray([item[0] for item in valid], dtype=float)
    pred_array = np.asarray([item[1] for item in valid], dtype=float)
    rmse = float(np.sqrt(np.mean((truth_array - pred_array) ** 2)))
    mae = float(np.mean(np.abs(truth_array - pred_array)))
    r_value: float | None
    if np.std(truth_array) < 1e-12 or np.std(pred_array) < 1e-12:
        r_value = None
    else:
        r_value = float(np.corrcoef(truth_array, pred_array)[0, 1])
    return {
        f"rmse.{prefix}": MetricValue(value=rmse, unit=unit, metric_ref=f"rmse.{prefix}"),
        f"mae.{prefix}": MetricValue(value=mae, unit=unit, metric_ref=f"mae.{prefix}"),
        f"r.{prefix}": MetricValue(value=r_value, unit=None, metric_ref=f"r.{prefix}"),
    }


def compute_metrics(
    parsed: ParsedPredictions,
    *,
    arms: tuple[str, str] | str,
    unit: str | None,
) -> dict[str, MetricValue]:
    metrics: dict[str, MetricValue] = {}
    if isinstance(arms, tuple):
        base_label, aug_label = arms
        metrics.update(_compute_arm_metrics(parsed.rows, arm=base_label, unit=unit))
        metrics.update(_compute_arm_metrics(parsed.rows, arm=aug_label, unit=unit))
        base_prefix, aug_prefix = _arm_label(base_label), _arm_label(aug_label)
        rmse_base = metrics.get(f"rmse.{base_prefix}")
        rmse_aug = metrics.get(f"rmse.{aug_prefix}")
        if rmse_base and rmse_aug and rmse_base.value is not None and rmse_aug.value is not None:
            metrics["gain_rmse"] = MetricValue(
                value=float(rmse_base.value - rmse_aug.value),
                unit=unit,
                reason="正值表示增强组 RMSE 更小。",
                metric_ref="gain_rmse",
            )
            if rmse_base.value:
                metrics["gain_rmse_relative"] = MetricValue(
                    value=float((rmse_base.value - rmse_aug.value) / rmse_base.value),
                    reason="相对 RMSE 改善；分子为 0 时仅显示原因。",
                    metric_ref="gain_rmse_relative",
                )
        mae_base = metrics.get(f"mae.{base_prefix}")
        mae_aug = metrics.get(f"mae.{aug_prefix}")
        if mae_base and mae_aug and mae_base.value is not None and mae_aug.value is not None:
            metrics["gain_mae"] = MetricValue(
                value=float(mae_base.value - mae_aug.value),
                unit=unit,
                reason="正值表示增强组 MAE 更小。",
                metric_ref="gain_mae",
            )
        r_base = metrics.get(f"r.{base_prefix}")
        r_aug = metrics.get(f"r.{aug_prefix}")
        if r_base and r_aug and r_base.value is not None and r_aug.value is not None:
            metrics["gain_r"] = MetricValue(
                value=float(r_aug.value - r_base.value),
                reason="正值表示增强组 Pearson r 更高。",
                metric_ref="gain_r",
            )
        base_keys = {_pair_key(row) for row in parsed.rows if str(row.get("arm")) == base_label}
        aug_keys = {_pair_key(row) for row in parsed.rows if str(row.get("arm")) == aug_label}
        metrics["common_sample_count"] = MetricValue(
            value=len(base_keys & aug_keys),
            reason="按 forecast_origin/target_time/horizon_days 计算的共同样本数。",
            metric_ref="common_sample_count",
        )
        metrics[f"sample_count.{base_prefix}"] = MetricValue(value=len(base_keys), metric_ref=f"sample_count.{base_prefix}")
        metrics[f"sample_count.{aug_prefix}"] = MetricValue(value=len(aug_keys), metric_ref=f"sample_count.{aug_prefix}")
    else:
        for arm in parsed.arm_values:
            metrics.update(_compute_arm_metrics(parsed.rows, arm=arm, unit=unit))
        metrics["common_sample_count"] = MetricValue(
            value=len(parsed.rows),
            reason="未提供第二组方案，共同样本数按单组全部记录登记。",
            metric_ref="common_sample_count",
        )
    return metrics


def build_evidence_packet(
    *,
    parsed: ParsedPredictions,
    protocol_hash_value: str | None,
    data_manifest_hash: str | None,
    run_id: str | None,
    claim_ids: list[str],
    target_id: str,
    unit: str | None,
    horizon_days: int | None,
    feature_columns: list[str] | None,
    base_arm: str | None,
    aug_arm: str | None,
    primary_artifact_id: str | None,
    primary_artifact_sha256: str | None,
    imported_metrics: dict[str, Any] | None = None,
    packet_id: str | None = None,
) -> EvidencePacket:
    if imported_metrics is not None:
        derived_arm_values = set(parsed.arm_values)
        for key in imported_metrics:
            lowered = str(key).lower()
            for prefix in ("rmse.", "mae.", "r."):
                if lowered.startswith(prefix):
                    suffix = str(key)[len(prefix):]
                    if suffix and suffix != "gain":
                        derived_arm_values.add(suffix)
        if base_arm:
            derived_arm_values.add(str(base_arm))
        if aug_arm:
            derived_arm_values.add(str(aug_arm))
        parsed.arm_values = sorted(derived_arm_values)

    arms_result = _resolve_arms(parsed, base_arm=base_arm, aug_arm=aug_arm)
    arms: tuple[str, str] | str = "only_one_arm" if arms_result == "only_one_arm" else arms_result
    if isinstance(arms, str) and arms not in {"only_one_arm"}:
        raise ValueError(arms)

    if imported_metrics is not None:
        metrics: dict[str, MetricValue] = {}
        for key, raw in imported_metrics.items():
            if isinstance(raw, dict):
                metrics[str(key)] = MetricValue(
                    value=_to_float(raw.get("value")),
                    unit=raw.get("unit"),
                    reason=raw.get("reason"),
                    metric_ref=raw.get("metric_ref") or str(key),
                )
            else:
                metrics[str(key)] = MetricValue(value=_to_float(raw), unit=unit, metric_ref=str(key))
        metrics["computed_elsewhere"] = MetricValue(
            value=1,
            reason="指标直接来自导入 JSON，未在本服务重新计算。",
            metric_ref="computed_elsewhere",
        )
    else:
        metrics = compute_metrics(parsed, arms=arms, unit=unit)

    checks = build_checks(
        parsed,
        horizon_days=horizon_days,
        protocol_hash_value=protocol_hash_value,
        target_id=target_id,
        feature_columns=feature_columns,
        arms=arms,
    )

    limitations: list[str] = []
    if imported_metrics is not None:
        limitations.append("导入模式：指标数值直接来自上传 JSON，未基于逐行预测记录重新计算。")
    if parsed.missing_columns:
        limitations.append(f"缺少字段：{', '.join(parsed.missing_columns)}；对应检查标记为 unknown。")
    if parsed.non_numeric_count:
        limitations.append(f"有 {parsed.non_numeric_count} 条记录的 y_true/y_pred 不是数值，已从指标计算中排除。")
    if imported_metrics is None and arms == "only_one_arm":
        limitations.append("当前证据包只有一组方案，不能进行公平的增量对比。")
    if isinstance(arms, tuple) and any(check.code == "PAIRED_SAMPLES" and check.status != "pass" for check in checks):
        limitations.append("两组预测键未全部匹配，不能据此得出直接增量结论。")

    evidence_items: list[EvidenceItem] = []
    for key, metric in metrics.items():
        if metric.value is None:
            continue
        evidence_items.append(
            EvidenceItem(
                evidence_id=f"EV_{_arm_label(key)}",
                artifact_id=primary_artifact_id,
                artifact_sha256=primary_artifact_sha256,
                locator=f"/metrics/{key}",
                value_type="metric",
                value=float(metric.value),
                unit=metric.unit,
                provenance=f"program evidence_builder -> metrics.{key}",
            )
        )

    if parsed.rows:
        evidence_items.append(
            EvidenceItem(
                evidence_id="EV_SAMPLE_ROWS",
                artifact_id=primary_artifact_id,
                artifact_sha256=primary_artifact_sha256,
                locator="/rows",
                value_type="count",
                value=len(parsed.rows),
                provenance="program evidence_builder row count",
            )
        )
    if isinstance(arms, tuple):
        evidence_items.append(
            EvidenceItem(
                evidence_id="EV_ARM_PAIRING",
                artifact_id=primary_artifact_id,
                artifact_sha256=primary_artifact_sha256,
                locator=f"/arms/{arms[0]}|{arms[1]}",
                value_type="assignment",
                value=f"{arms[0]} -> {arms[1]}",
                provenance="program evidence_builder arm resolution",
            )
        )

    seed: dict[str, Any] = {
        "schema_version": "research-review/1.0",
        "run_id": run_id,
        "protocol_hash": protocol_hash_value,
        "data_manifest_hash": data_manifest_hash,
        "claim_ids": sorted(claim_ids),
        "evidence_items": [item.model_dump(exclude_none=True) for item in evidence_items],
        "checks": [check.model_dump(exclude_none=True) for check in checks],
        "metrics": {key: metric.model_dump(exclude_none=True) for key, metric in sorted(metrics.items())},
        "limitations": limitations,
    }
    packet_hash = protocol_hash(seed)
    return EvidencePacket(
        packet_id=packet_id or f"EVP_{packet_hash[:12]}",
        packet_hash=packet_hash,
        run_id=run_id,
        protocol_hash=protocol_hash_value,
        data_manifest_hash=data_manifest_hash,
        claim_ids=sorted(claim_ids),
        evidence_items=evidence_items,
        checks=checks,
        metrics=metrics,
        limitations=limitations,
    )


def summarize_preflight(
    parsed: ParsedPredictions,
    protocol_hash_value: str | None,
) -> dict[str, Any]:
    return {
        "row_count": len(parsed.rows),
        "arm_values": parsed.arm_values,
        "columns": parsed.columns,
        "missing_columns": parsed.missing_columns,
        "non_numeric_count": parsed.non_numeric_count,
        "protocol_hash": protocol_hash_value,
        "parse_errors": parsed.parse_errors,
    }
