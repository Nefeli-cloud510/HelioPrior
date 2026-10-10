from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from core.protocol_bridge import ElasticNetDataSourceConfig, ElasticNetProtocolBridge
from core.protocol_hash import PROTOCOL_VERSION, protocol_hash
from core.method_registry import METHOD_REGISTRY
from core.result_unified import build_experiment_result, export_result_bundle
from core.unified_schema import (
    DataSourceCoverage,
    ExperimentProtocol,
    ExperimentResult,
    VisualizationArtifact,
)
from models.elasticnet_runner import resolve_alpha_for_model


def _values_differ(baseline_y: float | None, treatment_y: float | None) -> bool:
    if baseline_y is None or treatment_y is None:
        return baseline_y != treatment_y
    if np.isnan(baseline_y) or np.isnan(treatment_y):
        return not (np.isnan(baseline_y) and np.isnan(treatment_y))
    return not bool(
        np.isclose(baseline_y, treatment_y, rtol=1e-5, atol=1e-6)
    )


class UnifiedExperimentHarness:
    """Type-safe harness that executes the linear model runner via unified schemas."""

    def __init__(
        self,
        *,
        data_source: ElasticNetDataSourceConfig,
        project_root: Path | None = None,
        run_shap: bool = True,
    ) -> None:
        self.project_root = project_root or Path(__file__).resolve().parent.parent
        self.results_dir = self.project_root / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.latest_protocol_path = self.project_root / "config" / "latest_protocol.json"
        self.latest_metrics_path = self.project_root / "outputs" / "latest_metrics.json"
        self.bridge = ElasticNetProtocolBridge(data_source)
        self.run_shap = run_shap

    def run(self, protocol: ExperimentProtocol) -> ExperimentResult:
        if not METHOD_REGISTRY.has(protocol.model.name):
            raise ValueError(
                f"当前统一 harness 仅支持方法注册中心内的方法，收到: {protocol.model.name}"
            )
        if not protocol.features.treatment:
            raise ValueError("ExperimentProtocol.features.treatment 不能为空")
        is_baseline_single_arm = "experiment_mode:baseline_single_arm" in (protocol.notes or []) or not protocol.features.control
        if not is_baseline_single_arm and not protocol.features.control:
            raise ValueError("ExperimentProtocol.features.control 不能为空")

        round_dir = self.results_dir / f"round_{protocol.round_id:02d}"
        execution_dir = round_dir / protocol.experiment_id
        execution_dir.mkdir(parents=True, exist_ok=True)

        started = time.perf_counter()
        baseline_features = self._resolve_feature_columns(
            protocol=protocol,
            label="baseline",
            fallback=protocol.features.treatment if is_baseline_single_arm else protocol.features.control,
        )
        treatment_features = self._resolve_feature_columns(
            protocol=protocol,
            label="treatment",
            fallback=protocol.features.treatment,
        )
        if is_baseline_single_arm:
            baseline_output = self._execute_variant(
                protocol=protocol,
                feature_columns=baseline_features,
                label="baseline",
                output_dir=execution_dir / "baseline",
                only_keys=None,
            )
            treatment_output = baseline_output
        else:
            baseline_keys = self._build_retained_keys(
                protocol=protocol,
                feature_columns=baseline_features,
                label="baseline",
            )
            treatment_keys = self._build_retained_keys(
                protocol=protocol,
                feature_columns=treatment_features,
                label="treatment",
            )
            common_keys = baseline_keys & treatment_keys
            if not common_keys:
                raise ValueError(
                    "日历对齐后对照组与增强组没有共同样本，"
                    "不能进行同样本对比实验"
                )
            baseline_output = self._execute_variant(
                protocol=protocol,
                feature_columns=baseline_features,
                label="baseline",
                output_dir=execution_dir / "baseline",
                only_keys=common_keys,
            )
            treatment_output = self._execute_variant(
                protocol=protocol,
                feature_columns=treatment_features,
                label="treatment",
                output_dir=execution_dir / "treatment",
                only_keys=common_keys,
            )
        self._assert_paired_predictions(baseline_output, treatment_output)
        duration_seconds = time.perf_counter() - started

        predictions_dir = execution_dir / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        prediction_paths = {
            "baseline": self._to_relative_path(self._write_prediction_csv(
                predictions_dir / f"{protocol.experiment_id}_baseline.csv",
                baseline_output,
            )),
            "treatment": self._to_relative_path(self._write_prediction_csv(
                predictions_dir / f"{protocol.experiment_id}_treatment.csv",
                treatment_output,
            )),
        }

        visualizations = self._collect_visualizations(protocol, baseline_output, treatment_output)
        coverage_entries: list[DataSourceCoverage] = []
        outputs_to_track = [baseline_output] if is_baseline_single_arm else [baseline_output, treatment_output]
        for output in outputs_to_track:
            for entry in output.get("data_coverage", []):
                coverage_entries.append(DataSourceCoverage.model_validate(entry))
        result = build_experiment_result(
            protocol,
            baseline_output=baseline_output,
            treatment_output=treatment_output,
            prediction_paths=prediction_paths,
            visualizations=visualizations,
            duration_seconds=duration_seconds,
            data_coverage=coverage_entries,
            protocol_version=baseline_output.get("protocol_version"),
            protocol_hash=baseline_output.get("protocol_hash"),
        )
        export_result_bundle(
            result,
            protocol,
            round_dir=round_dir,
            latest_protocol_path=self.latest_protocol_path,
            latest_metrics_path=self.latest_metrics_path,
        )
        return result

    @staticmethod
    def _resolve_feature_columns(
        *,
        protocol: ExperimentProtocol,
        label: str,
        fallback: list[str],
    ) -> list[str]:
        feature_groups = protocol.model.parameters.get("feature_groups")
        if isinstance(feature_groups, dict):
            configured = feature_groups.get(label)
            if isinstance(configured, list):
                normalized = [str(item).strip() for item in configured if str(item).strip()]
                if normalized:
                    return normalized
        return [str(item).strip() for item in fallback if str(item).strip()]

    def _execute_variant(
        self,
        *,
        protocol: ExperimentProtocol,
        feature_columns: list[str],
        label: str,
        output_dir: Path,
        only_keys: set[tuple[str, str, int]] | None = None,
    ) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        method = METHOD_REGISTRY.get(protocol.model.name)
        runner = METHOD_REGISTRY.runner_for(method.method_id)
        arm_windows = protocol.model.parameters.get("arm_time_windows")
        arm_overrides = arm_windows.get(label) if isinstance(arm_windows, dict) else None
        runner_config = self.bridge.build_runner_config(
            protocol,
            feature_columns,
            arm_overrides=arm_overrides,
        )

        merged = runner.load_and_merge_data(runner_config)
        window_result = runner.build_sliding_window(
            merged.X,
            merged.y,
            merged.time,
            runner_config.time_window_config,
            only_keys=only_keys,
        )
        split = runner.split_dataset_by_date(
            window_result.X_all,
            window_result.y_all,
            window_result.time_all,
            runner_config.time_window_config.test_split_ratio,
            window_result.feature_names,
            origins_all=window_result.origins_all,
        )
        xtr, xte = split.X_train, split.X_test
        ytr, yte = split.y_train, split.y_test
        time_train, time_test = split.time_train, split.time_test
        origin_test = split.origin_test
        scaler = StandardScaler().fit(xtr)

        model_family = runner_config.elasticnet_config.model_family
        model = runner.train_model(xtr, ytr, runner_config.elasticnet_config)
        metrics, ytr_pred, yte_pred = runner.evaluate_model(model, xtr, xte, ytr, yte)
        metrics["model_family"] = model_family
        metrics["alpha_selected"] = resolve_alpha_for_model(
            model,
            runner_config.elasticnet_config.alpha,
        )
        coef_df = runner.make_coefficient_table(model, window_result.feature_names)
        coef_df.to_csv(output_dir / f"{label}_coefficients.csv", index=False, encoding="utf-8-sig")

        sparse_labels = set(merged.active_sparse_sources)
        coverage_entries: list[dict] = []
        # 只有本轮真正启用了稀疏数据源的缺失日处理时，才对外输出覆盖审计；
        # 本轮仅使用 OMNI + LHAASO 时，不把故意剔除 ICME 的日期误报为缺失。
        if sparse_labels:
            for cov in merged.coverage:
                if cov.source not in sparse_labels:
                    continue
                entry = cov.to_dict()
                entry["run_id"] = label
                entry["dropped_gap_windows"] = 0
                entry["note"] = "缺失记录已按行剔除（inner merge + dropna），窗口直接基于剩余行构造"
                coverage_entries.append(entry)

        full_time = np.concatenate([time_train, time_test])
        x_limits = (full_time.min(), full_time.max()) if len(full_time) > 0 else None
        split_time = time_test[0] if len(time_test) > 0 else None
        chart_label = f"{protocol.experiment_id} / {label}"
        figure_paths = {
            "coefficients": runner.plot_feature_coefficients(
                coef_df,
                output_dir,
                model_family=model_family,
            ),
            "scatter": runner.plot_prediction_scatter(
                yte,
                yte_pred,
                metrics,
                output_dir,
                experiment_label=chart_label,
                train_true=ytr,
                train_pred=ytr_pred,
                model_family=model_family,
            ),
            "timeseries": runner.plot_prediction_series(
                time_test,
                yte,
                yte_pred,
                output_dir,
                experiment_label=chart_label,
                time_train=time_train,
                train_true=ytr,
                train_pred=ytr_pred,
                x_limits=x_limits,
                split_time=split_time,
                model_family=model_family,
            ),
            "residual_scatter": runner.plot_residual_scatter(
                yte_pred,
                yte - yte_pred,
                output_dir,
                model_family=model_family,
            ),
            "residual_hist": runner.plot_residual_histogram(
                yte - yte_pred,
                output_dir,
                model_family=model_family,
            ),
        }

        shap_paths: dict[str, str] = {}
        shap_importance = None
        if self.run_shap:
            shap_result = runner.run_shap_analysis(
                model,
                xtr,
                xte,
                output_dir,
                model_family=model_family,
            )
            shap_importance = shap_result["importance"]
            shap_paths = {
                "summary": str(shap_result["summary_path"]),
                "bar": str(shap_result["bar_path"]),
                "waterfall": str(shap_result["waterfall_path"]),
                "importance_csv": str(shap_result["importance_path"]),
                "variable_importance_csv": str(shap_result["variable_importance_path"]),
                "lag_profile_csv": str(shap_result["lag_profile_path"]),
            }

        return {
            "label": label,
            "run_id": f"{protocol.experiment_id}_{label}",
            "config": runner_config.model_dump(mode="json"),
            "metrics": metrics,
            "coefficients": coef_df,
            "model": model,
            "scaler": scaler,
            "time_train": time_train,
            "time_test": time_test,
            "origin_test": origin_test,
            "horizon_days": int(runner_config.time_window_config.forecast_horizon_days),
            "protocol_version": PROTOCOL_VERSION,
            "protocol_hash": protocol_hash(runner_config.model_dump(mode="json")),
            "y_train": ytr,
            "y_test": yte,
            "y_train_pred": ytr_pred,
            "y_test_pred": yte_pred,
            "figure_paths": {name: str(path) for name, path in figure_paths.items()},
            "shap_paths": shap_paths,
            "shap_importance": shap_importance,
            "data_coverage": coverage_entries,
            "sliding_window_total": window_result.total_windows,
            "sliding_window_dropped": window_result.dropped_gap_windows,
            "sliding_window_common_sample_excluded": window_result.common_sample_excluded,
        }

    def _build_retained_keys(
        self,
        *,
        protocol: ExperimentProtocol,
        feature_columns: list[str],
        label: str,
    ) -> set[tuple[str, str, int]]:
        """Load each arm's calendar-eligible samples before execution and return retained keys."""

        method = METHOD_REGISTRY.get(protocol.model.name)
        runner = METHOD_REGISTRY.runner_for(method.method_id)
        arm_windows = protocol.model.parameters.get("arm_time_windows")
        arm_overrides = arm_windows.get(label) if isinstance(arm_windows, dict) else None
        runner_config = self.bridge.build_runner_config(
            protocol,
            feature_columns,
            arm_overrides=arm_overrides,
        )
        merged = runner.load_and_merge_data(runner_config)
        window_result = runner.build_sliding_window(
            merged.X,
            merged.y,
            merged.time,
            runner_config.time_window_config,
        )
        if window_result.sample_plan is None:
            raise ValueError("日历样本计划缺失，无法计算共同样本")
        return window_result.sample_plan.compact_keys()

    @staticmethod
    def _write_prediction_csv(file_path: Path, execution_output: dict) -> Path:
        target_time = pd.to_datetime(execution_output["time_test"])
        forecast_origin = pd.to_datetime(execution_output["origin_test"])
        frame = pd.DataFrame(
            {
                "run_id": execution_output["run_id"],
                "split": "test",
                "forecast_origin": forecast_origin.astype(str),
                "target_time": target_time.astype(str),
                "horizon_days": int(execution_output["horizon_days"]),
                "time": target_time.astype(str),
                "y_true": execution_output["y_test"],
                "y_pred": execution_output["y_test_pred"],
                "protocol_version": execution_output["protocol_version"],
                "protocol_hash": execution_output["protocol_hash"],
            }
        )
        frame.to_csv(file_path, index=False, encoding="utf-8-sig")
        return file_path

    @staticmethod
    def _assert_paired_predictions(baseline_output: dict, treatment_output: dict) -> None:
        """Block a comparison unless both arms cover identical test origins/targets."""

        if baseline_output is treatment_output:
            return
        baseline_pairs = UnifiedExperimentHarness._prediction_pairs(baseline_output)
        treatment_pairs = UnifiedExperimentHarness._prediction_pairs(treatment_output)
        baseline_keys = set(baseline_pairs)
        treatment_keys = set(treatment_pairs)
        if baseline_keys != treatment_keys:
            only_baseline = baseline_keys - treatment_keys
            only_treatment = treatment_keys - baseline_keys
            raise ValueError(
                "双路预测样本不一致，禁止比较："
                f"仅对照组 {len(only_baseline)} 条，仅增强组 {len(only_treatment)} 条"
            )
        for key in sorted(baseline_keys):
            baseline_y = baseline_pairs[key]
            treatment_y = treatment_pairs[key]
            if _values_differ(baseline_y, treatment_y):
                raise ValueError(
                    f"双路预测同一样本真实目标不一致: {key} "
                    f"baseline={baseline_y!r} treatment={treatment_y!r}"
                )

    @staticmethod
    def _prediction_pairs(execution_output: dict) -> dict[tuple, float | None]:
        origins = pd.to_datetime(execution_output["origin_test"])
        targets = pd.to_datetime(execution_output["time_test"])
        horizon = int(execution_output["horizon_days"])
        raw_ys = execution_output["y_test"]
        pairs: dict[tuple, float | None] = {}
        for origin, target, raw in zip(origins, targets, raw_ys):
            key = (
                origin.date().isoformat(),
                target.date().isoformat(),
                horizon,
            )
            pairs[key] = float(raw) if raw is not None else None
        return pairs

    def _collect_visualizations(
        self,
        protocol: ExperimentProtocol,
        baseline_output: dict,
        treatment_output: dict,
    ) -> list[VisualizationArtifact]:
        visualizations: list[VisualizationArtifact] = []

        for label, output in (("baseline", baseline_output), ("treatment", treatment_output)):
            for name, path in output["figure_paths"].items():
                visualizations.append(
                    VisualizationArtifact(
                        name=f"{protocol.experiment_id}_{label}_{name}",
                        path=self._to_relative_path(Path(path)),
                    )
                )
            for name, path in output["shap_paths"].items():
                visualizations.append(
                    VisualizationArtifact(
                        name=f"{protocol.experiment_id}_{label}_shap_{name}",
                        path=self._to_relative_path(Path(path)),
                    )
                )

        return visualizations

    def _to_relative_path(self, path: Path) -> str:
        try:
            relative = path.resolve().relative_to(self.project_root.resolve())
            return relative.as_posix()
        except ValueError:
            return path.as_posix()
