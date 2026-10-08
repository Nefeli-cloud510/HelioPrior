from __future__ import annotations

from importlib import import_module
from typing import Any, Literal

from pydantic import BaseModel, Field


ParameterType = Literal[
    "number",
    "integer",
    "boolean",
    "string",
    "number_list",
    "string_list",
]


class ParameterSpec(BaseModel):
    """Declaration of one tunable model parameter exposed to planners/UI."""

    name: str = Field(min_length=1)
    label: str = Field(min_length=1)
    type: ParameterType = "number"
    default: Any = None
    description: str = ""
    min: float | int | None = None
    max: float | int | None = None
    options: list[str] = Field(default_factory=list)


class MethodDescriptor(BaseModel):
    """One registered forecasting method backed by a runner plugin module."""

    method_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    family: str = Field(default="linear_regularized")
    runner_module: str = Field(min_length=1)
    description: str = ""
    capabilities: list[str] = Field(default_factory=list)
    default_parameters: dict[str, Any] = Field(default_factory=dict)
    parameter_schema: list[ParameterSpec] = Field(default_factory=list)
    enabled: bool = True


DEFAULT_METHOD_ID = "ridge"


class MethodRegistry:
    """Source of truth for method selection and plugin runner routing."""

    def __init__(self) -> None:
        self._methods: dict[str, MethodDescriptor] = {}
        self._aliases: dict[str, str] = {}

    def register(
        self,
        descriptor: MethodDescriptor,
        *,
        aliases: list[str] | None = None,
    ) -> None:
        normalized = descriptor.method_id.strip().lower()
        if not normalized:
            raise ValueError("method_id cannot be empty")
        self._methods[normalized] = descriptor
        self._aliases[normalized] = normalized
        for alias in aliases or []:
            alias_id = alias.strip().lower()
            if alias_id:
                self._aliases[alias_id] = normalized

    def resolve_method_id(self, value: Any | None) -> str:
        if value is None or str(value).strip().lower() in {"", "default", "auto"}:
            return DEFAULT_METHOD_ID
        raw = str(value or "").strip().lower()
        if raw in self._aliases:
            return self._aliases[raw]
        raise ValueError(
            f"未知方法: {value!r}；当前已注册方法: {', '.join(self.display_strings())}"
        )

    def get(self, method_id: Any | None) -> MethodDescriptor:
        resolved = self.resolve_method_id(method_id)
        descriptor = self._methods[resolved]
        if not descriptor.enabled:
            raise ValueError(f"方法已禁用: {descriptor.method_id}")
        return descriptor

    def has(self, method_id: Any | None) -> bool:
        try:
            self.resolve_method_id(method_id)
            return True
        except ValueError:
            return False

    def runner_for(self, method_id: Any | None):
        descriptor = self.get(method_id)
        return import_module(descriptor.runner_module)

    def default(self) -> MethodDescriptor:
        return self.get(DEFAULT_METHOD_ID)

    def default_parameters(self, method_id: Any | None = None) -> dict[str, Any]:
        return dict(self.get(method_id).default_parameters)

    def display_strings(self) -> list[str]:
        return [
            f"{item.display_name}（{item.method_id}）"
            for item in self._methods.values()
            if item.enabled
        ]

    def display_names(self) -> set[str]:
        return {item.display_name for item in self._methods.values() if item.enabled}

    def catalog(self) -> list[dict[str, Any]]:
        return [
            {
                "methodId": item.method_id,
                "displayName": item.display_name,
                "family": item.family,
                "runnerModule": item.runner_module,
                "description": item.description,
                "capabilities": list(item.capabilities),
                "defaultParameters": dict(item.default_parameters),
                "parameterSchema": [spec.model_dump(mode="json") for spec in item.parameter_schema],
            }
            for item in self._methods.values()
            if item.enabled
        ]

    def catalog_json(self) -> dict[str, Any]:
        return {
            "defaultMethodId": DEFAULT_METHOD_ID,
            "methods": self.catalog(),
        }

    def parameter_labels(self, method_id: Any | None = None) -> dict[str, str]:
        labels: dict[str, str] = {}
        descriptor = self.get(method_id) if method_id is not None else self.default()
        for spec in descriptor.parameter_schema:
            labels[spec.name] = spec.label
        return labels

    def filter_parameters(
        self,
        parameters: dict[str, Any],
        *,
        method_id: Any | None = None,
    ) -> dict[str, Any]:
        if not parameters:
            return {}
        effective_id = method_id if method_id is not None else parameters.get("model_family")
        try:
            descriptor = self.get(effective_id)
        except ValueError:
            if method_id is not None:
                raise
            descriptor = self.default()
        specs = {spec.name: spec for spec in descriptor.parameter_schema}
        filtered: dict[str, Any] = {}
        for key, value in parameters.items():
            spec = specs.get(key)
            if spec is None:
                continue
            if spec.type in {"number", "integer"}:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                if spec.type == "integer" and isinstance(value, float) and not value.is_integer():
                    continue
                filtered[key] = value
            elif spec.type == "boolean" and isinstance(value, bool):
                filtered[key] = value
            elif spec.type in {"string", "string_list"}:
                if isinstance(value, str) and value.strip():
                    if spec.options and value.strip() not in spec.options:
                        continue
                    filtered[key] = value.strip()
                elif spec.type == "string_list" and isinstance(value, list):
                    cleaned = [str(item).strip() for item in value if str(item).strip()]
                    if cleaned:
                        filtered[key] = cleaned
            elif spec.type == "number_list" and isinstance(value, list):
                try:
                    filtered[key] = [float(item) for item in value]
                except (TypeError, ValueError):
                    continue
        return filtered


def _build_default_schema() -> list[ParameterSpec]:
    return [
        ParameterSpec(
            name="window_size",
            label="窗口大小（天）",
            type="integer",
            default=14,
            min=1,
            max=196,
            description="滑动窗口天数",
        ),
        ParameterSpec(
            name="past_lag_days",
            label="样本滞后（天）",
            type="integer",
            default=None,
            min=1,
            max=196,
            description="仅保留最近 N 天滞后特征",
        ),
        ParameterSpec(
            name="forecast_horizon_days",
            label="预测视野（天）",
            type="integer",
            default=3,
            min=0,
            max=30,
            description="预测超前天数，0 表示同天",
        ),
        ParameterSpec(
            name="max_lag_day",
            label="最大滞后构造（天）",
            type="integer",
            default=3,
            min=0,
            max=10,
        ),
        ParameterSpec(
            name="test_split_ratio",
            label="测试集比例",
            type="number",
            default=0.2,
            min=0.1,
            max=0.5,
        ),
        ParameterSpec(
            name="model_family",
            label="方法族",
            type="string",
            default="ridge",
            options=["ridge", "elasticnet"],
        ),
        ParameterSpec(name="alpha", label="正则强度", type="number", default=1.0, min=0.0, max=2000.0),
        ParameterSpec(
            name="alpha_grid",
            label="alpha 候选网格",
            type="number_list",
            default=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        ),
        ParameterSpec(
            name="l1_ratio",
            label="L1 占比（ElasticNet）",
            type="number",
            default=0.5,
            min=0.0,
            max=1.0,
        ),
        ParameterSpec(
            name="ridge_solver",
            label="Ridge 求解器",
            type="string",
            default="svd",
            options=["svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"],
        ),
        ParameterSpec(
            name="fit_intercept",
            label="是否拟合截距",
            type="boolean",
            default=True,
        ),
        ParameterSpec(
            name="use_lag_feature",
            label="是否启用滞后特征",
            type="boolean",
            default=False,
        ),
        ParameterSpec(
            name="random_state",
            label="随机种子",
            type="integer",
            default=42,
        ),
    ]


_DEFAULT_PARAMETER_SCHEMA = _build_default_schema()


def _new_method(
    *,
    method_id: str,
    display_name: str,
    family: str,
    default_parameters: dict[str, Any],
    description: str,
    capabilities: list[str],
    aliases: list[str] | None = None,
) -> tuple[MethodDescriptor, list[str] | None]:
    return (
        MethodDescriptor(
            method_id=method_id,
            display_name=display_name,
            family=family,
            runner_module="models.elasticnet_runner",
            description=description,
            capabilities=capabilities,
            default_parameters=default_parameters,
            parameter_schema=[spec.model_copy(deep=True) for spec in _DEFAULT_PARAMETER_SCHEMA],
        ),
        aliases,
    )


METHOD_REGISTRY = MethodRegistry()

# 新增方法的收敛点：注册一个 MethodDescriptor（可配插件模块、默认参数、能力与参数表）。
for _descriptor, _aliases in (
    _new_method(
        method_id="ridge",
        display_name="Ridge",
        family="linear_regularized",
        default_parameters={
            "model_family": "ridge",
            "alpha": 1.0,
            "l1_ratio": 0.5,
            "alpha_grid": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            "ridge_solver": "svd",
            "fit_intercept": True,
        },
        description="L2 正则线性回归，当前默认主线；训练时按时序折选择 alpha 网格。",
        capabilities=["sliding_window", "dual_arm", "shap", "coverage_audit", "timeseries_alpha_cv"],
        aliases=["ridge_regression"],
    ),
    _new_method(
        method_id="elasticnet",
        display_name="ElasticNet",
        family="linear_regularized",
        default_parameters={
            "model_family": "elasticnet",
            "alpha": 1.0,
            "l1_ratio": 0.5,
            "alpha_grid": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            "ridge_solver": "svd",
            "fit_intercept": True,
        },
        description="L1/L2 混合正则线性模型，作为对照方法验证机制稳健性。",
        capabilities=["sliding_window", "dual_arm", "shap", "coverage_audit"],
        aliases=["enet", "elastic_net"],
    ),
):
    METHOD_REGISTRY.register(_descriptor, aliases=_aliases)


def resolve_method_id(value: Any | None) -> str:
    return METHOD_REGISTRY.resolve_method_id(value)


def filter_model_parameters(
    parameters: dict[str, Any],
    *,
    method_id: Any | None = None,
) -> dict[str, Any]:
    """Registry-driven filter shared by LLM planners and the central controller."""
    return METHOD_REGISTRY.filter_parameters(parameters, method_id=method_id)
