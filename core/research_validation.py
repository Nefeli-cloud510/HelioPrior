from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.calendar_samples import build_sample_plan
from core.llm_gateway import LLMGateway
from core.method_registry import METHOD_REGISTRY
from core.protocol_bridge import (
    ElasticNetDataSourceConfig,
    ElasticNetProtocolBridge,
)
from core.protocol_hash import PROTOCOL_VERSION, protocol_hash
from core.research_repository import ResearchRepository
from core.research_schema import (
    AgentCallRecord,
    Dispute,
    JobRecord,
    ResearchTarget,
    ProtocolSnapshot,
    ProtocolTimeSpec,
    ResearchDecision,
    ResearchRevision,
    ValidationCandidate,
    ValidationEstimate,
)
from core.runtime_config import get_llm_model_for_role
from core.unified_schema import ExperimentProtocol, ModelSpec


VALIDATION_TEMPLATES = {
    "paired_replay": {
        "description": "在相同日历样本上重放对照组与增强组，比较增量预测价值。",
    },
    "calendar_alignment": {
        "description": "冻结日历样本计划，核对两组共同样本、时间对齐与目标可用性。",
    },
}

ALLOWED_DESIGN_KEYS = {
    "control",
    "treatment",
    "target",
    "lags",
    "forecast_horizon_days",
    "past_lag_days",
    "window_size",
    "use_lag_feature",
    "max_lag_day",
    "alpha",
    "l1_ratio",
    "random_state",
    "test_split_ratio",
    "model",
    "model_family",
    "sample_condition",
    "probe_axis",
    "design_focus",
    "notes",
}

MAX_BATCH_CANDIDATES = 3

ACTIONABLE_DISPUTE_STATUSES = {"open", "proposed_validation"}
SUPERSEDED_CANDIDATE_STATUSES = {"draft", "pending_approval", "approved", "failed"}
EXECUTED_CANDIDATE_STATUSES = {"executed"}

UTILITY_WEIGHTS = {
    "ig": 0.4,
    "pg": 0.3,
    "risk": 0.15,
    "cost": 0.15,
}


class ResearchValidationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status: int,
        code: str,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code
        self.retryable = retryable
        self.details = details or {}


class PlannerCandidateOutput(BaseModel):
    """Structured candidate emitted by the planner role."""

    experiment_id: str = Field(min_length=1)
    dispute_ids: list[str] = Field(default_factory=list)
    hypothesis_ids: list[str] = Field(default_factory=list)
    template_id: str = Field(min_length=1)
    design: dict[str, Any] = Field(default_factory=dict)
    rationale: str | None = None


class ExperimentPlannerOutput(BaseModel):
    """Batch output from one experiment-planner LLM call."""

    batch_id: str = Field(min_length=1)
    candidates: list[PlannerCandidateOutput] = Field(default_factory=list)
    is_stub: bool = False


@dataclass(frozen=True)
class PlannerBatchResult:
    revision: ResearchRevision
    batch_id: str
    is_stub: bool
    candidate_ids: list[str]
    message: str


class ValidationPlanner:
    """Program-driven batch planning, approval, and gated execution."""

    def __init__(
        self,
        repository: ResearchRepository,
        *,
        gateway: Any | None = None,
        data_source: ElasticNetDataSourceConfig | None = None,
        harness_project_root: Path | None = None,
        run_shap: bool = False,
        experiment_planner_model: str | None = None,
    ) -> None:
        self.repository = repository
        self.gateway = gateway
        self.data_source = data_source
        self.harness_project_root = harness_project_root
        self.run_shap = run_shap
        self.experiment_planner_model = experiment_planner_model

    # ---------------------------------------------------------------- planning

    def plan_batch(
        self,
        research_id: str,
        expected_revision: int,
        payload: dict[str, Any],
    ) -> PlannerBatchResult:
        current = self._load(research_id)
        self._require_revision(current, expected_revision)

        raw_dispute_ids = payload.get("dispute_ids") or []
        if not isinstance(raw_dispute_ids, (list, tuple)):
            raise ResearchValidationError(
                "dispute_ids 必须是字符串数组。",
                status=422,
                code="invalid_dispute_ids",
            )
        requested_ids = [str(item) for item in raw_dispute_ids]
        existing_ids = {dispute.dispute_id for dispute in current.disputes}
        unknown_ids = [item for item in requested_ids if item not in existing_ids]
        if unknown_ids:
            raise ResearchValidationError(
                f"争议不存在：{', '.join(unknown_ids)}",
                status=422,
                code="invalid_dispute_ids",
            )

        actionable = [
            dispute
            for dispute in current.disputes
            if dispute.status in ACTIONABLE_DISPUTE_STATUSES
        ]
        if not actionable:
            raise ResearchValidationError(
                "批量规划需要至少一个待验证争议。",
                status=422,
                code="dispute_required",
            )
        if requested_ids:
            selected = [
                dispute
                for dispute in current.disputes
                if dispute.dispute_id in requested_ids
            ]
            for dispute in selected:
                if dispute.status not in ACTIONABLE_DISPUTE_STATUSES:
                    raise ResearchValidationError(
                        f"争议 {dispute.dispute_id} 当前状态 {dispute.status} 不可执行验证。",
                        status=422,
                        code="dispute_not_actionable",
                    )
        else:
            selected = actionable

        template_id = str(payload.get("template_id") or "").strip()
        hints = payload.get("design_hints") or {}
        if not isinstance(hints, dict):
            raise ResearchValidationError(
                "design_hints 必须是对象。",
                status=422,
                code="invalid_design_hints",
            )
        raw_model = payload.get("model") or payload.get("model_name") or "Ridge"

        model_id = self.experiment_planner_model or get_llm_model_for_role(
            "experiment_planner"
        )
        request_hash = ResearchRepository.request_hash(
            {
                "action": "plan_validations",
                "research_id": research_id,
                "expected_revision": current.revision,
                "dispute_ids": [dispute.dispute_id for dispute in selected],
                "template_id": template_id,
                "design_hints": hints,
                "model": str(raw_model),
            }
        )

        system_prompt = (
            "你是先羲科研协作系统的实验规划者。你的职责是把争议映射为可执行的受控实验，"
            "一次只输出一批最多 3 个候选。数值、科学检查、价值排序和状态变更由程序完成；"
            "你只提供结构化候选、设计草案和简短理由。不要编造实验结论，不要输出本批之外的"
            "长文案或调参建议。"
        )
        user_prompt = self._build_planner_prompt(
            project=current.project.model_dump(exclude_none=True),
            disputes=selected,
            template_id=template_id,
            hints=hints,
            allowed_template_ids=sorted(VALIDATION_TEMPLATES),
            max_candidates=MAX_BATCH_CANDIDATES,
            model_name=str(raw_model),
        )
        started = time.perf_counter()
        fallback = lambda: self._fallback_output(  # noqa: E731
            selected=selected,
            template_id=template_id,
            hints=hints,
            model_name=str(raw_model),
        )
        try:
            output = self._generate_planner(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                fallback_factory=fallback,
            )
        except ResearchValidationError:
            raise
        except Exception as exc:
            raise ResearchValidationError(
                f"实验批量规划调用失败：{exc}",
                status=502,
                code="planner_call_failed",
                retryable=True,
            ) from exc
        latency_ms = int(round((time.perf_counter() - started) * 1000))

        validated = self._validate_planner_output(
            output=output,
            selected_ids=[dispute.dispute_id for dispute in selected],
            project_id=current.project.research_id,
            template_hint=template_id,
            hints=hints,
            model_name=str(raw_model),
            expected_revision=current.revision,
        )
        agent_call = AgentCallRecord(
            call_id="CALL_" + uuid.uuid4().hex[:12].upper(),
            role="experiment_planner",
            model_id=model_id,
            request_hash=request_hash,
            attempt=1,
            status="succeeded",
            latency_ms=latency_ms,
            cache_hit=False,
        )

        def apply_planner(state: ResearchRevision) -> None:
            for dispute in state.disputes:
                if dispute.dispute_id in [d.dispute_id for d in selected]:
                    for candidate in state.candidates:
                        if (
                            any(
                                item in candidate.dispute_ids
                                for item in [d.dispute_id for d in selected]
                            )
                            and candidate.status in SUPERSEDED_CANDIDATE_STATUSES
                        ):
                            candidate.status = "superseded"
            state.candidates.extend(validated.candidates)
            state.protocols.extend(validated.protocols)
            state.agent_calls.append(agent_call)
            state.project.stage = "awaiting_approval"

        next_state = self.repository.mutate(
            research_id,
            expected_revision=current.revision,
            mutation=apply_planner,
        )
        message = (
            "批量规划成功（离线模板）"
            if output.is_stub
            else "批量规划成功"
        )
        return PlannerBatchResult(
            revision=next_state,
            batch_id=output.batch_id or ("BATCH_" + uuid.uuid4().hex[:12].upper()),
            is_stub=bool(output.is_stub),
            candidate_ids=[candidate.experiment_id for candidate in validated.candidates],
            message=message,
        )

    def _generate_planner(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        fallback_factory: Any,
    ) -> ExperimentPlannerOutput:
        if self.gateway is None:
            gateway = LLMGateway(client=None, allow_fallback=True)
        else:
            gateway = self.gateway
        return gateway.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=ExperimentPlannerOutput,
            fallback_factory=fallback_factory,
        )

    @staticmethod
    def _build_planner_prompt(
        *,
        project: dict[str, Any],
        disputes: list[Dispute],
        template_id: str,
        hints: dict[str, Any],
        allowed_template_ids: list[str],
        max_candidates: int,
        model_name: str,
    ) -> str:
        dispute_lines = []
        for dispute in disputes:
            dispute_lines.append(
                {
                    "dispute_id": dispute.dispute_id,
                    "question": dispute.question,
                    "issue_code": dispute.issue_code,
                    "claim_id": dispute.claim_id,
                    "scope": dispute.scope,
                    "opinion_ids": dispute.opinion_ids,
                    "evidence_ids": dispute.evidence_ids,
                    "validation_ids": dispute.validation_ids,
                }
            )
        return (
            f"研究项目：{project.get('title', '')}\n"
            f"科学问题：{project.get('question', '')}\n"
            f"目标变量：{project.get('target')}\n"
            f"预测提前量（天）：{project.get('horizon_days')}\n\n"
            f"待解决争议（JSON）：\n"
            f"{json.dumps(dispute_lines, ensure_ascii=False, indent=2)}\n\n"
            f"可用模板：{allowed_template_ids}\n"
            f"建议模板：{template_id or '由你选择'}\n"
            f"设计参数上限与模板白名单：{sorted(ALLOWED_DESIGN_KEYS)}\n"
            f"模型：{model_name}\n"
            f"设计提示（可合并，程序最终校验）：{json.dumps(hints, ensure_ascii=False)}\n\n"
            f"输出要求：一批最多 {max_candidates} 个候选，每个候选必须给出 experiment_id、"
            f"dispute_ids、template_id、design 和简短 rationale；design 只使用白名单键；"
            f"不输出 utility_score（由程序计算）。"
        )

    @staticmethod
    def _fallback_output(
        *,
        selected: list[Dispute],
        template_id: str,
        hints: dict[str, Any],
        model_name: str,
    ) -> dict[str, Any]:
        dispute_id = selected[0].dispute_id if selected else "DSP_UNKNOWN"
        template = template_id if template_id in VALIDATION_TEMPLATES else "paired_replay"
        design = dict(hints)
        defaults = {
            "control": ["By", "Bz"],
            "treatment": ["By", "Bz", "DeltaDec"],
            "forecast_horizon_days": 3,
            "past_lag_days": 2,
            "window_size": 4,
            "use_lag_feature": True,
            "max_lag_day": 2,
            "model": model_name,
        }
        for key, value in defaults.items():
            if key not in design:
                design[key] = value
        return {
            "batch_id": "BATCH_PROGRAM",
            "is_stub": True,
            "candidates": [
                {
                    "experiment_id": "VALID_STUB",
                    "dispute_ids": [dispute_id],
                    "hypothesis_ids": [],
                    "template_id": template,
                    "design": design,
                    "rationale": "离线模板：未调用模型，由程序生成最小验证候选。",
                }
            ],
        }

    # -------------------------------------------------------------- validation

    def _validate_planner_output(
        self,
        *,
        output: ExperimentPlannerOutput,
        selected_ids: list[str],
        project_id: str,
        template_hint: str,
        hints: dict[str, Any],
        model_name: str,
        expected_revision: int,
    ):
        candidates = list(output.candidates)
        if not candidates:
            raise ResearchValidationError(
                "批量规划未返回任何候选。",
                status=422,
                code="empty_planner_output",
            )
        if len(candidates) > MAX_BATCH_CANDIDATES:
            raise ResearchValidationError(
                f"批量规划候选数超过上限 {MAX_BATCH_CANDIDATES}。",
                status=422,
                code="candidate_limit_exceeded",
            )
        ids = [candidate.experiment_id for candidate in candidates]
        if len(ids) != len(set(ids)):
            raise ResearchValidationError(
                "批量规划候选 experiment_id 必须唯一。",
                status=422,
                code="duplicate_experiment_id",
            )
        sanitized_ids: list[str] = []
        for candidate in candidates:
            candidate.dispute_ids = [
                item for item in candidate.dispute_ids if item in selected_ids
            ]
            if not candidate.dispute_ids:
                raise ResearchValidationError(
                    f"候选 {candidate.experiment_id} 未关联任何待验证争议。",
                    status=422,
                    code="candidate_without_dispute",
                )
            template = candidate.template_id
            if template not in VALIDATION_TEMPLATES:
                raise ResearchValidationError(
                    f"不支持的实验模板：{template}",
                    status=422,
                    code="unsupported_template",
                )
            candidate.design = self._merge_and_validate_design(
                template=template,
                design=candidate.design,
                hints=hints,
            )
            sanitized = "".join(
                char if char.isalnum() else "_" for char in candidate.experiment_id
            )
            if not sanitized:
                sanitized = "VALID_%08X" % (uuid.uuid4().int & 0xFFFFFFFF)
            sanitized_ids.append(sanitized[:80])

        snapshot_list: list[ProtocolSnapshot] = []
        candidate_records: list[ValidationCandidate] = []
        for index, candidate in enumerate(candidates):
            experiment_id = sanitized_ids[index]
            protocol, snapshot = self._build_protocol(
                candidate=candidate,
                experiment_id=experiment_id,
                project_id=project_id,
                template=candidate.template_id,
                project_target=None,
            )
            estimates = self._estimate_candidate(
                template=candidate.template_id,
                design=candidate.design,
                dispute_count=len(candidate.dispute_ids),
                is_stub=bool(output.is_stub),
            )
            utility = self._utility_from_estimates(estimates)
            candidate_record = ValidationCandidate(
                experiment_id=experiment_id,
                dispute_ids=candidate.dispute_ids,
                hypothesis_ids=list(candidate.hypothesis_ids or []),
                template_id=candidate.template_id,
                round_id=1,
                design=dict(candidate.design),
                estimates=estimates,
                approval_required=True,
                status="pending_approval",
                protocol_hash=snapshot.protocol_hash,
                utility_score=utility,
                source="program" if output.is_stub else "llm",
                is_stub=bool(output.is_stub),
                rationale=candidate.rationale,
            )
            candidate_records.append(candidate_record)
            snapshot_list.append(snapshot)

        return _ValidatedBatch(
            candidates=candidate_records,
            protocols=snapshot_list,
        )

    def _merge_and_validate_design(
        self,
        *,
        template: str,
        design: dict[str, Any],
        hints: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(design)
        for key, value in (hints or {}).items():
            merged[key] = value
        unknown = sorted(set(merged) - ALLOWED_DESIGN_KEYS)
        if unknown:
            raise ResearchValidationError(
                f"模板 {template} 的设计包含未知参数：{', '.join(unknown)}",
                status=422,
                code="unsupported_design_key",
            )
        for key in ("control", "treatment"):
            value = merged.get(key, [])
            if not isinstance(value, (list, tuple)):
                raise ResearchValidationError(
                    f"design.{key} 必须是字符串数组。",
                    status=422,
                    code="invalid_design_value",
                )
            merged[key] = [str(item) for item in value]
        if not merged.get("control") or not merged.get("treatment"):
            raise ResearchValidationError(
                "设计必须同时给出对照组与增强组特征。",
                status=422,
                code="incomplete_design",
            )
        horizon = merged.get("forecast_horizon_days", 3)
        try:
            horizon = int(horizon)
        except (TypeError, ValueError):
            raise ResearchValidationError(
                "forecast_horizon_days 必须是整数。",
                status=422,
                code="invalid_design_value",
            )
        if not 1 <= horizon <= 30:
            raise ResearchValidationError(
                "forecast_horizon_days 必须在 1 到 30 之间。",
                status=422,
                code="invalid_design_value",
            )
        merged["forecast_horizon_days"] = horizon
        for key, minimum, maximum in (
            ("window_size", 1, 366),
            ("past_lag_days", 1, 366),
            ("max_lag_day", 0, 366),
        ):
            if key in merged:
                try:
                    value = int(merged[key])
                except (TypeError, ValueError):
                    raise ResearchValidationError(
                        f"{key} 必须是整数。",
                        status=422,
                        code="invalid_design_value",
                    )
                if not minimum <= value <= maximum:
                    raise ResearchValidationError(
                        f"{key} 必须在 {minimum} 到 {maximum} 之间。",
                        status=422,
                        code="invalid_design_value",
                    )
                merged[key] = value
        for key in ("alpha", "l1_ratio", "test_split_ratio", "random_state"):
            if key in merged and merged[key] is not None:
                try:
                    merged[key] = float(merged[key])
                except (TypeError, ValueError):
                    raise ResearchValidationError(
                        f"{key} 必须是数字。",
                        status=422,
                        code="invalid_design_value",
                    )
        if "test_split_ratio" in merged and not 0.0 < merged["test_split_ratio"] < 1.0:
            raise ResearchValidationError(
                "test_split_ratio 必须在 (0, 1) 区间。",
                status=422,
                code="invalid_design_value",
            )
        if self.data_source is not None:
            merged = self._resolve_feature_columns(merged)
        return merged

    def _resolve_feature_columns(self, design: dict[str, Any]) -> dict[str, Any]:
        bridge = ElasticNetProtocolBridge(self.data_source)
        try:
            control = bridge.validate_feature_columns(design["control"])
            treatment = bridge.validate_feature_columns(design["treatment"])
        except (KeyError, ValueError) as exc:
            raise ResearchValidationError(
                f"特征列无法在数据源中解析：{exc}",
                status=422,
                code="unresolved_feature_columns",
            ) from exc
        design["control"] = control
        design["treatment"] = treatment
        target = design.get("target")
        if target:
            resolved_target = bridge._resolve_column_name(str(target), bridge._available_columns())
            if resolved_target is None:
                raise ResearchValidationError(
                    f"目标列无法在数据源中解析：{target}",
                    status=422,
                    code="unresolved_target_column",
                )
            design["target"] = resolved_target
        return design

    def _build_protocol(
        self,
        *,
        candidate: PlannerCandidateOutput,
        experiment_id: str,
        project_id: str,
        template: str,
        project_target: Any,
    ):
        design = dict(candidate.design)
        target = design.get("target")
        if target is None and self.data_source is not None:
            target = self.data_source.target_column
        if target is None and project_target is not None:
            target = project_target.column or project_target.id
        if target is None:
            target = "target"
        model_name = str(design.get("model") or "Ridge").strip()
        if model_name.lower() in {"ridge", "elasticnet"}:
            method_id = METHOD_REGISTRY.resolve_method_id(model_name)
        else:
            method_id = METHOD_REGISTRY.resolve_method_id(None)
        model_params = {
            "window_size": int(design.get("window_size", 4)),
            "use_lag_feature": bool(design.get("use_lag_feature", True)),
            "max_lag_day": int(design.get("max_lag_day", 2)),
            "alpha": float(design.get("alpha", 1.0 if method_id == "ridge" else 0.1)),
            "l1_ratio": float(design.get("l1_ratio", 0.5)),
            "random_state": int(
                design.get(
                    "random_state",
                    self.data_source.random_state if self.data_source else 42,
                )
            ),
            "test_split_ratio": float(
                design.get(
                    "test_split_ratio",
                    self.data_source.test_split_ratio if self.data_source else 0.2,
                )
            ),
            "forecast_horizon_days": int(design.get("forecast_horizon_days", 3)),
            "past_lag_days": int(design.get("past_lag_days", 2)),
            "model_family": method_id,
        }
        protocol = ExperimentProtocol(
            experiment_id=experiment_id,
            round_id=1,
            task_id=project_id,
            target=str(target),
            features={
                "target": str(target),
                "control": list(design["control"]),
                "treatment": list(design["treatment"]),
                "lags": design.get("lags") or {},
            },
            model=ModelSpec(name=method_id, parameters=model_params),
            tested_hypotheses=list(candidate.hypothesis_ids or []),
            forecast_horizon_days=int(design.get("forecast_horizon_days", 3)),
            scientific_objective=candidate.rationale,
            notes=[
                f"template:{template}",
                "research_review_validation",
                *[str(item) for item in (design.get("notes") or [])],
            ],
            source_candidate_id=experiment_id,
            requires_human_review=True,
        )
        snapshot = self._build_snapshot(
            protocol=protocol,
            candidate=candidate,
            template=template,
            method=method_id,
        )
        return protocol, snapshot

    def _build_snapshot(
        self,
        *,
        protocol: ExperimentProtocol,
        candidate: PlannerCandidateOutput,
        template: str,
        method: str,
    ) -> ProtocolSnapshot:
        sample_plan: dict[str, Any] = {
            "template": template,
            "target": protocol.target,
            "feature_arms": {
                "control": list(protocol.features.control),
                "treatment": list(protocol.features.treatment),
            },
            "forecast_horizon_days": protocol.forecast_horizon_days,
            "past_lag_days": protocol.model.parameters.get("past_lag_days"),
            "window_size": protocol.model.parameters.get("window_size"),
            "sample_condition": candidate.design.get("sample_condition") or "complete_case",
            "checks": [
                "template_in_whitelist",
                "design_keys_whitelisted",
                "dispute_bound",
                "protocol_hash_bound",
            ],
        }
        if template == "calendar_alignment" and self.data_source is not None:
            try:
                built_plan = self._build_calendar_plan(protocol, candidate.design)
                sample_plan["sample_plan"] = built_plan
            except ResearchValidationError:
                raise
            except Exception as exc:
                raise ResearchValidationError(
                    f"日历样本计划构建失败：{exc}",
                    status=422,
                    code="sample_plan_failed",
                ) from exc
        proto_hash = protocol_hash(
            {
                "protocol_version": PROTOCOL_VERSION,
                "template": template,
                "experiment_id": protocol.experiment_id,
                "dispute_ids": sorted(candidate.dispute_ids),
                "hypothesis_ids": sorted(candidate.hypothesis_ids or []),
                "round_id": protocol.round_id,
                "method": method,
                "target": protocol.target,
                "features": protocol.features.model_dump(),
                "model": protocol.model.model_dump(),
            }
        )
        return ProtocolSnapshot(
            protocol_id="PRT_" + uuid.uuid4().hex[:12].upper(),
            version=1,
            protocol_hash=proto_hash,
            target=ResearchTarget(id=protocol.target, column=protocol.target),
            time_spec=ProtocolTimeSpec(
                calendar_offsets=True,
                availability_assumption="日历偏移，非记录条数偏移",
            ),
            sample_plan=sample_plan,
            feature_arms=[
                {
                    "label": "control",
                    "features": list(protocol.features.control),
                },
                {
                    "label": "treatment",
                    "features": list(protocol.features.treatment),
                },
            ],
            split_plan={
                "test_split_ratio": protocol.model.parameters.get("test_split_ratio"),
                "random_state": protocol.model.parameters.get("random_state"),
            },
            method=method,
            checks=list(sample_plan["checks"]),
        )

    def _build_calendar_plan(
        self,
        protocol: ExperimentProtocol,
        design: dict[str, Any],
    ) -> dict[str, Any]:
        if self.data_source is None:
            raise ResearchValidationError(
                "日历样本规划需要数据源配置。",
                status=409,
                code="data_source_required",
            )
        bridge = ElasticNetProtocolBridge(self.data_source)
        runner = METHOD_REGISTRY.runner_for(protocol.model.name)
        runner_config = bridge.build_runner_config(
            protocol,
            protocol.features.control,
        )
        merged = runner.load_and_merge_data(runner_config)
        window_size = int(protocol.model.parameters.get("window_size", 4))
        past_lag_days = int(protocol.model.parameters.get("past_lag_days", 2))
        horizon = int(protocol.model.parameters.get("forecast_horizon_days", 3))
        plan = build_sample_plan(
            merged.X,
            merged.y,
            merged.time,
            feature_names=protocol.features.control,
            target_name=str(protocol.target),
            window_size=window_size,
            past_lag_days=past_lag_days,
            forecast_horizon_days=horizon,
        )
        return plan.to_dict()

    # -------------------------------------------------------------- estimation

    @staticmethod
    def _estimate_candidate(
        *,
        template: str,
        design: dict[str, Any],
        dispute_count: int,
        is_stub: bool,
    ) -> list[ValidationEstimate]:
        if template == "calendar_alignment":
            ig = 0.50
            pg = 0.30
            risk = 0.10
            cost = 0.05
        else:
            ig = 0.65
            pg = 0.40
            risk = 0.25
            cost = 0.20
        if is_stub:
            ig = round(ig * 0.5, 4)
            pg = round(pg * 0.5, 4)
            risk = min(0.6, round(risk + 0.15, 4))
        estimates = [
            ValidationEstimate(
                component="IG",
                value=ig,
                unit="0-1",
                calculation_version="program/1.0",
                basis="区分竞争机制的信息能力估计",
            ),
            ValidationEstimate(
                component="PG",
                value=pg,
                unit="0-1",
                calculation_version="program/1.0",
                basis="预报性能提升期望估计",
            ),
            ValidationEstimate(
                component="Risk",
                value=risk,
                unit="0-1",
                calculation_version="program/1.0",
                basis="数据质量、过拟合与执行风险估计",
            ),
            ValidationEstimate(
                component="Cost",
                value=cost,
                unit="0-1",
                calculation_version="program/1.0",
                basis="模型调用与计算成本估计",
            ),
        ]
        return estimates

    @staticmethod
    def _utility_from_estimates(estimates: list[ValidationEstimate]) -> float:
        values = {
            item.component: float(item.value or 0.0) for item in estimates
        }
        raw = (
            UTILITY_WEIGHTS["ig"] * values.get("IG", 0.0)
            + UTILITY_WEIGHTS["pg"] * values.get("PG", 0.0)
            - UTILITY_WEIGHTS["risk"] * values.get("Risk", 0.0)
            - UTILITY_WEIGHTS["cost"] * values.get("Cost", 0.0)
        )
        return max(-1.0, min(1.0, round(raw, 6)))

    # ---------------------------------------------------------------- approval

    def approve(
        self,
        research_id: str,
        expected_revision: int,
        payload: dict[str, Any],
    ) -> ResearchRevision:
        current = self._load(research_id)
        self._require_revision(current, expected_revision)
        protocol_hash_value = str(payload.get("protocol_hash") or "").strip()
        candidate_id = str(payload.get("candidate_id") or "").strip()
        if not protocol_hash_value:
            raise ResearchValidationError(
                "审批必须携带 protocol_hash。",
                status=422,
                code="protocol_hash_required",
            )
        candidate = next(
            (
                item
                for item in current.candidates
                if item.protocol_hash == protocol_hash_value
                and (
                    not candidate_id or item.experiment_id == candidate_id
                )
                and item.status
                not in EXECUTED_CANDIDATE_STATUSES | {
                    "superseded",
                }
            ),
            None,
        )
        if candidate is None:
            raise ResearchValidationError(
                "没有可批准且哈希匹配的候选，参数修改后旧审批无效，请重新规划。",
                status=409,
                code="approval_hash_mismatch",
            )
        if candidate.status != "pending_approval":
            raise ResearchValidationError(
                f"候选当前状态 {candidate.status} 不可批准。",
                status=409,
                code="candidate_not_approvable",
            )
        decision = ResearchDecision(
            decision_id="DEC_" + uuid.uuid4().hex[:12].upper(),
            actor_type=str(payload.get("actor_type") or "human_pi"),
            action="approve_protocol",
            rationale=str(payload.get("rationale") or "") or None,
            protocol_hash=protocol_hash_value,
            next_stage="executing",
        )

        def apply_approval(state: ResearchRevision) -> None:
            for item in state.candidates:
                if item.experiment_id == candidate.experiment_id:
                    item.status = "approved"
            state.decisions.append(decision)
            state.project.stage = "executing"

        return self.repository.mutate(
            research_id,
            expected_revision=current.revision,
            mutation=apply_approval,
        )

    # --------------------------------------------------------------- execution

    def execute(
        self,
        research_id: str,
        expected_revision: int,
        payload: dict[str, Any],
    ) -> ResearchRevision:
        current = self._load(research_id)
        self._require_revision(current, expected_revision)
        protocol_hash_value = str(payload.get("protocol_hash") or "").strip()
        candidate_id = str(payload.get("candidate_id") or "").strip()
        if not protocol_hash_value:
            raise ResearchValidationError(
                "执行必须携带 protocol_hash。",
                status=422,
                code="protocol_hash_required",
            )
        candidate = next(
            (
                item
                for item in current.candidates
                if item.protocol_hash == protocol_hash_value
                and (not candidate_id or item.experiment_id == candidate_id)
                and item.status != "superseded"
            ),
            None,
        )
        if candidate is None:
            raise ResearchValidationError(
                "执行失败：没有匹配且未失效的候选协议，请先规划并审批。",
                status=409,
                code="protocol_unknown_or_invalidated",
            )
        if candidate.status != "approved":
            raise ResearchValidationError(
                "执行前必须有有效审批。",
                status=409,
                code="approval_required",
            )
        if self.data_source is None:
            raise ResearchValidationError(
                "当前部署未配置实验数据源，无法执行验证。",
                status=409,
                code="data_source_required",
            )

        protocol, _snapshot = self._build_protocol(
            candidate=PlannerCandidateOutput(
                experiment_id=candidate.experiment_id,
                dispute_ids=candidate.dispute_ids,
                hypothesis_ids=candidate.hypothesis_ids,
                template_id=candidate.template_id,
                design=candidate.design,
                rationale=candidate.rationale,
            ),
            experiment_id=candidate.experiment_id,
            project_id=research_id,
            template=candidate.template_id,
            project_target=current.project.target,
        )

        try:
            if candidate.template_id == "calendar_alignment":
                sample_plan = self._build_calendar_plan(protocol, candidate.design)
                result_payload: dict[str, Any] = {
                    "status": "completed",
                    "template": candidate.template_id,
                    "protocol_hash": protocol_hash_value,
                    "sample_plan": sample_plan,
                }
                result_ref = (
                    f"{protocol_hash_value[:16]}:{candidate.experiment_id}:sample_plan"
                )
            else:
                harness = self._harness()
                result = harness.run(protocol)
                result_payload = result.model_dump(exclude_none=True)
                result_ref = (
                    f"{protocol_hash_value[:16]}:{candidate.experiment_id}:"
                    f"{result.status}"
                )
        except Exception as exc:
            failure = self._record_failure(
                research_id=research_id,
                expected_revision=current.revision,
                candidate=candidate,
                protocol_hash_value=protocol_hash_value,
                message=f"验证执行失败：{exc}",
            )
            raise ResearchValidationError(
                f"验证执行失败：{exc}",
                status=422,
                code="execution_failed",
                retryable=True,
            ) from exc

        job = JobRecord(
            job_id="JOB_" + uuid.uuid4().hex[:12].upper(),
            kind="validation_execution",
            status="completed",
            stage="completed",
            research_id=research_id,
            request_hash=ResearchRepository.request_hash(
                {
                    "action": "execute_protocol",
                    "research_id": research_id,
                    "expected_revision": current.revision,
                    "protocol_hash": protocol_hash_value,
                }
            ),
            review_id=candidate.dispute_ids[0],
            result_ref=result_ref,
        )
        decision = ResearchDecision(
            decision_id="DEC_" + uuid.uuid4().hex[:12].upper(),
            actor_type="program",
            action="execute_protocol",
            rationale="程序完成受控验证执行",
            protocol_hash=protocol_hash_value,
            next_stage="evidence_ready",
        )

        def apply_execution(state: ResearchRevision) -> None:
            for item in state.candidates:
                if item.experiment_id == candidate.experiment_id:
                    item.status = "executed"
                    execution_marker = ValidationEstimate(
                        component="EXECUTION",
                        value=1.0 if result_payload.get("status") == "completed" else 0.0,
                        calculation_version="program/1.0",
                        basis="受控验证完成标记",
                    )
                    preserved = [
                        estimate
                        for estimate in item.estimates
                        if estimate.component != "EXECUTION"
                    ]
                    item.estimates = [*preserved, execution_marker]
            state.jobs.append(job)
            state.decisions.append(decision)
            state.agent_calls.append(
                AgentCallRecord(
                    call_id="CALL_" + uuid.uuid4().hex[:12].upper(),
                    role="validation_execution",
                    model_id="program",
                    request_hash=job.request_hash,
                    attempt=1,
                    status="succeeded",
                    latency_ms=0,
                    cache_hit=False,
                )
            )
            for dispute in state.disputes:
                if (
                    dispute.dispute_id in candidate.dispute_ids
                    and candidate.experiment_id not in dispute.validation_ids
                ):
                    dispute.validation_ids.append(candidate.experiment_id)
            state.project.stage = "evidence_ready"

        return self.repository.mutate(
            research_id,
            expected_revision=current.revision,
            mutation=apply_execution,
        )

    def _record_failure(
        self,
        *,
        research_id: str,
        expected_revision: int,
        candidate: ValidationCandidate,
        protocol_hash_value: str,
        message: str,
    ) -> None:
        job = JobRecord(
            job_id="JOB_" + uuid.uuid4().hex[:12].upper(),
            kind="validation_execution",
            status="failed",
            stage="failed",
            research_id=research_id,
            request_hash=ResearchRepository.request_hash(
                {
                    "action": "execute_protocol",
                    "research_id": research_id,
                    "expected_revision": expected_revision,
                    "protocol_hash": protocol_hash_value,
                }
            ),
            review_id=candidate.dispute_ids[0],
            error_code="execution_failed",
            message=message,
        )

        def apply_failure(state: ResearchRevision) -> None:
            for item in state.candidates:
                if item.experiment_id == candidate.experiment_id:
                    item.status = "failed"
            state.jobs.append(job)
            state.project.stage = "failed"

        self.repository.mutate(
            research_id,
            expected_revision=expected_revision,
            mutation=apply_failure,
        )
        return None

    def _harness(self):
        from core.harness_unified import UnifiedExperimentHarness

        return UnifiedExperimentHarness(
            data_source=self.data_source,
            project_root=self.harness_project_root,
            run_shap=self.run_shap,
        )

    # ------------------------------------------------------------------ helpers

    def _load(self, research_id: str) -> ResearchRevision:
        try:
            return self.repository.load_revision(research_id)
        except Exception as exc:
            raise ResearchValidationError(
                "研究项目不存在。",
                status=404,
                code="project_not_found",
            ) from exc

    @staticmethod
    def _require_revision(current: ResearchRevision, expected_revision: Any) -> None:
        try:
            expected = int(expected_revision)
        except (TypeError, ValueError):
            raise ResearchValidationError(
                "expected_revision 必须是整数。",
                status=422,
                code="invalid_revision",
            )
        if current.revision != expected:
            raise ResearchValidationError(
                f"版本冲突：当前 revision={current.revision}。",
                status=409,
                code="revision_conflict",
                retryable=True,
            )

@dataclass
class _ValidatedBatch:
    candidates: list[ValidationCandidate] = field(default_factory=list)
    protocols: list[ProtocolSnapshot] = field(default_factory=list)


__all__ = [
    "ALLOWED_DESIGN_KEYS",
    "ExperimentPlannerOutput",
    "MAX_BATCH_CANDIDATES",
    "PlannerBatchResult",
    "PlannerCandidateOutput",
    "ResearchValidationError",
    "VALIDATION_TEMPLATES",
    "ValidationPlanner",
]
