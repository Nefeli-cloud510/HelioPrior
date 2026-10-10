from __future__ import annotations

import hashlib
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from core.llm_gateway import LLMGateway
from core.research_repository import ResearchRepository
from core.research_schema import (
    SCHEMA_VERSION,
    AgentCallRecord,
    ArbitrationRecord,
    ClaimVersion,
    Dispute,
    EvidencePacket,
    JobRecord,
    ResearchProject,
    ResearchRevision,
    ReviewOpinion,
)
from core.runtime_config import get_llm_model_for_role


REVIEW_ROLES = ("scientific_questioner", "scientific_interpreter")
MAX_IN_FLIGHT = 2
MAX_RETRIES_PER_ROLE = 1


class ResearchReviewError(RuntimeError):
    """Actionable error raised inside the review orchestration.

    The HTTP layer maps it to a 4xx response; it never represents a
    fabricated scientific outcome.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str,
        status: int = 422,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status
        self.details = details or {}


class ReviewObservation(BaseModel):
    """One structured finding. Values of note are program facts, not LLM
    re-typed numbers."""

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    observation_id: str = Field(min_length=1)
    issue_code: str = Field(min_length=1)
    text: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)
    metric_refs: list[str] = Field(default_factory=list)
    consequence: str | None = None
    validation_template_id: str | None = None


class ReviewOpinionOutput(BaseModel):
    """Structured review opinion produced by one independent role."""

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    claim_id: str = Field(min_length=1)
    packet_hash: str = Field(min_length=1)
    stance: str = Field(default="unknown")
    observations: list[ReviewObservation] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    metric_refs: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    validation_proposals: list[str] = Field(default_factory=list)
    scope_notes: list[str] = Field(default_factory=list)
    is_stub: bool = False


class ArbitrationOutput(BaseModel):
    """Conditional central arbitrator recommendation."""

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    issue_summary: str = Field(min_length=1)
    recommended_action: str = Field(min_length=1)
    required_validation: str | None = None
    is_stub: bool = False


@dataclass(frozen=True)
class RoleAttemptResult:
    role: str
    attempt: int
    opinion: ReviewOpinion | None
    call: AgentCallRecord


@dataclass(frozen=True)
class DualReviewResult:
    status: str
    review_id: str
    packet_hash: str
    claim_id: str | None
    request_hash: str
    job: JobRecord
    opinions: list[ReviewOpinion] = field(default_factory=list)
    agent_calls: list[AgentCallRecord] = field(default_factory=list)
    disputes: list[Dispute] = field(default_factory=list)
    arbitration: ArbitrationOutput | None = None
    arbitration_call: AgentCallRecord | None = None
    message: str | None = None
    is_stub: bool = False


QUESTIONER_SYSTEM_PROMPT = (
    "你是科研复核协作中的科学质询者。你的职责是独立检查给定结论在指定证据包范围内是否得到足够支持。"
    "优先阅读程序检查结果；指出同样本、时间边界、基线、覆盖范围、稳健性与可重复性问题。"
    "每条意见必须绑定证据引用；没有证据时不得断言存在错误。"
    "不要读取或猜测另一名审读者的意见。数值引用只能使用证据包中已有的 metric_ref，不得自行填写指标值。"
)

INTERPRETER_SYSTEM_PROMPT = (
    "你是科研复核协作中的结果解释者。你的职责是独立归纳程序结果支持的结论范围。"
    "区分指标改善、跨条件稳定性和物理解释；为每条解释绑定证据。"
    "存在替代解释时给出可区分实验；对不显著、混合或负向结果如实描述。"
    "不要读取或猜测另一名审读者的意见。不得把已有论文成绩或其他项目的数值写成本轮结果。"
)

ARBITRATOR_SYSTEM_PROMPT = (
    "你是中央协调者，只在两路独立审读产生可执行冲突时被调用。"
    "你的任务不是重新审读证据，也不是投票：请归纳冲突焦点，给出一次可执行的澄清或最小补充验证建议。"
    "最多提供一条建议；不需要生成长解释。"
)


def review_request_hash(
    *,
    packet_hash: str,
    claim_id: str | None,
    claim_version: int | None,
    project_stage: str,
    frozen_revision: int,
    retry_roles: tuple[str, ...] = (),
) -> str:
    """Stable request hash for a review run.

    Project isolation and protocol version are already captured by the
    repository revision; the packet hash plus claim version prevent a stale
    opinion from being committed after evidence or claim edits.
    """

    payload = {
        "schema_version": SCHEMA_VERSION,
        "packet_hash": packet_hash,
        "claim_id": claim_id,
        "claim_version": claim_version,
        "project_stage": project_stage,
        "frozen_revision": frozen_revision,
        "retry_roles": sorted(retry_roles),
    }
    return hashlib.sha256(ResearchRepository.canonical_payload(payload).encode("utf-8")).hexdigest()


def validate_opinion_refs(
    output: ReviewOpinionOutput,
    packet: EvidencePacket,
    *,
    claim_id: str,
) -> None:
    """Reject LLM output whose references cannot be found in the packet.

    The packet itself is the whitelist: evidence_ids from ``evidence_items``
    plus metric dict keys for ``metric_refs``.
    """

    if output.claim_id != claim_id:
        raise ResearchReviewError(
            f"审读意见指向了错误的结论：{output.claim_id}",
            code="review_claim_mismatch",
        )
    if output.packet_hash != packet.packet_hash:
        raise ResearchReviewError(
            "审读意见的证据包哈希与当前证据包不一致。",
            code="review_packet_mismatch",
        )

    allowed_evidence = {item.evidence_id for item in packet.evidence_items}
    allowed_metrics = set(packet.metrics)
    bad_evidence: list[str] = []
    bad_metrics: list[str] = []

    def check_refs(
        evidence_refs: list[str],
        metric_refs: list[str],
    ) -> None:
        bad_evidence.extend(item for item in evidence_refs if item not in allowed_evidence)
        bad_metrics.extend(item for item in metric_refs if item not in allowed_metrics)

    check_refs(output.evidence_refs, output.metric_refs)
    for observation in output.observations:
        check_refs(observation.evidence_refs, observation.metric_refs)

    if bad_evidence or bad_metrics:
        raise ResearchReviewError(
            "审读意见包含不在证据包白名单内的引用。",
            code="invalid_review_refs",
            details={
                "evidence_refs": sorted(set(bad_evidence)),
                "metric_refs": sorted(set(bad_metrics)),
                "allowed_evidence": sorted(allowed_evidence),
                "allowed_metrics": sorted(allowed_metrics),
            },
        )


def _normalize_stance(value: str) -> str:
    lowered = str(value or "unknown").strip().lower()
    if lowered in {"support", "challenge", "mixed", "unknown"}:
        return lowered
    return "unknown"


def _fallback_factory(
    role: str,
    *,
    claim_id: str,
    packet_hash: str,
) -> Callable[[], dict[str, Any]]:
    def factory() -> dict[str, Any]:
        observation_id = "OBS_OFFLINE_QUESTIONER" if role == "scientific_questioner" else "OBS_OFFLINE_INTERPRETER"
        proposal = (
            "补充相同样本上的 paired_replay 离线复核"
            if role == "scientific_questioner"
            else "仅补充人工校准记录，不扩大结论范围"
        )
        return {
            "claim_id": claim_id,
            "packet_hash": packet_hash,
            "stance": "unknown",
            "observations": [
                {
                    "observation_id": observation_id,
                    "issue_code": "OFFLINE_STUB",
                    "text": "未调用真实模型：本意见是离线占位，不能作为科学结论。",
                    "evidence_refs": [],
                    "metric_refs": [],
                    "consequence": "当前结果未经过真实模型审读。",
                }
            ],
            "evidence_refs": [],
            "metric_refs": [],
            "missing_evidence": ["真实 LLM 审读结果"],
            "validation_proposals": [proposal],
            "scope_notes": ["占位意见，仅用于接口联调。"],
            "is_stub": True,
        }

    return factory


def _role_prompt(role: str, packet: EvidencePacket, claim: ClaimVersion | None) -> str:
    checks = [
        {
            "code": check.code,
            "status": check.status,
            "detail": check.detail,
            "blocking": check.blocking,
        }
        for check in packet.checks
    ]
    metrics = {
        key: {
            "value": metric.value,
            "unit": metric.unit,
            "reason": metric.reason,
        }
        for key, metric in packet.metrics.items()
    }
    evidence_items = [
        {
            "evidence_id": item.evidence_id,
            "value_type": item.value_type,
            "value": item.value,
            "unit": item.unit,
            "provenance": item.provenance,
        }
        for item in packet.evidence_items
    ]
    prompt = {
        "task": "请对以下证据包给出独立审读意见",
        "role": role,
        "claim": {
            "claim_id": claim.claim_id if claim else None,
            "version": claim.version if claim else None,
            "statement": claim.statement if claim else None,
            "scope": claim.scope.model_dump(exclude_none=True) if claim else None,
        },
        "packet": {
            "packet_id": packet.packet_id,
            "packet_hash": packet.packet_hash,
            "run_id": packet.run_id,
            "protocol_hash": packet.protocol_hash,
            "claim_ids": packet.claim_ids,
            "checks": checks,
            "metrics": metrics,
            "evidence_items": evidence_items,
            "limitations": packet.limitations,
        },
        "output_requirements": [
            "observations[].evidence_refs 必须是证据包 evidence_items 中的 evidence_id",
            "observations[].metric_refs 必须是证据包 metrics 的键",
            "evidence 相关数字只能来自程序结果，不得自行填写数值",
            "validation_proposals 必须是可以执行的对照或验证动作",
            "没有证据时 use stance=unknown，不虚构错误",
        ],
    }
    if role == "scientific_questioner":
        prompt["focus"] = [
            "同样本对照与样本配对",
            "时间边界、日历提前量与数据版本",
            "基线、覆盖范围、稳健性与可重复性",
            "程序检查为 fail/unknown 时的后果与最小补充验证",
        ]
    else:
        prompt["focus"] = [
            "归纳程序结果支持的结论范围",
            "区分指标改善、跨条件稳定性和物理解释",
            "替代解释及其可区分实验",
            "混合、负向或不显著结果的如实表述",
        ]
    return json.dumps(prompt, ensure_ascii=False, indent=2, default=str)


def _to_opinion(
    output: ReviewOpinionOutput,
    *,
    review_id: str,
    role: str,
    claim_id: str | None,
    packet_hash: str,
    now: datetime,
) -> ReviewOpinion:
    observations: list[str] = []
    for observation in output.observations:
        evidence_refs = ",".join(observation.evidence_refs or [])
        metric_refs = ",".join(observation.metric_refs or [])
        parts = [observation.text]
        if observation.consequence:
            parts.append(f"后果：{observation.consequence}")
        if evidence_refs:
            parts.append(f"证据引用：{evidence_refs}")
        if metric_refs:
            parts.append(f"指标引用：{metric_refs}")
        if observation.validation_template_id:
            parts.append(f"验证模板：{observation.validation_template_id}")
        observations.append(" | ".join(parts))
    return ReviewOpinion(
        opinion_id=f"OPN_{review_id[-14:]}_{_role_label(role)}",
        review_id=review_id,
        role=role,
        packet_hash=packet_hash,
        claim_id=claim_id,
        stance=_normalize_stance(output.stance),
        observations=observations,
        evidence_refs=list(dict.fromkeys(output.evidence_refs)),
        metric_refs=list(dict.fromkeys(output.metric_refs)),
        missing_evidence=list(dict.fromkeys(output.missing_evidence)),
        validation_proposals=list(dict.fromkeys(output.validation_proposals)),
        status="completed",
        is_stub=bool(output.is_stub),
        created_at=now,
        updated_at=now,
        schema_version=SCHEMA_VERSION,
        source_kind="review_orchestrator",
    )


def make_disputes(
    opinions: list[ReviewOpinion],
    *,
    review_id: str,
    claim_id: str | None,
    packet: EvidencePacket,
    now: datetime,
) -> list[Dispute]:
    """Program-side conflict detection, run before any arbitration call."""

    completed = [opinion for opinion in opinions if opinion.status == "completed"]
    if len(completed) < 2:
        return []

    questioner = next(
        (opinion for opinion in completed if opinion.role == "scientific_questioner"),
        None,
    )
    interpreter = next(
        (opinion for opinion in completed if opinion.role == "scientific_interpreter"),
        None,
    )
    if questioner is None or interpreter is None:
        return []

    conflicts: list[tuple[str, str, list[str], list[str], str]] = []
    stance_pairs = {
        ("support", "challenge"),
        ("challenge", "support"),
        ("support", "mixed"),
        ("mixed", "support"),
    }
    pair = (questioner.stance, interpreter.stance)
    if pair in stance_pairs:
        conflicts.append(
            (
                "STANCE_CONFLICT",
                "两路独立审读对同一证据包的总体判断直接相反。",
                questioner.opinion_id,
                interpreter.opinion_id,
                "需要在相同证据上安排可执行验证以区分两路判断。",
            )
        )

    q_proposals = {_strip_lower(item) for item in questioner.validation_proposals}
    i_proposals = {_strip_lower(item) for item in interpreter.validation_proposals}
    contradictory = _contradictory_pairs(q_proposals, i_proposals)
    if contradictory:
        conflicts.append(
            (
                "VALIDATION_CONFLICT",
                f"两路审读提出的后续验证动作互斥：{'; '.join(sorted(contradictory))}。",
                questioner.opinion_id,
                interpreter.opinion_id,
                "需要由人工或被批准的验证实验确认哪一个动作满足最小验证条件。",
            )
        )

    disputes: list[Dispute] = []
    for index, (issue_code, question, first_id, second_id, consequence) in enumerate(conflicts, start=1):
        dispute = Dispute(
            dispute_id=f"DSP_{review_id[-14:]}_{index}",
            review_id=review_id,
            claim_id=claim_id,
            question=question,
            issue_code=issue_code,
            scope={
                "packet_id": packet.packet_id,
                "packet_hash": packet.packet_hash,
                "consequence": consequence,
            },
            opinion_ids=[first_id, second_id],
            evidence_ids=[packet.packet_id],
            status="open",
            created_at=now,
            updated_at=now,
        )
        disputes.append(dispute)
    return disputes


def _strip_lower(value: str) -> str:
    return str(value or "").strip().lower()


def _contradictory_pairs(left: set[str], right: set[str]) -> set[str]:
    if not left or not right:
        return set()
    pairs: set[str] = set()
    for a in left:
        for b in right:
            if a == b:
                continue
            if "paired_replay" in (a + b) and any(
                marker in (a + b) for marker in ("同一样本", "相同样本", "更短窗口", "更长窗口")
            ):
                pairs.update({a, b})
    return pairs


def arbitration_prompt(
    packet: EvidencePacket,
    claim_id: str | None,
    opinions: list[ReviewOpinion],
    disputes: list[Dispute],
) -> str:
    payload = {
        "instruction": "请基于以下互斥冲突给出一次最小澄清或补充验证建议，不要重新审读全部证据。",
        "packet": {"packet_id": packet.packet_id, "packet_hash": packet.packet_hash},
        "claim_id": claim_id,
        "opinions": [
            {
                "role": opinion.role,
                "stance": opinion.stance,
                "observations": opinion.observations,
                "validation_proposals": opinion.validation_proposals,
            }
            for opinion in opinions
        ],
        "disputes": [
            {
                "dispute_id": dispute.dispute_id,
                "issue_code": dispute.issue_code,
                "question": dispute.question,
                "scope": dispute.scope,
            }
            for dispute in disputes
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _arbitration_fallback(claim_id: str | None, packet_hash: str) -> dict[str, Any]:
    return {
        "issue_summary": "未调用真实模型：仲裁意见为离线占位，不能替代人工决定。",
        "recommended_action": "保持分歧可见，等待人工决定或批准一项最小验证实验。",
        "required_validation": "在相同证据包上登记一条可追溯的人工决定。",
        "is_stub": True,
    }


def _call_id(role: str) -> str:
    return f"CAL_{uuid.uuid4().hex[:10].upper()}_{_role_label(role)}"


_ROLE_SHORT_CODES = {
    "scientific_questioner": "Q",
    "scientific_interpreter": "I",
    "central_arbitrator": "A",
}


def _role_label(role: str) -> str:
    return _ROLE_SHORT_CODES.get(role, role[:2].upper())


def _normalize_review_roles(roles: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(roles or REVIEW_ROLES))
    invalid = [role for role in normalized if role not in REVIEW_ROLES]
    if invalid:
        raise ResearchReviewError(
            f"未知审读角色：{', '.join(invalid)}",
            code="invalid_review_roles",
        )
    return normalized


class ReviewOrchestrator:
    """Runs the two independent review paths, conditional arbitration and
    program-side dispute detection. Numeric checks and state transitions stay
    in program code; the LLM only proposes structured opinions."""

    def __init__(
        self,
        *,
        gateway: LLMGateway | None = None,
        max_in_flight: int = MAX_IN_FLIGHT,
        allow_stub: bool = True,
    ) -> None:
        self.gateway = gateway or LLMGateway()
        self.max_in_flight = max(1, int(max_in_flight))
        self.semaphore = threading.BoundedSemaphore(self.max_in_flight)
        self.allow_stub = bool(allow_stub)

    def run_dual_review(
        self,
        *,
        packet: EvidencePacket,
        claim: ClaimVersion | None,
        project: ResearchProject,
        frozen_revision: int,
        retry_roles: tuple[str, ...] = (),
        roles: tuple[str, ...] = REVIEW_ROLES,
        existing_opinions: list[ReviewOpinion] | None = None,
        review_id: str | None = None,
        request_hash: str | None = None,
        attempt: int = 1,
    ) -> DualReviewResult:
        """Generate both first-round opinions independently with bounded
        concurrency, then retry failed roles once and arbitrate only when a
        substantive conflict exists.

        When ``existing_opinions`` is passed this is a retry of an already
        started review: only roles still missing an opinion are called again,
        the same ``review_id``/``request_hash`` are reused, and the calls carry
        ``attempt`` (2).
        """

        now = datetime.now()
        role_list = _normalize_review_roles(roles)
        existing = list(existing_opinions or [])
        existing_by_role = {opinion.role: opinion for opinion in existing}
        is_retry = bool(existing)

        review_id = review_id or ("RVR_" + uuid.uuid4().hex[:12].upper())
        claim_id = claim.claim_id if claim else None
        if request_hash is None:
            request_hash = review_request_hash(
                packet_hash=packet.packet_hash,
                claim_id=claim_id,
                claim_version=claim.version if claim else None,
                project_stage=project.stage,
                frozen_revision=frozen_revision,
                retry_roles=retry_roles,
            )
        job = JobRecord(
            job_id=f"JOB_{review_id[-20:]}",
            kind="dual_review",
            status="running",
            stage="retry_running" if is_retry else "running",
            research_id=project.research_id,
            request_hash=request_hash,
            review_id=review_id,
            created_at=now,
            updated_at=now,
        )

        opinions: list[ReviewOpinion] = []
        calls: list[AgentCallRecord] = []
        failed_roles: list[str] = []

        if is_retry:
            missing = [role for role in role_list if role not in existing_by_role]
            if not missing:
                raise ResearchReviewError(
                    "该复核已完成，没有失败或缺失的审读角色。",
                    code="review_not_retryable",
                    status=409,
                )
            selected = [role for role in missing if not retry_roles or role in retry_roles]
            if not selected:
                raise ResearchReviewError(
                    "指定角色均已完成，没有可重试的失败路径。",
                    code="review_not_retryable",
                    status=409,
                )
            retried = self._generate_roles(
                tuple(selected),
                packet=packet,
                claim=claim,
                review_id=review_id,
                request_hash=request_hash,
                attempt=max(1, int(attempt)),
            )
            for role in role_list:
                existing_opinion = existing_by_role.get(role)
                if existing_opinion is not None:
                    opinions.append(existing_opinion)
                    continue
                role_result = retried.get(role)
                if role_result is None:
                    failed_roles.append(role)
                else:
                    calls.append(role_result.call)
                    if role_result.opinion is not None:
                        opinions.append(role_result.opinion)
                    else:
                        failed_roles.append(role)
        else:
            first_round = self._generate_roles(
                role_list,
                packet=packet,
                claim=claim,
                review_id=review_id,
                request_hash=request_hash,
                attempt=1,
            )
            for role in role_list:
                first = first_round[role]
                calls.append(first.call)
                if first.opinion is not None:
                    opinions.append(first.opinion)
                    continue
                if not retry_roles or role in retry_roles:
                    retried = self._generate_role(
                        role,
                        packet=packet,
                        claim=claim,
                        review_id=review_id,
                        request_hash=request_hash,
                        attempt=first.attempt + 1,
                    )
                    calls.append(retried.call)
                    if retried.opinion is not None:
                        opinions.append(retried.opinion)
                        continue
                failed_roles.append(role)

        is_stub = any(opinion.is_stub for opinion in opinions)
        if is_stub and not self.allow_stub:
            raise ResearchReviewError(
                "当前运行环境没有可用 LLM，且不允许写入占位意见。",
                code="llm_stub_not_allowed",
                status=503,
            )

        disputes: list[Dispute] = []
        arbitration: ArbitrationOutput | None = None
        arbitration_call: AgentCallRecord | None = None
        if not failed_roles and len(opinions) == len(role_list):
            disputes = make_disputes(
                opinions,
                review_id=review_id,
                claim_id=claim_id,
                packet=packet,
                now=datetime.now(),
            )
            if disputes:
                arbitration_call = AgentCallRecord(
                    call_id=_call_id("central_arbitrator"),
                    role="central_arbitrator",
                    model_id=get_llm_model_for_role("central_arbitrator"),
                    request_hash=request_hash,
                    attempt=1,
                    status="succeeded",
                    created_at=now,
                    updated_at=now,
                )
                try:
                    output = self.gateway.generate_structured(
                        system_prompt=ARBITRATOR_SYSTEM_PROMPT,
                        user_prompt=arbitration_prompt(packet, claim_id, opinions, disputes),
                        response_model=ArbitrationOutput,
                        fallback_factory=lambda: _arbitration_fallback(claim_id, packet.packet_hash),
                    )
                    arbitration = output
                    if output.is_stub:
                        arbitration_call.updated_at = datetime.now()
                except Exception as exc:
                    arbitration_call.status = "failed"
                    arbitration_call.error_code = "arbitration_failed"
                    arbitration_call.error_message = str(exc)
                    arbitration_call.updated_at = datetime.now()
                    arbitration = ArbitrationOutput(
                        issue_summary="仲裁调用失败，冲突保持未解决。",
                        recommended_action="保留两路意见与争议清单，交由人工决定。",
                        required_validation="登记一条人工决定或批准一项验证实验。",
                        is_stub=True,
                    )
                calls.append(arbitration_call)

        if failed_roles:
            status = "failed" if not opinions else "partial"
            message = f"审读角色未全部完成：{', '.join(failed_roles)}；已失败的轮次不生成共识。"
        else:
            status = "completed"
            message = None

        # JobStatus has no partial literal; the finer review state is carried
        # in stage so a failed review never claims a completed job state.
        job.status = "completed" if status == "completed" else "failed"
        job.updated_at = datetime.now()
        job.message = message
        return DualReviewResult(
            status=status,
            review_id=review_id,
            packet_hash=packet.packet_hash,
            claim_id=claim_id,
            request_hash=request_hash,
            job=job,
            opinions=opinions,
            agent_calls=calls,
            disputes=disputes,
            arbitration=arbitration,
            arbitration_call=arbitration_call,
            message=message,
            is_stub=is_stub,
        )

    def _generate_roles(
        self,
        roles: tuple[str, ...],
        *,
        packet: EvidencePacket,
        claim: ClaimVersion | None,
        review_id: str,
        request_hash: str,
        attempt: int,
    ) -> dict[str, RoleAttemptResult]:
        roles = tuple(dict.fromkeys(roles))
        results: dict[str, RoleAttemptResult] = {}
        if not roles:
            return results
        with ThreadPoolExecutor(
            max_workers=min(max(1, self.max_in_flight), len(roles)),
            thread_name_prefix="hp-review",
        ) as executor:
            futures = {
                executor.submit(
                    self._generate_role,
                    role,
                    packet=packet,
                    claim=claim,
                    review_id=review_id,
                    request_hash=request_hash,
                    attempt=attempt,
                ): role
                for role in roles
            }
            for future in futures:
                role = futures[future]
                results[role] = future.result()
        return results

    def _generate_role(
        self,
        role: str,
        *,
        packet: EvidencePacket,
        claim: ClaimVersion | None,
        review_id: str,
        request_hash: str,
        attempt: int,
    ) -> RoleAttemptResult:
        call_id = _call_id(role)
        now = datetime.now()
        call = AgentCallRecord(
            call_id=call_id,
            role=role,
            model_id=get_llm_model_for_role(role),
            request_hash=request_hash,
            attempt=attempt,
            status="running",
            created_at=now,
            updated_at=now,
        )
        with self.semaphore:
            claim_id = claim.claim_id if claim else None
            try:
                system_prompt = (
                    QUESTIONER_SYSTEM_PROMPT
                    if role == "scientific_questioner"
                    else INTERPRETER_SYSTEM_PROMPT
                )
                output = self.gateway.generate_structured(
                    system_prompt=system_prompt,
                    user_prompt=_role_prompt(role, packet, claim),
                    response_model=ReviewOpinionOutput,
                    fallback_factory=_fallback_factory(role, claim_id=claim_id or "CLM_UNKNOWN", packet_hash=packet.packet_hash),
                )
                validate_opinion_refs(
                    output,
                    packet,
                    claim_id=claim_id or "",
                )
                opinion = _to_opinion(
                    output,
                    review_id=review_id,
                    role=role,
                    claim_id=claim_id,
                    packet_hash=packet.packet_hash,
                    now=datetime.now(),
                )
                call.status = "succeeded"
                call.updated_at = datetime.now()
                return RoleAttemptResult(role=role, attempt=attempt, opinion=opinion, call=call)
            except ResearchReviewError as exc:
                call.status = "failed"
                call.error_code = exc.code
                message = str(exc)
                if exc.details:
                    metric_refs = exc.details.get("metric_refs") or []
                    evidence_refs = exc.details.get("evidence_refs") or []
                    if metric_refs:
                        message += f" 指标引用：{', '.join(metric_refs)}"
                    if evidence_refs:
                        message += f" 证据引用：{', '.join(evidence_refs)}"
                call.error_message = message
                call.updated_at = datetime.now()
                return RoleAttemptResult(role=role, attempt=attempt, opinion=None, call=call)
            except Exception as exc:
                call.status = "failed"
                call.error_code = "review_generation_failed"
                call.error_message = str(exc)
                call.updated_at = datetime.now()
                return RoleAttemptResult(role=role, attempt=attempt, opinion=None, call=call)


def build_review_mutation(
    *,
    review_id: str,
    result: DualReviewResult,
) -> Callable[[ResearchRevision], None]:
    """Produce a repository-compatible mutation that appends the whole run.

    State transitions are program-owned: no LLM output can change check
    statuses or pretend a partial review reached consensus.
    """

    def mutation(state: ResearchRevision) -> None:
        existing_job = next(
            (job for job in state.jobs if job.job_id == result.job.job_id),
            None,
        )
        if existing_job is not None:
            existing_job.status = result.job.status
            existing_job.stage = result.job.stage
            existing_job.message = result.job.message
            existing_job.updated_at = result.job.updated_at
        else:
            state.jobs.append(result.job)

        known_call_ids = {call.call_id for call in state.agent_calls}
        for call in result.agent_calls:
            if call.call_id not in known_call_ids:
                state.agent_calls.append(call)
                known_call_ids.add(call.call_id)

        known_opinion_ids = {opinion.opinion_id for opinion in state.opinions}
        for opinion in result.opinions:
            if opinion.opinion_id not in known_opinion_ids:
                state.opinions.append(opinion)
                known_opinion_ids.add(opinion.opinion_id)

        known_dispute_ids = {dispute.dispute_id for dispute in state.disputes}
        for dispute in result.disputes:
            if dispute.dispute_id not in known_dispute_ids:
                state.disputes.append(dispute)
                known_dispute_ids.add(dispute.dispute_id)

        if result.arbitration is not None:
            arbitration = ArbitrationRecord(
                arbitration_id=f"ARB_{result.review_id[-16:]}",
                review_id=result.review_id,
                dispute_ids=[dispute.dispute_id for dispute in result.disputes],
                issue_summary=result.arbitration.issue_summary,
                recommended_action=result.arbitration.recommended_action,
                required_validation=result.arbitration.required_validation,
                is_stub=result.arbitration.is_stub,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            if not any(
                item.arbitration_id == arbitration.arbitration_id
                and item.review_id == arbitration.review_id
                for item in state.arbitrations
            ):
                state.arbitrations.append(arbitration)

        if result.status == "completed":
            state.project.stage = "awaiting_decision"
            result.job.stage = "awaiting_decision"
        elif result.status == "partial":
            state.project.stage = "reviewing"
            result.job.stage = "partial_review"
        else:
            state.project.stage = "reviewing"
            result.job.stage = "review_failed"

    return mutation
