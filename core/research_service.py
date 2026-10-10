from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from core.evidence_builder import (
    build_evidence_packet,
    normalize_prediction_rows,
    parse_source_content,
    summarize_preflight,
)
from core.research_repository import (
    ResearchProjectNotFoundError,
    ResearchRepository,
)
from core.review_orchestrator import (
    REVIEW_ROLES,
    DualReviewResult,
    ResearchReviewError,
    ReviewOrchestrator,
    build_review_mutation,
    review_request_hash,
)
from core.research_schema import (
    SCHEMA_VERSION,
    ArtifactKind,
    ArtifactRecord,
    ClaimScope,
    ClaimVersion,
    EvidencePacket,
    ResearchProject,
    ResearchRevision,
    ResearchTarget,
)


class ResearchApiError(RuntimeError):
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


@dataclass(frozen=True)
class ApiResult:
    status: int
    payload: dict[str, Any]


class ResearchService:
    """State transitions and API actions for the research-review workflow."""

    def __init__(
        self,
        repository: ResearchRepository,
        *,
        gateway: Any | None = None,
    ) -> None:
        self.repository = repository
        self.gateway = gateway

    def _orchestrator(self) -> ReviewOrchestrator:
        return ReviewOrchestrator(gateway=self.gateway)

    def create_project(
        self,
        payload: dict[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> ApiResult:
        request_hash = ResearchRepository.request_hash(payload)
        if idempotency_key:
            existing = self.repository.read_idempotent(idempotency_key)
            if existing is not None:
                if existing.request_hash != request_hash:
                    raise ResearchApiError(
                        "幂等键已被不同请求使用。",
                        status=409,
                        code="idempotency_conflict",
                        retryable=False,
                    )
                return ApiResult(status=existing.status, payload=dict(existing.payload))

        project, initial_claim = self._build_project(payload)
        revision = self.repository.create_project(project)
        if initial_claim is not None:
            next_state = self.repository.mutate(
                project.research_id,
                expected_revision=1,
                mutation=lambda state: state.claims.append(initial_claim),
            )
            revision = next_state
        result = self._project_response(revision, created=True)
        if idempotency_key:
            self.repository.save_idempotent(
                idempotency_key,
                request_hash=request_hash,
                status=result.status,
                payload=result.payload,
            )
        return result

    def list_projects(self) -> ApiResult:
        projects = self.repository.list_projects()
        items = [
            {
                "research_id": item.research_id,
                "title": item.title,
                "question": item.question,
                "target": item.target.model_dump(exclude_none=True),
                "horizon_days": item.horizon_days,
                "data_mode": item.data_mode,
                "workflow_mode": item.workflow_mode,
                "stage": item.stage,
                "revision": item.revision,
                "updated_at": item.updated_at.isoformat(),
            }
            for item in projects
        ]
        return ApiResult(status=200, payload={"data": items})

    def get_project(self, research_id: str) -> ApiResult:
        try:
            revision = self.repository.load_revision(research_id)
        except ResearchProjectNotFoundError:
            raise ResearchApiError("研究项目不存在。", status=404, code="project_not_found")
        return self._project_response(revision, created=False)

    def import_artifact(
        self,
        research_id: str,
        expected_revision: int,
        payload: dict[str, Any],
    ) -> ApiResult:
        try:
            current = self.repository.load_revision(research_id)
        except ResearchProjectNotFoundError:
            raise ResearchApiError("研究项目不存在。", status=404, code="project_not_found")
        try:
            expected = int(expected_revision)
        except (TypeError, ValueError):
            raise ResearchApiError(
                "expected_revision 必须是整数。",
                status=422,
                code="invalid_revision",
            )
        if current.revision != expected:
            raise ResearchApiError(
                f"版本冲突：当前 revision={current.revision}。",
                status=409,
                code="revision_conflict",
                retryable=True,
            )

        file_name = str(payload.get("file_name") or "").strip()
        content = payload.get("content")
        if not file_name:
            raise ResearchApiError("file_name 不能为空。", status=422, code="missing_required_field")
        if content is None:
            raise ResearchApiError("content 不能为空。", status=422, code="missing_required_field")
        if isinstance(content, str):
            if len(content) > 20_000_000:
                raise ResearchApiError(
                    "导入产物过大，单文件上限为 20MB。",
                    status=422,
                    code="artifact_too_large",
                )
        elif not isinstance(content, (bytes, bytearray)):
            raise ResearchApiError(
                "content 必须是字符串或字节。",
                status=422,
                code="invalid_artifact_content",
            )

        kind = str(payload.get("kind") or "prediction_csv").lower()
        allowed_kinds = set(ArtifactKind.__args__)
        if kind not in allowed_kinds:
            raise ResearchApiError(
                f"artifact kind 不在白名单内：{kind}",
                status=422,
                code="invalid_artifact_kind",
            )

        text_content = content.decode("utf-8", errors="replace") if isinstance(content, (bytes, bytearray)) else content
        parsed_source = parse_source_content(kind, text_content, file_name)
        parsed = normalize_prediction_rows(parsed_source)
        artifact_id = "ART_" + uuid.uuid4().hex[:12].upper()
        stored_path = self.repository.store_artifact(research_id, artifact_id, text_content)
        now = datetime.now()
        artifact = ArtifactRecord(
            artifact_id=artifact_id,
            file_name=file_name,
            sha256=hashlib.sha256(text_content.encode("utf-8")).hexdigest(),
            size_bytes=len(text_content.encode("utf-8")),
            kind=kind,
            stored_path=str(stored_path),
            content_type=payload.get("content_type"),
            parsed=not bool(parsed_source.parse_errors),
            parse_errors=list(parsed_source.parse_errors),
            created_at=now,
            updated_at=now,
            source_kind="research_service",
            schema_version=SCHEMA_VERSION,
        )

        def apply_artifact(state: ResearchRevision) -> None:
            state.artifacts.append(artifact)

        next_state = self.repository.mutate(
            research_id,
            expected_revision=expected,
            mutation=apply_artifact,
        )
        result = self._project_response(next_state, created=False)
        result.payload["data"]["artifact"] = artifact.model_dump(exclude_none=True)
        return ApiResult(status=201, payload=result.payload)

    def preflight(
        self,
        research_id: str,
        expected_revision: int,
        payload: dict[str, Any],
    ) -> ApiResult:
        try:
            current = self.repository.load_revision(research_id)
        except ResearchProjectNotFoundError:
            raise ResearchApiError("研究项目不存在。", status=404, code="project_not_found")
        try:
            expected = int(expected_revision)
        except (TypeError, ValueError):
            raise ResearchApiError(
                "expected_revision 必须是整数。",
                status=422,
                code="invalid_revision",
            )
        if current.revision != expected:
            raise ResearchApiError(
                f"版本冲突：当前 revision={current.revision}。",
                status=409,
                code="revision_conflict",
                retryable=True,
            )

        artifact_id = str(payload.get("artifact_id") or "").strip()
        artifact = next(
            (item for item in current.artifacts if item.artifact_id == artifact_id),
            None,
        )
        if artifact is None:
            artifact = next(
                (item for item in reversed(current.artifacts) if item.kind == "prediction_csv"),
                None,
            )
        if artifact is None:
            raise ResearchApiError(
                "预检需要至少一条已导入产物。",
                status=422,
                code="artifact_required",
            )

        try:
            content = self.repository.read_artifact(research_id, artifact.artifact_id)
        except ResearchProjectNotFoundError:
            raise ResearchApiError(
                "产物文件不可读，请重新上传。",
                status=410,
                code="artifact_storage_missing",
            )
        text_content = content.decode("utf-8", errors="replace")
        parsed_source = parse_source_content(artifact.kind, text_content, artifact.file_name)
        parsed = normalize_prediction_rows(
            parsed_source,
            target_column=(payload.get("parse_options") or {}).get("target_column"),
        )

        project = current.project
        claim_ids = payload.get("claim_ids") or []
        claim_ids = [str(item) for item in claim_ids]
        existing_claim_ids = {claim.claim_id for claim in current.claims}
        unknown_claim_ids = [claim_id for claim_id in claim_ids if claim_id not in existing_claim_ids]
        if unknown_claim_ids:
            raise ResearchApiError(
                f"claim_ids 包含不存在的研究结论：{', '.join(unknown_claim_ids)}",
                status=422,
                code="invalid_claim_ids",
            )
        if not claim_ids and current.claims:
            claim_ids = [current.claims[0].claim_id]

        protocol_hash_value = payload.get("protocol_hash_value")
        if not protocol_hash_value and current.protocols:
            protocol_hash_value = current.protocols[-1].protocol_hash
        data_manifest_hash = payload.get("data_manifest_hash")
        feature_columns = payload.get("feature_columns")
        if feature_columns is not None and not isinstance(feature_columns, list):
            raise ResearchApiError(
                "feature_columns 必须是字符串数组。",
                status=422,
                code="invalid_feature_columns",
            )
        horizon_days = payload.get("horizon_days")
        if horizon_days is None:
            horizon_days = project.horizon_days
        try:
            horizon_days = int(horizon_days)
        except (TypeError, ValueError):
            raise ResearchApiError(
                "horizon_days 必须是整数。",
                status=422,
                code="invalid_horizon",
            )

        packet = build_evidence_packet(
            parsed=parsed,
            protocol_hash_value=str(protocol_hash_value) if protocol_hash_value else None,
            data_manifest_hash=data_manifest_hash,
            run_id=payload.get("run_id"),
            claim_ids=claim_ids,
            target_id=str(payload.get("target_id") or project.target.id),
            unit=project.target.unit,
            horizon_days=horizon_days,
            feature_columns=feature_columns,
            base_arm=payload.get("base_arm"),
            aug_arm=payload.get("aug_arm"),
            primary_artifact_id=artifact.artifact_id,
            primary_artifact_sha256=artifact.sha256,
            imported_metrics=parsed_source.imported_metrics,
        )
        if not packet.packet_id:
            packet.packet_id = f"EVP_{packet.packet_hash[:12]}"

        blocking_issues = [
            {
                "code": check.code,
                "status": check.status,
                "detail": check.detail,
                "reason": check.reason,
            }
            for check in packet.checks
            if check.blocking and check.status != "pass"
        ]
        if parsed_source.parse_errors or parsed.missing_columns:
            preflight_status = "needs_input"
            stage = "needs_input"
        elif blocking_issues:
            preflight_status = "needs_input"
            stage = "needs_input"
        else:
            preflight_status = "ready"
            stage = "evidence_ready"

        def apply_packet(state: ResearchRevision) -> None:
            state.evidence_packets.append(packet)
            for claim in state.claims:
                if claim.claim_id in claim_ids and packet.packet_id not in claim.evidence_ids:
                    claim.evidence_ids.append(packet.packet_id)
            state.project.stage = stage

        next_state = self.repository.mutate(
            research_id,
            expected_revision=expected,
            mutation=apply_packet,
        )
        result = self._project_response(next_state, created=False)
        result.payload["data"]["pc_stage"] = next_state.project.stage
        result.payload["data"]["preflight"] = {
            "status": preflight_status,
            "packet": packet.model_dump(exclude_none=True),
            "summary": summarize_preflight(parsed, str(protocol_hash_value) if protocol_hash_value else None),
            "blocking_issues": blocking_issues,
        }
        return ApiResult(status=200, payload=result.payload)

    def _resolve_review_targets(
        self,
        revision: ResearchRevision,
        payload: dict[str, Any],
    ) -> tuple[EvidencePacket, ClaimVersion]:
        packet_id = str(payload.get("packet_id") or "").strip()
        if packet_id:
            packet = next(
                (item for item in revision.evidence_packets if item.packet_id == packet_id),
                None,
            )
            if packet is None:
                raise ResearchApiError(
                    f"证据包不存在：{packet_id}",
                    status=404,
                    code="packet_not_found",
                )
        else:
            packet = next(reversed(revision.evidence_packets), None)
        if packet is None:
            raise ResearchApiError(
                "复核需要至少一个证据包，请先运行证据预检。",
                status=422,
                code="evidence_packet_required",
            )

        claim_id = str(payload.get("claim_id") or "").strip()
        if claim_id:
            claim = next(
                (item for item in revision.claims if item.claim_id == claim_id),
                None,
            )
            if claim is None:
                raise ResearchApiError(
                    f"研究结论不存在：{claim_id}",
                    status=404,
                    code="claim_not_found",
                )
        else:
            by_packet = [
                claim
                for claim in revision.claims
                if not packet.claim_ids or claim.claim_id in packet.claim_ids
            ]
            claim = by_packet[0] if by_packet else next(iter(revision.claims), None)
        if claim is None:
            raise ResearchApiError(
                "复核需要至少一条研究结论。",
                status=422,
                code="claim_required",
            )
        return packet, claim

    def start_review(
        self,
        research_id: str,
        expected_revision: int,
        payload: dict[str, Any],
    ) -> ApiResult:
        try:
            current = self.repository.load_revision(research_id)
        except ResearchProjectNotFoundError:
            raise ResearchApiError("研究项目不存在。", status=404, code="project_not_found")
        self._require_revision(current.revision, expected_revision)
        packet, claim = self._resolve_review_targets(current, payload)

        orchestrator = self._orchestrator()
        result = orchestrator.run_dual_review(
            packet=packet,
            claim=claim,
            project=current.project,
            frozen_revision=current.revision,
        )
        next_state = self.repository.mutate(
            research_id,
            expected_revision=current.revision,
            mutation=build_review_mutation(
                review_id=result.review_id,
                result=result,
            ),
        )
        return self._review_response(next_state, result)

    def retry_review(
        self,
        research_id: str,
        expected_revision: int,
        review_id: str,
        payload: dict[str, Any],
    ) -> ApiResult:
        try:
            current = self.repository.load_revision(research_id)
        except ResearchProjectNotFoundError:
            raise ResearchApiError("研究项目不存在。", status=404, code="project_not_found")
        self._require_revision(current.revision, expected_revision)

        review_opinions = [
            opinion
            for opinion in current.opinions
            if opinion.review_id == review_id
        ]
        review_jobs = [
            job
            for job in current.jobs
            if job.review_id == review_id and job.kind == "dual_review"
        ]
        if not review_opinions and not review_jobs:
            raise ResearchApiError(
                f"复核不存在：{review_id}",
                status=404,
                code="review_not_found",
            )

        completed_roles = {
            opinion.role
            for opinion in review_opinions
            if opinion.status == "completed"
        }
        missing_roles = [
            role for role in REVIEW_ROLES if role not in completed_roles
        ]
        if not missing_roles:
            raise ResearchApiError(
                "该复核已完成，没有失败或缺失的审读角色。",
                status=409,
                code="review_not_retryable",
            )

        raw_roles = payload.get("retry_roles") or missing_roles
        if not isinstance(raw_roles, (list, tuple)):
            raise ResearchApiError(
                "retry_roles 必须是角色数组。",
                status=422,
                code="invalid_retry_roles",
            )
        retry_roles = tuple(dict.fromkeys(str(item) for item in raw_roles))
        invalid_roles = [role for role in retry_roles if role not in REVIEW_ROLES]
        if invalid_roles:
            raise ResearchApiError(
                f"未知审读角色：{', '.join(invalid_roles)}",
                status=422,
                code="invalid_retry_roles",
            )

        packet_hash = str(payload.get("packet_hash") or "").strip()
        if packet_hash:
            packet = next(
                (
                    item
                    for item in current.evidence_packets
                    if item.packet_hash == packet_hash
                ),
                None,
            )
            if packet is None:
                raise ResearchApiError(
                    "重试指定的证据包不存在。",
                    status=404,
                    code="packet_not_found",
                )
        else:
            opinion_hashes = {opinion.packet_hash for opinion in review_opinions}
            packet = next(
                reversed(current.evidence_packets),
                None,
            )
            if packet is not None and packet.packet_hash not in opinion_hashes and opinion_hashes:
                packet = next(
                    (
                        item
                        for item in reversed(current.evidence_packets)
                        if item.packet_hash in opinion_hashes
                    ),
                    packet,
                )
        if packet is None:
            raise ResearchApiError(
                "重试需要证据包，请先运行证据预检。",
                status=422,
                code="evidence_packet_required",
            )

        claim_id = str(payload.get("claim_id") or "").strip()
        if claim_id:
            claim = next(
                (item for item in current.claims if item.claim_id == claim_id),
                None,
            )
        else:
            claim = next(
                (item for item in current.claims if item.claim_id in packet.claim_ids),
                None,
            ) or next(iter(current.claims), None)
        if claim is None:
            raise ResearchApiError(
                "重试需要研究结论。",
                status=422,
                code="claim_required",
            )

        request_hash = next(
            (
                job.request_hash
                for job in reversed(review_jobs)
                if job.request_hash
            ),
            None,
        )
        if request_hash is None:
            request_hash = next(
                (
                    call.request_hash
                    for call in reversed(current.agent_calls)
                    if call.request_hash and call.role in REVIEW_ROLES
                ),
                None,
            )
        if request_hash is None:
            request_hash = review_request_hash(
                packet_hash=packet.packet_hash,
                claim_id=claim.claim_id,
                claim_version=claim.version,
                project_stage=current.project.stage,
                frozen_revision=current.revision,
                retry_roles=retry_roles,
            )

        orchestrator = self._orchestrator()
        try:
            attempt = max(2, int(payload.get("attempt") or 2))
        except (TypeError, ValueError):
            attempt = 2
        result = orchestrator.run_dual_review(
            packet=packet,
            claim=claim,
            project=current.project,
            frozen_revision=current.revision,
            retry_roles=retry_roles,
            existing_opinions=review_opinions,
            review_id=review_id,
            request_hash=request_hash,
            attempt=attempt,
        )
        next_state = self.repository.mutate(
            research_id,
            expected_revision=current.revision,
            mutation=build_review_mutation(
                review_id=review_id,
                result=result,
            ),
        )
        return self._review_response(next_state, result)

    def list_disputes(self, research_id: str) -> ApiResult:
        try:
            revision = self.repository.load_revision(research_id)
        except ResearchProjectNotFoundError:
            raise ResearchApiError("研究项目不存在。", status=404, code="project_not_found")
        dispute_ids = {dispute.dispute_id for dispute in revision.disputes}
        return ApiResult(
            status=200,
            payload={
                "research_id": research_id,
                "revision": revision.revision,
                "data": {
                    "research_id": research_id,
                    "revision": revision.revision,
                    "project": revision.project.model_dump(exclude_none=True),
                    "disputes": [
                        dispute.model_dump(exclude_none=True)
                        for dispute in revision.disputes
                    ],
                    "opinions": [
                        opinion.model_dump(exclude_none=True)
                        for opinion in revision.opinions
                        if not dispute_ids or opinion.opinion_id in {
                            opinion_id
                            for dispute in revision.disputes
                            for opinion_id in dispute.opinion_ids
                        }
                    ],
                    "arbitrations": [
                        arbitration.model_dump(exclude_none=True)
                        for arbitration in revision.arbitrations
                    ],
                    "open_dispute_count": len(
                        [
                            dispute
                            for dispute in revision.disputes
                            if dispute.status
                            not in {"dismissed", "validated", "superseded"}
                        ]
                    ),
                },
            },
        )

    def _require_revision(self, current_revision: int, expected_revision: Any) -> None:
        try:
            expected = int(expected_revision)
        except (TypeError, ValueError):
            raise ResearchApiError(
                "expected_revision 必须是整数。",
                status=422,
                code="invalid_revision",
            )
        if current_revision != expected:
            raise ResearchApiError(
                f"版本冲突：当前 revision={current_revision}。",
                status=409,
                code="revision_conflict",
                retryable=True,
            )

    def _review_response(
        self,
        revision: ResearchRevision,
        result: DualReviewResult,
    ) -> ApiResult:
        base = self._project_response(revision, created=False)
        data = base.payload["data"]
        review_opinions = [
            opinion
            for opinion in revision.opinions
            if opinion.review_id == result.review_id
        ]
        review_disputes = [
            dispute
            for dispute in revision.disputes
            if dispute.review_id == result.review_id
        ]
        review_arbitrations = [
            arbitration
            for arbitration in revision.arbitrations
            if arbitration.review_id == result.review_id
        ]
        review_calls = [
            call
            for call in revision.agent_calls
            if call.request_hash == result.request_hash
        ]
        data["review"] = {
            "review_id": result.review_id,
            "status": result.status,
            "stage": result.job.stage,
            "packet_hash": result.packet_hash,
            "claim_id": result.claim_id,
            "request_hash": result.request_hash,
            "is_stub": result.is_stub,
            "message": result.message,
            "job": result.job.model_dump(exclude_none=True),
            "opinions": [
                opinion.model_dump(exclude_none=True)
                for opinion in review_opinions
            ],
            "disputes": [
                dispute.model_dump(exclude_none=True)
                for dispute in review_disputes
            ],
            "arbitrations": [
                arbitration.model_dump(exclude_none=True)
                for arbitration in review_arbitrations
            ],
            "agent_calls": [
                call.model_dump(exclude_none=True)
                for call in review_calls
            ],
        }
        return ApiResult(status=200, payload=base.payload)

    def _build_project(
        self,
        payload: dict[str, Any],
    ) -> tuple[ResearchProject, ClaimVersion | None]:
        workflow_mode = str(payload.get("workflow_mode") or "research_review")
        if workflow_mode != "research_review":
            raise ResearchApiError(
                "v2 仅支持 research_review 模式。",
                status=422,
                code="invalid_workflow_mode",
            )
        title = str(payload.get("title") or "").strip()
        question = str(payload.get("question") or "").strip()
        if not title or not question:
            raise ResearchApiError(
                "title 和 question 不能为空。",
                status=422,
                code="missing_required_field",
            )
        target_payload = payload.get("target")
        if not isinstance(target_payload, dict):
            raise ResearchApiError(
                "target 必须是对象。",
                status=422,
                code="invalid_target",
            )
        try:
            target = ResearchTarget.model_validate(target_payload)
        except Exception as exc:
            raise ResearchApiError(
                f"target 字段不合法：{exc}",
                status=422,
                code="invalid_target",
            )
        horizon_days = payload.get("horizon_days")
        if horizon_days is not None:
            try:
                horizon_days = int(horizon_days)
            except (TypeError, ValueError):
                raise ResearchApiError(
                    "horizon_days 必须是整数。",
                    status=422,
                    code="invalid_horizon",
                )
        else:
            horizon_days = 3

        entry_mode = str(payload.get("entry_mode") or "new_question")
        if entry_mode not in {"new_question", "import_claim"}:
            raise ResearchApiError(
                "entry_mode 仅支持 new_question 或 import_claim。",
                status=422,
                code="invalid_entry_mode",
            )
        data_mode = str(payload.get("data_mode") or "unknown")
        if data_mode not in {"authorized_data", "synthetic_fixture", "historical_replay", "unknown"}:
            raise ResearchApiError(
                "data_mode 不在白名单内。",
                status=422,
                code="invalid_data_mode",
            )
        now = datetime.now()
        research_id = "RSR_" + uuid.uuid4().hex[:12].upper()
        project = ResearchProject(
            research_id=research_id,
            revision=1,
            title=title,
            question=question,
            target=target,
            horizon_days=horizon_days,
            data_mode=data_mode,
            workflow_mode="research_review",
            entry_mode=entry_mode,
            stage="draft",
            summary=payload.get("summary"),
            created_at=now,
            updated_at=now,
            schema_version=SCHEMA_VERSION,
            source_kind="research_service",
        )
        claim: ClaimVersion | None = None
        if entry_mode == "import_claim":
            claim_statement = str(payload.get("claim_statement") or question)
            scope_payload = payload.get("claim_scope") or {}
            claim_scope = ClaimScope(
                target_id=target.id,
                unit=target.unit,
                coordinate_system=target.coordinate_system,
                horizon_days=horizon_days,
                sample_condition=scope_payload.get("sample_condition"),
                evaluation_period=scope_payload.get("evaluation_period"),
                baseline_protocol_id=scope_payload.get("baseline_protocol_id"),
                created_at=now,
                updated_at=now,
            )
            claim = ClaimVersion(
                claim_id=f"CLM_{research_id[-10:]}",
                version=1,
                statement=claim_statement,
                scope=claim_scope,
                status="unassessed",
                created_at=now,
                updated_at=now,
            )
        return project, claim

    def _project_response(self, revision: ResearchRevision, *, created: bool) -> ApiResult:
        project = revision.project
        claims = [claim.model_dump(exclude_none=True) for claim in revision.claims]
        artifacts = [artifact.model_dump(exclude_none=True) for artifact in revision.artifacts]
        evidence_packets = [
            packet.model_dump(exclude_none=True)
            for packet in revision.evidence_packets
        ]
        checks = [
            {
                **check.model_dump(exclude_none=True),
                "packet_id": packet.packet_id,
            }
            for packet in revision.evidence_packets
            for check in packet.checks
        ]
        metrics = {
            f"{packet.packet_id}#{key}": {
                **metric.model_dump(exclude_none=True),
                "packet_id": packet.packet_id,
            }
            for packet in revision.evidence_packets
            for key, metric in packet.metrics.items()
        }
        reviews = [
            {
                "review_id": job.review_id,
                "job_id": job.job_id,
                "status": job.status,
                "stage": job.stage,
                "request_hash": job.request_hash,
                "kind": job.kind,
                "message": job.message,
            }
            for job in revision.jobs
            if job.kind == "dual_review"
        ]
        data = {
            "project": project.model_dump(exclude_none=True),
            "revision": revision.revision,
            "claims": claims,
            "artifacts": artifacts,
            "evidence_packets": evidence_packets,
            "checks": checks,
            "metrics": metrics,
            "opinions": [
                opinion.model_dump(exclude_none=True)
                for opinion in revision.opinions
            ],
            "disputes": [
                dispute.model_dump(exclude_none=True)
                for dispute in revision.disputes
            ],
            "arbitrations": [
                arbitration.model_dump(exclude_none=True)
                for arbitration in revision.arbitrations
            ],
            "agent_calls": [
                call.model_dump(exclude_none=True)
                for call in revision.agent_calls
            ],
            "jobs": [
                job.model_dump(exclude_none=True)
                for job in revision.jobs
            ],
            "reviews": reviews,
            "pc_stage": project.stage,
            "claim_count": len(revision.claims),
            "evidence_packet_count": len(revision.evidence_packets),
            "open_dispute_count": len(
                [d for d in revision.disputes if d.status not in {"dismissed", "validated", "superseded"}]
            ),
            "awaiting_decision": bool(
                project.stage == "awaiting_decision"
                or any(c.status == "unassessed" for c in revision.claims)
            ),
        }
        return ApiResult(
            status=201 if created else 200,
            payload={
                "research_id": project.research_id,
                "revision": revision.revision,
                "data": data,
            },
        )
