from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from core.research_repository import (
    ResearchProjectNotFoundError,
    ResearchRepository,
)
from core.research_schema import (
    SCHEMA_VERSION,
    ClaimScope,
    ClaimVersion,
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

    def __init__(self, repository: ResearchRepository) -> None:
        self.repository = repository

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
        data = {
            "project": project.model_dump(exclude_none=True),
            "revision": revision.revision,
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
