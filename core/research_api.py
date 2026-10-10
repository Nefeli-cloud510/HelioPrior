from __future__ import annotations

import re
import uuid
from typing import Any
from urllib.parse import urlsplit

from core.research_repository import ResearchRepository, ResearchVersionConflictError
from core.review_orchestrator import ResearchReviewError
from core.research_service import ApiResult, ResearchApiError, ResearchService
from core.research_validation import ResearchValidationError


RESEARCH_BASE = "/api/v2/research"


class ResearchV2Api:
    """HTTP layer for the research-review API; no scientific logic lives here."""

    def __init__(
        self,
        repository: ResearchRepository | None = None,
        *,
        gateway: Any | None = None,
        data_source: Any | None = None,
        harness_project_root: Any | None = None,
        run_shap: bool = False,
    ) -> None:
        self.service = ResearchService(
            repository or ResearchRepository(),
            gateway=gateway,
            data_source=data_source,
            harness_project_root=harness_project_root,
            run_shap=run_shap,
        )

    def dispatch(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        idempotency_key: str | None = None,
    ) -> ApiResult:
        request_id = "REQ_" + uuid.uuid4().hex[:12].upper()
        normalized_path = urlsplit(path).path
        try:
            if method == "POST" and normalized_path == RESEARCH_BASE:
                return self._with_request_id(
                    request_id,
                    self.service.create_project(payload or {}, idempotency_key=idempotency_key),
                )
            if method == "GET" and normalized_path == RESEARCH_BASE:
                return self._with_request_id(request_id, self.service.list_projects())

            match = re.fullmatch(rf"{re.escape(RESEARCH_BASE)}/(?P<research_id>[^/]+)", normalized_path)
            if match and method == "GET":
                return self._with_request_id(request_id, self.service.get_project(match.group("research_id")))

            if normalized_path.startswith(f"{RESEARCH_BASE}/"):
                suffix = normalized_path[len(RESEARCH_BASE) + 1 :]
                operation_parts = suffix.split("/")[1:]
                if not operation_parts:
                    raise ResearchApiError("v2 接口不存在。", status=404, code="route_not_found")
                operation = operation_parts[0]
                research_id = suffix.split("/")[0]
                review_retry_match = re.fullmatch(
                    rf"reviews/(?P<review_id>[^/]+)/retry",
                    "/".join(operation_parts),
                )
                if review_retry_match and method == "POST":
                    return self._with_request_id(
                        request_id,
                        self.service.retry_review(
                            research_id=research_id,
                            expected_revision=(payload or {}).get("expected_revision"),
                            review_id=review_retry_match.group("review_id"),
                            payload=payload or {},
                        ),
                    )
                if operation == "artifacts" and method == "POST":
                    return self._with_request_id(
                        request_id,
                        self.service.import_artifact(
                            research_id=research_id,
                            expected_revision=(payload or {}).get("expected_revision"),
                            payload=payload or {},
                        ),
                    )
                if operation == "preflight" and method == "POST":
                    return self._with_request_id(
                        request_id,
                        self.service.preflight(
                            research_id=research_id,
                            expected_revision=(payload or {}).get("expected_revision"),
                            payload=payload or {},
                        ),
                    )
                if operation == "reviews" and method == "POST":
                    return self._with_request_id(
                        request_id,
                        self.service.start_review(
                            research_id=research_id,
                            expected_revision=(payload or {}).get("expected_revision"),
                            payload=payload or {},
                        ),
                    )
                if operation == "disputes" and method == "GET":
                    return self._with_request_id(
                        request_id,
                        self.service.list_disputes(research_id),
                    )
                if operation == "plans" and method == "POST":
                    return self._with_request_id(
                        request_id,
                        self.service.plan_validations(
                            research_id=research_id,
                            expected_revision=(payload or {}).get("expected_revision"),
                            payload=payload or {},
                        ),
                    )
                if operation == "approvals" and method == "POST":
                    return self._with_request_id(
                        request_id,
                        self.service.approve_protocol(
                            research_id=research_id,
                            expected_revision=(payload or {}).get("expected_revision"),
                            payload=payload or {},
                        ),
                    )
                if operation == "executions" and method == "POST":
                    return self._with_request_id(
                        request_id,
                        self.service.execute_protocol(
                            research_id=research_id,
                            expected_revision=(payload or {}).get("expected_revision"),
                            payload=payload or {},
                        ),
                    )
                claim_versions_match = re.fullmatch(
                    r"claims/(?P<claim_id>[^/]+)/versions",
                    "/".join(operation_parts),
                )
                if claim_versions_match:
                    claim_id = claim_versions_match.group("claim_id")
                    if method == "GET":
                        return self._with_request_id(
                            request_id,
                            self.service.list_claim_versions(research_id, claim_id),
                        )
                    if method == "POST":
                        return self._with_request_id(
                            request_id,
                            self.service.create_claim_version(
                                research_id=research_id,
                                expected_revision=(payload or {}).get("expected_revision"),
                                claim_id=claim_id,
                                payload=payload or {},
                            ),
                        )
                if operation == "report" and method == "GET":
                    return self._with_request_id(
                        request_id,
                        self.service.export_report(research_id),
                    )
                if operation in {
                    "artifacts",
                    "preflight",
                    "hypotheses",
                    "evidence",
                    "decisions",
                    "claims",
                    "jobs",
                    "report",
                }:
                    raise ResearchApiError(
                        "该 v2 接口已列入路线图，但尚未在本任务接线。",
                        status=501,
                        code="not_implemented",
                        retryable=False,
                    )
                raise ResearchApiError("v2 接口不存在。", status=404, code="route_not_found")
            raise ResearchApiError("接口不存在。", status=404, code="route_not_found")
        except ResearchApiError as exc:
            return self._error_response(request_id, exc)
        except ResearchReviewError as exc:
            error = ResearchApiError(
                exc.message,
                status=exc.status,
                code=exc.code,
                retryable=exc.status >= 500,
                details=exc.details,
            )
            return self._error_response(request_id, error)
        except ResearchValidationError as exc:
            error = ResearchApiError(
                exc.message,
                status=exc.status,
                code=exc.code,
                retryable=exc.retryable,
                details=exc.details,
            )
            return self._error_response(request_id, error)
        except ResearchVersionConflictError as exc:
            error = ResearchApiError(
                str(exc),
                status=409,
                code="revision_conflict",
                retryable=True,
            )
            return self._error_response(request_id, error)
        except Exception as exc:
            error = ResearchApiError(
                f"研究接口执行失败：{exc}",
                status=500,
                code="internal_error",
                retryable=True,
            )
            return self._error_response(request_id, error)

    @staticmethod
    def _with_request_id(request_id: str, result: ApiResult) -> ApiResult:
        payload = dict(result.payload)
        payload["request_id"] = request_id
        return ApiResult(status=result.status, payload=payload)

    @staticmethod
    def _error_response(request_id: str, error: ResearchApiError) -> ApiResult:
        return ApiResult(
            status=error.status,
            payload={
                "request_id": request_id,
                "code": error.code,
                "message": error.message,
                "retryable": error.retryable,
                "details": error.details,
            },
        )
