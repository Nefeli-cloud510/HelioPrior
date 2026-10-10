from __future__ import annotations

import re
import uuid
from typing import Any
from urllib.parse import urlsplit

from core.research_repository import ResearchRepository, ResearchVersionConflictError
from core.research_service import ApiResult, ResearchApiError, ResearchService


RESEARCH_BASE = "/api/v2/research"


class ResearchV2Api:
    """HTTP layer for the research-review API; no scientific logic lives here."""

    def __init__(self, repository: ResearchRepository | None = None) -> None:
        self.service = ResearchService(repository or ResearchRepository())

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
                if operation_parts and operation_parts[0] in {
                    "artifacts",
                    "preflight",
                    "hypotheses",
                    "plans",
                    "approvals",
                    "executions",
                    "evidence",
                    "reviews",
                    "disputes",
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
