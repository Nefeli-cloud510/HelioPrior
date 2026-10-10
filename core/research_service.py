from __future__ import annotations

import csv
import hashlib
import io
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from core.evidence_builder import (
    build_evidence_packet,
    normalize_prediction_rows,
    parse_source_content,
    summarize_preflight,
)
from core.protocol_hash import protocol_hash
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
    ClaimStatus,
    ClaimVersion,
    EvidencePacket,
    ResearchProject,
    ResearchDecision,
    ResearchRevision,
    ResearchTarget,
)
from core.research_validation import (
    ResearchValidationError,
    ValidationPlanner,
)


LEGACY_SESSION_BASE = Path(__file__).resolve().parent.parent / "runtime" / "live_session"


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
        data_source: Any | None = None,
        harness_project_root: Path | None = None,
        run_shap: bool = False,
        legacy_session_root: Path | None = None,
    ) -> None:
        self.repository = repository
        self.gateway = gateway
        self.data_source = data_source
        self.harness_project_root = harness_project_root
        self.run_shap = run_shap
        self.legacy_session_root = (
            legacy_session_root or LEGACY_SESSION_BASE
        ).resolve()
        self.validation_planner = ValidationPlanner(
            repository,
            gateway=gateway,
            data_source=data_source,
            harness_project_root=harness_project_root,
            run_shap=run_shap,
        )

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

    def import_legacy_round(
        self,
        research_id: str,
        expected_revision: int,
        payload: dict[str, Any],
    ) -> ApiResult:
        try:
            current = self.repository.load_revision(research_id)
        except ResearchProjectNotFoundError:
            raise ResearchApiError(
                "研究项目不存在。",
                status=404,
                code="project_not_found",
            )
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

        session_root = self._resolve_legacy_session(payload)
        round_number = self._resolve_legacy_round_number(session_root, payload)
        round_dir = session_root / "results" / f"round_{round_number:02d}"
        protocol_path = round_dir / "protocol.json"
        if not protocol_path.is_file():
            protocol_path = session_root / "config" / "latest_protocol.json"
        if not protocol_path.is_file():
            raise ResearchApiError(
                f"legacy 第 {round_number} 轮缺少 protocol.json，无法生成协议哈希。",
                status=422,
                code="legacy_protocol_missing",
            )
        try:
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ResearchApiError(
                f"legacy 协议文件不可读：{protocol_path.name}",
                status=422,
                code="legacy_protocol_invalid",
                details={"error": str(exc)},
            ) from exc

        experiment_id = str(
            payload.get("experiment_id") or protocol.get("experiment_id") or ""
        ).strip() or None
        baseline_path, treatment_path, resolved_experiment_id = (
            self._resolve_legacy_experiment(
                session_root,
                round_dir,
                experiment_id,
            )
        )
        try:
            horizon_days = int(
                payload.get("horizon_days")
                or protocol.get("forecast_horizon_days")
                or current.project.horizon_days
            )
        except (TypeError, ValueError):
            raise ResearchApiError(
                "legacy 协议中的 forecast_horizon_days 不是整数。",
                status=422,
                code="invalid_horizon",
            )

        protocol_hash_value = protocol_hash(protocol)
        prediction_csv = self._build_legacy_prediction_csv(
            baseline_path,
            treatment_path,
            experiment_id=resolved_experiment_id,
            horizon_days=horizon_days,
            protocol_hash_value=protocol_hash_value,
        )
        prediction_file_name = (
            f"legacy_round_{round_number:02d}_{resolved_experiment_id}_predictions.csv"
        )
        prediction_artifact = self._make_artifact_record(
            research_id,
            file_name=prediction_file_name,
            kind="prediction_csv",
            content=prediction_csv,
            source_kind="legacy_import",
        )
        protocol_artifact = self._make_artifact_record(
            research_id,
            file_name=f"legacy_round_{round_number:02d}_protocol.json",
            kind="protocol_json",
            content=json.dumps(protocol, ensure_ascii=False, indent=2),
            source_kind="legacy_import",
        )

        evidence_bundle = self._build_legacy_evidence_bundle(
            session_root,
            round_dir,
            round_number=round_number,
            experiment_id=resolved_experiment_id,
            protocol=protocol,
        )
        evidence_artifact = self._make_artifact_record(
            research_id,
            file_name=f"legacy_round_{round_number:02d}_evidence_bundle.json",
            kind="evidence_json",
            content=json.dumps(evidence_bundle, ensure_ascii=False, indent=2),
            source_kind="legacy_import",
        )

        target_label = str(
            payload.get("target_id") or protocol.get("target") or current.project.target.id
        ).strip()
        target_unit = _unit_for_target(target_label) or current.project.target.unit

        def apply_import(state: ResearchRevision) -> None:
            state.project.target = ResearchTarget(
                id=target_label,
                column=target_label,
                unit=target_unit,
                coordinate_system=current.project.target.coordinate_system
                or "GSE",
            )
            state.project.horizon_days = horizon_days
            state.artifacts.extend(
                [prediction_artifact, protocol_artifact, evidence_artifact]
            )

        next_state = self.repository.mutate(
            research_id,
            expected_revision=expected,
            mutation=apply_import,
        )
        result = self.preflight(
            research_id,
            expected_revision=next_state.revision,
            payload={
                "artifact_id": prediction_artifact.artifact_id,
                "protocol_hash_value": protocol_hash_value,
                "run_id": resolved_experiment_id,
                "horizon_days": horizon_days,
                "base_arm": "baseline",
                "aug_arm": "treatment",
                "feature_columns": _protocol_feature_columns(protocol),
            },
        )
        result.payload["data"]["legacy_import"] = {
            "session_name": session_root.name,
            "session_root": str(session_root),
            "round_number": round_number,
            "experiment_id": resolved_experiment_id,
            "protocol_hash": protocol_hash_value,
            "prediction_artifact_id": prediction_artifact.artifact_id,
            "protocol_artifact_id": protocol_artifact.artifact_id,
            "evidence_artifact_id": evidence_artifact.artifact_id,
            "preflight_status": result.payload["data"]["preflight"]["status"],
        }
        return ApiResult(status=200, payload=result.payload)

    def _resolve_legacy_session(self, payload: dict[str, Any]) -> Path:
        base = self.legacy_session_root
        if not base.is_dir():
            raise ResearchApiError(
                f"legacy 会话目录不存在：{base}",
                status=409,
                code="legacy_session_missing",
            )
        candidates: list[Path] = []
        if self._has_round_results(base):
            candidates.append(base)
        current = base / "current"
        if current.is_dir() and self._has_round_results(current):
            candidates.append(current)
        for child in sorted(
            base.iterdir(),
            key=lambda path: path.stat().st_mtime if path.is_dir() else 0,
            reverse=True,
        ):
            if (
                child.is_dir()
                and child.name != "current"
                and self._has_round_results(child)
            ):
                candidates.append(child)
        if not candidates:
            raise ResearchApiError(
                f"当前没有可导入的 legacy 科研会话：{base} 下未找到含轮次产物的目录。",
                status=409,
                code="legacy_round_missing",
            )
        requested = str(payload.get("session_name") or "").strip()
        if requested:
            chosen = next(
                (candidate for candidate in candidates if candidate.name == requested),
                None,
            )
            if chosen is None:
                raise ResearchApiError(
                    f"指定的 legacy 会话不存在或没有轮次产物：{requested}",
                    status=404,
                    code="legacy_session_not_found",
                )
            return chosen
        return candidates[0]

    @staticmethod
    def _has_round_results(session_root: Path) -> bool:
        results = session_root / "results"
        return results.is_dir() and any(
            child.is_dir() for child in results.glob("round_*")
        )

    @staticmethod
    def _resolve_legacy_round_number(
        session_root: Path,
        payload: dict[str, Any],
    ) -> int:
        raw_round = payload.get("round_number")
        results = session_root / "results"
        if raw_round is not None:
            try:
                round_number = int(raw_round)
            except (TypeError, ValueError):
                raise ResearchApiError(
                    "round_number 必须是整数。",
                    status=422,
                    code="invalid_round_number",
                )
            if round_number < 1:
                raise ResearchApiError(
                    "round_number 必须大于等于 1。",
                    status=422,
                    code="invalid_round_number",
                )
            round_dir = results / f"round_{round_number:02d}"
            if not round_dir.is_dir():
                raise ResearchApiError(
                    f"legacy 会话中不存在第 {round_number} 轮。",
                    status=404,
                    code="legacy_round_not_found",
                )
            return round_number
        matches: list[int] = []
        for path in results.glob("round_*"):
            if not path.is_dir():
                continue
            try:
                matches.append(int(path.name.removeprefix("round_")))
            except ValueError:
                continue
        if not matches:
            raise ResearchApiError(
                "legacy 会话没有可导入的轮次。",
                status=409,
                code="legacy_round_missing",
            )
        return max(matches)

    def _resolve_legacy_experiment(
        self,
        session_root: Path,
        round_dir: Path,
        experiment_id: str | None,
    ) -> tuple[Path, Path, str]:
        if experiment_id:
            experiment_dir = round_dir / experiment_id
            baseline_path, treatment_path = self._prediction_pair(experiment_dir)
            if baseline_path.is_file() and treatment_path.is_file():
                return baseline_path, treatment_path, experiment_id

        result_path = round_dir / "result_unified.json"
        if result_path.is_file():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                result = {}
            result_experiment_id = str(result.get("experiment_id") or "").strip()
            run_paths: dict[str, Path] = {}
            for run in result.get("runs") or []:
                run_id = str(run.get("run_id") or "").strip().lower()
                relative = str(run.get("prediction_artifact") or "").strip()
                if not run_id or not relative:
                    continue
                candidate = session_root / relative.lstrip("/\\")
                if not candidate.is_file():
                    candidate = round_dir / relative.lstrip("/\\")
                if candidate.is_file():
                    run_paths[run_id] = candidate
            baseline_path = run_paths.get("baseline")
            treatment_path = run_paths.get("treatment")
            if baseline_path and treatment_path and baseline_path.is_file() and treatment_path.is_file():
                return (
                    baseline_path,
                    treatment_path,
                    result_experiment_id or experiment_id or round_dir.name,
                )

        candidates: list[tuple[Path, Path]] = []
        for baseline in sorted(round_dir.glob("E_*/predictions/*_baseline.csv")):
            treatment = baseline.with_name(
                baseline.name.replace("_baseline.csv", "_treatment.csv")
            )
            if treatment.is_file():
                candidates.append((baseline, treatment))
        if not candidates:
            raise ResearchApiError(
                f"legacy 第 {round_dir.name} 轮没有可导入的 baseline/treatment 预测 CSV。",
                status=422,
                code="legacy_predictions_missing",
            )
        baseline_path, treatment_path = max(
            candidates,
            key=lambda pair: pair[0].stat().st_mtime,
        )
        return (
            baseline_path,
            treatment_path,
            baseline_path.parent.parent.name,
        )

    @staticmethod
    def _prediction_pair(experiment_dir: Path) -> tuple[Path, Path]:
        predictions = experiment_dir / "predictions"
        baselines = sorted(predictions.glob("*_baseline.csv")) if predictions.is_dir() else []
        if not baselines:
            return Path(""), Path("")
        baseline_path = baselines[0]
        treatment_path = baseline_path.with_name(
            baseline_path.name.replace("_baseline.csv", "_treatment.csv")
        )
        if not treatment_path.is_file():
            return Path(""), Path("")
        return baseline_path, treatment_path

    @staticmethod
    def _build_legacy_prediction_csv(
        baseline_path: Path,
        treatment_path: Path,
        *,
        experiment_id: str,
        horizon_days: int,
        protocol_hash_value: str,
    ) -> str:
        rows: list[dict[str, Any]] = []
        for arm, path in (
            ("baseline", baseline_path),
            ("treatment", treatment_path),
        ):
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                for raw in reader:
                    row = {str(key): value for key, value in raw.items()}
                    row["arm"] = arm
                    if not str(row.get("split") or "").strip():
                        row["split"] = "test"
                    if not str(row.get("run_id") or "").strip():
                        row["run_id"] = arm
                    if not str(row.get("horizon_days") or "").strip():
                        row["horizon_days"] = str(horizon_days)
                    if not str(row.get("protocol_hash") or "").strip():
                        row["protocol_hash"] = protocol_hash_value
                    target_time = str(
                        row.get("target_time") or row.get("time") or row.get("date") or ""
                    ).strip() or None
                    if target_time and not str(
                        row.get("target_time") or ""
                    ).strip():
                        row["target_time"] = target_time
                    if not str(row.get("forecast_origin") or "").strip() and target_time:
                        row["forecast_origin"] = _calendar_shift(
                            target_time,
                            -horizon_days,
                        )
                    rows.append(row)

        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        text = output.getvalue()
        if not text.strip():
            raise ResearchApiError(
                "legacy 预测文件为空，无法生成配对证据。",
                status=422,
                code="legacy_predictions_missing",
            )
        return text

    def _make_artifact_record(
        self,
        research_id: str,
        *,
        file_name: str,
        kind: str,
        content: str,
        source_kind: str,
    ) -> ArtifactRecord:
        text_content = (
            content.decode("utf-8", errors="replace")
            if isinstance(content, (bytes, bytearray))
            else content
        )
        parsed_source = parse_source_content(kind, text_content, file_name)
        artifact_id = "ART_" + uuid.uuid4().hex[:12].upper()
        stored_path = self.repository.store_artifact(
            research_id,
            artifact_id,
            text_content,
        )
        now = datetime.now()
        return ArtifactRecord(
            artifact_id=artifact_id,
            file_name=file_name,
            sha256=hashlib.sha256(text_content.encode("utf-8")).hexdigest(),
            size_bytes=len(text_content.encode("utf-8")),
            kind=kind,
            stored_path=str(stored_path),
            parsed=not bool(parsed_source.parse_errors),
            parse_errors=list(parsed_source.parse_errors),
            created_at=now,
            updated_at=now,
            source_kind=source_kind,
            schema_version=SCHEMA_VERSION,
        )

    @staticmethod
    def _build_legacy_evidence_bundle(
        session_root: Path,
        round_dir: Path,
        *,
        round_number: int,
        experiment_id: str,
        protocol: dict[str, Any],
    ) -> dict[str, Any]:
        def read_json(path: Path) -> Any:
            if not path.is_file():
                return None
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None

        evaluation = read_json(round_dir / "evaluation_unified.json")
        result = read_json(round_dir / "result_unified.json")
        hypothesis_tree = read_json(session_root / "state" / "hypothesis_tree.json")
        decision_log = read_json(session_root / "state" / "decision_log.json")
        task = read_json(session_root / "state" / "task.json")
        return {
            "bundle_kind": "legacy_round_import",
            "round_number": round_number,
            "experiment_id": experiment_id,
            "protocol": protocol,
            "evaluation": evaluation,
            "result": result,
            "hypothesis_tree": hypothesis_tree,
            "decision_log": decision_log,
            "task": task,
            "metrics": (evaluation or {}).get("metrics"),
            "source_files": {
                "protocol": str(round_dir / "protocol.json"),
                "evaluation": str(round_dir / "evaluation_unified.json"),
                "result": str(round_dir / "result_unified.json"),
                "hypothesis_tree": str(session_root / "state" / "hypothesis_tree.json"),
                "task": str(session_root / "state" / "task.json"),
            },
        }

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

    def plan_validations(
        self,
        research_id: str,
        expected_revision: int,
        payload: dict[str, Any],
    ) -> ApiResult:
        result = self.validation_planner.plan_batch(
            research_id=research_id,
            expected_revision=expected_revision,
            payload=payload,
        )
        return self._validation_plan_response(
            result.revision,
            batch_id=result.batch_id,
            is_stub=result.is_stub,
            candidate_ids=result.candidate_ids,
            message=result.message,
        )

    def approve_protocol(
        self,
        research_id: str,
        expected_revision: int,
        payload: dict[str, Any],
    ) -> ApiResult:
        revision = self.validation_planner.approve(
            research_id=research_id,
            expected_revision=expected_revision,
            payload=payload,
        )
        candidate = self._find_candidate(revision, payload)
        decision = (
            revision.decisions[-1]
            if revision.decisions
            and revision.decisions[-1].action == "approve_protocol"
            else None
        )
        result = self._project_response(revision, created=False)
        result.payload["data"]["approval"] = {
            "candidate_id": candidate.experiment_id if candidate else None,
            "protocol_hash": payload.get("protocol_hash"),
            "status": candidate.status if candidate else None,
            "decision": decision.model_dump(exclude_none=True) if decision else None,
        }
        return result

    def execute_protocol(
        self,
        research_id: str,
        expected_revision: int,
        payload: dict[str, Any],
    ) -> ApiResult:
        revision = self.validation_planner.execute(
            research_id=research_id,
            expected_revision=expected_revision,
            payload=payload,
        )
        candidate = self._find_candidate(revision, payload)
        jobs = [
            job
            for job in revision.jobs
            if job.kind == "validation_execution"
            and (candidate is None or job.review_id in candidate.dispute_ids)
        ]
        result = self._project_response(revision, created=False)
        result.payload["data"]["execution"] = {
            "candidate_id": candidate.experiment_id if candidate else None,
            "protocol_hash": payload.get("protocol_hash"),
            "status": candidate.status if candidate else None,
            "job": jobs[-1].model_dump(exclude_none=True) if jobs else None,
        }
        return result

    @staticmethod
    def _find_candidate(
        revision: ResearchRevision,
        payload: dict[str, Any],
    ) -> Any | None:
        protocol_hash_value = str(payload.get("protocol_hash") or "").strip()
        candidate_id = str(payload.get("candidate_id") or "").strip()
        for candidate in revision.candidates:
            if candidate.protocol_hash != protocol_hash_value:
                continue
            if candidate_id and candidate.experiment_id != candidate_id:
                continue
            return candidate
        return None

    def list_claim_versions(
        self,
        research_id: str,
        claim_id: str,
    ) -> ApiResult:
        try:
            revision = self.repository.load_revision(research_id)
        except ResearchProjectNotFoundError:
            raise ResearchApiError("研究项目不存在。", status=404, code="project_not_found")
        versions = [
            claim
            for claim in revision.claims
            if claim.claim_id == claim_id
        ]
        if not versions:
            raise ResearchApiError(
                f"研究结论不存在：{claim_id}",
                status=404,
                code="claim_not_found",
            )
        return ApiResult(
            status=200,
            payload={
                "research_id": research_id,
                "revision": revision.revision,
                "data": {
                    "claim_id": claim_id,
                    "versions": [
                        version.model_dump(exclude_none=True)
                        for version in versions
                    ],
                    "current_version": versions[-1].version,
                    "version_count": len(versions),
                },
            },
        )

    def create_claim_version(
        self,
        research_id: str,
        expected_revision: int,
        claim_id: str,
        payload: dict[str, Any],
    ) -> ApiResult:
        try:
            current = self.repository.load_revision(research_id)
        except ResearchProjectNotFoundError:
            raise ResearchApiError("研究项目不存在。", status=404, code="project_not_found")
        self._require_revision(current.revision, expected_revision)

        versions = [
            claim
            for claim in current.claims
            if claim.claim_id == claim_id
        ]
        if not versions:
            raise ResearchApiError(
                f"研究结论不存在：{claim_id}",
                status=404,
                code="claim_not_found",
            )
        current_version = versions[-1]

        evidence_ids: list[str]
        raw_evidence = payload.get("evidence_ids")
        if raw_evidence is None:
            evidence_ids = list(current_version.evidence_ids)
        else:
            if not isinstance(raw_evidence, (list, tuple)):
                raise ResearchApiError(
                    "evidence_ids 必须是字符串数组。",
                    status=422,
                    code="invalid_evidence_ids",
                )
            evidence_ids = [str(item) for item in raw_evidence]
            packet_ids = {packet.packet_id for packet in current.evidence_packets}
            unknown = [item for item in evidence_ids if item not in packet_ids]
            if unknown:
                raise ResearchApiError(
                    f"证据包引用不存在：{', '.join(unknown)}",
                    status=422,
                    code="unknown_evidence_packet",
                )

        statement = payload.get("statement")
        if statement is None:
            statement = current_version.statement
        else:
            statement = str(statement).strip()
            if not statement:
                raise ResearchApiError(
                    "statement 不能为空。",
                    status=422,
                    code="claim_statement_required",
                )

        revision_note = str(payload.get("revision_note") or "").strip()
        if not revision_note:
            raise ResearchApiError(
                "revision_note 不能为空：结论版本必须说明修改理由。",
                status=422,
                code="revision_note_required",
            )

        raw_scope = payload.get("scope")
        if raw_scope is None:
            scope = current_version.scope.model_copy(deep=True)
        else:
            if not isinstance(raw_scope, dict):
                raise ResearchApiError(
                    "scope 必须是对象。",
                    status=422,
                    code="invalid_claim_scope",
                )
            scope_data = current_version.scope.model_dump(exclude_none=True)
            scope_data.update(raw_scope)
            try:
                scope = ClaimScope.model_validate(scope_data)
            except Exception as exc:
                raise ResearchApiError(
                    f"scope 字段不合法：{exc}",
                    status=422,
                    code="invalid_claim_scope",
                ) from exc

        raw_status = payload.get("status")
        if raw_status is None:
            status_value = current_version.status
        else:
            status_value = str(raw_status)
            if status_value not in ClaimStatus.__args__:
                raise ResearchApiError(
                    f"status 不在白名单内：{status_value}",
                    status=422,
                    code="invalid_claim_status",
                )

        program_recommendation = payload.get("program_recommendation")
        if program_recommendation is not None:
            program_recommendation = str(program_recommendation)
            if program_recommendation not in ClaimStatus.__args__:
                raise ResearchApiError(
                    f"program_recommendation 不在白名单内：{program_recommendation}",
                    status=422,
                    code="invalid_claim_status",
                )

        human_decision = payload.get("human_decision")
        if human_decision is not None:
            human_decision = str(human_decision).strip() or None
        next_stage = payload.get("next_stage")
        if next_stage is not None:
            next_stage = str(next_stage).strip() or None

        new_version_number = max(version.version for version in versions) + 1
        decision = ResearchDecision(
            decision_id="DEC_" + uuid.uuid4().hex[:12].upper(),
            actor_type="human_pi",
            action="update_claim_version",
            rationale=revision_note,
            claim_version=new_version_number,
            protocol_hash=current_version.scope.baseline_protocol_id,
            next_stage=next_stage,
        )
        new_version = ClaimVersion(
            claim_id=claim_id,
            version=new_version_number,
            statement=statement,
            scope=scope,
            evidence_ids=evidence_ids,
            status=status_value,
            parent_version=current_version.version,
            decision_id=decision.decision_id,
            program_recommendation=program_recommendation,
            human_decision=human_decision,
            revision_note=revision_note,
        )

        def apply_version(state: ResearchRevision) -> None:
            state.claims.append(new_version)
            state.decisions.append(decision)
            if next_stage:
                state.project.stage = next_stage

        next_state = self.repository.mutate(
            research_id,
            expected_revision=current.revision,
            mutation=apply_version,
        )
        result = self._project_response(next_state, created=False)
        result.payload["data"]["claim_version"] = new_version.model_dump(exclude_none=True)
        result.payload["data"]["decision"] = decision.model_dump(exclude_none=True)
        return result

    def export_report(self, research_id: str) -> ApiResult:
        try:
            revision = self.repository.load_revision(research_id)
        except ResearchProjectNotFoundError:
            raise ResearchApiError("研究项目不存在。", status=404, code="project_not_found")

        claim_versions_by_id: dict[str, list[ClaimVersion]] = {}
        for claim in revision.claims:
            claim_versions_by_id.setdefault(claim.claim_id, []).append(claim)
        current_claims = [
            claim_versions[-1]
            for claim_versions in claim_versions_by_id.values()
            if claim_versions
        ]
        evidence_locators: dict[str, dict[str, Any]] = {}
        for packet in revision.evidence_packets:
            for item in packet.evidence_items:
                evidence_locators[item.evidence_id] = {
                    "evidence_id": item.evidence_id,
                    "packet_id": packet.packet_id,
                    "packet_hash": packet.packet_hash,
                    "artifact_id": item.artifact_id,
                    "artifact_sha256": item.artifact_sha256,
                    "locator": item.locator,
                    "value_type": item.value_type,
                    "value": item.value,
                    "unit": item.unit,
                    "provenance": item.provenance,
                }
            for metric_key, metric in packet.metrics.items():
                if metric.metric_ref and metric.metric_ref not in evidence_locators:
                    evidence_locators[metric.metric_ref] = {
                        "metric_key": metric_key,
                        "packet_id": packet.packet_id,
                        "packet_hash": packet.packet_hash,
                        "value": metric.value,
                        "unit": metric.unit,
                        "reason": metric.reason,
                    }

        unresolved_disputes = [
            dispute
            for dispute in revision.disputes
            if dispute.status not in {"dismissed", "validated", "superseded"}
        ]
        unresolved_claims = [
            claim
            for claim in current_claims
            if claim.status in {"unassessed", "insufficient", "mixed"}
        ]
        failed_jobs = [
            job
            for job in revision.jobs
            if job.status in {"failed", "expired", "cancelled"}
        ]
        packet_limitations = [
            {
                "packet_id": packet.packet_id,
                "packet_hash": packet.packet_hash,
                "limitations": list(packet.limitations),
            }
            for packet in revision.evidence_packets
            if packet.limitations
        ]

        agent_calls = [
            call.model_dump(exclude_none=True)
            for call in revision.agent_calls
        ]
        model_usage: dict[str, Any] = {
            "total_calls": len(agent_calls),
            "succeeded_calls": sum(
                1 for call in agent_calls if call.get("status") == "succeeded"
            ),
            "failed_calls": sum(
                1 for call in agent_calls if call.get("status") == "failed"
            ),
            "unfinished_calls": sum(
                1
                for call in agent_calls
                if call.get("status") in {"queued", "running", "cancelled"}
            ),
            "stub_opinions": len(
                [op for op in revision.opinions if op.is_stub]
            ),
            "stub_arbitrations": len(
                [item for item in revision.arbitrations if item.is_stub]
            ),
            "stub_candidates": len(
                [item for item in revision.candidates if item.is_stub]
            ),
            "roles": {},
        }
        for call in agent_calls:
            role = str(call.get("role") or "other")
            bucket = model_usage["roles"].setdefault(
                role,
                {"total": 0, "succeeded": 0, "failed": 0},
            )
            bucket["total"] += 1
            if call.get("status") == "succeeded":
                bucket["succeeded"] += 1
            elif call.get("status") == "failed":
                bucket["failed"] += 1

        report_payload = {
            "report_kind": "research_review_report",
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now().isoformat(),
            "research_id": research_id,
            "revision": revision.revision,
            "stage": revision.project.stage,
            "project": revision.project.model_dump(exclude_none=True),
            "claims": [
                claim.model_dump(exclude_none=True)
                for claim in current_claims
            ],
            "claim_versions": [
                claim.model_dump(exclude_none=True)
                for claim in revision.claims
            ],
            "versions_by_claim": {
                claim_id: [
                    version.model_dump(exclude_none=True)
                    for version in claim_versions
                ]
                for claim_id, claim_versions in claim_versions_by_id.items()
            },
            "evidence_locators": evidence_locators,
            "protocols": [
                protocol.model_dump(exclude_none=True)
                for protocol in revision.protocols
            ],
            "artifacts": [
                artifact.model_dump(exclude_none=True)
                for artifact in revision.artifacts
            ],
            "evidence_packets": [
                packet.model_dump(exclude_none=True)
                for packet in revision.evidence_packets
            ],
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
            "candidates": [
                candidate.model_dump(exclude_none=True)
                for candidate in revision.candidates
            ],
            "decisions": [
                decision.model_dump(exclude_none=True)
                for decision in revision.decisions
            ],
            "jobs": [
                job.model_dump(exclude_none=True)
                for job in revision.jobs
            ],
            "agent_calls": agent_calls,
            "unresolved_issues": {
                "open_disputes": [
                    dispute.model_dump(exclude_none=True)
                    for dispute in unresolved_disputes
                ],
                "unresolved_claims": [
                    claim.model_dump(exclude_none=True)
                    for claim in unresolved_claims
                ],
                "packet_limitations": packet_limitations,
                "failed_jobs": [
                    job.model_dump(exclude_none=True)
                    for job in failed_jobs
                ],
            },
            "model_usage": model_usage,
            "current_confirmed_versions": [
                claim.model_dump(exclude_none=True)
                for claim in current_claims
            ],
        }
        return ApiResult(
            status=200,
            payload={
                "research_id": research_id,
                "revision": revision.revision,
                "data": report_payload,
            },
        )

    def _validation_plan_response(
        self,
        revision: ResearchRevision,
        *,
        batch_id: str,
        is_stub: bool,
        candidate_ids: list[str],
        message: str,
    ) -> ApiResult:
        result = self._project_response(revision, created=False)
        selected = [
            candidate
            for candidate in revision.candidates
            if candidate.experiment_id in set(candidate_ids)
        ]
        selected_hashes = {candidate.protocol_hash for candidate in selected}
        result.payload["data"]["validation_plan"] = {
            "batch_id": batch_id,
            "is_stub": is_stub,
            "candidate_ids": candidate_ids,
            "message": message,
            "candidates": [
                candidate.model_dump(exclude_none=True)
                for candidate in selected
            ],
            "protocols": [
                protocol.model_dump(exclude_none=True)
                for protocol in revision.protocols
                if protocol.protocol_hash in selected_hashes
            ],
        }
        return result

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
        protocols = [
            protocol.model_dump(exclude_none=True)
            for protocol in revision.protocols
        ]
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
            "protocols": protocols,
            "candidates": [
                candidate.model_dump(exclude_none=True)
                for candidate in revision.candidates
            ],
            "decisions": [
                decision.model_dump(exclude_none=True)
                for decision in revision.decisions
            ],
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


def _unit_for_target(target_label: str) -> str | None:
    label = str(target_label or "").lower()
    if "speed" in label:
        return "km/s"
    if "density" in label:
        return "n/cm^3"
    if "temperature" in label:
        return "K"
    if any(token in label for token in ("by", "bz", "bx", "scalar b", "imf", "磁场")):
        return "nT"
    return None


def _protocol_feature_columns(protocol: dict[str, Any]) -> list[str]:
    features = protocol.get("features") or {}
    control = [str(item).strip() for item in features.get("control") or []]
    treatment = [str(item).strip() for item in features.get("treatment") or []]
    seen: set[str] = set()
    columns: list[str] = []
    for item in control + treatment:
        if item and item not in seen:
            seen.add(item)
            columns.append(item)
    return columns


def _calendar_shift(value: Any, days: int) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        original = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    shifted = original + timedelta(days=int(days))
    if (
        original.hour == 0
        and original.minute == 0
        and original.second == 0
        and original.microsecond == 0
    ):
        return shifted.date().isoformat()
    return shifted.isoformat()
