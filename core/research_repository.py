from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from core.research_schema import (
    SCHEMA_VERSION,
    ResearchProject,
    ResearchRevision,
)


RESEARCH_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
REVISION_DIR_PATTERN = re.compile(r"^\d{4}$")


class ResearchVersionConflictError(RuntimeError):
    pass


class ResearchIdempotencyConflictError(RuntimeError):
    pass


class ResearchProjectNotFoundError(KeyError):
    pass


@dataclass(frozen=True)
class IdempotencyRecord:
    key: str
    request_hash: str
    status: int
    payload: dict[str, Any]
    created_at: datetime


class ResearchRepository:
    """Versioned, project-isolated storage under ``runtime/research``.

    A project owns an independent directory. API-level mutation takes an
    ``expected_revision`` and writes a new immutable revision; background
    products commit through the same repository and never replace a newer
    revision. Idempotency keys are persisted so browser retries cannot cause
    duplicate research objects.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or Path(__file__).resolve().parent.parent / "runtime" / "research").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.RLock] = {}
        self._guard = threading.Lock()

    def _lock_for(self, research_id: str) -> threading.RLock:
        with self._guard:
            return self._locks.setdefault(research_id, threading.RLock())

    def validate_research_id(self, research_id: str) -> str:
        value = str(research_id).strip()
        if not RESEARCH_ID_PATTERN.fullmatch(value):
            raise ValueError("research_id 仅允许字母、数字、下划线和短横线。")
        return value

    def project_dir(self, research_id: str) -> Path:
        return self.root / self.validate_research_id(research_id)

    def revision_dir(self, research_id: str, revision: int) -> Path:
        return self.project_dir(research_id) / "revisions" / f"{int(revision):04d}"

    def _manifest_path(self, research_id: str) -> Path:
        return self.project_dir(research_id) / "manifest.json"

    def _revision_path(self, research_id: str, revision: int) -> Path:
        return self.revision_dir(research_id, revision) / "revision.json"

    def exists(self, research_id: str) -> bool:
        try:
            return self._manifest_path(research_id).is_file()
        except ValueError:
            return False

    def current_revision(self, research_id: str) -> int:
        manifest = self._read_manifest(research_id)
        return int(manifest["current_revision"])

    def _read_manifest(self, research_id: str) -> dict[str, Any]:
        path = self._manifest_path(research_id)
        if not path.is_file():
            raise ResearchProjectNotFoundError(research_id)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"项目清单损坏：{research_id}") from exc

    def _write_manifest(self, research_id: str, revision: int) -> None:
        manifest = {
            "research_id": research_id,
            "schema_version": SCHEMA_VERSION,
            "current_revision": revision,
            "updated_at": datetime.now().isoformat(),
        }
        self._atomic_write_json(self._manifest_path(research_id), manifest)

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        os.replace(tmp, path)

    @staticmethod
    def canonical_payload(payload: dict[str, Any] | list[Any]) -> str:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)

    @staticmethod
    def request_hash(payload: dict[str, Any]) -> str:
        return hashlib.sha256(ResearchRepository.canonical_payload(payload).encode("utf-8")).hexdigest()

    def create_project(self, project: ResearchProject) -> ResearchRevision:
        research_id = self.validate_research_id(project.research_id)
        with self._lock_for(research_id):
            if self.exists(research_id):
                raise RuntimeError(f"研究项目已存在：{research_id}")
            project_dir = self.project_dir(research_id)
            project_dir.mkdir(parents=True, exist_ok=True)
            revision = ResearchRevision(revision=1, project=project)
            self._save_revision_locked(research_id, revision)
            self._write_manifest(research_id, 1)
            return revision

    def load_project(self, research_id: str) -> ResearchProject:
        return self.load_revision(research_id).project

    def load_revision(self, research_id: str, revision: int | None = None) -> ResearchRevision:
        research_id = self.validate_research_id(research_id)
        with self._lock_for(research_id):
            target = self.current_revision(research_id) if revision is None else int(revision)
            path = self._revision_path(research_id, target)
            if not path.is_file():
                raise ResearchProjectNotFoundError(research_id)
            try:
                return ResearchRevision.model_validate(json.loads(path.read_text(encoding="utf-8")))
            except Exception as exc:
                raise RuntimeError(f"修订版本不可读：{research_id}@{target}") from exc

    def artifact_path(self, research_id: str, artifact_id: str) -> Path:
        research_id = self.validate_research_id(research_id)
        artifact_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(artifact_id))
        return self.project_dir(research_id) / "data" / "artifacts" / f"{artifact_id}.artifact"

    def store_artifact(self, research_id: str, artifact_id: str, content: str | bytes) -> Path:
        path = self.artifact_path(research_id, artifact_id)
        payload = content.encode("utf-8") if isinstance(content, str) else bytes(content)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
        tmp.write_bytes(payload)
        os.replace(tmp, path)
        return path

    def read_artifact(self, research_id: str, artifact_id: str) -> bytes:
        path = self.artifact_path(research_id, artifact_id)
        if not path.is_file():
            raise ResearchProjectNotFoundError(artifact_id)
        return path.read_bytes()

    def list_projects(self) -> list[ResearchProject]:
        projects: list[ResearchProject] = []
        for manifest_path in sorted(self.root.glob("*/manifest.json")):
            research_id = manifest_path.parent.name
            if not RESEARCH_ID_PATTERN.fullmatch(research_id):
                continue
            try:
                projects.append(self.load_project(research_id))
            except Exception:
                continue
        return sorted(projects, key=lambda item: item.updated_at, reverse=True)

    def mutate(
        self,
        research_id: str,
        *,
        expected_revision: int,
        mutation: Callable[[ResearchRevision], None],
    ) -> ResearchRevision:
        research_id = self.validate_research_id(research_id)
        with self._lock_for(research_id):
            current = self.load_revision(research_id)
            if current.revision != int(expected_revision):
                raise ResearchVersionConflictError(
                    f"版本冲突：当前 revision={current.revision}，请求 expected_revision={expected_revision}"
                )
            next_state = current.model_copy(deep=True)
            mutation(next_state)
            next_state.revision = current.revision + 1
            next_state.project.revision = next_state.revision
            next_state.updated_at = datetime.now()
            self._save_revision_locked(research_id, next_state)
            self._write_manifest(research_id, next_state.revision)
            return next_state

    def _save_revision_locked(self, research_id: str, revision: ResearchRevision) -> None:
        path = self._revision_path(research_id, revision.revision)
        self._atomic_write_json(path, json.loads(revision.model_dump_json(exclude_none=True)))

    def read_idempotent(self, key: str) -> IdempotencyRecord | None:
        key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
        path = self.root / "idempotency" / f"{key_hash}.json"
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return IdempotencyRecord(**payload)
        except Exception:
            return None

    def save_idempotent(
        self,
        key: str,
        *,
        request_hash: str,
        status: int,
        payload: dict[str, Any],
    ) -> None:
        created_at = datetime.now()
        record = IdempotencyRecord(
            key=key,
            request_hash=request_hash,
            status=int(status),
            payload=payload,
            created_at=created_at,
        )
        key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
        path = self.root / "idempotency" / f"{key_hash}.json"
        self._atomic_write_json(path, {
            "key": record.key,
            "request_hash": record.request_hash,
            "status": record.status,
            "payload": record.payload,
            "created_at": record.created_at.isoformat(),
        })
