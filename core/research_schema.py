from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


SCHEMA_VERSION = "research-review/1.0"

WorkflowMode = Literal["legacy", "research_review"]
DataMode = Literal["authorized_data", "synthetic_fixture", "historical_replay", "unknown"]
EntryMode = Literal["new_question", "import_claim"]
ClaimStatus = Literal[
    "unassessed",
    "insufficient",
    "supported_in_scope",
    "not_supported_in_scope",
    "mixed",
    "retracted",
]
ReviewRole = Literal["scientific_questioner", "scientific_interpreter", "central_arbitrator"]
CheckStatus = Literal["pass", "fail", "unknown", "not_applicable"]
ReviewStance = Literal["support", "challenge", "mixed", "unknown"]
DisputeStatus = Literal["open", "proposed_validation", "validated", "dismissed", "superseded"]
JobStatus = Literal["queued", "running", "completed", "failed", "cancelled", "expired"]
ArtifactKind = Literal["prediction_csv", "protocol_json", "data_csv", "evidence_json", "report", "other"]


class ResearchBaseModel(BaseModel):
    """Shared base for all research-review persisted objects."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    schema_version: str = SCHEMA_VERSION
    source_kind: str = "research_review_service"
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def to_pretty_json(self) -> str:
        return self.model_dump_json(indent=2, exclude_none=True)


class ResearchTarget(ResearchBaseModel):
    id: str = Field(min_length=1)
    column: str | None = None
    unit: str | None = None
    coordinate_system: str | None = None


class ClaimScope(ResearchBaseModel):
    target_id: str | None = None
    unit: str | None = None
    coordinate_system: str | None = None
    horizon_days: int | None = Field(default=None, ge=1, le=30)
    evaluation_period: dict[str, str] | None = None
    sample_condition: str | None = None
    baseline_protocol_id: str | None = None


class ResearchProject(ResearchBaseModel):
    research_id: str = Field(min_length=1)
    revision: int = Field(default=1, ge=1)
    title: str = Field(min_length=1)
    question: str = Field(min_length=1)
    target: ResearchTarget
    horizon_days: int = Field(default=3, ge=1, le=30)
    data_mode: DataMode = "unknown"
    workflow_mode: WorkflowMode = "research_review"
    entry_mode: EntryMode = "new_question"
    stage: str = "draft"
    summary: str | None = None


class ClaimVersion(ResearchBaseModel):
    claim_id: str = Field(min_length=1)
    version: int = Field(default=1, ge=1)
    statement: str = Field(min_length=1)
    scope: ClaimScope = Field(default_factory=ClaimScope)
    evidence_ids: list[str] = Field(default_factory=list)
    status: ClaimStatus = "unassessed"
    parent_version: int | None = None
    decision_id: str | None = None
    program_recommendation: ClaimStatus | None = None
    human_decision: str | None = None
    revision_note: str | None = None


class ProtocolTimeSpec(ResearchBaseModel):
    calendar_offsets: bool = True
    forecast_origin_column: str | None = None
    target_time_column: str | None = None
    availability_assumption: str | None = None


class ProtocolSnapshot(ResearchBaseModel):
    protocol_id: str = Field(min_length=1)
    version: int = Field(default=1, ge=1)
    protocol_hash: str = Field(min_length=1)
    target: ResearchTarget
    time_spec: ProtocolTimeSpec = Field(default_factory=ProtocolTimeSpec)
    sample_plan: dict[str, Any] = Field(default_factory=dict)
    feature_arms: list[dict[str, Any]] = Field(default_factory=list)
    split_plan: dict[str, Any] = Field(default_factory=dict)
    method: str = "ridge"
    checks: list[str] = Field(default_factory=list)


class ArtifactRecord(ResearchBaseModel):
    artifact_id: str = Field(min_length=1)
    file_name: str = Field(min_length=1)
    sha256: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    kind: ArtifactKind = "other"
    stored_path: str = Field(min_length=1)
    content_type: str | None = None
    parsed: bool = False
    parse_errors: list[str] = Field(default_factory=list)


class EvidenceItem(ResearchBaseModel):
    evidence_id: str = Field(min_length=1)
    artifact_id: str | None = None
    artifact_sha256: str | None = None
    locator: str | None = None
    value_type: str = "unknown"
    value: float | int | str | None = None
    unit: str | None = None
    provenance: str | None = None


class EvidenceCheck(ResearchBaseModel):
    code: str = Field(min_length=1)
    status: CheckStatus = "unknown"
    detail: str | None = None
    reason: str | None = None
    blocking: bool = False


class MetricValue(ResearchBaseModel):
    value: float | int | None = None
    unit: str | None = None
    reason: str | None = None
    metric_ref: str | None = None


class EvidencePacket(ResearchBaseModel):
    packet_id: str = Field(min_length=1)
    packet_hash: str = Field(min_length=1)
    run_id: str | None = None
    protocol_hash: str | None = None
    data_manifest_hash: str | None = None
    claim_ids: list[str] = Field(default_factory=list)
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    checks: list[EvidenceCheck] = Field(default_factory=list)
    metrics: dict[str, MetricValue] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)


class ReviewOpinion(ResearchBaseModel):
    opinion_id: str = Field(min_length=1)
    role: ReviewRole
    packet_hash: str = Field(min_length=1)
    claim_id: str | None = None
    stance: ReviewStance = "unknown"
    observations: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    metric_refs: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    validation_proposals: list[str] = Field(default_factory=list)
    status: Literal["queued", "running", "completed", "failed", "partial"] = "queued"
    error_code: str | None = None
    error_message: str | None = None


class Dispute(ResearchBaseModel):
    dispute_id: str = Field(min_length=1)
    claim_id: str | None = None
    question: str = Field(min_length=1)
    issue_code: str = "OPINION_CONFLICT"
    scope: dict[str, Any] = Field(default_factory=dict)
    opinion_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    status: DisputeStatus = "open"
    validation_ids: list[str] = Field(default_factory=list)


class ValidationEstimate(ResearchBaseModel):
    component: str = Field(min_length=1)
    value: float | None = None
    unit: str | None = None
    calculation_version: str = "program/1.0"
    basis: str | None = None


class ValidationCandidate(ResearchBaseModel):
    experiment_id: str = Field(min_length=1)
    dispute_ids: list[str] = Field(default_factory=list)
    hypothesis_ids: list[str] = Field(default_factory=list)
    template_id: str = Field(min_length=1)
    design: dict[str, Any] = Field(default_factory=dict)
    estimates: list[ValidationEstimate] = Field(default_factory=list)
    approval_required: bool = True
    status: Literal["draft", "pending_approval", "approved", "executed", "failed", "superseded"] = "draft"
    protocol_hash: str | None = None


class ResearchDecision(ResearchBaseModel):
    decision_id: str = Field(min_length=1)
    actor_type: Literal["human_pi", "program", "llm_suggestion"] = "program"
    action: str = Field(min_length=1)
    rationale: str | None = None
    claim_version: int | None = None
    protocol_hash: str | None = None
    next_stage: str | None = None


class AgentCallRecord(ResearchBaseModel):
    call_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    model_id: str = "unknown"
    request_hash: str | None = None
    attempt: int = Field(default=1, ge=1)
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"] = "queued"
    latency_ms: int | None = Field(default=None, ge=0)
    usage: dict[str, int] = Field(default_factory=dict)
    error_code: str | None = None
    cache_hit: bool = False


class JobRecord(ResearchBaseModel):
    job_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    status: JobStatus = "queued"
    stage: str = "queued"
    research_id: str = Field(min_length=1)
    request_hash: str | None = None
    error_code: str | None = None
    message: str | None = None
    result_ref: str | None = None


class ResearchRevision(ResearchBaseModel):
    revision: int = Field(ge=1)
    project: ResearchProject
    claims: list[ClaimVersion] = Field(default_factory=list)
    protocols: list[ProtocolSnapshot] = Field(default_factory=list)
    artifacts: list[ArtifactRecord] = Field(default_factory=list)
    evidence_packets: list[EvidencePacket] = Field(default_factory=list)
    opinions: list[ReviewOpinion] = Field(default_factory=list)
    disputes: list[Dispute] = Field(default_factory=list)
    candidates: list[ValidationCandidate] = Field(default_factory=list)
    decisions: list[ResearchDecision] = Field(default_factory=list)
    jobs: list[JobRecord] = Field(default_factory=list)
    agent_calls: list[AgentCallRecord] = Field(default_factory=list)

