export type ResearchWorkflowMode = 'legacy' | 'research_review'
export type ResearchDataMode = 'authorized_data' | 'synthetic_fixture' | 'historical_replay' | 'unknown'
export type ResearchEntryMode = 'new_question' | 'import_claim'
export type ArtifactKind = 'prediction_csv' | 'protocol_json' | 'data_csv' | 'evidence_json' | 'report' | 'other'
export type CheckStatus = 'pass' | 'fail' | 'unknown' | 'not_applicable'
export type ClaimStatus =
  | 'unassessed'
  | 'insufficient'
  | 'supported_in_scope'
  | 'not_supported_in_scope'
  | 'mixed'
  | 'retracted'

export interface ResearchTarget {
  id: string
  column?: string
  unit?: string
  coordinate_system?: string
}

export interface CreateResearchPayload {
  workflow_mode: ResearchWorkflowMode
  entry_mode: ResearchEntryMode
  data_mode: ResearchDataMode
  title: string
  question: string
  target: ResearchTarget
  horizon_days: number
  summary?: string
  claim_statement?: string
  claim_scope?: {
    sample_condition?: string
    evaluation_period?: Record<string, string>
    baseline_protocol_id?: string
  }
}

export interface ResearchProjectSummary {
  research_id: string
  title: string
  question: string
  target: ResearchTarget
  horizon_days: number
  data_mode: ResearchDataMode
  workflow_mode: ResearchWorkflowMode
  stage: string
  revision: number
  updated_at: string
}

export interface ResearchProjectDetail {
  project: ResearchProjectSummary & {
    entry_mode: ResearchEntryMode
    summary?: string
    created_at: string
    schema_version: string
  }
  revision: number
  claims: ClaimVersion[]
  artifacts: ArtifactRecord[]
  evidence_packets: EvidencePacket[]
  checks: EvidenceCheck[]
  metrics: Record<string, MetricValue>
  pc_stage: string
  claim_count: number
  evidence_packet_count: number
  open_dispute_count: number
  awaiting_decision: boolean
}

export interface ClaimScope {
  target_id?: string
  unit?: string
  coordinate_system?: string
  horizon_days?: number
  evaluation_period?: Record<string, string>
  sample_condition?: string
  baseline_protocol_id?: string
}

export interface ClaimVersion {
  claim_id: string
  version: number
  statement: string
  scope?: ClaimScope
  evidence_ids: string[]
  status: ClaimStatus
  parent_version?: number
  decision_id?: string
  program_recommendation?: ClaimStatus
  human_decision?: string
  revision_note?: string
  created_at: string
  updated_at: string
}

export interface ArtifactRecord {
  artifact_id: string
  file_name: string
  sha256?: string
  size_bytes?: number
  kind: ArtifactKind
  stored_path: string
  content_type?: string
  parsed: boolean
  parse_errors: string[]
  created_at: string
  updated_at: string
}

export interface EvidenceCheck {
  code: string
  status: CheckStatus
  detail?: string
  reason?: string
  blocking: boolean
  packet_id?: string
}

export interface MetricValue {
  value: number | null
  unit?: string
  reason?: string
  metric_ref?: string
  packet_id?: string
}

export interface EvidenceItem {
  evidence_id: string
  artifact_id?: string
  artifact_sha256?: string
  locator?: string
  value_type: string
  value?: number | string | null
  unit?: string
  provenance?: string
}

export interface EvidencePacket {
  packet_id: string
  packet_hash: string
  run_id?: string
  protocol_hash?: string
  data_manifest_hash?: string
  claim_ids: string[]
  evidence_items: EvidenceItem[]
  checks: EvidenceCheck[]
  metrics: Record<string, MetricValue>
  limitations: string[]
  created_at: string
  updated_at: string
}

export interface PreflightBlockingIssue {
  code: string
  status: CheckStatus
  detail?: string
  reason?: string
}

export interface PreflightResult {
  status: 'ready' | 'needs_input'
  packet: EvidencePacket
  summary: {
    row_count: number
    arm_values: string[]
    columns: string[]
    missing_columns: string[]
    non_numeric_count: number
    protocol_hash?: string
    parse_errors: string[]
  }
  blocking_issues: PreflightBlockingIssue[]
}

export type ResearchProjectView = ResearchProjectDetail

export interface V2ApiEnvelope<T> {
  data?: T
  research_id?: string
  revision?: number
  request_id: string
  code?: string
  message?: string
  retryable?: boolean
  details?: Record<string, unknown>
}

export class ResearchApiError extends Error {
  code: string
  retryable: boolean
  details: Record<string, unknown>

  constructor(message: string, code: string, retryable: boolean, details: Record<string, unknown> = {}) {
    super(message)
    this.name = 'ResearchApiError'
    this.code = code
    this.retryable = retryable
    this.details = details
  }
}
