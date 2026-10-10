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
  disputes?: ResearchDispute[]
  arbitrations?: ResearchArbitration[]
  candidates?: ValidationCandidate[]
  decisions?: ResearchDecision[]
  jobs?: JobRecord[]
  agent_calls?: AgentCallRecord[]
  opinions?: ReviewOpinion[]
  protocols?: ProtocolSnapshot[]
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

export interface ClaimVersionHistory {
  claim_id: string
  versions: ClaimVersion[]
  current_version: number
  version_count: number
}

export interface ResearchDispute {
  dispute_id: string
  review_id?: string
  claim_id?: string
  question: string
  issue_code: string
  scope?: Record<string, unknown>
  opinion_ids?: string[]
  evidence_ids?: string[]
  status: string
  validation_ids?: string[]
}

export interface ResearchArbitration {
  arbitration_id: string
  review_id?: string
  dispute_ids?: string[]
  issue_summary: string
  recommended_action: string
  required_validation?: string
  is_stub?: boolean
}

export interface ValidationEstimate {
  component: string
  value?: number | null
  unit?: string
  calculation_version?: string
  basis?: string
}

export interface ValidationCandidate {
  experiment_id: string
  dispute_ids?: string[]
  hypothesis_ids?: string[]
  template_id: string
  round_id: number
  design?: Record<string, unknown>
  estimates?: ValidationEstimate[]
  approval_required: boolean
  status: string
  protocol_hash?: string
  utility_score?: number | null
  source?: string
  is_stub?: boolean
  rationale?: string
}

export interface ValidationPlan {
  batch_id: string
  is_stub: boolean
  candidate_ids: string[]
  message?: string
  candidates: ValidationCandidate[]
  protocols: ProtocolSnapshot[]
}

export interface ResearchApproval {
  candidate_id?: string | null
  protocol_hash?: string
  status?: string
  decision?: ResearchDecision | null
}

export interface ResearchExecution {
  candidate_id?: string | null
  protocol_hash?: string
  status?: string
  job?: JobRecord | null
}

export interface ResearchDecision {
  decision_id: string
  actor_type: string
  action: string
  rationale?: string
  claim_version?: number
  protocol_hash?: string
  next_stage?: string
}

export interface JobRecord {
  job_id: string
  kind: string
  status: string
  stage: string
  research_id: string
  request_hash?: string
  review_id?: string
  error_code?: string
  message?: string
  result_ref?: string
}

export interface AgentCallRecord {
  call_id: string
  role: string
  model_id: string
  request_hash?: string
  attempt: number
  status: string
  latency_ms?: number
  usage?: Record<string, number>
  error_code?: string
  error_message?: string
  cache_hit?: boolean
}

export interface ResearchReviewRecord {
  review_id: string
  status: string
  stage?: string
  packet_hash?: string
  claim_id?: string
  request_hash?: string
  is_stub: boolean
  message?: string
  job?: JobRecord
  opinions: ReviewOpinion[]
  disputes: ResearchDispute[]
  arbitrations: ResearchArbitration[]
  agent_calls: AgentCallRecord[]
}

export interface ReviewOpinion {
  opinion_id: string
  review_id?: string
  role: string
  packet_hash: string
  claim_id?: string
  stance: string
  observations?: string[]
  evidence_refs?: string[]
  metric_refs?: string[]
  missing_evidence?: string[]
  validation_proposals?: string[]
  status: string
  is_stub: boolean
  error_code?: string
  error_message?: string
}

export interface ProtocolSnapshot {
  protocol_id: string
  version: number
  protocol_hash: string
  target: ResearchTarget
  time_spec?: Record<string, unknown>
  sample_plan?: Record<string, unknown>
  feature_arms?: Array<Record<string, unknown>>
  split_plan?: Record<string, unknown>
  method?: string
  checks?: string[]
}

export interface EvidenceLocator {
  evidence_id?: string
  packet_id: string
  packet_hash: string
  artifact_id?: string
  artifact_sha256?: string
  locator?: string
  value_type?: string
  value?: number | string | null
  unit?: string
  provenance?: string
  metric_key?: string
  reason?: string
}

export interface UnresolvedIssueBucket {
  open_disputes: ResearchDispute[]
  unresolved_claims: ClaimVersion[]
  packet_limitations: Array<{ packet_id: string; packet_hash: string; limitations: string[] }>
  failed_jobs: JobRecord[]
}

export interface ModelUsageRecord {
  total_calls: number
  succeeded_calls: number
  failed_calls: number
  unfinished_calls: number
  stub_opinions: number
  stub_arbitrations: number
  stub_candidates: number
  roles: Record<string, { total: number; succeeded: number; failed: number }>
}

export interface ResearchReport {
  report_kind: 'research_review_report'
  schema_version: string
  generated_at: string
  research_id: string
  revision: number
  stage: string
  project: ResearchProjectDetail['project']
  claims: ClaimVersion[]
  claim_versions: ClaimVersion[]
  versions_by_claim: Record<string, ClaimVersion[]>
  evidence_locators: Record<string, EvidenceLocator>
  protocols: ProtocolSnapshot[]
  artifacts: ArtifactRecord[]
  evidence_packets: EvidencePacket[]
  opinions: ReviewOpinion[]
  disputes: ResearchDispute[]
  arbitrations: ResearchArbitration[]
  candidates: ValidationCandidate[]
  decisions: ResearchDecision[]
  jobs: JobRecord[]
  agent_calls: AgentCallRecord[]
  unresolved_issues: UnresolvedIssueBucket
  model_usage: ModelUsageRecord
  current_confirmed_versions: ClaimVersion[]
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

export interface DisputeListResponse {
  research_id: string
  revision: number
  project: ResearchProjectDetail['project']
  disputes: ResearchDispute[]
  opinions: ReviewOpinion[]
  arbitrations: ResearchArbitration[]
  open_dispute_count: number
}

export type StartReviewResponse = ResearchProjectDetail & { review: ResearchReviewRecord }
export type ValidationPlanResponse = ResearchProjectDetail & { validation_plan: ValidationPlan }
export type ResearchApprovalResponse = ResearchProjectDetail & { approval: ResearchApproval }
export type ResearchExecutionResponse = ResearchProjectDetail & { execution: ResearchExecution }

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
