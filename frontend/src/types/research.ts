export type ResearchWorkflowMode = 'legacy' | 'research_review'
export type ResearchDataMode = 'authorized_data' | 'synthetic_fixture' | 'historical_replay' | 'unknown'
export type ResearchEntryMode = 'new_question' | 'import_claim'
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
  claim_count: number
  evidence_packet_count: number
  open_dispute_count: number
  awaiting_decision: boolean
}

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

