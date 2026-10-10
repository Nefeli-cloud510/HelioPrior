import { API_ROOT } from './liveWorkflow'
import type {
  ArtifactKind,
  ArtifactRecord,
  ClaimStatus,
  ClaimVersionHistory,
  CreateResearchPayload,
  DisputeListResponse,
  PreflightResult,
  ResearchApproval,
  ResearchApprovalResponse,
  ResearchDecision,
  ResearchExecution,
  ResearchExecutionResponse,
  ResearchReport,
  ResearchReviewRecord,
  ResearchProjectDetail,
  ResearchProjectSummary,
  StartReviewResponse,
  ValidationPlan,
  ValidationPlanResponse,
  V2ApiEnvelope,
} from '../types/research'
import { ResearchApiError } from '../types/research'

type UnknownRecord = Record<string, unknown>

async function v2Request<T>(
  path: string,
  init: RequestInit = {},
  idempotencyKey?: string,
): Promise<V2ApiEnvelope<T>> {
  if (!API_ROOT) {
    throw new ResearchApiError('本地研究服务未配置，无法访问 v2 接口。', 'service_unavailable', true)
  }
  const headers = new Headers(init.headers ?? {})
  if (init.body && !(init.body instanceof FormData)) {
    headers.set('Content-Type', 'application/json; charset=utf-8')
  }
  if (idempotencyKey) {
    headers.set('Idempotency-Key', idempotencyKey)
  }

  const response = await fetch(`${API_ROOT}/v2${path}`, {
    ...init,
    headers,
  })
  let payload: V2ApiEnvelope<T>
  try {
    payload = (await response.json()) as V2ApiEnvelope<T>
  } catch {
    payload = {
      request_id: '',
      code: 'invalid_response',
      message: 'v2 接口返回了无法解析的响应。',
      retryable: true,
    }
  }
  if (!response.ok || payload.code) {
    throw new ResearchApiError(
      payload.message ?? 'v2 接口调用失败。',
      payload.code ?? 'http_error',
      payload.retryable ?? false,
      (payload.details as UnknownRecord | undefined) ?? {},
    )
  }
  return payload
}

export async function listResearchProjects(): Promise<ResearchProjectSummary[]> {
  const response = await v2Request<ResearchProjectSummary[]>('/research')
  return response.data ?? []
}

export async function createResearchProject(
  payload: CreateResearchPayload,
  idempotencyKey?: string,
): Promise<{ researchId: string; revision: number; detail: ResearchProjectDetail }> {
  const response = await v2Request<ResearchProjectDetail>('/research', {
    method: 'POST',
    body: JSON.stringify(payload),
  }, idempotencyKey)
  if (!response.research_id || !response.data) {
    throw new ResearchApiError('创建研究失败：服务未返回项目标识。', 'missing_research_id', false)
  }
  return {
    researchId: response.research_id,
    revision: response.revision ?? response.data.revision,
    detail: response.data,
  }
}

export async function getResearchProject(researchId: string): Promise<ResearchProjectDetail> {
  const response = await v2Request<ResearchProjectDetail>(`/research/${encodeURIComponent(researchId)}`)
  if (!response.data) {
    throw new ResearchApiError('研究项目数据为空。', 'empty_project', false)
  }
  return response.data
}

export interface UploadResearchArtifactPayload {
  file_name: string
  kind: ArtifactKind
  content: string
  content_type?: string
}

export async function uploadResearchArtifact(
  researchId: string,
  expectedRevision: number,
  payload: UploadResearchArtifactPayload,
): Promise<{ project: ResearchProjectDetail; artifact: ArtifactRecord }> {
  const response = await v2Request<ResearchProjectDetail & { artifact: ArtifactRecord }>(
    `/research/${encodeURIComponent(researchId)}/artifacts`,
    {
      method: 'POST',
      body: JSON.stringify({
        expected_revision: expectedRevision,
        ...payload,
      }),
    },
  )
  if (!response.data?.artifact) {
    throw new ResearchApiError('产物导入后未返回文件记录。', 'missing_artifact', false)
  }
  return {
    project: response.data,
    artifact: response.data.artifact,
  }
}

export interface RunResearchPreflightPayload {
  artifact_id?: string
  feature_columns?: string[]
  claim_ids?: string[]
  protocol_hash_value?: string
  data_manifest_hash?: string
  run_id?: string
  horizon_days?: number
  base_arm?: string
  aug_arm?: string
}

export async function runResearchPreflight(
  researchId: string,
  expectedRevision: number,
  payload: RunResearchPreflightPayload,
): Promise<{ project: ResearchProjectDetail; preflight: PreflightResult }> {
  const response = await v2Request<ResearchProjectDetail & { preflight: PreflightResult }>(
    `/research/${encodeURIComponent(researchId)}/preflight`,
    {
      method: 'POST',
      body: JSON.stringify({
        expected_revision: expectedRevision,
        ...payload,
      }),
    },
  )
  if (!response.data?.preflight) {
    throw new ResearchApiError('预检未返回证据包结果。', 'missing_preflight', false)
  }
  return {
    project: response.data,
    preflight: response.data.preflight,
  }
}

export interface RunResearchReviewPayload {
  artifact_id?: string
  packet_hash?: string
  claim_id?: string
}

export async function runResearchReview(
  researchId: string,
  expectedRevision: number,
  payload: RunResearchReviewPayload = {},
): Promise<{ project: ResearchProjectDetail; review: ResearchReviewRecord }> {
  const response = await v2Request<StartReviewResponse>(
    `/research/${encodeURIComponent(researchId)}/reviews`,
    {
      method: 'POST',
      body: JSON.stringify({
        expected_revision: expectedRevision,
        ...payload,
      }),
    },
  )
  if (!response.data?.review) {
    throw new ResearchApiError('复核已启动但未返回复核记录。', 'missing_review', false)
  }
  return {
    project: response.data,
    review: response.data.review,
  }
}

export interface RetryResearchReviewPayload {
  retry_roles?: string[]
  attempt?: number
  packet_hash?: string
  claim_id?: string
}

export async function retryResearchReview(
  researchId: string,
  expectedRevision: number,
  reviewId: string,
  payload: RetryResearchReviewPayload = {},
): Promise<{ project: ResearchProjectDetail; review: ResearchReviewRecord }> {
  const response = await v2Request<StartReviewResponse>(
    `/research/${encodeURIComponent(researchId)}/reviews/${encodeURIComponent(reviewId)}/retry`,
    {
      method: 'POST',
      body: JSON.stringify({
        expected_revision: expectedRevision,
        retry_roles: payload.retry_roles,
        attempt: payload.attempt,
        packet_hash: payload.packet_hash,
        claim_id: payload.claim_id,
      }),
    },
  )
  if (!response.data?.review) {
    throw new ResearchApiError('复核重试后未返回复核记录。', 'missing_review', false)
  }
  return {
    project: response.data,
    review: response.data.review,
  }
}

export async function listResearchDisputes(researchId: string): Promise<DisputeListResponse> {
  const response = await v2Request<DisputeListResponse>(
    `/research/${encodeURIComponent(researchId)}/disputes`,
  )
  if (!response.data) {
    throw new ResearchApiError('分歧清单数据为空。', 'empty_disputes', false)
  }
  return response.data
}

export interface PlanResearchValidationsPayload {
  dispute_ids?: string[]
  template_id?: string
  design_hints?: string[]
  model?: string
}

export async function planResearchValidations(
  researchId: string,
  expectedRevision: number,
  payload: PlanResearchValidationsPayload = {},
): Promise<{ project: ResearchProjectDetail; plan: ValidationPlan }> {
  const response = await v2Request<ValidationPlanResponse>(
    `/research/${encodeURIComponent(researchId)}/plans`,
    {
      method: 'POST',
      body: JSON.stringify({
        expected_revision: expectedRevision,
        ...payload,
      }),
    },
  )
  if (!response.data?.validation_plan) {
    throw new ResearchApiError('验证任务规划后未返回计划。', 'missing_validation_plan', false)
  }
  return {
    project: response.data,
    plan: response.data.validation_plan,
  }
}

export interface ApproveResearchProtocolPayload {
  protocol_hash: string
  candidate_id?: string
  rationale?: string
  actor_type?: string
}

export async function approveResearchProtocol(
  researchId: string,
  expectedRevision: number,
  payload: ApproveResearchProtocolPayload,
): Promise<{ project: ResearchProjectDetail; approval: ResearchApproval }> {
  const response = await v2Request<ResearchApprovalResponse>(
    `/research/${encodeURIComponent(researchId)}/approvals`,
    {
      method: 'POST',
      body: JSON.stringify({
        expected_revision: expectedRevision,
        ...payload,
      }),
    },
  )
  if (!response.data?.approval) {
    throw new ResearchApiError('批准提交后未返回结果。', 'missing_approval', false)
  }
  return {
    project: response.data,
    approval: response.data.approval,
  }
}

export interface ExecuteResearchProtocolPayload {
  protocol_hash: string
  candidate_id?: string
}

export async function executeResearchProtocol(
  researchId: string,
  expectedRevision: number,
  payload: ExecuteResearchProtocolPayload,
): Promise<{ project: ResearchProjectDetail; execution: ResearchExecution }> {
  const response = await v2Request<ResearchExecutionResponse>(
    `/research/${encodeURIComponent(researchId)}/executions`,
    {
      method: 'POST',
      body: JSON.stringify({
        expected_revision: expectedRevision,
        ...payload,
      }),
    },
  )
  if (!response.data?.execution) {
    throw new ResearchApiError('验证执行后未返回结果。', 'missing_execution', false)
  }
  return {
    project: response.data,
    execution: response.data.execution,
  }
}

export async function listClaimVersions(
  researchId: string,
  claimId: string,
): Promise<ClaimVersionHistory> {
  const response = await v2Request<ClaimVersionHistory>(
    `/research/${encodeURIComponent(researchId)}/claims/${encodeURIComponent(claimId)}/versions`,
  )
  if (!response.data) {
    throw new ResearchApiError('结论版本数据为空。', 'empty_claim_versions', false)
  }
  return response.data
}

export interface CreateClaimVersionPayload {
  statement?: string
  evidence_ids?: string[]
  status?: ClaimStatus
  revision_note: string
  scope?: {
    sample_condition?: string
    evaluation_period?: Record<string, string>
    baseline_protocol_id?: string
    horizon_days?: number
  }
  program_recommendation?: ClaimStatus
  human_decision?: string
  next_stage?: string
}

export async function createClaimVersion(
  researchId: string,
  expectedRevision: number,
  claimId: string,
  payload: CreateClaimVersionPayload,
): Promise<{
  project: ResearchProjectDetail
  claimVersion: ClaimVersionHistory['versions'][number]
  decision: ResearchDecision
}> {
  const response = await v2Request<
    ResearchProjectDetail & {
      claim_version: ClaimVersionHistory['versions'][number]
      decision: ResearchDecision
    }
  >(`/research/${encodeURIComponent(researchId)}/claims/${encodeURIComponent(claimId)}/versions`, {
    method: 'POST',
    body: JSON.stringify({
      expected_revision: expectedRevision,
      ...payload,
    }),
  })
  if (!response.data?.claim_version || !response.data?.decision) {
    throw new ResearchApiError('结论版本保存后缺少版本或决定记录。', 'missing_claim_version', false)
  }
  return {
    project: response.data,
    claimVersion: response.data.claim_version,
    decision: response.data.decision,
  }
}

export async function exportResearchReport(researchId: string): Promise<ResearchReport> {
  const response = await v2Request<ResearchReport>(`/research/${encodeURIComponent(researchId)}/report`)
  if (!response.data) {
    throw new ResearchApiError('研究报告数据为空。', 'empty_report', false)
  }
  return response.data
}
