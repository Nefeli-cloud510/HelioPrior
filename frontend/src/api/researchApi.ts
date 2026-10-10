import { API_ROOT } from './liveWorkflow'
import type {
  CreateResearchPayload,
  ResearchProjectDetail,
  ResearchProjectSummary,
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
