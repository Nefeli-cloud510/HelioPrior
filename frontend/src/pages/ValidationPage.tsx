import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import {
  approveResearchProtocol,
  executeResearchProtocol,
  getResearchProject,
  planResearchValidations,
} from '../api/researchApi'
import { ResearchNav } from '../components/ResearchNav'
import type {
  ProtocolSnapshot,
  ResearchApproval,
  ResearchProjectDetail,
  ValidationCandidate,
  ValidationPlan,
} from '../types/research'

function prettyValue(value: unknown): string {
  if (value === null || value === undefined) {
    return '—'
  }
  if (typeof value === 'string') {
    return value
  }
  if (typeof value === 'number') {
    return Number.isFinite(value) ? String(value) : '—'
  }
  return JSON.stringify(value)
}

function ProtocolPreview({ protocol }: { protocol: ProtocolSnapshot }) {
  return (
    <div className="protocol-preview">
      <div className="detail-card__title-row">
        <h3>协议预览</h3>
        <span className="badge">v{protocol.version}</span>
      </div>
      <dl className="evidence-drawer__grid">
        <div className="evidence-drawer__item">
          <dt>协议哈希</dt>
          <dd className="research-page__mono">{protocol.protocol_hash}</dd>
        </div>
        <div className="evidence-drawer__item">
          <dt>目标</dt>
          <dd>
            {protocol.target.id}
            {protocol.target.unit ? ` · ${protocol.target.unit}` : ''}
            {protocol.target.coordinate_system ? ` · ${protocol.target.coordinate_system}` : ''}
          </dd>
        </div>
        <div className="evidence-drawer__item">
          <dt>方法</dt>
          <dd>{protocol.method ?? '未指定'}</dd>
        </div>
      </dl>
      {Object.keys(protocol.time_spec ?? {}).length > 0 ? (
        <div className="protocol-preview__block">
          <strong>时间规格</strong>
          <pre>{JSON.stringify(protocol.time_spec, null, 2)}</pre>
        </div>
      ) : null}
      {Object.keys(protocol.feature_arms ?? {}).length > 0 || (protocol.feature_arms ?? []).length > 0 ? (
        <div className="protocol-preview__block">
          <strong>特征组</strong>
          <pre>{JSON.stringify(protocol.feature_arms, null, 2)}</pre>
        </div>
      ) : null}
      {Object.keys(protocol.split_plan ?? {}).length > 0 ? (
        <div className="protocol-preview__block">
          <strong>划分计划</strong>
          <pre>{JSON.stringify(protocol.split_plan, null, 2)}</pre>
        </div>
      ) : null}
      {protocol.checks && protocol.checks.length > 0 ? (
        <ul className="research-check-list">
          {protocol.checks.map((check) => (
            <li key={check} className="research-check-item">
              {check}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  )
}

function CandidateCard({
  candidate,
  protocol,
  approval,
  onApprove,
  onExecute,
  busy,
  error,
}: {
  candidate: ValidationCandidate
  protocol?: ProtocolSnapshot
  approval?: ResearchApproval
  onApprove: () => void
  onExecute: () => void
  busy: boolean
  error?: string | null
}) {
  const estimates = (candidate.estimates ?? []).map((item) => ({
    ...item,
    key: `${candidate.experiment_id}-${item.component}`,
  }))
  const isApproved = candidate.status === 'approved'
  const isExecuted = candidate.status === 'executed'
  return (
    <article className={`candidate-card candidate-card--${candidate.status}`}>
      <header className="candidate-card__head">
        <div>
          <h3>{candidate.experiment_id}</h3>
          <p>
            {candidate.template_id} · 轮次 {candidate.round_id} · 来源 {candidate.source}
          </p>
        </div>
        <span className={`badge ${isApproved || isExecuted ? 'badge--success' : candidate.status === 'failed' ? 'badge--high' : 'badge--warning'}`}>
          {candidate.status}
        </span>
      </header>
      {candidate.is_stub ? <span className="badge">离线 stub 方案</span> : null}
      {candidate.rationale ? <p className="candidate-card__rationale">{candidate.rationale}</p> : null}
      {candidate.dispute_ids && candidate.dispute_ids.length > 0 ? (
        <div className="detail-card__meta">
          回应分歧：{candidate.dispute_ids.join('、')}
        </div>
      ) : null}
      {Object.keys(candidate.design ?? {}).length > 0 ? (
        <div className="protocol-preview__block">
          <strong>设计要点</strong>
          <pre>{JSON.stringify(candidate.design, null, 2)}</pre>
        </div>
      ) : null}
      {candidate.utility_score !== null && candidate.utility_score !== undefined ? (
        <div className="candidate-card__score">
          程序估计价值：<strong>{Number(candidate.utility_score).toFixed(3)}</strong>
        </div>
      ) : null}
      {estimates.length > 0 ? (
        <div className="research-metrics">
          {estimates.map((estimate) => (
            <div key={estimate.key} className="metric-box">
              <div className="metric-box__label">{estimate.component}</div>
              <div className="metric-box__value">{prettyValue(estimate.value)}</div>
              <div className="detail-card__meta">
                {estimate.unit ?? '无单位'} · {estimate.calculation_version ?? 'program/1.0'}
              </div>
            </div>
          ))}
        </div>
      ) : null}
      {protocol ? <ProtocolPreview protocol={protocol} /> : null}
      {error ? <div className="research-banner research-banner--error">{error}</div> : null}
      <div className="research-actions">
        <button
          className="research-primary"
          type="button"
          disabled={busy || !protocol || isApproved || isExecuted}
          onClick={onApprove}
        >
          {isApproved ? '已批准' : isExecuted ? '已执行' : '批准协议'}
        </button>
        <button
          className="research-secondary"
          type="button"
          disabled={busy || !protocol || !isApproved || isExecuted}
          onClick={onExecute}
        >
          执行验证
        </button>
      </div>
      {approval ? (
        <div className="candidate-card__approval">
          最近批准：{approval.decision?.decision_id ?? '—'} · {approval.status ?? '—'}
        </div>
      ) : null}
    </article>
  )
}

export function ValidationPage() {
  const { researchId } = useParams<{ researchId: string }>()
  const [project, setProject] = useState<ResearchProjectDetail | null>(null)
  const [plan, setPlan] = useState<ValidationPlan | null>(null)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [perCandidateError, setPerCandidateError] = useState<Record<string, string>>({})

  async function refresh() {
    if (!researchId) {
      setError('缺少研究项目标识。')
      setLoading(false)
      return
    }
    try {
      setLoading(true)
      setError(null)
      setProject(await getResearchProject(researchId))
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : '验证实验读取失败。')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void refresh()
  }, [researchId])

  async function planValidations() {
    if (!project || !researchId) {
      return
    }
    try {
      setBusy(true)
      setError(null)
      setNotice(null)
      setPlan(null)
      const result = await planResearchValidations(researchId, project.revision, {
        dispute_ids: (project.disputes ?? [])
          .filter((item) => item.status === 'open')
          .map((item) => item.dispute_id),
      })
      setProject(result.project)
      setPlan(result.plan)
      setNotice(
        result.plan.is_stub
          ? `验证规划 ${result.plan.batch_id} 完成（离线 stub）。${result.plan.message ?? ''}`
          : `验证规划 ${result.plan.batch_id} 完成，生成 ${result.plan.candidate_ids.length} 个候选实验。`,
      )
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : '验证规划失败。')
    } finally {
      setBusy(false)
    }
  }

  async function approve(candidate: ValidationCandidate, protocol: ProtocolSnapshot | undefined) {
    if (!project || !researchId || !protocol) {
      return
    }
    try {
      setBusy(true)
      setError(null)
      setPerCandidateError((prev) => ({ ...prev, [candidate.experiment_id]: '' }))
      const result = await approveResearchProtocol(researchId, project.revision, {
        protocol_hash: protocol.protocol_hash,
        candidate_id: candidate.experiment_id,
        rationale: '人工 PI 审阅候选实验与协议预览后批准。',
      })
      setProject(result.project)
      setNotice(`协议已批准：${protocol.protocol_hash.slice(0, 16)}…（revision ${result.project.revision}）`)
    } catch (caught) {
      setPerCandidateError((prev) => ({
        ...prev,
        [candidate.experiment_id]: caught instanceof Error ? caught.message : '协议批准失败。',
      }))
    } finally {
      setBusy(false)
    }
  }

  async function execute(candidate: ValidationCandidate, protocol: ProtocolSnapshot | undefined) {
    if (!project || !researchId || !protocol) {
      return
    }
    try {
      setBusy(true)
      setError(null)
      setPerCandidateError((prev) => ({ ...prev, [candidate.experiment_id]: '' }))
      const result = await executeResearchProtocol(researchId, project.revision, {
        protocol_hash: protocol.protocol_hash,
        candidate_id: candidate.experiment_id,
      })
      setProject(result.project)
      setNotice(
        `验证任务 ${result.execution.candidate_id ?? '—'} 状态：${result.execution.status ?? '—'}。未配置数据源时会停留在失败状态，不会伪造成功。`,
      )
    } catch (caught) {
      setPerCandidateError((prev) => ({
        ...prev,
        [candidate.experiment_id]: caught instanceof Error ? caught.message : '验证执行失败。',
      }))
    } finally {
      setBusy(false)
    }
  }

  if (!researchId) {
    return <Link className="research-secondary" to="/research">返回研究工作台</Link>
  }

  const approval = (candidate: ValidationCandidate) =>
    [...(project?.decisions ?? [])]
      .reverse()
      .find((item) => item.action === 'approve_protocol' && item.protocol_hash === candidate.protocol_hash)
  const candidates = plan?.candidates ?? project?.candidates ?? []
  const executionJobs = (project?.jobs ?? []).filter((item) => item.kind === 'validation_execution')

  return (
    <div className="timeline-shell">
      <div className="timeline-shell__frame timeline-shell__frame--detail">
        <main className="research-page">
          <header className="research-page__header">
            <div>
              <h1>验证实验</h1>
              <p>
                把分歧映射为可执行验证：每项候选实验说明解决什么问题、比较哪两组、受哪些条件约束。批准和运行情况可追溯。
              </p>
              <div className="research-page__chips">
                <span className="badge">{candidates.length} 个候选</span>
                <span className="badge">revision {project?.revision ?? '—'}</span>
                <span className="badge">{project?.pc_stage ?? '未加载'}</span>
              </div>
            </div>
            <div className="research-page__aside">
              <Link className="research-primary" to={`/research/${researchId}/review`}>
                返回协作复核
              </Link>
              <button className="research-secondary" type="button" onClick={() => void refresh()}>
                刷新
              </button>
            </div>
          </header>

          <ResearchNav researchId={researchId} active="validate" />

          {error ? <div className="research-banner research-banner--error">{error}</div> : null}
          {notice ? <div className="research-banner research-banner--success">{notice}</div> : null}
          {loading && !project ? <div className="research-banner">正在读取验证实验……</div> : null}

          <section className="detail-card">
            <div className="detail-card__title-row">
              <h2>生成候选验证任务</h2>
              <span className="badge">程序价值排序</span>
            </div>
            <p className="detail-card__meta">
              选择所有未关闭分歧后调用规划器；不填写任何参数时由程序选择可执行的争议并生成候选实验。无 API Key 时使用明确标注的 stub。
            </p>
            <div className="research-actions">
              <button className="research-primary" type="button" disabled={busy || !project} onClick={() => void planValidations()}>
                {busy ? '规划中……' : '生成候选验证任务'}
              </button>
              {plan ? (
                <span className="badge">
                  batch {plan.batch_id}
                  {plan.is_stub ? ' · stub' : ''}
                </span>
              ) : null}
            </div>
          </section>

          {candidates.length > 0 ? (
            <div className="candidate-grid" style={{ marginTop: 18 }}>
              {candidates.map((candidate) => {
                const protocol = (plan?.protocols ?? project?.protocols ?? []).find(
                  (item) => item.protocol_hash === candidate.protocol_hash,
                )
                const approvalRecord = approval(candidate)
                return (
                  <CandidateCard
                    key={candidate.experiment_id}
                    candidate={candidate}
                    protocol={protocol}
                    approval={
                      approvalRecord
                        ? {
                            candidate_id: candidate.experiment_id,
                            protocol_hash: candidate.protocol_hash,
                            status: candidate.status,
                            decision: approvalRecord,
                          }
                        : undefined
                    }
                    onApprove={() => void approve(candidate, protocol)}
                    onExecute={() => void execute(candidate, protocol)}
                    busy={busy}
                    error={perCandidateError[candidate.experiment_id]}
                  />
                )
              })}
            </div>
          ) : (
            <div className="research-empty">
              还没有候选验证任务。先运行“生成候选验证任务”，或者先到协作复核页产出分歧。
            </div>
          )}

          {executionJobs.length > 0 ? (
            <section className="detail-card" style={{ marginTop: 18 }}>
              <div className="detail-card__title-row">
                <h2>执行记录</h2>
                <span className="badge">{executionJobs.length} 项</span>
              </div>
              <ul className="research-material-list">
                {executionJobs
                  .slice()
                  .reverse()
                  .map((job) => (
                    <li key={job.job_id} className="research-material-item">
                      <div className="research-check-item__top">
                        <strong>{job.job_id}</strong>
                        <span className={`badge ${job.status === 'succeeded' ? 'badge--success' : job.status === 'failed' ? 'badge--high' : 'badge--warning'}`}>
                          {job.status}
                        </span>
                      </div>
                      <p>{job.stage}{job.message ? ` · ${job.message}` : ''}</p>
                      {job.error_code ? (
                        <div className="detail-card__meta">
                          {job.error_code}
                          {job.result_ref ? ` · ${job.result_ref}` : ''}
                        </div>
                      ) : null}
                    </li>
                  ))}
              </ul>
            </section>
          ) : null}
        </main>
      </div>
    </div>
  )
}
