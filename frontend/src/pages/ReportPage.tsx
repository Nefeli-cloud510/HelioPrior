import { useEffect, useMemo, useState } from 'react'
import { useParams } from 'react-router-dom'
import { exportResearchReport, getResearchProject } from '../api/researchApi'
import { ResearchNav } from '../components/ResearchNav'
import type {
  ClaimVersion,
  ResearchProjectDetail,
  ResearchReport,
  ReviewOpinion,
} from '../types/research'

interface DiffToken {
  text: string
  kind: 'same' | 'removed' | 'added'
}

function diffWords(oldText: string, newText: string): DiffToken[] {
  const oldTokens = oldText.split(/(\s+)/).filter(Boolean)
  const newTokens = newText.split(/(\s+)/).filter(Boolean)
  const oldLen = oldTokens.length
  const newLen = newTokens.length
  const dp: number[][] = Array.from({ length: oldLen + 1 }, () => new Array<number>(newLen + 1).fill(0))
  for (let i = oldLen - 1; i >= 0; i -= 1) {
    for (let j = newLen - 1; j >= 0; j -= 1) {
      dp[i][j] =
        oldTokens[i] === newTokens[j]
          ? dp[i + 1][j + 1] + 1
          : Math.max(dp[i + 1][j], dp[i][j + 1])
    }
  }
  const result: DiffToken[] = []
  let i = 0
  let j = 0
  while (i < oldLen && j < newLen) {
    if (oldTokens[i] === newTokens[j]) {
      result.push({ text: oldTokens[i], kind: 'same' })
      i += 1
      j += 1
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      result.push({ text: oldTokens[i], kind: 'removed' })
      i += 1
    } else {
      result.push({ text: newTokens[j], kind: 'added' })
      j += 1
    }
  }
  while (i < oldLen) {
    result.push({ text: oldTokens[i], kind: 'removed' })
    i += 1
  }
  while (j < newLen) {
    result.push({ text: newTokens[j], kind: 'added' })
    j += 1
  }
  return result
}

function StatementDiff({ oldText, newText }: { oldText: string; newText: string }) {
  const tokens = useMemo(() => diffWords(oldText, newText), [oldText, newText])
  return (
    <div className="claim-diff-text">
      {tokens.map((token, index) => (
        <span key={`${token.kind}-${index}`} className={`claim-diff-${token.kind}`}>
          {token.text}
        </span>
      ))}
    </div>
  )
}

function ClaimDiffView({
  claimId,
  versions,
}: {
  claimId: string
  versions: ClaimVersion[]
}) {
  const [leftIndex, setLeftIndex] = useState(() => Math.max(0, versions.length - 2))
  const [rightIndex, setRightIndex] = useState(() => versions.length - 1)

  if (versions.length < 2) {
    return (
      <div className="research-empty">
        结论 {claimId} 只有 {versions.length} 个版本，还没有可对比的新旧版本。保存新结论版本后会自动激活对比视图。
      </div>
    )
  }

  const left = versions[Math.min(leftIndex, versions.length - 1)]
  const right = versions[Math.min(rightIndex, versions.length - 1)]

  return (
    <div className="claim-diff">
      <div className="claim-diff__controls">
        <label>
          旧版本
          <select value={left.version} onChange={(event) => setLeftIndex(Number(event.target.value))}>
            {versions.map((version, index) => (
              <option key={version.version} value={index}>
                v{version.version}
              </option>
            ))}
          </select>
        </label>
        <label>
          新版本
          <select value={right.version} onChange={(event) => setRightIndex(Number(event.target.value))}>
            {versions.map((version, index) => (
              <option key={version.version} value={index}>
                v{version.version}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="claim-diff__grid">
        <section className="claim-diff__side claim-diff__side--old">
          <div className="detail-card__title-row">
            <h3>旧表述 v{left.version}</h3>
            <span className="badge">{left.status}</span>
          </div>
          <p>{left.statement}</p>
          <div className="detail-card__meta">
            证据：{left.evidence_ids.join('、') || '暂无'}
            {left.scope?.sample_condition ? ` · ${left.scope.sample_condition}` : ''}
          </div>
        </section>
        <section className="claim-diff__side claim-diff__side--new">
          <div className="detail-card__title-row">
            <h3>新表述 v{right.version}</h3>
            <span className="badge">{right.status}</span>
          </div>
          <p>{right.statement}</p>
          <div className="detail-card__meta">
            证据：{right.evidence_ids.join('、') || '暂无'}
            {right.scope?.sample_condition ? ` · ${right.scope.sample_condition}` : ''}
          </div>
        </section>
      </div>
      {left.version !== right.version ? (
        <section className="claim-diff__detail">
          <h3>逐段差异</h3>
          <div className="claim-diff-text">
            <StatementDiff oldText={left.statement} newText={right.statement} />
          </div>
          {right.parent_version === left.version && right.revision_note ? (
            <p className="claim-diff__reason">修订说明：{right.revision_note}</p>
          ) : null}
          {right.human_decision ? (
            <p className="claim-diff__reason">人工决定：{right.human_decision}</p>
          ) : null}
        </section>
      ) : null}
    </div>
  )
}

function reprStance(stance: string): string {
  const map: Record<string, string> = {
    supported: '支持',
    unsupported: '不支持',
    insufficient: '证据不足',
    conflict: '存在分歧',
    unknown: '未判断',
  }
  return map[stance] ?? stance
}

export function ReportPage() {
  const { researchId } = useParams<{ researchId: string }>()
  const [project, setProject] = useState<ResearchProjectDetail | null>(null)
  const [report, setReport] = useState<ResearchReport | null>(null)
  const [reportBusy, setReportBusy] = useState(false)
  const [selectedClaimId, setSelectedClaimId] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  async function refresh() {
    if (!researchId) {
      setError('缺少研究项目标识。')
      setLoading(false)
      return
    }
    try {
      setLoading(true)
      setError(null)
      const next = await getResearchProject(researchId)
      setProject(next)
      if (selectedClaimId && next.claims.some((claim) => claim.claim_id === selectedClaimId)) {
        return
      }
      if (next.claims.length > 0) {
        setSelectedClaimId(next.claims[0].claim_id)
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : '研究报告读取失败。')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void refresh()
  }, [researchId])

  async function exportReport() {
    if (!researchId) {
      return
    }
    try {
      setReportBusy(true)
      setError(null)
      setReport(await exportResearchReport(researchId))
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : '研究报告导出失败。')
    } finally {
      setReportBusy(false)
    }
  }

  if (loading && !project) {
    return (
      <div className="timeline-shell">
        <div className="timeline-shell__frame timeline-shell__frame--detail">
          <main className="research-page">
            <div className="research-banner">正在读取研究报告……</div>
          </main>
        </div>
      </div>
    )
  }

  if (error || !project || !researchId) {
    return (
      <div className="timeline-shell">
        <div className="timeline-shell__frame timeline-shell__frame--detail">
          <main className="research-page">
            <div className="research-banner research-banner--error">
              {error ?? '研究项目不存在。'}
              <div className="research-actions">
                <button className="research-secondary" type="button" onClick={() => void refresh()}>
                  重试
                </button>
              </div>
            </div>
          </main>
        </div>
      </div>
    )
  }

  const selectedVersions = (project.claims ?? []).filter((claim) => claim.claim_id === selectedClaimId)
  const opinions: ReviewOpinion[] = project.opinions ?? []

  return (
    <div className="timeline-shell">
      <div className="timeline-shell__frame timeline-shell__frame--detail">
        <main className="research-page">
          <header className="research-page__header">
            <div>
              <h1>研究报告</h1>
              <p>
                结论版本变化、证据链接、复核意见、尚未解决的限制与人工决定。旧结论保留，新版本可追溯。
              </p>
              <div className="research-page__chips">
                <span className="badge">{project.claim_count} 条结论</span>
                <span className="badge">revision {project.revision}</span>
                <span className="badge">{opinions.length} 条复核意见</span>
              </div>
            </div>
            <div className="research-page__aside">
              <button className="research-primary" type="button" disabled={reportBusy} onClick={() => void exportReport()}>
                {reportBusy ? '导出中……' : '生成研究报告'}
              </button>
            </div>
          </header>

          <ResearchNav researchId={researchId} active="report" />

          {error ? <div className="research-banner research-banner--error">{error}</div> : null}

          <section className="detail-card">
            <div className="detail-card__title-row">
              <h2>结论版本对比</h2>
              <span className="badge">不可变版本</span>
            </div>
            {project.claims.length > 0 ? (
              <>
                <div className="research-field" style={{ maxWidth: 420 }}>
                  <label htmlFor="report-claim">选择结论</label>
                  <select
                    id="report-claim"
                    value={selectedClaimId}
                    onChange={(event) => setSelectedClaimId(event.target.value)}
                  >
                    {project.claims.map((claim) => (
                      <option key={claim.claim_id} value={claim.claim_id}>
                        {claim.claim_id} · v{claim.version}
                      </option>
                    ))}
                  </select>
                </div>
                <ClaimDiffView claimId={selectedClaimId} versions={selectedVersions} />
              </>
            ) : (
              <div className="research-empty">当前项目没有结论，无法生成版本对比。</div>
            )}
          </section>

          <div className="research-page__grid" style={{ marginTop: 18 }}>
            <section className="detail-card">
              <div className="detail-card__title-row">
                <h2>复核意见摘要</h2>
                <span className="badge">{opinions.length} 条</span>
              </div>
              {opinions.length > 0 ? (
                <ul className="research-material-list">
                  {opinions.slice().reverse().map((opinion) => (
                    <li key={opinion.opinion_id} className="research-material-item">
                      <div className="research-check-item__top">
                        <strong>{opinion.role}</strong>
                        <span className={`badge ${opinion.status === 'completed' ? 'badge--success' : opinion.status === 'failed' ? 'badge--high' : 'badge--warning'}`}>
                          {opinion.status}
                          {opinion.is_stub ? ' · stub' : ''}
                        </span>
                      </div>
                      <p>立场：{reprStance(opinion.stance)}</p>
                      <div className="detail-card__meta">
                        {opinion.observations?.length ? `观察 ${opinion.observations.length} 条` : '无观察'}
                        {opinion.metric_refs?.length ? ` · 指标引用 ${opinion.metric_refs.length} 项` : ''}
                      </div>
                    </li>
                  ))}
                </ul>
              ) : (
                <div className="research-empty">尚无复核意见。</div>
              )}
            </section>

            <section className="detail-card">
              <div className="detail-card__title-row">
                <h2>未解决事项与模型使用</h2>
                <span className="badge">审计记录</span>
              </div>
              <div className="research-fact-list">
                <div className="research-fact-item">待处理分歧：{project.open_dispute_count}</div>
                <div className="research-fact-item">等待人工决定：{project.awaiting_decision ? '是' : '否'}</div>
                <div className="research-fact-item">模型调用：{(project.agent_calls ?? []).length} 次</div>
                <div className="research-fact-item">验证任务：{(project.jobs ?? []).filter((job) => job.kind === 'validation_execution').length} 项</div>
              </div>
              {(project.agent_calls ?? []).length > 0 ? (
                <ul className="research-check-list" style={{ marginTop: 14 }}>
                  {(project.agent_calls ?? []).slice().reverse().slice(0, 6).map((call) => (
                    <li key={call.call_id} className="research-check-item">
                      <div className="research-check-item__top">
                        <code>{call.role}</code>
                        <span className={`badge ${call.status === 'succeeded' ? 'badge--success' : call.status === 'failed' ? 'badge--high' : 'badge--warning'}`}>
                          {call.status}
                        </span>
                      </div>
                      <p>
                        {call.model_id} · attempt {call.attempt}
                        {call.cache_hit ? ' · 缓存命中' : ''}
                        {call.error_message ? ` · ${call.error_message}` : ''}
                      </p>
                    </li>
                  ))}
                </ul>
              ) : null}
            </section>
          </div>

          {report ? (
            <section className="detail-card" style={{ marginTop: 18 }}>
              <div className="detail-card__title-row">
                <h2>报告 JSON</h2>
                <span className="badge">research_review_report</span>
              </div>
              <p className="detail-card__meta">
                报告由后端在同一 revision 上生成：结论版本、证据引用、模型使用与未解决事项都来自真实记录。
              </p>
              <pre className="research-report-preview">{JSON.stringify(report, null, 2)}</pre>
            </section>
          ) : null}
        </main>
      </div>
    </div>
  )
}
