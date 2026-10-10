import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { getResearchProject, retryResearchReview, runResearchReview } from '../api/researchApi'
import { EvidenceDrawer } from '../components/EvidenceDrawer'
import { ResearchNav } from '../components/ResearchNav'
import type {
  ResearchArbitration,
  ResearchDispute,
  ResearchProjectDetail,
  ResearchReviewRecord,
  ReviewOpinion,
} from '../types/research'

const ROLE_META: Record<string, { label: string; intro: string }> = {
  scientific_questioner: {
    label: '科学质询者',
    intro: '独立检查结论在证据包范围内是否得到足够支持：共同样本、时间边界、基线、覆盖与稳健性。',
  },
  scientific_interpreter: {
    label: '结果解释者',
    intro: '独立解释结果方向、适用范围与中介干扰：避免将相关、预测增益与因果机制混为一谈。',
  },
}

const STATUS_LABELS: Record<string, string> = {
  queued: '排队',
  running: '运行中',
  completed: '完成',
  failed: '失败',
  partial: '部分完成',
}

function OpinionCard({
  opinion,
  onOpenEvidence,
}: {
  opinion: ReviewOpinion
  onOpenEvidence: (opinion: ReviewOpinion) => void
}) {
  const meta = ROLE_META[opinion.role] ?? { label: opinion.role, intro: '' }
  return (
    <article className={`review-opinion review-opinion--${opinion.status}`}>
      <header className="review-opinion__head">
        <div>
          <strong>{meta.label}</strong>
          <span className={`badge ${opinion.status === 'completed' ? 'badge--success' : opinion.status === 'failed' ? 'badge--high' : opinion.status === 'running' || opinion.status === 'queued' ? 'badge--warning' : ''}`}>
            {STATUS_LABELS[opinion.status] ?? opinion.status}
          </span>
          {opinion.is_stub ? <span className="badge">离线 stub</span> : null}
        </div>
        <button className="research-secondary" type="button" onClick={() => onOpenEvidence(opinion)}>
          查看证据引用
        </button>
      </header>
      <p className="detail-card__meta">{meta.intro}</p>
      {opinion.stance ? (
        <div className="review-opinion__stance">
          立场：<strong>{opinion.stance}</strong>
          {opinion.metric_refs?.length ? ` · 指标引用 ${opinion.metric_refs.length} 项` : ''}
        </div>
      ) : null}
      {opinion.error_message ? <div className="research-banner research-banner--error">{opinion.error_message}</div> : null}
      {opinion.observations && opinion.observations.length > 0 ? (
        <ul className="research-material-list">
          {opinion.observations.map((text, index) => (
            <li key={`${opinion.opinion_id}-${index}`} className="research-material-item">
              {text}
            </li>
          ))}
        </ul>
      ) : (
        <div className="research-empty">该路尚未产出结构化意见。</div>
      )}
      {opinion.missing_evidence && opinion.missing_evidence.length > 0 ? (
        <div className="review-opinion__missing">
          <strong>缺失证据</strong>
          <ul>
            {opinion.missing_evidence.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      ) : null}
      {opinion.validation_proposals && opinion.validation_proposals.length > 0 ? (
        <div className="review-opinion__missing">
          <strong>建议验证</strong>
          <ul>
            {opinion.validation_proposals.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </article>
  )
}

function DisputeCard({
  dispute,
  arbitration,
  onOpenEvidence,
}: {
  dispute: ResearchDispute
  arbitration?: ResearchArbitration
  onOpenEvidence: (packetId: string) => void
}) {
  return (
    <article className="dispute-card">
      <div className="research-check-item__top">
        <strong>{dispute.dispute_id}</strong>
        <span className={`badge ${dispute.status === 'validated' || dispute.status === 'dismissed' ? 'badge--success' : dispute.status === 'open' ? 'badge--warning' : ''}`}>
          {dispute.status}
        </span>
      </div>
      <h3>{dispute.question}</h3>
      <div className="detail-card__meta">
        <code>{dispute.issue_code}</code>
        {dispute.opinion_ids?.length ? ` · 意见 ${dispute.opinion_ids.join('、')}` : ''}
      </div>
      {dispute.evidence_ids && dispute.evidence_ids.length > 0 ? (
        <div className="research-actions">
          {dispute.evidence_ids.map((id) => (
            <button key={id} className="research-secondary" type="button" onClick={() => onOpenEvidence(id)}>
              打开证据包 {id}
            </button>
          ))}
        </div>
      ) : null}
      {arbitration ? (
        <div className="review-opinion__missing">
          <strong>仲裁建议：{arbitration.recommended_action}</strong>
          <p>{arbitration.issue_summary}</p>
          {arbitration.required_validation ? <p>需要验证：{arbitration.required_validation}</p> : null}
        </div>
      ) : null}
    </article>
  )
}

export function ReviewPage() {
  const { researchId } = useParams<{ researchId: string }>()
  const [project, setProject] = useState<ResearchProjectDetail | null>(null)
  const [review, setReview] = useState<ResearchReviewRecord | null>(null)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [drawer, setDrawer] = useState<{ packetId: string; evidenceKey?: string } | null>(null)

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
      setError(caught instanceof Error ? caught.message : '协作复核读取失败。')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void refresh()
  }, [researchId])

  async function startReview() {
    if (!project || !researchId) {
      return
    }
    try {
      setBusy(true)
      setError(null)
      setNotice(null)
      const result = await runResearchReview(researchId, project.revision)
      setProject(result.project)
      setReview(result.review)
      setNotice(
        `复核 ${result.review.review_id} 已启动：${result.review.is_stub ? '当前使用离线 stub，未调用真实模型。' : '使用已配置模型。'}`,
      )
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : '复核启动失败。')
    } finally {
      setBusy(false)
    }
  }

  async function retryReview(reviewId: string, roles: string[]) {
    if (!project || !researchId) {
      return
    }
    try {
      setBusy(true)
      setError(null)
      setNotice(null)
      const result = await retryResearchReview(researchId, project.revision, reviewId, {
        retry_roles: roles,
      })
      setProject(result.project)
      setReview(result.review)
      setNotice(`已按 ${roles.join('、')} 重试，复核记录更新为 revision ${result.project.revision}。`)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : '复核重试失败。')
    } finally {
      setBusy(false)
    }
  }

  if (!researchId) {
    return <Link className="research-secondary" to="/research">返回研究工作台</Link>
  }

  const opinions = (project?.opinions ?? []).map((item) => ({ ...item }))
  const questionerOpinions = opinions.filter((item) => item.role === 'scientific_questioner').reverse()
  const interpreterOpinions = opinions.filter((item) => item.role === 'scientific_interpreter').reverse()
  const openDisputes = (project?.disputes ?? []).filter(
    (item) => item.status !== 'validated' && item.status !== 'dismissed' && item.status !== 'superseded',
  )
  const failedRoles = opinions.filter((item) => item.status === 'failed').map((item) => item.role)
  const reviewJob = project?.jobs?.find((item) => item.kind === 'dual_review')

  const openDrawerForOpinion = (opinion: ReviewOpinion) => {
    setDrawer({ packetId: opinion.packet_hash })
  }
  const openDrawerForPacket = (packetId: string) => {
    setDrawer({ packetId })
  }
  const displayedPacket =
    project?.evidence_packets.find((packet) => packet.packet_hash === drawer?.packetId) ??
    project?.evidence_packets.find((packet) => packet.packet_id === drawer?.packetId) ??
    null

  return (
    <div className="timeline-shell">
      <div className="timeline-shell__frame timeline-shell__frame--detail">
        <main className="research-page">
          <header className="research-page__header">
            <div>
              <h1>协作复核</h1>
              <p>
                同一证据包、两条独立审读路线。意见、引用与分歧都登记在项目 revision 上，失败和部分状态可见。
              </p>
              <div className="research-page__chips">
                <span className="badge">{reviewJob?.status ?? '未运行'}</span>
                <span className="badge">{openDisputes.length} 个待处理分歧</span>
                <span className="badge">revision {project?.revision ?? '—'}</span>
              </div>
            </div>
            <div className="research-page__aside">
              <Link className="research-primary" to={`/research/${researchId}/validate`}>
                进入验证实验
              </Link>
              <button className="research-secondary" type="button" onClick={() => void refresh()}>
                刷新
              </button>
            </div>
          </header>

          <ResearchNav researchId={researchId} active="review" />

          {error ? <div className="research-banner research-banner--error">{error}</div> : null}
          {notice ? <div className="research-banner research-banner--success">{notice}</div> : null}
          {loading && !project ? <div className="research-banner">正在读取复核状态……</div> : null}

          <section className="detail-card">
            <div className="detail-card__title-row">
              <h2>启动或重试双路复核</h2>
              <span className="badge">最多 2 个并发请求</span>
            </div>
            <p className="detail-card__meta">
              复核前需要至少一个证据包和一条结论。数值引用、状态判定由程序执行，LLM 只负责解释与审查意见。
            </p>
            <div className="research-actions">
              <button className="research-primary" type="button" disabled={busy || !project} onClick={() => void startReview()}>
                {busy ? '复核处理中……' : '启动双路复核'}
              </button>
              {review ? (
                <button
                  className="research-secondary"
                  type="button"
                  disabled={busy || failedRoles.length === 0}
                  onClick={() => void retryReview(review.review_id, failedRoles)}
                >
                  重试失败角色（{failedRoles.join('、') || '无'}）
                </button>
              ) : null}
            </div>
            {review ? (
              <div className="research-fact-list" style={{ marginTop: 14 }}>
                <div className="research-fact-item">复核：{review.review_id}</div>
                <div className="research-fact-item">状态：{review.status} · {review.stage}</div>
                <div className="research-fact-item">
                  证据包哈希：<span className="research-page__mono">{review.packet_hash ?? '未指定'}</span>
                </div>
                <div className="research-fact-item">是否离线 stub：{review.is_stub ? '是' : '否'}</div>
              </div>
            ) : null}
          </section>

          <div className="review-columns">
            {[
              { role: 'scientific_questioner', list: questionerOpinions },
              { role: 'scientific_interpreter', list: interpreterOpinions },
            ].map(({ role, list }) => (
              <section key={role} className="detail-card review-column">
                <div className="detail-card__title-row">
                  <h2>{ROLE_META[role]?.label ?? role}</h2>
                  <span className="badge">{list.length} 条</span>
                </div>
                <p className="detail-card__meta">{ROLE_META[role]?.intro ?? ''}</p>
                {list.length > 0 ? (
                  <div className="review-opinion-stack">
                    {list.map((opinion) => (
                      <OpinionCard key={opinion.opinion_id} opinion={opinion} onOpenEvidence={openDrawerForOpinion} />
                    ))}
                  </div>
                ) : (
                  <div className="research-empty">该路还没有意见，启动复核后由对应角色独立生成。</div>
                )}
              </section>
            ))}
          </div>

          <section className="detail-card" style={{ marginTop: 18 }}>
            <div className="detail-card__title-row">
              <h2>争议清单</h2>
              <span className="badge">{openDisputes.length} 个未关闭</span>
            </div>
            {project?.disputes && project.disputes.length > 0 ? (
              <div className="dispute-grid">
                {project.disputes.map((dispute) => (
                  <DisputeCard
                    key={dispute.dispute_id}
                    dispute={dispute}
                    arbitration={project.arbitrations?.find((item) =>
                      item.dispute_ids?.includes(dispute.dispute_id),
                    )}
                    onOpenEvidence={openDrawerForPacket}
                  />
                ))}
              </div>
            ) : (
              <div className="research-empty">复核后没有产生可执行分歧，或尚未启动复核。</div>
            )}
          </section>

          {drawer ? (
            <EvidenceDrawer packet={displayedPacket} onClose={() => setDrawer(null)} />
          ) : null}
        </main>
      </div>
    </div>
  )
}
