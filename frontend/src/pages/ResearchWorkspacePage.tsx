import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import {
  createResearchProject,
  getResearchProject,
  listResearchProjects,
  runResearchPreflight,
  uploadResearchArtifact,
} from '../api/researchApi'
import type {
  ArtifactKind,
  ArtifactRecord,
  ClaimVersion,
  EvidenceCheck,
  EvidencePacket,
  PreflightResult,
  ResearchDataMode,
  ResearchEntryMode,
  ResearchProjectDetail,
  ResearchProjectSummary,
} from '../types/research'

const STAGE_LABELS: Record<string, string> = {
  draft: '草稿',
  evidence_ready: '证据就绪',
  needs_input: '需要补充',
  awaiting_decision: '待人工决定',
  reviewing: '复核中',
  validated: '已验证',
}

const CHECK_STATUS_LABELS: Record<string, string> = {
  pass: '通过',
  fail: '未通过',
  unknown: '未知',
  not_applicable: '不适用',
}

const targetPreview = (project: { target: ResearchProjectDetail['project']['target'] }) => {
  const parts = [project.target.id]
  if (project.target.unit) {
    parts.push(project.target.unit)
  }
  if (project.target.coordinate_system) {
    parts.push(project.target.coordinate_system)
  }
  return parts.join(' · ')
}

function ResearchProjectList() {
  const navigate = useNavigate()
  const [projects, setProjects] = useState<ResearchProjectSummary[] | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [creating, setCreating] = useState(false)
  const [createError, setCreateError] = useState<string | null>(null)
  const [form, setForm] = useState({
    entryMode: 'new_question' as ResearchEntryMode,
    title: '',
    question: '',
    claimStatement: '',
    targetId: 'imf_by_gse',
    targetColumn: 'by_gse',
    unit: 'nT',
    coordinateSystem: 'GSE',
    horizonDays: '3',
    dataMode: 'historical_replay' as ResearchDataMode,
  })

  async function refresh() {
    try {
      setLoading(true)
      setError(null)
      setProjects(await listResearchProjects())
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : '研究项目列表读取失败。')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void refresh()
  }, [])

  async function createProject() {
    const title = form.title.trim()
    const question = form.question.trim()
    if (!title || !question) {
      setCreateError('请填写研究标题与研究问题。')
      return
    }
    if (form.entryMode === 'import_claim' && !form.claimStatement.trim()) {
      setCreateError('导入结论模式需要填写结论原文。')
      return
    }
    try {
      setCreating(true)
      setCreateError(null)
      const result = await createResearchProject({
        workflow_mode: 'research_review',
        entry_mode: form.entryMode,
        data_mode: form.dataMode,
        title,
        question,
        claim_statement: form.entryMode === 'import_claim' ? form.claimStatement.trim() : undefined,
        target: {
          id: form.targetId.trim(),
          column: form.targetColumn.trim() || undefined,
          unit: form.unit.trim() || undefined,
          coordinate_system: form.coordinateSystem.trim() || undefined,
        },
        horizon_days: Number(form.horizonDays) || 3,
      })
      navigate(`/research/${result.researchId}`)
    } catch (caught) {
      setCreateError(caught instanceof Error ? caught.message : '研究项目创建失败。')
    } finally {
      setCreating(false)
    }
  }

  return (
    <div className="timeline-shell">
      <div className="timeline-shell__frame timeline-shell__frame--detail">
        <main className="research-page">
          <header className="research-page__header">
            <div>
              <h1>研究工作台</h1>
              <p>
                将竞争解释转化为实验，用可追溯证据推进研究。以下项目均使用标准科研复核流程，
                数据模式、结论版本和证据包始终可追溯。
              </p>
            </div>
            <div className="research-page__aside">
              <span className="badge">research_review</span>
              <Link className="research-secondary" to="/workflow">
                返回原有闭环
              </Link>
            </div>
          </header>

          {error ? (
            <div className="research-banner research-banner--error">
              项目列表加载失败：{error}
              <div className="research-actions">
                <button className="research-secondary" type="button" onClick={() => void refresh()}>
                  重试
                </button>
              </div>
            </div>
          ) : null}

          <section className="detail-card research-page__grid__first">
            <div className="detail-card__title-row">
              <h2>已有研究</h2>
              <span className="badge">{loading ? '读取中' : `${projects?.length ?? 0} 项`}</span>
            </div>
            {loading ? (
              <div className="research-empty">正在从本地研究仓库读取项目……</div>
            ) : projects && projects.length > 0 ? (
              <ul className="research-material-list">
                {projects.map((item) => (
                  <li key={item.research_id} className="research-material-item">
                    <Link to={`/research/${item.research_id}`}>
                      <strong>{item.title}</strong>
                    </Link>
                    <p>{item.question}</p>
                    <div className="detail-card__meta">
                      {item.research_id} · {targetPreview(item)} · 提前量 {item.horizon_days} 天 ·
                      revision {item.revision} · {STAGE_LABELS[item.stage] ?? item.stage}
                      {item.data_mode !== 'authorized_data' ? ` · 数据模式 ${item.data_mode}` : ''}
                    </div>
                  </li>
                ))}
              </ul>
            ) : (
              <div className="research-empty">还没有研究项目，可以新建一个问题或导入一条已有结论。</div>
            )}
          </section>

          <section className="detail-card">
            <div className="detail-card__title-row">
              <h2>新建研究 / 导入结论</h2>
              <span className="badge">项目隔离</span>
            </div>
            <div className="research-form-grid">
              <div className="research-field">
                <label htmlFor="entry-mode">进入方式</label>
                <select
                  id="entry-mode"
                  value={form.entryMode}
                  onChange={(event) =>
                    setForm((prev) => ({ ...prev, entryMode: event.target.value as ResearchEntryMode }))
                  }
                >
                  <option value="new_question">提出新研究问题</option>
                  <option value="import_claim">导入已有结论</option>
                </select>
              </div>
              <div className="research-field">
                <label htmlFor="data-mode">数据模式</label>
                <select
                  id="data-mode"
                  value={form.dataMode}
                  onChange={(event) =>
                    setForm((prev) => ({ ...prev, dataMode: event.target.value as ResearchDataMode }))
                  }
                >
                  <option value="historical_replay">历史回放</option>
                  <option value="authorized_data">授权真实数据</option>
                  <option value="synthetic_fixture">合成夹具</option>
                  <option value="unknown">未声明</option>
                </select>
              </div>
              <div className="research-field research-field--wide">
                <label htmlFor="research-title">研究标题</label>
                <input
                  id="research-title"
                  value={form.title}
                  onChange={(event) => setForm((prev) => ({ ...prev, title: event.target.value }))}
                  placeholder="例如：LHAASO 日影对未来 IMF By 的增量预测研究"
                />
              </div>
              <div className="research-field research-field--wide">
                <label htmlFor="research-question">研究问题</label>
                <textarea
                  id="research-question"
                  value={form.question}
                  onChange={(event) => setForm((prev) => ({ ...prev, question: event.target.value }))}
                  placeholder="请描述想要回答的科学问题，以及当前证据与竞争解释。"
                />
              </div>
              {form.entryMode === 'import_claim' ? (
                <div className="research-field research-field--wide">
                  <label htmlFor="claim-statement">结论原文</label>
                  <textarea
                    id="claim-statement"
                    value={form.claimStatement}
                    onChange={(event) => setForm((prev) => ({ ...prev, claimStatement: event.target.value }))}
                    placeholder="粘贴需要登记和复核的结论原文。"
                  />
                </div>
              ) : null}
              <div className="research-field">
                <label htmlFor="target-id">目标标识</label>
                <input
                  id="target-id"
                  value={form.targetId}
                  onChange={(event) => setForm((prev) => ({ ...prev, targetId: event.target.value }))}
                />
              </div>
              <div className="research-field">
                <label htmlFor="target-column">目标列</label>
                <input
                  id="target-column"
                  value={form.targetColumn}
                  onChange={(event) => setForm((prev) => ({ ...prev, targetColumn: event.target.value }))}
                />
              </div>
              <div className="research-field">
                <label htmlFor="target-unit">单位</label>
                <input id="target-unit" value={form.unit} onChange={(event) => setForm((prev) => ({ ...prev, unit: event.target.value }))} />
              </div>
              <div className="research-field">
                <label htmlFor="coordinate-system">坐标系</label>
                <input
                  id="coordinate-system"
                  value={form.coordinateSystem}
                  onChange={(event) => setForm((prev) => ({ ...prev, coordinateSystem: event.target.value }))}
                />
              </div>
              <div className="research-field">
                <label htmlFor="horizon-days">预测提前量（天）</label>
                <input
                  id="horizon-days"
                  type="number"
                  min={1}
                  max={30}
                  value={form.horizonDays}
                  onChange={(event) => setForm((prev) => ({ ...prev, horizonDays: event.target.value }))}
                />
              </div>
            </div>
            {createError ? <div className="research-banner research-banner--error">{createError}</div> : null}
            <div className="research-actions">
              <button className="research-primary" type="button" disabled={creating} onClick={() => void createProject()}>
                {creating ? '创建中……' : '创建研究项目'}
              </button>
            </div>
          </section>
        </main>
      </div>
    </div>
  )
}

interface UploadDraft {
  file_name: string
  kind: ArtifactKind
  content: string
  artifact_id: string
}

function ClaimCard({ claim }: { claim: ClaimVersion }) {
  return (
    <li className="research-material-item">
      <div className="research-check-item__top">
        <strong>{claim.claim_id} v{claim.version}</strong>
        <span className="badge">{claim.status}</span>
      </div>
      <p>{claim.statement}</p>
      <div className="detail-card__meta">
        关联证据：{claim.evidence_ids.length > 0 ? claim.evidence_ids.join('、') : '暂无'}
        {claim.scope?.sample_condition ? ` · 样本条件：${claim.scope.sample_condition}` : ''}
        {claim.scope?.evaluation_period ? ` · 期限：${JSON.stringify(claim.scope.evaluation_period)}` : ''}
      </div>
    </li>
  )
}

function CheckRow({ check }: { check: EvidenceCheck }) {
  return (
    <li className="research-check-item">
      <div className="research-check-item__top">
        <code>{check.code}</code>
        <span
          className={`badge ${
            check.status === 'pass'
              ? 'badge--success'
              : check.status === 'fail'
                ? 'badge--high'
                : check.status === 'unknown'
                  ? 'badge--warning'
                  : ''
          }`}
        >
          {CHECK_STATUS_LABELS[check.status] ?? check.status}
        </span>
      </div>
      <p>{check.detail ?? check.reason ?? '暂无说明。'}</p>
    </li>
  )
}

function PacketMetrics({ packet }: { packet: EvidencePacket }) {
  const entries = Object.entries(packet.metrics)
  if (entries.length === 0) {
    return <div className="research-empty">该证据包没有已登记指标；未计算的数据不会被虚报为 0。</div>
  }
  return (
    <div className="research-metrics">
      {entries.map(([key, metric]) => (
        <div key={key} className="metric-box">
          <div className="metric-box__label">{key}</div>
          <div className="metric-box__value">
            {metric.value === null || metric.value === undefined ? '—' : Number(metric.value).toPrecision(4)}
          </div>
          <div className="detail-card__meta">
            {metric.unit ?? '无单位'}
            {metric.metric_ref ? ` · ref ${metric.metric_ref}` : ''}
          </div>
        </div>
      ))}
    </div>
  )
}

export function ResearchWorkspacePage() {
  const { researchId } = useParams<{ researchId?: string }>()
  if (!researchId) {
    return <ResearchProjectList />
  }
  return <ResearchProjectDetailView researchId={researchId} />
}

function ResearchProjectDetailView({ researchId }: { researchId: string }) {
  const [project, setProject] = useState<ResearchProjectDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [uploadDraft, setUploadDraft] = useState<UploadDraft>({
    file_name: '',
    kind: 'prediction_csv',
    content: '',
    artifact_id: '',
  })
  const [featureColumns, setFeatureColumns] = useState('shadow_shift')
  const [baseArm, setBaseArm] = useState('base')
  const [augArm, setAugArm] = useState('aug')
  const [protocolHash, setProtocolHash] = useState('')
  const [runId, setRunId] = useState('')
  const [uploadBusy, setUploadBusy] = useState(false)
  const [preflightBusy, setPreflightBusy] = useState(false)
  const [actionError, setActionError] = useState<string | null>(null)
  const [actionNotice, setActionNotice] = useState<string | null>(null)
  const [preflightResult, setPreflightResult] = useState<PreflightResult | null>(null)

  async function refresh() {
    try {
      setLoading(true)
      setError(null)
      setProject(await getResearchProject(researchId))
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : '研究项目读取失败。')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void refresh()
  }, [researchId])

  const latestArtifact = useMemo(() => {
    const list = project?.artifacts ?? []
    return list.length > 0 ? list[list.length - 1] : null
  }, [project?.artifacts])

  const latestPacket = useMemo(() => {
    const list = project?.evidence_packets ?? []
    return list.length > 0 ? list[list.length - 1] : null
  }, [project?.evidence_packets])

  async function handleFileSelected(file: File | undefined) {
    if (!file) {
      return
    }
    setActionError(null)
    const text = await file.text()
    setUploadDraft((prev) => ({
      ...prev,
      file_name: file.name,
      content: text,
      kind: file.name.toLowerCase().endsWith('.json') ? 'evidence_json' : 'prediction_csv',
      artifact_id: '',
    }))
  }

  async function uploadArtifact() {
    if (!project) {
      return
    }
    if (!uploadDraft.file_name || !uploadDraft.content.trim()) {
      setActionError('请先选择或粘贴证据产物内容。')
      return
    }
    try {
      setUploadBusy(true)
      setActionError(null)
      setActionNotice(null)
      setPreflightResult(null)
      const result = await uploadResearchArtifact(researchId, project.revision, {
        file_name: uploadDraft.file_name,
        kind: uploadDraft.kind,
        content: uploadDraft.content,
      })
      setProject(result.project)
      setUploadDraft((prev) => ({ ...prev, artifact_id: result.artifact.artifact_id }))
      setActionNotice(`产物已登记：${result.artifact.artifact_id}（revision ${result.project.revision}）。`)
    } catch (caught) {
      setActionError(caught instanceof Error ? caught.message : '产物导入失败。')
    } finally {
      setUploadBusy(false)
    }
  }

  async function runPreflight() {
    if (!project) {
      return
    }
    try {
      setPreflightBusy(true)
      setActionError(null)
      setActionNotice(null)
      const artifactId = uploadDraft.artifact_id || latestArtifact?.artifact_id
      if (!artifactId) {
        throw new Error('请先导入至少一份预测 CSV。')
      }
      const result = await runResearchPreflight(researchId, project.revision, {
        artifact_id: artifactId,
        feature_columns: featureColumns
          .split(/[,，\n]/)
          .map((item) => item.trim())
          .filter(Boolean),
        base_arm: baseArm.trim() || undefined,
        aug_arm: augArm.trim() || undefined,
        protocol_hash_value: protocolHash.trim() || undefined,
        run_id: runId.trim() || undefined,
      })
      setProject(result.project)
      setPreflightResult(result.preflight)
      setActionNotice(
        result.preflight.status === 'ready'
          ? '预检完成，证据包已生成并通过阻塞检查。'
          : '预检完成但存在需要补充的检查项，证据包仍已登记。',
      )
    } catch (caught) {
      setActionError(caught instanceof Error ? caught.message : '证据预检失败。')
    } finally {
      setPreflightBusy(false)
    }
  }

  if (loading && !project) {
    return (
      <div className="timeline-shell">
        <div className="timeline-shell__frame timeline-shell__frame--detail">
          <main className="research-page">
            <div className="research-banner">正在读取研究项目 {researchId} ……</div>
          </main>
        </div>
      </div>
    )
  }

  if (error || !project) {
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
                <Link className="research-secondary" to="/research">
                  返回研究工作台
                </Link>
              </div>
            </div>
          </main>
        </div>
      </div>
    )
  }

  const blockingIssues = preflightResult?.blocking_issues ?? project.checks.filter((check) => check.blocking && check.status !== 'pass')

  return (
    <div className="timeline-shell">
      <div className="timeline-shell__frame timeline-shell__frame--detail">
        <main className="research-page">
          <header className="research-page__header">
            <div>
              <h1>{project.project.title}</h1>
              <p>{project.project.question}</p>
              <div className="research-page__chips">
                <span className="badge">{STAGE_LABELS[project.pc_stage] ?? project.pc_stage}</span>
                <span className="badge">revision {project.revision}</span>
                <span className="badge">数据模式 {project.project.data_mode}</span>
                <span className="badge">提前量 {project.project.horizon_days} 天</span>
                <span className="badge">{targetPreview(project.project)}</span>
              </div>
            </div>
            <div className="research-page__aside">
              <Link className="research-primary" to={`/research/${researchId}/evidence`}>
                打开证据中心
              </Link>
              <button className="research-secondary" type="button" onClick={() => void refresh()}>
                刷新项目
              </button>
              <span className="detail-card__meta">{project.project.research_id}</span>
            </div>
          </header>

          {actionError ? <div className="research-banner research-banner--error">{actionError}</div> : null}
          {actionNotice ? <div className="research-banner research-banner--success">{actionNotice}</div> : null}

          <div className="research-page__grid">
            <section className="detail-card">
              <div className="detail-card__title-row">
                <h2>当前结论</h2>
                <span className="badge">{project.claim_count} 条</span>
              </div>
              <p className="detail-card__meta">
                结论未经双路复核前保持 unassessed，系统不会默认判定成立。
              </p>
              {project.claims.length > 0 ? (
                <ul className="research-material-list">
                  {project.claims.map((claim) => (
                    <ClaimCard key={claim.claim_id} claim={claim} />
                  ))}
                </ul>
              ) : (
                <div className="research-empty">尚未登记结论。可在研究工作台通过“导入已有结论”创建带初始版本的项目。</div>
              )}
              <div className="research-fact-list" style={{ marginTop: 14 }}>
                <div className="research-fact-item">待处理分歧：{project.open_dispute_count}</div>
                <div className="research-fact-item">是否等待人工决定：{project.awaiting_decision ? '是' : '否'}</div>
              </div>
            </section>

            <section className="detail-card">
              <div className="detail-card__title-row">
                <h2>证据导入与预检</h2>
                <span className="badge">{project.evidence_packet_count} 个证据包</span>
              </div>
              <div className="research-form-grid">
                <div className="research-field research-field--wide">
                  <label htmlFor="artifact-file">预测 CSV / 证据 JSON</label>
                  <input
                    id="artifact-file"
                    type="file"
                    accept=".csv,.txt,.json"
                    onChange={(event) => void handleFileSelected(event.target.files?.[0])}
                  />
                </div>
                <div className="research-field">
                  <label htmlFor="artifact-kind">产物类型</label>
                  <select
                    id="artifact-kind"
                    value={uploadDraft.kind}
                    onChange={(event) =>
                      setUploadDraft((prev) => ({ ...prev, kind: event.target.value as ArtifactKind }))
                    }
                  >
                    <option value="prediction_csv">预测结果 CSV</option>
                    <option value="evidence_json">证据 JSON</option>
                    <option value="data_csv">数据 CSV</option>
                    <option value="protocol_json">协议 JSON</option>
                    <option value="report">报告</option>
                  </select>
                </div>
                <div className="research-field">
                  <label htmlFor="artifact-name">文件名</label>
                  <input
                    id="artifact-name"
                    value={uploadDraft.file_name}
                    onChange={(event) => setUploadDraft((prev) => ({ ...prev, file_name: event.target.value }))}
                  />
                </div>
                <div className="research-field research-field--wide">
                  <label htmlFor="artifact-content">内容（或粘贴文本）</label>
                  <textarea
                    id="artifact-content"
                    value={uploadDraft.content}
                    onChange={(event) => setUploadDraft((prev) => ({ ...prev, content: event.target.value }))}
                    placeholder="选择文件后自动填充；也可以直接粘贴 CSV/JSON 原文。"
                  />
                </div>
              </div>
              <div className="research-actions">
                <button className="research-primary" type="button" disabled={uploadBusy} onClick={() => void uploadArtifact()}>
                  {uploadBusy ? '导入中……' : '登记产物'}
                </button>
              </div>
              {latestArtifact ? (
                <p className="detail-card__meta" style={{ marginTop: 10 }}>
                  最近产物：{latestArtifact.file_name} · SHA-256{' '}
                  <span className="research-page__mono">{latestArtifact.sha256?.slice(0, 16) ?? '未登记'}</span>
                </p>
              ) : null}
            </section>
          </div>

          <div className="research-page__grid">
            <section className="detail-card">
              <div className="detail-card__title-row">
                <h2>预检配置</h2>
                <span className="badge">程序检查</span>
              </div>
              <div className="research-form-grid">
                <div className="research-field">
                  <label htmlFor="preflight-artifact">证据产物</label>
                  <select
                    id="preflight-artifact"
                    value={uploadDraft.artifact_id || latestArtifact?.artifact_id || ''}
                    onChange={(event) => setUploadDraft((prev) => ({ ...prev, artifact_id: event.target.value }))}
                  >
                    {(project.artifacts.length > 0
                      ? project.artifacts
                      : [{ artifact_id: '', file_name: '（无产物）' } as ArtifactRecord]
                    ).map((item) => (
                      <option key={item.artifact_id} value={item.artifact_id}>
                        {item.file_name} · {item.artifact_id}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="research-field">
                  <label htmlFor="preflight-features">特征列（逗号分隔）</label>
                  <input
                    id="preflight-features"
                    value={featureColumns}
                    onChange={(event) => setFeatureColumns(event.target.value)}
                  />
                </div>
                <div className="research-field">
                  <label htmlFor="preflight-base">基准组</label>
                  <input id="preflight-base" value={baseArm} onChange={(event) => setBaseArm(event.target.value)} />
                </div>
                <div className="research-field">
                  <label htmlFor="preflight-aug">增强组</label>
                  <input id="preflight-aug" value={augArm} onChange={(event) => setAugArm(event.target.value)} />
                </div>
                <div className="research-field">
                  <label htmlFor="preflight-protocol">协议哈希（可选）</label>
                  <input
                    id="preflight-protocol"
                    value={protocolHash}
                    onChange={(event) => setProtocolHash(event.target.value)}
                    placeholder="留空则使用项目最后登记的协议"
                  />
                </div>
                <div className="research-field">
                  <label htmlFor="preflight-run">run_id（可选）</label>
                  <input id="preflight-run" value={runId} onChange={(event) => setRunId(event.target.value)} />
                </div>
              </div>
              <div className="research-actions">
                <button className="research-primary" type="button" disabled={preflightBusy} onClick={() => void runPreflight()}>
                  {preflightBusy ? '预检中……' : '运行证据预检'}
                </button>
              </div>
            </section>

            <section className="detail-card">
              <div className="detail-card__title-row">
                <h2>证据就绪状态</h2>
                <span className="badge">{blockingIssues.length > 0 ? '需补充' : '就绪'}</span>
              </div>
              <p className="detail-card__meta">
                数值/状态均由程序检查生成；未提供的字段显示 unknown，而不是假定通过。
              </p>
              {preflightResult ? (
                <div className="research-fact-list">
                  <div className="research-fact-item">预检状态：{preflightResult.status}</div>
                  <div className="research-fact-item">证据包：{preflightResult.packet.packet_id}</div>
                  <div className="research-fact-item">
                    证据哈希：<span className="research-page__mono">{preflightResult.packet.packet_hash.slice(0, 24)}</span>
                  </div>
                  <div className="research-fact-item">行数：{preflightResult.summary.row_count}</div>
                  <div className="research-fact-item">组别：{preflightResult.summary.arm_values.join('、') || '未识别'}</div>
                </div>
              ) : (
                <div className="research-empty">尚未运行预检；结果会与导出 JSON 保持一致。</div>
              )}
            </section>
          </div>

          <section className="detail-card" style={{ marginTop: 18 }}>
            <div className="detail-card__title-row">
              <h2>程序检查</h2>
              <span className="badge">{project.checks.length} 项</span>
            </div>
            {project.checks.length > 0 ? (
              <ul className="research-check-list">
                {project.checks.map((check) => (
                  <CheckRow key={`${check.packet_id ?? ''}-${check.code}`} check={check} />
                ))}
              </ul>
            ) : (
              <div className="research-empty">暂无检查结果，请先导入产物并运行预检。</div>
            )}
          </section>

          <section className="detail-card" style={{ marginTop: 18 }}>
            <div className="detail-card__title-row">
              <h2>最近证据包</h2>
              {latestPacket ? <span className="badge">{latestPacket.packet_id}</span> : null}
            </div>
            {latestPacket ? (
              <>
                <PacketMetrics packet={latestPacket} />
                {latestPacket.limitations.length > 0 ? (
                  <ul className="detail-list" style={{ marginTop: 14 }}>
                    {latestPacket.limitations.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                ) : null}
              </>
            ) : (
              <div className="research-empty">暂无证据包。导入预测 CSV 并完成预检后会在此显示真实指标。</div>
            )}
          </section>
        </main>
      </div>
    </div>
  )
}
