import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { getResearchProject } from '../api/researchApi'
import { EvidenceDrawer } from '../components/EvidenceDrawer'
import { EvidenceTimeline } from '../components/EvidenceTimeline'
import { ResearchNav } from '../components/ResearchNav'
import type {
  EvidenceCheck,
  EvidencePacket,
  ResearchProjectDetail,
} from '../types/research'

type EvidenceTab = 'data_time' | 'protocol' | 'results' | 'sources'

const TAB_LABELS: Array<{ id: EvidenceTab; label: string }> = [
  { id: 'data_time', label: '数据与时间' },
  { id: 'protocol', label: '实验协议' },
  { id: 'results', label: '结果与稳健性' },
  { id: 'sources', label: '材料来源' },
]

const CHECK_LABELS: Record<string, string> = {
  TIME_TARGET: '日历提前量',
  FEATURE_AVAILABLE: '特征可用时间',
  TARGET_ALIAS: '目标别名泄漏',
  SPLIT_BOUNDARY: '训练/评价边界',
  FIT_SCOPE: '折内预处理',
  PAIRED_SAMPLES: '共同样本配对',
  TARGET_EQUAL: '真实目标一致',
  DATA_VERSION: '数据与协议版本',
  QUALITY_COVERAGE: '日影质量覆盖',
}

const STATUS_LABELS: Record<string, string> = {
  pass: '通过',
  fail: '未通过',
  unknown: '未知',
  not_applicable: '不适用',
}

function CheckLine({ check }: { check: EvidenceCheck }) {
  return (
    <li className="research-check-item">
      <div className="research-check-item__top">
        <code>{CHECK_LABELS[check.code] ?? check.code}</code>
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
          {STATUS_LABELS[check.status] ?? check.status}
        </span>
      </div>
      <p>{check.detail ?? check.reason ?? '暂无说明。'}</p>
    </li>
  )
}

export function EvidenceCenterPage() {
  const { researchId } = useParams<{ researchId: string }>()
  const [project, setProject] = useState<ResearchProjectDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<EvidenceTab>('data_time')
  const [drawer, setDrawer] = useState<{ packetHash?: string; evidenceKey?: string } | null>(null)

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
      setError(caught instanceof Error ? caught.message : '证据中心读取失败。')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void refresh()
  }, [researchId])

  const packets = useMemo(() => project?.evidence_packets ?? [], [project?.evidence_packets])
  const packet: EvidencePacket | null = useMemo(() => {
    if (drawer?.packetHash) {
      return packets.find(
        (item) => item.packet_hash === drawer.packetHash || item.packet_id === drawer.packetHash,
      ) ?? null
    }
    return packets.length > 0 ? packets[packets.length - 1] : null
  }, [packets, drawer?.packetHash])

  const drawerEvidence = useMemo(() => {
    if (!drawer?.evidenceKey) {
      return null
    }
    return packet?.evidence_items.find((item) => item.evidence_id === drawer.evidenceKey) ?? null
  }, [drawer?.evidenceKey, packet])

  if (loading && !project) {
    return (
      <div className="timeline-shell">
        <div className="timeline-shell__frame timeline-shell__frame--detail">
          <main className="research-page">
            <div className="research-banner">正在读取证据中心……</div>
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
                {researchId ? (
                  <Link className="research-secondary" to={`/research/${researchId}`}>
                    返回项目总览
                  </Link>
                ) : null}
              </div>
            </div>
          </main>
        </div>
      </div>
    )
  }

  return (
    <div className="timeline-shell">
      <div className="timeline-shell__frame timeline-shell__frame--detail">
        <main className="research-page">
          <header className="research-page__header">
            <div>
              <h1>证据中心</h1>
              <p>{project.project.title}：证据包、检查结果与指标引用均来自真实 API 数据。</p>
              <div className="research-page__chips">
                <span className="badge">{project.evidence_packet_count} 个证据包</span>
                <span className="badge">{project.artifacts.length} 份产物</span>
                <span className="badge">revision {project.revision}</span>
              </div>
            </div>
            <div className="research-page__aside">
              <Link className="research-primary" to={`/research/${project.project.research_id}`}>
                返回项目总览
              </Link>
              <button className="research-secondary" type="button" onClick={() => void refresh()}>
                刷新
              </button>
            </div>
          </header>

          <ResearchNav researchId={project.project.research_id} active="evidence" />

          <div className="research-tabbar" style={{ marginTop: 18 }}>
            {TAB_LABELS.map((tab) => (
              <button
                key={tab.id}
                type="button"
                aria-selected={activeTab === tab.id}
                onClick={() => setActiveTab(tab.id)}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {!packet ? (
            <div className="research-empty">
              当前项目还没有证据包。请先在项目总览导入预测 CSV 并运行证据预检。
            </div>
          ) : null}

          {activeTab === 'data_time' ? (
            <section className="detail-card">
              <div className="detail-card__title-row">
                <h2>证据时间轴</h2>
                <span className="badge">{packets.length} 个证据包</span>
              </div>
              <EvidenceTimeline packets={packets} onOpenPacket={(item) => setDrawer({ packetHash: item.packet_hash })} />
            </section>
          ) : null}

          {activeTab === 'data_time' ? (
            <section className="detail-card">
              <div className="detail-card__title-row">
                <h2>数据与时间</h2>
                <span className="badge">{packet?.packet_id ?? '—'}</span>
              </div>
              {packet ? (
                <>
                  <p className="detail-card__meta">
                    只上传预测 CSV 时，训练来源、折内预处理和特征发布时间会标记 unknown，
                    不能被自动当作已验证。
                  </p>
                  <ul className="research-check-list">
                    {packet.checks
                      .filter((check) =>
                        ['TIME_TARGET', 'FEATURE_AVAILABLE', 'SPLIT_BOUNDARY', 'FIT_SCOPE', 'PAIRED_SAMPLES', 'TARGET_EQUAL'].includes(
                          check.code,
                        ),
                      )
                      .map((check) => (
                        <CheckLine key={check.code} check={check} />
                      ))}
                  </ul>
                  <div className="research-fact-list" style={{ marginTop: 14 }}>
                    <div className="research-fact-item">run_id：{packet.run_id ?? '未登记'}</div>
                    <div className="research-fact-item">
                      证据哈希（packet_hash）：
                      <span className="research-page__mono">{packet.packet_hash}</span>
                    </div>
                  </div>
                </>
              ) : null}
            </section>
          ) : null}

          {activeTab === 'protocol' ? (
            <section className="detail-card">
              <div className="detail-card__title-row">
                <h2>实验协议</h2>
                <span className="badge">版本绑定</span>
              </div>
              {packet ? (
                <>
                  <p className="detail-card__meta">
                    证据包保存了执行时绑定的协议与数据版本；协议快照、时间规格与样本计划将在复核/验证阶段扩展。
                  </p>
                  <div className="research-fact-list">
                    <div className="research-fact-item">
                      protocol_hash：
                      <span className="research-page__mono">{packet.protocol_hash ?? '未提供'}</span>
                    </div>
                    <div className="research-fact-item">
                      data_manifest_hash：
                      <span className="research-page__mono">{packet.data_manifest_hash ?? '未提供'}</span>
                    </div>
                    <div className="research-fact-item">关联结论：{packet.claim_ids.join('、') || '未绑定'}</div>
                    <div className="research-fact-item">目标：{project.project.target.id}</div>
                    <div className="research-fact-item">提前量：{project.project.horizon_days} 天</div>
                  </div>
                </>
              ) : null}
            </section>
          ) : null}

          {activeTab === 'results' ? (
            <section className="detail-card">
              <div className="detail-card__title-row">
                <h2>结果与稳健性</h2>
                <span className="badge">程序值</span>
              </div>
              {packet ? (
                <>
                  <p className="detail-card__meta">
                    RMSE、MAE 等指标由证据构建程序在同一共同样本上计算；文本说明不替代数值引用。
                  </p>
                  <div className="research-metrics">
                    {Object.entries(packet.metrics).map(([key, metric]) => (
                      <button
                        key={key}
                        type="button"
                        className="metric-box metric-box--button"
                        onClick={() => setDrawer({ packetHash: packet.packet_hash, evidenceKey: key })}
                      >
                        <div className="metric-box__label">{key}</div>
                        <div className="metric-box__value">
                          {metric.value === null || metric.value === undefined
                            ? '—'
                            : Number(metric.value).toPrecision(4)}
                        </div>
                        <div className="detail-card__meta">
                          {metric.unit ?? '无单位'}
                          {metric.reason ? ` · ${metric.reason}` : ''}
                        </div>
                      </button>
                    ))}
                  </div>
                  {packet.limitations.length > 0 ? (
                    <ul className="detail-list" style={{ marginTop: 14 }}>
                      {packet.limitations.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  ) : null}
                </>
              ) : null}
            </section>
          ) : null}

          {activeTab === 'sources' ? (
            <section className="detail-card">
              <div className="detail-card__title-row">
                <h2>材料来源</h2>
                <span className="badge">{project.artifacts.length} 份</span>
              </div>
              {project.artifacts.length > 0 ? (
                <ul className="research-material-list">
                  {project.artifacts.map((artifact) => (
                    <li key={artifact.artifact_id} className="research-material-item">
                      <div className="research-check-item__top">
                        <strong>{artifact.file_name}</strong>
                        <span className={`badge ${artifact.parsed ? 'badge--success' : 'badge--high'}`}>
                          {artifact.parsed ? '可解析' : `解析失败 ${artifact.parse_errors.length} 项`}
                        </span>
                      </div>
                      <p>{artifact.artifact_id} · {artifact.kind}</p>
                      <div className="detail-card__meta">
                        SHA-256 <span className="research-page__mono">{artifact.sha256 ?? '未登记'}</span>
                        {artifact.parse_errors.length > 0 ? (
                          <ul className="detail-list">
                            {artifact.parse_errors.map((item) => (
                              <li key={item}>{item}</li>
                            ))}
                          </ul>
                        ) : null}
                      </div>
                    </li>
                  ))}
                </ul>
              ) : (
                <div className="research-empty">尚未登记证据产物。</div>
              )}
              {packet && packet.evidence_items.length > 0 ? (
                <>
                  <h2 style={{ fontSize: 18, margin: '18px 0 10px' }}>证据项</h2>
                  <ul className="research-material-list">
                    {packet.evidence_items.map((item) => (
                      <li key={item.evidence_id} className="research-material-item evidence-item-button">
                        <button type="button" onClick={() => setDrawer({ packetHash: packet.packet_hash, evidenceKey: item.evidence_id })}>
                          <div className="research-check-item__top">
                            <code>{item.evidence_id}</code>
                            <span className="badge">{item.value_type}</span>
                          </div>
                          <p>
                            {String(item.value ?? '—')}
                            {item.unit ? ` ${item.unit}` : ''}
                          </p>
                          <div className="detail-card__meta">
                            {item.locator ?? '无定位'} · {item.provenance ?? '来源未知'} · 打开证据抽屉
                          </div>
                        </button>
                      </li>
                    ))}
                  </ul>
                </>
              ) : null}
            </section>
          ) : null}

          {drawer ? (
            <EvidenceDrawer
              packet={packet}
              evidence={drawerEvidence}
              metricKey={drawer.evidenceKey}
              onClose={() => setDrawer(null)}
            />
          ) : null}
        </main>
      </div>
    </div>
  )
}
