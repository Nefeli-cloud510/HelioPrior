import type { EvidenceItem, EvidencePacket, MetricValue } from '../types/research'

interface EvidenceDrawerProps {
  packet: EvidencePacket | null
  evidence?: EvidenceItem | null
  metricKey?: string
  onClose: () => void
}

function Handle({ label, value }: { label: string; value: string | number | null | undefined }) {
  return (
    <div className="evidence-drawer__item">
      <dt>{label}</dt>
      <dd className={typeof value === 'string' && value.length > 18 ? 'research-page__mono' : undefined}>
        {value === null || value === undefined || value === '' ? '—' : String(value)}
      </dd>
    </div>
  )
}

export function EvidenceDrawer({ packet, evidence, metricKey, onClose }: EvidenceDrawerProps) {
  const metric: MetricValue | undefined = packet?.metrics?.[metricKey ?? '']
  return (
    <div className="evidence-drawer__backdrop" role="presentation" onClick={onClose}>
      <aside
        className="evidence-drawer"
        role="dialog"
        aria-label="证据抽屉"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="evidence-drawer__head">
          <div>
            <h3>{evidence ? `证据项 ${evidence.evidence_id}` : metricKey ? `指标 ${metricKey}` : '证据包详情'}</h3>
            <p>{packet?.packet_id ?? '未绑定证据包'}</p>
          </div>
          <button type="button" className="evidence-drawer__close" onClick={onClose} aria-label="关闭证据抽屉">
            关闭
          </button>
        </div>

        {packet ? (
          <dl className="evidence-drawer__grid">
            <Handle label="证据哈希" value={packet.packet_hash} />
            <Handle label="数据清单哈希" value={packet.data_manifest_hash} />
            <Handle label="协议哈希" value={packet.protocol_hash} />
            <Handle label="run_id" value={packet.run_id} />
            <Handle label="关联结论" value={packet.claim_ids.join('、') || '未绑定'} />
            <Handle label="生成时间" value={packet.created_at} />
          </dl>
        ) : null}

        {metric ? (
          <section className="evidence-drawer__section">
            <h4>指标引用</h4>
            <dl className="evidence-drawer__grid">
              <Handle label="数值" value={metric.value} />
              <Handle label="单位" value={metric.unit} />
              <Handle label="口径引用" value={metric.metric_ref} />
            </dl>
            {metric.reason ? <p className="evidence-drawer__note">{metric.reason}</p> : null}
          </section>
        ) : null}

        {evidence ? (
          <section className="evidence-drawer__section">
            <h4>证据项</h4>
            <dl className="evidence-drawer__grid">
              <Handle label="类型" value={evidence.value_type} />
              <Handle label="数值" value={evidence.value} />
              <Handle label="单位" value={evidence.unit} />
              <Handle label="定位" value={evidence.locator} />
              <Handle label="来源" value={evidence.provenance} />
              <Handle label="来源文件" value={evidence.artifact_id} />
              <Handle label="产物哈希" value={evidence.artifact_sha256} />
            </dl>
          </section>
        ) : null}

        {packet && packet.checks.length > 0 ? (
          <section className="evidence-drawer__section">
            <h4>程序检查</h4>
            <ul className="evidence-drawer__checks">
              {packet.checks.map((check) => (
                <li key={check.code}>
                  <span className={`badge ${check.status === 'pass' ? 'badge--success' : check.status === 'fail' ? 'badge--high' : 'badge--warning'}`}>
                    {check.status}
                  </span>
                  <span>
                    <code>{check.code}</code>
                    {check.detail || check.reason ? ` · ${check.detail ?? check.reason}` : ''}
                  </span>
                </li>
              ))}
            </ul>
          </section>
        ) : null}

        {packet && packet.limitations.length > 0 ? (
          <section className="evidence-drawer__section">
            <h4>局限说明</h4>
            <ul className="evidence-drawer__checks">
              {packet.limitations.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </section>
        ) : null}

        <div className="evidence-drawer__foot">
          <span>
            引用白名单：仅展示证据包内已登记的值；未计算的指标不会显示为结论。
          </span>
        </div>
      </aside>
    </div>
  )
}
