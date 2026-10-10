import type { EvidencePacket } from '../types/research'

interface EvidenceTimelineProps {
  packets: EvidencePacket[]
  onOpenPacket: (packet: EvidencePacket) => void
}

function formatTime(value?: string): string {
  if (!value) {
    return '时间未登记'
  }
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return value
  }
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`
}

export function EvidenceTimeline({ packets, onOpenPacket }: EvidenceTimelineProps) {
  if (packets.length === 0) {
    return (
      <div className="research-empty">
        还没有证据包的时间轴。导入预测 CSV 并运行预检后，这里会显示证据包、检查与局限的位置。
      </div>
    )
  }
  return (
    <div className="evidence-timeline">
      <div className="evidence-timeline__rail" aria-hidden="true">
        <span className="evidence-timeline__line" />
        {packets.map((packet) => (
          <button
            key={packet.packet_id}
            type="button"
            className="evidence-timeline__mark"
            title={`打开 ${packet.packet_id}`}
            onClick={() => onOpenPacket(packet)}
          />
        ))}
      </div>
      <div className="evidence-timeline__items">
        {packets.map((packet) => {
          const checkCount = packet.checks.length
          const failCount = packet.checks.filter((check) => check.status === 'fail').length
          return (
            <button
              key={packet.packet_id}
              type="button"
              className="evidence-timeline__item"
              onClick={() => onOpenPacket(packet)}
            >
              <strong>{formatTime(packet.created_at)}</strong>
              <span>{packet.packet_id}</span>
              <span className="detail-card__meta">
                {checkCount} 项检查
                {failCount > 0 ? ` · ${failCount} 项失败` : ''}
                {packet.limitations.length > 0 ? ` · ${packet.limitations.length} 项局限` : ''}
              </span>
            </button>
          )
        })}
      </div>
    </div>
  )
}
