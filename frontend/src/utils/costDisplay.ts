type NumericLike =
  | number
  | string
  | { value?: number | string; rationale?: string }
  | undefined
  | null

export type CostBreakdownInfo = {
  score: number
  fee?: string
  tokenTotal?: number
  duration?: string
  humanReview: boolean
  humanReviewDominant: boolean
  displayText: string
  tooltip: string
  paragraph: string
}

function toNumber(value: NumericLike): number {
  if (typeof value === 'number') {
    return value
  }
  if (typeof value === 'string') {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : 0
  }
  if (value && typeof value === 'object') {
    return toNumber(value.value)
  }
  return 0
}

function readRationale(value: NumericLike): string {
  if (value && typeof value === 'object' && typeof value.rationale === 'string') {
    return value.rationale
  }
  return ''
}

function firstMatch(source: string, patterns: RegExp[]): string | undefined {
  for (const pattern of patterns) {
    const match = source.match(pattern)
    if (match?.[1]) {
      return match[1]
    }
  }
  return undefined
}

export function getCostBreakdown(value: NumericLike): CostBreakdownInfo {
  const score = toNumber(value)
  const rationale = readRationale(value)
  const fee = firstMatch(rationale, [
    /估算约\s*([0-9]+(?:\.[0-9]+)?)\s*元/,
    /费用约\s*([0-9]+(?:\.[0-9]+)?)\s*元/,
  ])
  const tokenTotal = Number(firstMatch(rationale, [/合计\s*([0-9]+)/])) || undefined
  const duration = firstMatch(rationale, [
    /估算耗时(?:为)?\s*([0-9]+(?:\.[0-9]+)?)\s*s/,
    /计算时间\s*([0-9]+(?:\.[0-9]+)?)\s*s/,
  ])
  const humanReview = /人工审核/.test(rationale)
  const humanReviewDominant =
    /人工审核成本项(?:为|是)\s*1\.0+/.test(rationale) ||
    /C3=\s*1\.0+/.test(rationale) ||
    /主要成本压力来自人工审核/.test(rationale) ||
    /成本分主要由人工审核/.test(rationale)

  const detailParts: string[] = []
  if (fee) {
    detailParts.push(`真实调用约 ¥${fee}`)
  }
  if (tokenTotal) {
    detailParts.push(`合计 ${tokenTotal} tokens`)
  }
  if (duration) {
    detailParts.push(`估算耗时 ${duration}`)
  }
  if (humanReview) {
    detailParts.push('需人工审核')
  }

  const displayText = fee ? `¥${fee}` : detailParts[0] ?? `成本分 ${score.toFixed(3)}`
  const tooltipParts = [`成本分 ${score.toFixed(3)}`]
  if (fee) {
    tooltipParts.push(`真实调用约 ¥${fee}`)
  }
  if (tokenTotal) {
    tooltipParts.push(`合计 ${tokenTotal} tokens`)
  }
  if (duration) {
    tooltipParts.push(`估算耗时 ${duration}`)
  }
  if (humanReview) {
    tooltipParts.push('需人工审核')
  }

  const paragraph = detailParts.length > 0
    ? `成本评价：${detailParts.join('，')}。${humanReviewDominant ? '当前成本主要由人工审核项构成，模型调用与计算时间占比很低。' : ''}`
    : `成本评价：成本分 ${score.toFixed(3)}，量化比例见成本指标。`

  return {
    score,
    fee,
    tokenTotal,
    duration,
    humanReview,
    humanReviewDominant,
    displayText,
    tooltip: tooltipParts.join('；'),
    paragraph,
  }
}

export function getCostDisplayText(value: NumericLike): string {
  return getCostBreakdown(value).displayText
}

export function getCostTooltip(value: NumericLike): string {
  return getCostBreakdown(value).tooltip
}

export function getCostParagraph(value: NumericLike): string {
  return getCostBreakdown(value).paragraph
}

export function maskCostScores(text: string, info: CostBreakdownInfo): string {
  if (!text || !Number.isFinite(info.score) || info.score <= 0) {
    return text
  }
  const variants = Array.from(new Set([
    info.score.toFixed(4),
    info.score.toFixed(3),
    info.score.toFixed(2),
  ]))
  let output = text
  for (const variant of variants) {
    const escaped = variant.replace('.', '\\.')
    output = output.replace(
      new RegExp(`成本(?:分|分数)?(?:为|是)?\\s*${escaped}(?![0-9.])`, 'g'),
      '成本评价（真实费用与资源消耗见成本指标）',
    )
  }
  return output
}
