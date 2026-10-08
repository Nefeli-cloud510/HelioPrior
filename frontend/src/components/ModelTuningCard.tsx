import { useEffect, useMemo, useState } from 'react'

import { fetchMethodCatalog } from '../api/liveWorkflow'
import type { ExperimentEvaluationViewModel } from '../types/timeline'

type TuningEntry = NonNullable<ExperimentEvaluationViewModel['tuningEntries']>[number]

const STATIC_PARAMETER_LABELS: Record<string, string> = {
  window_size: '窗口大小（天）',
  past_lag_days: '样本滞后（天）',
  forecast_horizon_days: '预测视野（天）',
  max_lag_day: '最大滞后构造（天）',
  test_split_ratio: '测试集比例',
  model_family: '模型族（Ridge/ElasticNet）',
  alpha: '正则强度',
  alpha_grid: 'Ridge alpha 候选网格',
  l1_ratio: 'L1 占比（ElasticNet）',
  ridge_solver: 'Ridge 求解器（svd）',
  fit_intercept: '是否拟合截距',
  use_lag_feature: '是否启用滞后特征',
  random_state: '随机种子',
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return '--'
  if (typeof value === 'boolean') return value ? '启用' : '不启用'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

function ParameterList({
  parameters,
  labels,
}: {
  parameters: Record<string, unknown>
  labels: Record<string, string>
}) {
  const entries = Object.entries(parameters).filter(
    ([, value]) => value === null || typeof value !== 'object',
  )
  if (entries.length === 0) {
    return null
  }
  return (
    <ul className="model-parameter-list">
      {entries.map(([key, value]) => (
        <li key={key}>
          <strong>{labels[key] ?? key}</strong>
          <span>{formatValue(value)}</span>
        </li>
      ))}
    </ul>
  )
}

export function ModelTuningCard({
  modelParameters,
  tuningEntries,
  tuningNarrative,
}: {
  modelParameters?: Record<string, unknown>
  tuningEntries?: TuningEntry[]
  tuningNarrative?: string
}) {
  const [methodLabels, setMethodLabels] = useState<Record<string, string>>({})
  useEffect(() => {
    let cancelled = false
    fetchMethodCatalog()
      .then((catalog) => {
        if (cancelled || !catalog) return
        const labels: Record<string, string> = {}
        for (const method of catalog.methods) {
          for (const spec of method.parameterSchema) {
            labels[spec.name] ??= spec.label
          }
        }
        setMethodLabels(labels)
      })
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [])
  const parameterLabels = useMemo(
    () => ({ ...STATIC_PARAMETER_LABELS, ...methodLabels }),
    [methodLabels],
  )
  const hasTuning = Boolean(tuningEntries && tuningEntries.length > 0)
  const hasFinalParameters = Boolean(modelParameters && Object.keys(modelParameters).length > 0)
  return (
    <section className="detail-card detail-card--wide model-tuning-card">
      <span className="detail-card__eyebrow">Model Parameters</span>
      <h2>本轮模型参数与调参理由</h2>
      {tuningNarrative ? (
        <div className="evaluation-summary-card">
          <p>{tuningNarrative}</p>
        </div>
      ) : null}
      {hasTuning ? (
        <div className="collection-grid">
          {tuningEntries!.map((entry, index) => (
            <article key={`${entry.refinementType ?? 'tuning'}-${index}`} className="collection-card">
              <strong>{entry.refinementType ?? '模型调参'}</strong>
              {!tuningNarrative ? <p>{entry.rationale || '未记录调整理由。'}</p> : null}
              {entry.modelParameters && Object.keys(entry.modelParameters).length > 0 ? (
                <ParameterList parameters={entry.modelParameters} labels={parameterLabels} />
              ) : (
                <div className="collection-card__meta">
                  <span>无额外参数变更</span>
                </div>
              )}
              {!tuningNarrative && entry.protocolNotes && entry.protocolNotes.length > 0 ? (
                <ul className="detail-list">
                  {entry.protocolNotes.map((note) => (
                    <li key={note}>{note}</li>
                  ))}
                </ul>
              ) : null}
            </article>
          ))}
        </div>
      ) : null}
      {hasFinalParameters ? (
        <div className="evaluation-summary-card">
          <h3>最终执行参数</h3>
          <ParameterList parameters={modelParameters!} labels={parameterLabels} />
        </div>
      ) : (
        <p className="detail-card__meta">本轮无显式模型参数调整，沿用协议默认参数。</p>
      )}
    </section>
  )
}
