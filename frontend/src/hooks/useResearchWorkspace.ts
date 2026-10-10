import { useCallback, useEffect, useState } from 'react'

const CACHE_PREFIX = 'helioprior.research.v1'

function cacheKey(researchId: string, slot: string): string {
  const safeId = encodeURIComponent(researchId)
  const safeSlot = encodeURIComponent(slot)
  return `${CACHE_PREFIX}.${safeId}.${safeSlot}`
}

/**
 * Local browser cache scoped to a research project. The same slot name never
 * crosses research_id boundaries, so opening another project cannot show the
 * previous project's draft or selection state.
 */
export function useResearchWorkspace<T>(
  researchId: string | undefined,
  slot: string,
  initialValue: () => T,
) {
  const [value, setValue] = useState<T>(() => {
    if (!researchId) {
      return initialValue()
    }
    const key = cacheKey(researchId, slot)
    const cached = window.localStorage.getItem(key)
    if (cached === null) {
      return initialValue()
    }
    try {
      return JSON.parse(cached) as T
    } catch {
      window.localStorage.removeItem(key)
      return initialValue()
    }
  })

  useEffect(() => {
    if (!researchId) {
      return
    }
    const key = cacheKey(researchId, slot)
    try {
      window.localStorage.setItem(key, JSON.stringify(value))
    } catch {
      // Storage may be unavailable in private mode; the workspace still works
      // in memory for the current tab.
    }
  }, [researchId, slot, value])

  const clear = useCallback(() => {
    if (researchId) {
      window.localStorage.removeItem(cacheKey(researchId, slot))
    }
    setValue(initialValue())
  }, [initialValue, researchId, slot])

  return [value, setValue, clear] as const
}

