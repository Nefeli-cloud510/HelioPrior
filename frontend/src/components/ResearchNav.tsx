import { Link, useLocation } from 'react-router-dom'

interface ResearchNavProps {
  researchId: string
  active: 'overview' | 'evidence' | 'review' | 'validate' | 'report'
}

const NAV_ITEMS: Array<{ id: ResearchNavProps['active']; label: string }> = [
  { id: 'overview', label: '项目总览' },
  { id: 'evidence', label: '证据中心' },
  { id: 'review', label: '协作复核' },
  { id: 'validate', label: '验证实验' },
  { id: 'report', label: '研究报告' },
]

export function ResearchNav({ researchId, active }: ResearchNavProps) {
  const location = useLocation()
  return (
    <nav className="research-nav" aria-label="研究项目导航">
      {NAV_ITEMS.map((item) => {
        const to =
          item.id === 'overview'
            ? `/research/${researchId}`
            : `/research/${researchId}/${item.id}`
        const selected = active === item.id
        return (
          <Link
            key={item.id}
            className={`research-nav__item${selected ? ' research-nav__item--active' : ''}`}
            to={to}
            state={location.state}
            aria-current={selected ? 'page' : undefined}
          >
            {item.label}
          </Link>
        )
      })}
    </nav>
  )
}
