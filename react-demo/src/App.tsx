import { useState } from 'react'
import Flexibility from './sections/Flexibility'
import Procurement from './sections/Procurement'
import Forecasting from './sections/Forecasting'

const TABS = [
  { id: 'fcst', label: '📈 Forecasting', el: <Forecasting /> },
  { id: 'proc', label: '💶 Procurement', el: <Procurement /> },
  { id: 'flex', label: '🏭 Flexibility', el: <Flexibility /> },
] as const

export default function App() {
  const [tab, setTab] = useState<(typeof TABS)[number]['id']>('fcst')

  return (
    <div className="wrap">
      <div className="topbar">
        <div>
          <h1>Industrial energy optimiser</h1>
          <p className="muted">A prototype exploring the industrial energy problem — forecast → procurement → flexibility</p>
        </div>
        <span className="badge">React · TS · real EPEX</span>
      </div>

      <div className="tabs">
        {TABS.map((t) => (
          <button key={t.id} className={`tab${tab === t.id ? ' active' : ''}`}
            onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </div>

      {TABS.find((t) => t.id === tab)!.el}
    </div>
  )
}
