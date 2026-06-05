import { useMemo, useState } from 'react'
import data from '../data/procurement.json'
import { eur } from '../theme'

function std(xs: number[]) {
  const m = xs.reduce((a, b) => a + b, 0) / xs.length
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / xs.length)
}

export default function Procurement() {
  const [loadMw, setLoadMw] = useState(12)
  const [premium, setPremium] = useState(6)
  const [hedge, setHedge] = useState(80)

  const c = useMemo(() => {
    const avg = data.avg_spot
    const sigma = std(data.weekly_unit_prices)
    const fixedPrice = avg * (1 + premium / 100)
    const energy = loadMw * 8760
    const spotCost = avg * energy
    const fixedCost = fixedPrice * energy
    const r = hedge / 100
    return {
      avg, sigma, spotCost, fixedCost,
      cost: r * fixedCost + (1 - r) * spotCost,
      risk: (1 - r) * sigma,
    }
  }, [loadMw, premium, hedge])

  const premiumPct = (c.cost / c.spotCost - 1) * 100
  const riskCut = (1 - c.risk / c.sigma) * 100

  return (
    <div>
      <h2>How should an industrial buyer split spot vs fixed?</h2>
      <p>
        Buying energy is a risk-adjusted decision. Drag the <span className="accent">hedge ratio</span> to
        trade cost against budget certainty.
      </p>

      <div className="controls">
        <div>
          <label className="label">Average load (MW)</label>
          <input type="number" min={1} max={500} value={loadMw}
            onChange={(e) => setLoadMw(Number(e.target.value))} />
        </div>
        <div>
          <label className="label">Fixed premium — <span className="slider-val">{premium}%</span></label>
          <input type="range" min={0} max={20} step={0.5} value={premium}
            onChange={(e) => setPremium(Number(e.target.value))} />
        </div>
        <div>
          <label className="label">Hedge ratio — <span className="slider-val">{hedge}%</span></label>
          <input type="range" min={0} max={100} step={5} value={hedge}
            onChange={(e) => setHedge(Number(e.target.value))} />
        </div>
      </div>

      <div className="cards">
        <div className="card">
          <div className="k">All-spot</div>
          <div className="v">{eur(c.spotCost)}/yr</div>
          <div className="d">cheapest · most volatile</div>
        </div>
        <div className="card">
          <div className="k">All-fixed</div>
          <div className="v">{eur(c.fixedCost)}/yr</div>
          <div className="d">certain · pays the premium</div>
        </div>
        <div className="card hl">
          <div className="k">Your hedge — {hedge}%</div>
          <div className="v">{eur(c.cost)}/yr</div>
          <div className="d">+{premiumPct.toFixed(1)}% cost · −{riskCut.toFixed(0)}% volatility</div>
        </div>
      </div>

      <p className="caption">
        Real DE day-ahead prices ({data.start} → {data.end}), avg €{c.avg.toFixed(0)}/MWh. Fixed =
        avg spot × (1 + premium), representative. The recommendation: the smallest hedge that cuts
        volatility to ≤25% of all-spot — here ~80%.
      </p>
    </div>
  )
}
