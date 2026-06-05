import {
  Area, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import fc from '../data/forecast.json'
import { C, eur, mono } from '../theme'

const data = fc.series.map((d) => ({
  t: d.t, actual: d.actual, p50: d.p50, band: [d.p10, d.p90] as [number, number],
}))

const TIP = {
  background: '#ffffff', border: `1px solid ${C.axis}`, borderRadius: 4,
  fontFamily: mono, fontSize: 12, boxShadow: '0 1px 6px rgba(20,23,28,0.10)',
}

export default function Forecasting() {
  return (
    <div>
      <h2>Does the forecast capture the value, not just the price?</h2>
      <p>
        The forecaster is an <span className="accent">XGBoost quantile model</span> on real DE features,
        gated point-in-time (issue = D-1 12:00 Berlin — no leakage). But the business question isn't
        MAE. It's: when we dispatch on the <i>forecast</i>, how much of the perfect-foresight money do
        we actually keep? That's the <span className="accent">Value Capture Ratio</span>.
      </p>

      <div className="cards">
        {fc.assets.map((a) => (
          <div className="card hl" key={a.name}>
            <div className="k">{a.name} battery — VCR</div>
            <div className="v">{a.vcr}%</div>
            <div className="d">{eur(a.forecast)} of {eur(a.oracle)} oracle €/MW/yr</div>
          </div>
        ))}
        <div className="card">
          <div className="k">Forecast MAE</div>
          <div className="v">€{fc.mae}</div>
          <div className="d">per MWh — the <i>secondary</i> metric</div>
        </div>
      </div>

      <p className="caption">
        Why 2h captures more (91% vs 85%): a longer battery is forgiving of timing errors — it can wait
        out a mis-ranked hour. VCR is a <span className="accent">ranking</span> metric (Kendall τ),
        not an accuracy one. A forecast can have a worse MAE yet a better VCR if it gets the cheap/expensive
        <i> order</i> right.
      </p>

      <h2>Forecast vs actual — {fc.period.label}</h2>
      <div className="chart-wrap">
        <ResponsiveContainer width="100%" height={340}>
          <ComposedChart data={data} margin={{ top: 10, right: 16, bottom: 4, left: 4 }}>
            <CartesianGrid stroke={C.grid} vertical={false} />
            <XAxis dataKey="t" tick={{ fill: C.muted, fontSize: 10, fontFamily: mono }} stroke={C.axis}
              interval={95} tickFormatter={(t: string) => t.slice(5, 10)} />
            <YAxis tick={{ fill: C.muted, fontSize: 11, fontFamily: mono }} stroke={C.axis}
              label={{ value: '€/MWh', angle: -90, position: 'insideLeft', fill: C.muted, fontSize: 11 }} />
            <Area dataKey="band" name="Forecast P10–P90" stroke="none" fill={C.accent}
              fillOpacity={0.18} isAnimationActive={false} />
            <Line dataKey="actual" name="Actual" stroke={C.naive} strokeWidth={1.3}
              dot={false} isAnimationActive={false} />
            <Line dataKey="p50" name="Forecast P50" stroke={C.accent} strokeWidth={1.3}
              dot={false} isAnimationActive={false} />
            <Tooltip contentStyle={TIP} labelStyle={{ color: C.muted }} />
            <Legend wrapperStyle={{ fontSize: 12, fontFamily: mono }} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <p className="caption">
        <span className="accent">Lilac band</span> = forecast P10–P90 uncertainty; dark line = realised
        actual, hourly across February. Actual stays mostly inside the band — calibrated uncertainty.
      </p>
    </div>
  )
}
