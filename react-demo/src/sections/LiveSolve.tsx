import { useRef, useState } from 'react'
import {
  Area, Bar, BarChart, CartesianGrid, Cell, ComposedChart, Legend, Line, ReferenceArea,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { C, eur, mono } from '../theme'

const API = 'http://localhost:8000'

function segments(on: number[]) {
  const segs: { s: number; e: number }[] = []
  let start = -1
  on.forEach((v, i) => {
    if (v > 0 && start < 0) start = i
    if (v === 0 && start >= 0) { segs.push({ s: start - 0.5, e: i - 0.5 }); start = -1 }
  })
  if (start >= 0) segs.push({ s: start - 0.5, e: on.length - 0.5 })
  return segs
}

const TIP = {
  background: '#ffffff', border: `1px solid ${C.axis}`, borderRadius: 4,
  fontFamily: mono, fontSize: 12, boxShadow: '0 1px 6px rgba(20,23,28,0.10)',
}

type Mode = 'perfect' | 'forecast'
type WinId = 'all' | 'jan' | 'feb'
type Params = {
  date: string; baseline_mw: number; proc_mw: number; proc_hours: number;
  batt_mwh: number; grid_limit_mw: number; mode: Mode
}
type DayResult = {
  feasible: boolean; message?: string; date?: string; savings_eur?: number; savings_pct?: number;
  opt_eur?: number; naive_eur?: number; vcr?: number | null;
  price?: number[]; price_forecast?: number[] | null; grid_opt?: number[]; grid_naive?: number[]; proc_on?: number[]
}
type DailyRow = { date: string; savings: number; naive: number; naive_mwh: number; savings_nb: number; naive_nb: number }
type WindowResult = { feasible: boolean; daily?: DailyRow[]; peak_load_mw?: number }

const FIELDS: [keyof Params, string, number, number, number][] = [
  ['baseline_mw', 'Baseline (MW)', 0, 20, 1],
  ['proc_mw', 'Flexible (MW)', 0, 20, 1],
  ['proc_hours', 'Run-hours / day', 0, 24, 1],
  ['batt_mwh', 'Battery (MWh)', 0, 40, 5],
  ['grid_limit_mw', 'Grid limit (MW)', 5, 40, 1],
]
const WINDOWS = [
  { id: 'all', label: 'Full window' }, { id: 'jan', label: 'January' }, { id: 'feb', label: 'February' },
] as const
const inWindow = (date: string, w: WinId) =>
  w === 'all' ? true : w === 'jan' ? date < '2026-02-01' : date >= '2026-02-01'

export default function LiveSolve() {
  const [p, setP] = useState<Params>({
    date: '2026-01-08', baseline_mw: 8, proc_mw: 6, proc_hours: 8, batt_mwh: 20,
    grid_limit_mw: 25, mode: 'perfect',
  })
  const [day, setDay] = useState<DayResult | null>(null)
  const [win, setWin] = useState<WindowResult | null>(null)
  const [winSel, setWinSel] = useState<WinId>('all')
  const [dayLoading, setDayLoading] = useState(false)
  const [winLoading, setWinLoading] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const reqId = useRef(0)

  const set = (k: keyof Params, v: string) =>
    setP((s) => ({ ...s, [k]: k === 'date' || k === 'mode' ? v : Number(v) }))

  function solve() {
    const id = ++reqId.current
    setErr(null); setDayLoading(true); setWinLoading(true)
    const body = JSON.stringify({ ...p, pv_mwp: 10, batt_mw: 10 })
    const opts = { method: 'POST', headers: { 'Content-Type': 'application/json' }, body }
    fetch(`${API}/solve`, opts).then((r) => r.json()).then((d: DayResult) => {
      if (id !== reqId.current) return
      if (!d.feasible) { setErr(d.message || 'Infeasible.'); setDay(null) } else setDay(d)
    }).catch(() => {
      if (id !== reqId.current) return
      setErr(`Can't reach the optimiser API at ${API}. Start it:  uv run python serve/flex_api.py`)
    }).finally(() => { if (id === reqId.current) setDayLoading(false) })
    fetch(`${API}/window`, opts).then((r) => r.json()).then((w: WindowResult) => {
      if (id !== reqId.current) return
      setWin(w.feasible ? w : null)
    }).catch(() => {}).finally(() => { if (id === reqId.current) setWinLoading(false) })
  }

  const dayData = day?.price
    ? day.price.map((pr, i) => ({
      slot: i, price: pr, gridOpt: day.grid_opt![i], gridNaive: day.grid_naive![i],
      priceFc: day.price_forecast ? day.price_forecast[i] : null,
    }))
    : []
  const segs = day?.proc_on ? segments(day.proc_on) : []
  const modeLabel = p.mode === 'forecast' ? 'forecast' : 'perfect foresight'

  const rows = (win?.daily ?? []).filter((d) => inWindow(d.date, winSel))
  const peak = win?.peak_load_mw ?? 14
  const wagg = rows.length ? (() => {
    const sum = (f: (d: DailyRow) => number) => rows.reduce((a, d) => a + f(d), 0)
    const savings = sum((d) => d.savings), naive = sum((d) => d.naive)
    const naiveMwh = sum((d) => d.naive_mwh), savingsNb = sum((d) => d.savings_nb), naiveNb = sum((d) => d.naive_nb)
    const days = rows.length, eurYr = (savings * 365) / days
    return {
      days, pct: (savings / naive) * 100, perMwh: savings / naiveMwh, eurYr,
      keurPerMwYr: eurYr / 1000 / peak, flexOnlyPct: naiveNb ? (savingsNb / naiveNb) * 100 : 0,
    }
  })() : null

  return (
    <div className="live">
      <h2 style={{ marginTop: 0 }}>Run the optimiser live</h2>
      <p>
        Set the site, pick a day, and click <b>Solve</b> — it runs the MILP for <b>one day</b> and the
        <b> full 59-day window</b> at these settings. Toggle <span className="accent">perfect foresight</span> vs
        the <span className="accent">forecast</span> (one-day) to see the value the forecast captures.
      </p>

      <div className="chips">
        <button className={`chip${p.mode === 'perfect' ? ' active' : ''}`}
          onClick={() => set('mode', 'perfect')}>Perfect foresight</button>
        <button className={`chip${p.mode === 'forecast' ? ' active' : ''}`}
          onClick={() => set('mode', 'forecast')}>Forecast (P50)</button>
      </div>

      <div className="live-controls">
        <div className="field">
          <label className="label">Day</label>
          <input type="date" min="2026-01-01" max="2026-02-28" value={p.date}
            onChange={(e) => set('date', e.target.value)} />
        </div>
        {FIELDS.map(([k, label, min, max, step]) => (
          <div className="field" key={k}>
            <label className="label">{label}</label>
            <input type="number" min={min} max={max} step={step} value={p[k]}
              onChange={(e) => set(k, e.target.value)} />
          </div>
        ))}
      </div>

      <button className="solve-btn" onClick={solve} disabled={dayLoading || winLoading}>
        {dayLoading || winLoading ? 'Solving…' : 'Solve ▸'}
      </button>

      {err && <div className="live-err">{err}</div>}
      {!day && !dayLoading && !err && (
        <div className="live-empty">Set the parameters and click <b>Solve</b> to run the MILP.</div>
      )}

      {day && day.feasible && (
        <>
          <h3>One day — {day.date}</h3>
          <div className="live-stats">
            <div className="stat hl">
              <div className="k">Saved ({modeLabel})</div>
              <div className="v">{eur(day.savings_eur!)} <span className="pct">({day.savings_pct}%)</span></div>
            </div>
            <div className="stat"><div className="k">Optimised bill</div><div className="v">{eur(day.opt_eur!)}</div></div>
            <div className="stat"><div className="k">Naïve bill</div><div className="v">{eur(day.naive_eur!)}</div></div>
            <div className="stat"><div className="k">Value capture (VCR)</div>
              <div className="v">{day.vcr != null ? `${day.vcr}%` : '—'}</div></div>
          </div>
          <div className="chart-wrap" style={{ marginTop: '0.75rem' }}>
            <ResponsiveContainer width="100%" height={290}>
              <ComposedChart data={dayData} margin={{ top: 10, right: 16, bottom: 4, left: 4 }}>
                <CartesianGrid stroke={C.grid} vertical={false} />
                <XAxis dataKey="slot" tick={{ fill: C.muted, fontSize: 11, fontFamily: mono }} stroke={C.axis}
                  label={{ value: '15-min slot', position: 'insideBottom', offset: -2, fill: C.muted, fontSize: 11 }} />
                <YAxis yAxisId="mw" tick={{ fill: C.muted, fontSize: 11, fontFamily: mono }} stroke={C.axis}
                  label={{ value: 'MW', angle: -90, position: 'insideLeft', fill: C.muted, fontSize: 11 }} />
                <YAxis yAxisId="eur" orientation="right" tick={{ fill: C.muted, fontSize: 11, fontFamily: mono }}
                  stroke={C.axis} label={{ value: '€/MWh', angle: 90, position: 'insideRight', fill: C.muted, fontSize: 11 }} />
                {segs.map((s, i) => (
                  <ReferenceArea key={i} yAxisId="mw" x1={s.s} x2={s.e} fill={C.green} fillOpacity={0.16} />
                ))}
                <Area yAxisId="mw" dataKey="gridOpt" name="Grid (optimised)" stroke={C.grey}
                  fill={C.grey} fillOpacity={0.4} isAnimationActive={false} />
                <Line yAxisId="mw" dataKey="gridNaive" name="Grid (naïve)" stroke={C.naive}
                  strokeWidth={1.4} strokeDasharray="4 3" dot={false} isAnimationActive={false} />
                <Line yAxisId="eur" dataKey="price" name="Spot price (actual)" stroke={C.accent}
                  strokeWidth={1.8} dot={false} isAnimationActive={false} />
                <Line yAxisId="eur" dataKey="priceFc" name="Forecast price (P50)" stroke={C.blue}
                  strokeWidth={1.4} strokeDasharray="5 3" dot={false} connectNulls isAnimationActive={false} />
                <Tooltip contentStyle={TIP} labelStyle={{ color: C.muted }} />
                <Legend wrapperStyle={{ fontSize: 12, fontFamily: mono }} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          <p className="caption">
            <b>Perfect foresight</b> optimises on realised prices (an upper bound); <b>forecast</b> optimises
            on the D-1 P50, then settles at actual — the drop is the cost of forecast error.{' '}
            <span className="accent">VCR</span> = forecast ÷ perfect savings, the same lens as the battery.
          </p>
        </>
      )}

      {winLoading && !win && (
        <div className="live-empty">Computing the full 59-day window for these settings… (~20s)</div>
      )}

      {win && wagg && (
        <>
          <h3>Full window — these settings {winLoading && <span style={{ color: C.accent, fontSize: '0.8rem' }}>· recomputing…</span>}</h3>
          <div className="chips">
            {WINDOWS.map((w) => (
              <button key={w.id} className={`chip${winSel === w.id ? ' active' : ''}`}
                onClick={() => setWinSel(w.id)}>{w.label}</button>
            ))}
          </div>
          <div className="cards">
            <div className="card hl">
              <div className="k">Savings · {wagg.days} days</div>
              <div className="v">{wagg.pct.toFixed(1)}%</div>
              <div className="d">€{wagg.perMwh.toFixed(1)}/MWh · {wagg.keurPerMwYr.toFixed(0)} k€/MW·yr</div>
            </div>
            <div className="card">
              <div className="k">Annualised saving</div>
              <div className="v">{eur(wagg.eurYr)}/yr</div>
              <div className="d">perfect foresight · scaled from {wagg.days} days</div>
            </div>
            <div className="card">
              <div className="k">Flexibility-only</div>
              <div className="v">{wagg.flexOnlyPct.toFixed(1)}%</div>
              <div className="d">robust floor · battery adds the rest</div>
            </div>
          </div>
          <div className="chart-wrap" style={{ marginTop: '0.75rem' }}>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={rows} margin={{ top: 10, right: 16, bottom: 4, left: 8 }}>
                <CartesianGrid stroke={C.grid} vertical={false} />
                <XAxis dataKey="date" tick={{ fill: C.muted, fontSize: 10, fontFamily: mono }}
                  stroke={C.axis} interval={6} tickFormatter={(d: string) => d.slice(5)} />
                <YAxis tick={{ fill: C.muted, fontSize: 11, fontFamily: mono }} stroke={C.axis}
                  tickFormatter={(v: number) => '€' + (v / 1000).toFixed(0) + 'k'}
                  label={{ value: 'saved / day', angle: -90, position: 'insideLeft', fill: C.muted, fontSize: 11 }} />
                <Tooltip contentStyle={TIP} labelStyle={{ color: C.muted }}
                  formatter={(v) => [eur(Number(v)), 'saved']} />
                <Bar dataKey="savings" isAnimationActive={false}>
                  {rows.map((b, i) => (<Cell key={i} fill={b.savings >= 0 ? C.green : C.red} />))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="caption">
            Each bar = one day's saving for these settings (<span style={{ color: C.green }}>green = money kept</span>),
            perfect foresight. Change a parameter and re-Solve to watch the whole window move.
          </p>
        </>
      )}
    </div>
  )
}
