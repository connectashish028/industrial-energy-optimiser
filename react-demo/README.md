# React demo — Industrial energy optimiser (React/TS proof-of-concept)

A **React + TypeScript (Vite)** app covering the three pillars —
**flexibility → procurement → forecasting** — themed to match the Streamlit dashboard
(the xAI-inspired dark theme). Built as a ~few-hour proof that the Python prototype can move
into a React/TypeScript stack, scoped to the parts I can explain end-to-end in 15 minutes.

```bash
cd react-demo
npm install        # once
npm run dev        # → http://localhost:5173  (launch.json "react-demo" uses 5174)
```

## The three tabs

1. **🏭 Flexibility** — a flexible >10 MW industrial consumer scheduled into the cheapest/sunniest
   hours by a **MILP** (HiGHS). Optimal-day chart: green bands = process ON, grey area = optimised
   grid draw, dashed = naïve fixed shift, lilac = spot price. Honest decomposition: the *scheduling*
   itself is a steady ~5% (€438k/yr); the battery adds the rest to reach 9.8% (€862k/yr).
2. **💶 Procurement** — spot vs fixed vs hedged. Drag the **hedge ratio** and the cost / risk /
   frontier recompute *live* (procurement is pure arithmetic → runs client-side). Risk = weekly
   unit-price volatility, so fully-fixed = zero risk. Hedge 80% → +4.8% cost, −80% volatility.
3. **📈 Forecasting** — the XGBoost quantile price model, judged the way dispatch cares: not MAE but
   **VCR (Value Capture Ratio)** — the share of perfect-foresight money the forecast actually keeps
   (84.7% for 1h, 91% for 2h). Forecast-vs-actual line chart shows it ranks the day's *shape* right.

All numbers are exported from the Python project to `src/data/{flex,forecast,procurement}.json`.
Charts use Recharts except the procurement frontier (hand-rolled SVG — the maths is light enough).

## The architecture point (say this in the demo)
The optimiser and forecaster **stay in Python** (they're numerical — a `FastAPI` service exposes
them). **React is the operator's decision surface.** Here procurement is light enough to run in the
browser; the flexibility MILP and the forecaster would be a `fetch()` to the Python endpoint. That
split — `React (TS) ↔ FastAPI (Python) ↔ Postgres` — is a clean production stack.
