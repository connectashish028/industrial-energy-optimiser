# Industrial Energy Optimiser

A German (DE-LU) power-market optimiser across the **forecast → procurement → flexibility** lifecycle,
plus a merchant-battery value-capture engine. A Python core (linopy + HiGHS, XGBoost) behind a FastAPI
service, with React/TypeScript and Streamlit front-ends.

## Interactive demo — forecast → procurement → flexibility

A **React + TypeScript** front-end over a **Python / FastAPI** solver:

- **Forecasting** — XGBoost quantile day-ahead price model, scored by **value capture (VCR)**, not MAE.
- **Procurement** — spot vs fixed vs hedged: the cost-vs-certainty frontier.
- **Flexibility** — a **MILP** (HiGHS) scheduling a >10 MW industrial consumer into the cheapest hours;
  solve one day or the full window live, **perfect-foresight vs forecast**.

```bash
uv sync                                            # Python deps (CPython 3.11, via uv)
uv run python -m bessopt.data.refresh --rebuild    # fetch DE market data (no API token; ~1 min)
uv run python serve/flex_api.py                    # solver API -> http://localhost:8000
# then, in a second terminal:
cd react-demo && npm install && npm run dev        # app -> http://localhost:5173
```

> The market parquet is gitignored — the refresh above rebuilds it (public SMARD / Energy-Charts /
> Open-Meteo, no token). The bundled price-model checkpoint means the forecaster runs out of the box.

A production-grade merchant battery optimiser for the German (DE-LU) power market, built to mirror how real merchant trading desks operate. Target asset: a **10 MW battery in two configurations — 1h (10 MW / 10 MWh) and 2h (10 MW / 20 MWh)** — operating as a merchant price-taker.

## The asset (and why two of it)

The 1h and 2h assets differ **only** in energy capacity (10 vs 20 MWh). Running the identical optimiser at both durations is the cleanest way to surface the duration economics — and, once the PICASSO reserve buffer is modelled (Phase 5), to *derive* (not assert) why the 2h asset is worth meaningfully more.

## The headline deliverable

One chart: **revenue under perfect price foresight (the "oracle") vs revenue the forecast actually achieves**, in €/MW/year, for both the 1h and 2h asset, with the **Value Capture Ratio (VCR = R_forecast / R_oracle)** annotated. That gap is the *cost of forecast error*, and it is exactly how a desk thinks.

## Loud assumptions (state them, don't hide them)

This MVP makes deliberate simplifications. Naming them is what separates understanding the model from running it:

- **Price-taker.** No market impact; our dispatch never moves the clearing price.
- **Perfect fills at the clearing price.** Every committed MWh settles at the day-ahead price.
- **Market scope.** Day-ahead (full) + FCR + aFRR capacity (M5, representative reserve prices). The **intraday (IDC)** layer is a *simplified, perfect-foresight ceiling* on the value of intraday access (`market/intraday.py`, representative DA-ID spread) — not the full continuous order book or event-driven re-optimisation. aFRR *energy* activation, intraday auctions (IDA), and imbalance (reBAP) are out of scope. DA-only revenue is **lower than published full-stack benchmarks** — correct, not a bug. State the scope of every number.
- **Charge/discharge mutual-exclusivity is relaxed** (LP, not MILP) — with round-trip efficiency < 1 the LP optimum never charges and discharges simultaneously at positive prices. Documented, not silent.
- **15-min MTU structural break (2025-09-30).** Models trained across it mix two regimes; default evaluation is post-break only.

## Architecture (six layers, built bottom-up)

```
data/      L0 ingestion → canonical 15-min DE-LU parquet   (vendored; no API token)
features/  leakage-safe point-in-time feature builders      (vendored; corrupt-future tested)
forecast/  probabilistic day-ahead price forecast           (vendored XGBoost quantile P10/P50/P90)
optimiser/ SoC-aware LP/MILP dispatch (linopy + HiGHS)       ← the heart
backtest/  oracle / value-capture replay + rolling-horizon MPC
market/    sequential five-market simulator + PICASSO buffer (Phase 5)
risk/      CVaR + linear degradation overlay                 (Phase 6)
```

The data, feature, and forecast layers are **vendored and adapted** from the sibling `loadforecast` project (a deployed German day-ahead load+price forecaster whose price model already captures ~97% of perfect-foresight battery P&L). The genuine net-new engineering here is the **SoC-aware optimiser, the honest value-capture backtest, the MPC loop, the five-market simulator, and the risk overlay.**

Every prediction respects an **issue-time cutoff of D-1 12:00 Europe/Berlin** (the EPEX day-ahead gate). A "corrupt-future" test scrambles every post-cutoff value and asserts the resulting features are byte-identical — leakage is tested, not hoped for.

## Quickstart

Dependencies are managed with [**uv**](https://docs.astral.sh/uv/). `uv sync` creates
`.venv` (CPython 3.11, pinned in `.python-version`), installs runtime + dev deps from the
committed `uv.lock`, and builds `bessopt` editable.

```bash
# from bess-optimiser/
uv sync                       # create .venv + install everything (reproducible from uv.lock)

# 1. Verify install + all tests (uses the bundled parquet + price model)
uv run pytest -q

# 2. (Optional) refresh the parquet from public APIs (no token needed)
uv run python -m bessopt.data.refresh --rebuild --start 2022-01-01

# 3. Phase 1 — perfect-foresight oracle: €/MW/yr + SoC plots, 1h & 2h
uv run python scripts/run_oracle.py

# 4. Phase 3 — the headline VCR chart
uv run python scripts/run_vcr.py

# 5. Phase 4 — rolling-horizon MPC vs static-daily uplift
uv run python scripts/run_mpc.py

# 6. Phase 5 — value stack: revenue by market (DA + FCR + aFRR), 1h vs 2h
uv run python scripts/run_valuestack.py

# 7. Phase 6 — risk (CVaR efficient frontier) + degradation sensitivity
uv run python scripts/run_risk.py

# 7b. Intraday (IDC) — value of intraday access vs DA-ID spread (simplified)
uv run python scripts/run_idc.py

# 8. Phase 7 — daily pipeline (score + next-day dispatch + log to MLflow)
uv run python -m bessopt.pipeline --window-days 60

# 9. Phase 7 — operator dashboard (incl. the "Industrial flex" tab)
uv run streamlit run dashboards/app.py

# 10. Industrial flexibility — cost-optimal scheduling of a >10 MW consumer (MILP)
uv run python scripts/run_flex.py
```

## Two products, one engine

The same MILP/HiGHS core serves both sides of the meter:
- **Merchant battery** (revenue-max): day-ahead + reserves value stack, VCR backtest, MPC, CVaR.
- **Industrial consumer** (`bessopt/flex/`, cost-min): schedule a flexible process + on-site PV + battery against EPEX spot to minimise annual procurement cost — €0.9–1.6M/yr saved on a representative >10 MW site. See the dashboard's **🏭 Industrial flex** tab.

## Productionisation (lightweight, gate-closure paced)

- **Daily pipeline** (`bessopt/pipeline.py`): load → score the trailing window (VCR/τ/€per-MW-yr, 1h & 2h) → next-day forecast + dispatch → log to **MLflow** (SQLite backend) → write `outputs/results.json` + charts. No Airflow — a `GitHub Actions` cron (`.github/workflows/daily.yml`) runs it after the day-ahead settles and commits the refreshed dashboard artifacts; `tests.yml` runs ruff + pytest on every push.
- **Dashboard** (`dashboards/app.py`): a Streamlit app with two parts — (1) **"Simulate your BESS"**: an interactive what-if where you set power / energy / efficiency / SoC band / **degradation cost (0–40 €/MWh slider)** and get live economics (€/MW/yr, cycling, SoC plot, DA+FCR+aFRR split) via `bessopt/simulate.py`; and (2) a viewer of the daily pipeline outputs (value capture, value stack, CVaR frontier, next-day dispatch, freshness badge).
- The workflows activate once the repo is pushed to a GitHub remote (`git init` + push). Latency is not a concern — the loop is paced by the D-1 12:00 gate.

Common uv commands: `uv add <pkg>` (add a runtime dep), `uv add --dev <pkg>` (dev dep),
`uv lock --upgrade` (refresh the lockfile), `uv sync --no-dev` (production install).

## Roadmap

| Milestone | Deliverable |
|---|---|
| **M0** Data layer + as-of helper | Point-in-time-correct DE-LU 15-min dataset |
| **M1** Perfect-foresight optimiser | SoC plots + oracle €/MW/yr, 1h vs 2h |
| **M2** Probabilistic price forecast | P10/P50/P90 through the as-of helper |
| **M3** ⭐ Headline VCR chart | Oracle vs forecast revenue, VCR annotated |
| **M4** Rolling-horizon MPC | Uplift over static daily solve |
| **M5** ⭐ Value-stack + PICASSO | ✅ Revenue-by-market (DA+FCR+aFRR); 1h-vs-2h mix derived from the buffer |
| **M6** Risk + degradation | ✅ CVaR efficient frontier + linear throughput-cost sensitivity |
| **M7** Productionisation | ✅ Daily pipeline + MLflow + Streamlit dashboard + GitHub Actions cron |

Reserve prices in M5 are **representative** (`data/sources/regelleistung.py`) until the live
regelleistung.net feed is wired — the reserve revenue magnitudes illustrate the mechanism, not actuals.

## License

MIT. Data: CC-BY 4.0 (SMARD / Bundesnetzagentur, Energy-Charts / Fraunhofer ISE, ENTSO-E).
