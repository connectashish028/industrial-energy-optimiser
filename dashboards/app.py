"""Streamlit dashboard — a focused prototype of the industrial energy problem.

Three interactive tabs, the demand-side ones first (the role), then a curiosity
aside on the supply side:
  - Flexibility: cost-optimal scheduling of a >10 MW industrial consumer (MILP).
  - Procurement: spot vs fixed vs hedged — cost-vs-certainty frontier.
  - Battery & forecasting: the same engine flipped to revenue-max, fed by an ML
    price forecast (oracle vs forecast-driven, the value-capture ratio).

    uv run streamlit run dashboards/app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
import styles  # local: dashboards/styles.py (xAI-inspired dark theme)

from bessopt.config import REPO_ROOT, load_data_config
from bessopt.data.loader import load_de_lu_15min
from bessopt.flex import ConsumerSpec, run_flex
from bessopt.optimiser.spec import BatterySpec
from bessopt.procurement import procurement_analysis
from bessopt.reporting.charts import (
    plot_flex_schedule,
    plot_procurement_frontier,
    plot_soc_dispatch,
)
from bessopt.simulate import default_window, simulate_asset

OUT = REPO_ROOT / "outputs"

st.set_page_config(page_title="bessopt — energy optimiser", page_icon="⚡", layout="wide")
styles.inject(st)


@st.cache_resource(show_spinner=False)
def load_market_data():
    cfg = load_data_config()
    return load_de_lu_15min(cfg["parquet_path"]), cfg["price_col"]


@st.cache_data(show_spinner=False)
def run_sim(power, duration_h, eta, soc_min, soc_max, deg, window_days, with_vs, with_fc):
    df, price_col = load_market_data()
    spec = BatterySpec(power_mw=power, energy_mwh=power * duration_h, eta_rt=eta,
                       soc_min_frac=soc_min, soc_max_frac=soc_max, deg_cost_eur_per_mwh=deg)
    start, end = default_window(df, window_days, price_col)
    return simulate_asset(spec, df, start, end, price_col=price_col,
                          with_value_stack=with_vs, with_forecast=with_fc)


@st.cache_data(show_spinner=False)
def run_flex_sim(baseline, proc_mw, proc_hours, pv_mwp, batt_mw, batt_mwh, window_days):
    df, price_col = load_market_data()
    battery = BatterySpec(power_mw=batt_mw, energy_mwh=batt_mwh) if batt_mw > 0 else None
    spec = ConsumerSpec(baseline_load_mw=baseline, proc_power_mw=proc_mw,
                        proc_hours_per_day=proc_hours, pv_capacity_mwp=pv_mwp, battery=battery)
    start, end = default_window(df, window_days, price_col)
    return spec, run_flex(df, spec, start, end, price_col=price_col)


@st.cache_data(show_spinner=False)
def run_proc(load_mw, premium, window_days):
    df, price_col = load_market_data()
    start, end = default_window(df, window_days, price_col)
    return procurement_analysis(df, load_mw, start, end, price_col=price_col,
                                fixed_premium=premium)


st.title("⚡ Industrial energy optimiser")
st.caption("A prototype exploring the industrial energy problem — cost-optimal scheduling and "
           "procurement for an industrial site, on real German market data (EPEX + weather).")

tab_flex, tab_proc, tab_bess = st.tabs(
    ["🏭 Flexibility", "💶 Procurement", "🔋 Battery & forecasting"]
)

# ----------------------------------------------------------------------------- Industrial flex
with tab_flex:
    st.subheader("Cost-optimal scheduling for a >10 MW industrial consumer")
    st.caption("Minimise EPEX day-ahead procurement cost by shifting a flexible process into the "
               "cheapest / sunniest hours, with on-site PV and a battery. MILP solved with HiGHS.")
    with st.form("flex"):
        st.markdown("**Your site**")
        f1, f2, f3 = st.columns(3)
        baseline = f1.number_input("Baseline load (MW)", 0.0, 500.0, 8.0, 1.0,
                                   help="Inflexible base load that always runs.")
        proc_mw = f2.number_input("Flexible process (MW)", 0.0, 500.0, 6.0, 1.0,
                                  help="A load that must run a set number of hours/day but can be shifted.")
        proc_hours = f3.number_input("Run-hours / day", 0.0, 24.0, 8.0, 1.0)
        f4, f5, f6 = st.columns(3)
        pv_mwp = f4.number_input("On-site PV (MWp)", 0.0, 500.0, 10.0, 1.0)
        batt_mw = f5.number_input("Battery power (MW)", 0.0, 500.0, 10.0, 1.0)
        batt_mwh = f6.number_input("Battery energy (MWh)", 0.0, 2000.0, 20.0, 1.0)
        flex_window = st.slider("Window (days, to latest data)", 14, 90, 45)
        go_flex = st.form_submit_button("▶ Optimise schedule", type="primary")

    if go_flex:
        try:
            with st.spinner("Solving the MILP day-by-day…"):
                fspec, frun = run_flex_sim(baseline, proc_mw, proc_hours, pv_mwp,
                                           batt_mw, batt_mwh, flex_window)
        except FileNotFoundError:
            st.error("Market data parquet not found. Run "
                     "`uv run python -m bessopt.data.refresh --rebuild` first.")
            st.stop()

        st.success(f"Optimised a **{fspec.peak_load_mw:.0f} MW-peak** site over {frun.n_days} days "
                   f"({frun.best_day} window).")
        fc = st.columns(3)
        fc[0].metric("Procurement — naive shift", f"€{frun.naive_eur_per_year:,.0f}/yr")
        fc[1].metric("Procurement — optimised", f"€{frun.optimised_eur_per_year:,.0f}/yr")
        fc[2].metric("Annual savings", f"€{frun.savings_eur_per_year:,.0f}/yr",
                     delta=f"{frun.savings_pct:.1f}% of the bill")
        r = frun.best_day_result
        png = plot_flex_schedule(r, frun.best_day_naive_grid, fspec,
                                 title=f"Highest-saving day — {frun.best_day}",
                                 out_path=OUT / "flex_schedule.png")
        st.image(str(png), use_container_width=True)
        st.caption("Day-ahead spot only · PV from real Open-Meteo irradiance · representative site. "
                   "The flexible process is scheduled into the cheapest/sunniest hours and the "
                   "battery peak-shaves. Add min-runtime / tariff / grid-fee terms for a real site.")
    else:
        st.info("Enter your site parameters and press **Optimise schedule** to see the annual "
                "saving and the optimal day-schedule. This is the core problem: turning a "
                "MILP into €/year of procurement savings.")

# ----------------------------------------------------------------------------- Procurement
with tab_proc:
    st.subheader("Procurement strategy — spot vs fixed vs hedged")
    st.caption("Buying energy is a risk-adjusted decision: spot is cheapest on average but "
               "volatile; a fixed contract is certain but pays a forward premium. Find the hedge "
               "that balances cost against budget certainty.")
    with st.form("proc"):
        p1, p2, p3 = st.columns(3)
        load_mw = p1.number_input("Average load (MW)", 0.5, 1000.0, 12.0, 1.0)
        premium = p2.slider("Fixed-contract premium (%)", 0.0, 20.0, 6.0, 0.5,
                            help="Forward contracts price above expected spot to cover risk — "
                                 "representative ~5–8% for DE.") / 100.0
        proc_window = p3.slider("History window (days)", 30, 90, 90)
        go_proc = st.form_submit_button("▶ Analyse procurement", type="primary")

    if go_proc:
        try:
            with st.spinner("Costing spot vs hedged…"):
                pr = run_proc(load_mw, premium, proc_window)
        except FileNotFoundError:
            st.error("Market data parquet not found. Run "
                     "`uv run python -m bessopt.data.refresh --rebuild` first.")
            st.stop()

        ri = min(range(len(pr.ratios)), key=lambda i: abs(pr.ratios[i] - pr.recommended_ratio))
        rec_premium = (pr.recommended_cost_per_year / pr.spot_eur_per_year - 1) * 100
        risk_cut = (1 - pr.risk_eur[ri] / pr.risk_eur[0]) * 100 if pr.risk_eur[0] else 0.0
        st.success(f"Analysed a **{load_mw:g} MW** load over {pr.n_days} days "
                   f"({pr.volume_mwh:,.0f} MWh) · avg spot €{pr.avg_spot_eur_mwh:.0f}/MWh.")
        pc = st.columns(3)
        pc[0].metric("All-spot", f"€{pr.spot_eur_per_year:,.0f}/yr",
                     delta="cheapest · most volatile", delta_color="off")
        pc[1].metric("All-fixed", f"€{pr.fixed_eur_per_year:,.0f}/yr",
                     delta="certain · pays the premium", delta_color="off")
        pc[2].metric(f"Recommended — hedge {pr.recommended_ratio * 100:.0f}%",
                     f"€{pr.recommended_cost_per_year:,.0f}/yr",
                     delta=f"+{rec_premium:.1f}% cost, −{risk_cut:.0f}% volatility")
        st.image(str(plot_procurement_frontier(pr, OUT / "procurement_frontier.png")),
                 use_container_width=True)
        st.caption("Fixed price = avg spot × (1 + premium), representative — a real desk prices off "
                   "the traded forward curve. Risk = weekly unit-price volatility (€/MWh).")
    else:
        st.info("Set your average load and the fixed-contract premium, then **Analyse procurement** "
                "for the cost-vs-certainty frontier and a recommended hedge ratio. This is the "
                "'procurement' half of the forecast → procurement → flexibility lifecycle.")

# ----------------------------------------------------------------------------- Battery & forecasting
with tab_bess:
    st.subheader("Same engine, supply side — a merchant battery, with an ML price forecast")
    st.caption("Where my curiosity went next: the *same* optimiser flipped to revenue-max, fed by "
               "an XGBoost price forecast. Toggle **Forecast-driven** to see the honest revenue vs "
               "the perfect-foresight ceiling (the value-capture ratio).")
    with st.form("bess"):
        st.markdown("**Battery size** — spec it the way a desk does, MW × duration")
        c1, c2, c3 = st.columns(3)
        power = c1.number_input("Power (MW)", 0.5, 2000.0, 10.0, 0.5)
        duration_h = c2.number_input("Duration (h)", 0.25, 12.0, 2.0, 0.25)
        eta = c3.slider("Round-trip efficiency (%)", 70, 99, 90) / 100.0
        st.caption(f"→ Energy = Power × Duration = **{power:g} MW × {duration_h:g} h = "
                   f"{power * duration_h:g} MWh**")
        c4, c5 = st.columns(2)
        soc = c4.slider("Usable SoC band (%)", 0, 100, (10, 90))
        deg = c5.slider(
            "Degradation cost (€/MWh of throughput)", 0.0, 40.0, 10.0, 0.5,
            help="LFP linear throughput cost is typically 2–4 €/MWh; 10–40 models "
                 "costlier chemistries or amortised cell capex. Drag it to see cycling "
                 "and revenue respond.",
        )
        c6, c7, c8 = st.columns(3)
        window_days = c6.slider("Window (days, to latest data)", 14, 90, 30)
        with_vs = c7.checkbox("Reserve value stack (DA+FCR+aFRR)", value=True)
        with_fc = c8.checkbox("Forecast-driven VCR (slower)", value=False,
                              help="Run the honest forecast→optimise→settle replay for this "
                                   "battery — the realistic revenue and value-capture ratio.")
        go = st.form_submit_button("▶ Run simulation", type="primary")

    if go:
        try:
            with st.spinner("Optimising dispatch over the window…"):
                sim = run_sim(power, duration_h, eta, soc[0] / 100, soc[1] / 100, deg,
                              window_days, with_vs, with_fc)
        except FileNotFoundError:
            st.error("Market data parquet not found. Run "
                     "`uv run python -m bessopt.data.refresh --rebuild` first.")
            st.stop()

        energy = power * duration_h
        cyc_yr = sim.avg_cycles_per_day * 365.0
        st.success(f"Simulated **{power:g} MW / {energy:g} MWh** ({duration_h:g}h) over "
                   f"{sim.n_days} days ({sim.start} → {sim.end}).")

        # Revenue row.
        r = st.columns(3)
        r[0].metric("Perfect-foresight (oracle)", f"€{sim.oracle_eur_per_mw_yr:,.0f}/MW/yr",
                    help="Upper bound — knows future prices exactly.")
        if sim.forecast_eur_per_mw_yr is not None:
            r[1].metric("Forecast-driven (realistic)", f"€{sim.forecast_eur_per_mw_yr:,.0f}/MW/yr",
                        delta=f"VCR {sim.vcr * 100:.0f}%", delta_color="off")
        if sim.value_stack_total_eur_per_mw_yr is not None:
            r[2].metric("Value-stack total (DA+reserves)",
                        f"€{sim.value_stack_total_eur_per_mw_yr:,.0f}/MW/yr")
        # Operations row.
        o = st.columns(3)
        o[0].metric("Avg cycles / day", f"{sim.avg_cycles_per_day:.2f}")
        o[1].metric("Full-equiv cycles / yr", f"{cyc_yr:,.0f}",
                    delta="within ~700 warranty" if cyc_yr <= 700 else "above ~700 warranty",
                    delta_color="normal" if cyc_yr <= 700 else "inverse")
        o[2].metric("Duration", f"{duration_h:g} h")

        st.divider()
        left, right = st.columns([3, 2])
        with left:
            st.markdown(f"**Optimal dispatch — highest-spread day ({sim.best_day})**")
            png = plot_soc_dispatch(sim.best_day_dispatch, sim.spec,
                                    title=f"{power:g} MW / {energy:g} MWh ({duration_h:g}h) — {sim.best_day}",
                                    out_path=OUT / "sim_soc.png")
            st.image(str(png), use_container_width=True)
        with right:
            if sim.value_stack_eur_per_mw_yr is not None:
                st.markdown("**Revenue by market (€/MW/yr)**")
                pretty = {"day_ahead": "Day-ahead", "fcr": "FCR",
                          "afrr_pos": "aFRR POS", "afrr_neg": "aFRR NEG"}
                vs = pd.DataFrame(
                    {"€/MW/yr": {pretty[k]: v for k, v in sim.value_stack_eur_per_mw_yr.items()}}
                )
                st.bar_chart(vs)
                st.caption("Reserve prices are representative — the duration-driven mix is "
                           "real, the magnitudes illustrative.")
        st.caption("Perfect-foresight oracle (an upper bound). Forecast-driven revenue is lower "
                   "by the value-capture ratio (~70–90%). Day-ahead + reserves; intraday extra.")
    else:
        st.info("Set your battery parameters above and press **Run simulation**. "
                "Try dragging the **degradation cost** from 10 to 40 €/MWh to watch cycling fall.")

with st.expander("Honest assumptions (stated, not hidden)"):
    st.markdown(
        "- Real EPEX day-ahead prices + Open-Meteo weather; the rest of the site (load, PV, "
        "tariffs) is **representative** — a real deployment plugs in the customer's actuals.\n"
        "- Flexibility savings are vs an **unoptimised baseline** (the value of the optimisation).\n"
        "- Procurement: fixed price = avg spot × (1 + premium), representative — a desk prices off "
        "the **traded forward curve**.\n"
        "- The battery view shows the **perfect-foresight oracle** (an upper bound); the "
        "forecast-driven number is lower by the value-capture ratio.\n"
        "- A prototype I built to understand the problem — not a finished product."
    )
