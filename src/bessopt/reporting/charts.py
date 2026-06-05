"""Reporting charts — SoC/dispatch plots and the headline VCR chart.

Static matplotlib PNGs (no kaleido needed). These are the persuasion: you can
literally see the battery charge in cheap/negative hours and discharge into
spikes, and the VCR chart is the number that lands the project.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..optimiser.spec import BatterySpec, DispatchResult  # noqa: E402

# xAI-inspired dark theme to match the Streamlit dashboard (loadforecast palette).
_BG = "#1f2228"
ACCENT = "#B8A1FF"   # lilac primary
plt.rcParams.update({
    "figure.facecolor": _BG, "axes.facecolor": _BG,
    "savefig.facecolor": _BG, "savefig.edgecolor": _BG,
    "text.color": "#e8e8ea", "axes.titlecolor": "#ffffff", "axes.labelcolor": "#e8e8ea",
    "axes.edgecolor": "#3a3d44", "xtick.color": "#a8a8ad", "ytick.color": "#a8a8ad",
    "grid.color": "#2f323a", "legend.edgecolor": "#3a3d44",
    "font.family": "monospace",
})


def plot_soc_dispatch(
    result: DispatchResult,
    spec: BatterySpec,
    *,
    title: str,
    out_path: str | Path,
) -> Path:
    """Dual-axis plot: price (right) with SoC trajectory (left), charge slots
    shaded green and discharge slots shaded red."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prices = result.prices_used
    H = len(prices)
    x = np.arange(H)
    dt = spec.slot_hours

    fig, ax_price = plt.subplots(figsize=(13, 5))
    ax_price.plot(x, prices, color="#1f77b4", lw=1.3, label="Price (€/MWh)", zorder=3)
    ax_price.axhline(0.0, color="#888", lw=0.6, ls="--", zorder=1)
    ax_price.set_ylabel("Price (€/MWh)", color="#1f77b4")
    ax_price.set_xlabel(f"Slot ({int(60 * dt)}-min) over delivery window")

    # Shade charge / discharge slots.
    for i in range(H):
        if result.charge_mw[i] > 1e-6:
            ax_price.axvspan(i - 0.5, i + 0.5, color="#2ca02c",
                             alpha=0.10 + 0.20 * result.charge_mw[i] / spec.power_mw, zorder=0)
        if result.discharge_mw[i] > 1e-6:
            ax_price.axvspan(i - 0.5, i + 0.5, color="#d62728",
                             alpha=0.10 + 0.20 * result.discharge_mw[i] / spec.power_mw, zorder=0)

    ax_soc = ax_price.twinx()
    ax_soc.plot(np.arange(H + 1) - 0.5, result.soc_mwh, color="#ff7f0e", lw=1.8,
                label="SoC (MWh)", zorder=4)
    ax_soc.axhline(spec.soc_min_mwh, color="#ff7f0e", lw=0.6, ls=":", alpha=0.6)
    ax_soc.axhline(spec.soc_max_mwh, color="#ff7f0e", lw=0.6, ls=":", alpha=0.6)
    ax_soc.set_ylabel("SoC (MWh)", color="#ff7f0e")
    ax_soc.set_ylim(-0.5, spec.energy_mwh + 0.5)

    ax_price.set_title(
        f"{title}\nrevenue €{result.revenue_eur:,.0f}  ·  "
        f"{result.cycles(spec):.2f} cycles  ·  green=charge red=discharge"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_headline_vcr(summary: dict, out_path: str | Path) -> Path:
    """The headline chart: oracle vs forecast-driven revenue (€/MW/yr) per asset,
    with VCR annotated.

    `summary` maps asset label → {"oracle": €/MW/yr, "forecast": €/MW/yr, "vcr": ratio}.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = list(summary.keys())
    oracle = [summary[k]["oracle"] for k in labels]
    forecast = [summary[k]["forecast"] for k in labels]
    x = np.arange(len(labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - w / 2, oracle, w, label="Perfect foresight (oracle)", color="#4c78a8")
    ax.bar(x + w / 2, forecast, w, label="Forecast-driven", color="#f58518")

    for i, k in enumerate(labels):
        ax.text(x[i] - w / 2, oracle[i], f"€{oracle[i]:,.0f}", ha="center", va="bottom", fontsize=9)
        ax.text(x[i] + w / 2, forecast[i], f"€{forecast[i]:,.0f}", ha="center", va="bottom", fontsize=9)
        ymax = max(oracle[i], forecast[i])
        ax.text(x[i], ymax * 1.08, f"VCR {summary[k]['vcr'] * 100:.1f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold", color=ACCENT)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Revenue (€/MW/year)")
    ax.set_title("Value Capture: perfect-foresight vs forecast-driven revenue")
    ax.legend()
    ax.margins(y=0.18)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_revenue_by_market(summary: dict, out_path: str | Path) -> Path:
    """Stacked bar of revenue-by-market (€/MW/yr) per asset — the M5 deliverable.

    `summary` maps asset label → {stream: €/MW/yr, ...} for streams
    day_ahead / fcr / afrr_pos / afrr_neg. Shows how the optimal mix shifts with
    duration: the 1h asset leans on reserve, the 2h stacks reserve + arbitrage.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = list(summary.keys())
    streams = ["day_ahead", "fcr", "afrr_pos", "afrr_neg"]
    pretty = {"day_ahead": "Day-ahead arbitrage", "fcr": "FCR",
              "afrr_pos": "aFRR capacity (POS)", "afrr_neg": "aFRR capacity (NEG)"}
    colours = {"day_ahead": "#f58518", "fcr": "#4c78a8",
               "afrr_pos": "#54a24b", "afrr_neg": "#b279a2"}

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 6))
    bottom = np.zeros(len(labels))
    for s in streams:
        vals = np.array([summary[k].get(s, 0.0) for k in labels])
        ax.bar(x, vals, 0.55, bottom=bottom, label=pretty[s], color=colours[s])
        bottom += vals
    for i in range(len(labels)):
        ax.text(x[i], bottom[i], f"€{bottom[i]:,.0f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Revenue (€/MW/year)")
    ax.set_title("Value stack: revenue by market, 1h vs 2h (PICASSO-driven mix)")
    ax.legend()
    ax.margins(y=0.12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_efficient_frontier(frontier: list[dict], out_path: str | Path) -> Path:
    """CVaR efficient frontier — expected revenue vs downside risk as β varies.

    `frontier` is a list of {beta, expected, cvar_loss} dicts (per-day-average
    EUR). Downside risk is plotted as CVaR_α(loss); lower-left = safer, lower
    return; upper-right = risk-neutral, higher return.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    betas = [f["beta"] for f in frontier]
    exp = [f["expected"] for f in frontier]
    risk = [f["cvar_loss"] for f in frontier]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(risk, exp, "-o", color="#4c78a8", zorder=2)
    for b, x, y in zip(betas, risk, exp, strict=True):
        ax.annotate(f"β={b:g}", (x, y), textcoords="offset points", xytext=(7, 4), fontsize=9)
    ax.set_xlabel("Downside risk — CVaR$_α$(loss), €/day (lower = safer)")
    ax.set_ylabel("Expected revenue, €/day")
    ax.set_title("CVaR efficient frontier — risk-adjusted dispatch")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_procurement_frontier(result, out_path: str | Path) -> Path:
    """Procurement cost-vs-risk frontier across hedge ratios (spot → fixed)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    risk = result.risk_eur
    cost = result.cost_per_year
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(risk, cost, "-o", color="#4c78a8", zorder=2)
    for r, x, y in zip(result.ratios, risk, cost, strict=True):
        if r in (0.0, result.recommended_ratio, 1.0):
            label = {0.0: "100% spot", 1.0: "100% fixed"}.get(r, f"{int(r * 100)}% hedged ★")
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(8, 4), fontsize=9,
                        fontweight="bold" if r == result.recommended_ratio else "normal")
    ri = list(result.ratios).index(result.recommended_ratio)
    ax.scatter([risk[ri]], [cost[ri]], s=140, facecolors="none", edgecolors="#e45756",
               linewidths=2, zorder=3)
    ax.set_xlabel("Risk — weekly price volatility (€/MWh, lower = more certain)")
    ax.set_ylabel("Expected annual procurement cost (€/year)")
    ax.set_title("Procurement strategy — cost vs certainty across hedge ratios")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_flex_schedule(result, naive_grid, spec, *, title: str, out_path: str | Path) -> Path:
    """Industrial-flex day: optimised vs naive grid draw, with the flexible process
    shifted into cheap/sunny hours and PV/price overlaid — the savings made visible."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    H = len(result.spot)
    x = np.arange(H)
    fig, ax_mw = plt.subplots(figsize=(13, 5.5))

    ax_mw.fill_between(x, result.grid_mw, step="mid", color="#9e9e9e", alpha=0.45,
                       label="Grid draw — optimised", zorder=2)
    ax_mw.plot(x, naive_grid, color="#c8ccd4", lw=1.4, ls="--", drawstyle="steps-mid",
               label="Grid draw — naive (fixed shift)", zorder=3)
    ax_mw.fill_between(x, result.pv_avail_mw, step="mid", color="#f6c343", alpha=0.5,
                       label="PV available", zorder=1)
    for i in range(H):
        if result.proc_on[i] > 0.5:
            ax_mw.axvspan(i - 0.5, i + 0.5, color="#2ca02c", alpha=0.10, zorder=0)
    ax_mw.plot([], [], color="#2ca02c", alpha=0.35, lw=8, label="Flexible process ON (optimised)")
    ax_mw.set_ylabel("Power (MW)")
    ax_mw.set_xlabel("Slot (15-min) over delivery day")
    ax_mw.set_ylim(bottom=0)

    ax_p = ax_mw.twinx()
    ax_p.plot(x, result.spot, color="#1f77b4", lw=1.3, label="Spot price", zorder=4)
    ax_p.set_ylabel("Spot price (€/MWh)", color="#1f77b4")

    ax_mw.set_title(title)
    h1, l1 = ax_mw.get_legend_handles_labels()
    h2, l2 = ax_p.get_legend_handles_labels()
    ax_mw.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_intraday_value(sensitivity: dict, out_path: str | Path, *, typical_spread: float = 12.0) -> Path:
    """Value of intraday access (€/MW/yr) vs the DA–ID RMS spread, per asset.

    `sensitivity` maps asset label → list of (spread_eur, uplift_eur_per_mw_yr).
    The near-linear slope is the structural result; the dashed line marks the
    typical DE spread to read off an estimate.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    for label, pts in sensitivity.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, "-o", label=label)
    ax.axvline(typical_spread, color="#9aa0aa", ls="--", lw=1)
    ax.annotate(f"typical DE ≈ {typical_spread:g} €/MWh", (typical_spread, ax.get_ylim()[1]),
                textcoords="offset points", xytext=(6, -14), fontsize=9, color="#a8a8ad")
    ax.set_xlabel("DA–ID RMS spread (€/MWh) — representative")
    ax.set_ylabel("Value of intraday access (€/MW/year)")
    ax.set_title("Intraday (IDC) value — perfect-foresight ceiling vs DA-only")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


__all__ = [
    "plot_efficient_frontier",
    "plot_flex_schedule",
    "plot_headline_vcr",
    "plot_intraday_value",
    "plot_procurement_frontier",
    "plot_revenue_by_market",
    "plot_soc_dispatch",
]
