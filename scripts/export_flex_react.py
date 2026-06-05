"""Export per-day flex savings for the React demo's Flexibility tab.

Runs the industrial-flex MILP day-by-day over the window and writes a richer
JSON than the headline aggregates: a daily savings series (so the React tab can
show savings over a *selectable* window instead of a single cherry-picked day),
window/annual totals, a per-MWh and k€/MW-peak/yr normalisation, the honest
flexibility-only decomposition (battery disabled), and the best day's dispatch
**for each window** (full / January / February) so the day chart follows the
selected window.

    uv run python scripts/export_flex_react.py
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np

from bessopt.config import REPO_ROOT, load_data_config
from bessopt.data.loader import load_de_lu_15min, target_index_for
from bessopt.flex import ConsumerSpec, naive_cost, optimise_day
from bessopt.flex.pv import pv_available

START = date(2026, 1, 1)
END = date(2026, 2, 28)
FEB_START = date(2026, 2, 1)
DAYS_PER_YEAR = 365.0


def _daterange(s: date, e: date):
    d = s
    while d <= e:
        yield d
        d += timedelta(days=1)


def _day_dict(res, ngrid):
    return {
        "price": [round(x, 2) for x in res.spot.tolist()],
        "grid_opt": [round(x, 3) for x in res.grid_mw.tolist()],
        "grid_naive": [round(x, 3) for x in ngrid.tolist()],
        "proc_on": [int(round(x)) for x in res.proc_on.tolist()],
        "pv": [round(x, 3) for x in res.pv_avail_mw.tolist()],
    }


def _series(df, spec, price_col, *, with_battery=True, track_best=False):
    """Per-day naive vs optimised cost over the window.

    Returns (daily, totals, bests) where bests maps window→(date, day_dict).
    """
    dt = spec.slot_hours
    soc = spec.battery.soc_min_mwh if (with_battery and spec.battery is not None) else 0.0
    daily = []
    opt_tot = naive_tot = naive_mwh_tot = 0.0
    # best[window] = (savings, date, res, ngrid)
    best: dict[str, tuple] = {}

    for d in _daterange(START, END):
        idx = target_index_for(d)
        spot = df[price_col].reindex(idx).to_numpy(dtype=float)
        if np.isnan(spot).any():
            continue
        pv = pv_available(df, idx, spec.pv_capacity_mwp)
        res = optimise_day(spot, pv, spec, soc_init=soc)
        nc, ngrid = naive_cost(spot, pv, spec)
        if with_battery and spec.battery is not None:
            soc = res.soc_end_mwh
        sav = nc - res.cost_eur
        naive_mwh = float(np.sum(ngrid * dt))
        daily.append({
            "date": d.isoformat(), "naive": round(nc, 2), "opt": round(res.cost_eur, 2),
            "savings": round(sav, 2), "naive_mwh": round(naive_mwh, 3),
        })
        opt_tot += res.cost_eur
        naive_tot += nc
        naive_mwh_tot += naive_mwh

        if track_best:
            wins = ["all", "jan" if d < FEB_START else "feb"]
            for w in wins:
                if w not in best or sav > best[w][0]:
                    best[w] = (sav, d, res, ngrid)

    return daily, (opt_tot, naive_tot, naive_mwh_tot), best


def main() -> int:
    cfg = load_data_config()
    df = load_de_lu_15min(cfg["parquet_path"])
    price_col = cfg["price_col"]

    spec = ConsumerSpec()  # 8 MW base + 6 MW flex ×8h, 10 MWp PV, 10 MW/20 MWh battery
    daily, (opt_tot, naive_tot, naive_mwh_tot), best = _series(
        df, spec, price_col, track_best=True)
    n = len(daily)
    if n == 0:
        raise RuntimeError("No complete flex days in window — check data coverage.")
    scale = DAYS_PER_YEAR / n
    savings_tot = naive_tot - opt_tot

    # Honest decomposition: same site with the battery disabled = flexibility-only.
    _, (opt_nb, naive_nb, _), _ = _series(
        df, ConsumerSpec(battery=None), price_col, with_battery=False)
    flex_only = naive_nb - opt_nb

    days = {w: _day_dict(b[2], b[3]) for w, b in best.items()}
    best_day = {w: b[1].isoformat() for w, b in best.items()}

    out = {
        "spec": {
            "baseline_mw": spec.baseline_load_mw, "proc_mw": spec.proc_power_mw,
            "proc_hours": spec.proc_hours_per_day, "pv_mwp": spec.pv_capacity_mwp,
            "batt_mw": spec.battery.power_mw, "batt_mwh": spec.battery.energy_mwh,
            "peak_load_mw": spec.peak_load_mw,
        },
        "window": {"start": START.isoformat(), "end": END.isoformat(), "days": n},
        "daily": daily,
        "totals": {
            "naive_eur_window": round(naive_tot, 2), "opt_eur_window": round(opt_tot, 2),
            "savings_eur_window": round(savings_tot, 2),
            "naive_eur_yr": round(naive_tot * scale), "opt_eur_yr": round(opt_tot * scale),
            "savings_eur_yr": round(savings_tot * scale),
            "savings_pct": round(savings_tot / naive_tot * 100, 1),
            "naive_mwh_window": round(naive_mwh_tot, 1),
            "savings_eur_per_mwh": round(savings_tot / naive_mwh_tot, 2),
            "savings_keur_per_mw_yr": round(savings_tot * scale / 1000 / spec.peak_load_mw, 1),
        },
        "flex_only": {
            "savings_eur_yr": round(flex_only * scale),
            "savings_pct": round(flex_only / naive_nb * 100, 1),
        },
        "best_day": best_day,  # {all, jan, feb}
        "days": days,          # {all, jan, feb} → dispatch arrays
    }

    out_path = REPO_ROOT / "react-demo" / "src" / "data" / "flex.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    t = out["totals"]
    print(f"wrote {out_path}")
    print(f"  {n} days · savings {t['savings_pct']}% · flex-only {out['flex_only']['savings_pct']}%")
    print(f"  €{t['savings_eur_yr']:,}/yr · €{t['savings_eur_per_mwh']}/MWh · "
          f"{t['savings_keur_per_mw_yr']} k€/MW-peak/yr")
    print(f"  best days: {best_day}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
