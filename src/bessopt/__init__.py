"""bessopt — production-grade German BESS optimiser for the DE-LU power market.

Layers (built bottom-up, sequenced for "always something demonstrable"):
  data/      L0 ingestion + canonical 15-min parquet (vendored, no API token)
  features/  leakage-safe, point-in-time-correct feature builders (vendored)
  forecast/  probabilistic day-ahead price forecaster (vendored XGBoost quantile)
  optimiser/ SoC-aware LP/MILP dispatch (linopy + HiGHS)        ← the heart
  backtest/  oracle / value-capture replay engine + rolling-horizon MPC
  market/    sequential five-market clearing simulator (FCR→aFRR→DA→intraday)
  risk/      CVaR + linear degradation overlay
  reporting/ headline VCR chart, SoC plots
"""

__version__ = "0.1.0"
