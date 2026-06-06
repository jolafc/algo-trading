# algo-trading

Backtesting framework for long-only US equity strategies using Alpha Vantage historical price data. Includes one built-in strategy (weekly rotation on the S&P 500), a fee-aware backtester, and a Bayesian-optimization tuning loop with walk-forward cross-validation.

## Quick start

```bash
# 1. Create and activate the conda env
conda env create -f environment.yaml
conda activate trading

# 2. Export your Alpha Vantage credentials
export AV_KEY=<your_alphavantage_api_key>
export AV_RQM=<your_requests_per_minute_quota>

# 3. Verify the setup
pytest

# 4. Run the end-to-end example (downloads SP500 history on first run, then backtests + tunes)
python examples/run_end_to_end.py
```

The S&P 500 price download can take an hour on a free-tier AV key — it caches to `data/sp500.pkl` and subsequent runs reuse it.

## Further documentation

Detailed docs live in [`doc/`](doc/):

- [`doc/ARCHITECTURE.md`](doc/ARCHITECTURE.md) — control flow from `cv_opt_driver` down to `Backtesting`, including a sequence diagram and the walk-forward CV layout.
- [`doc/data_schema.md`](doc/data_schema.md) — exact columns, indices, and dtypes for every DataFrame passed between modules (AV per-ticker, wide per-feature, `positions_df`, `trades_df`, metrics dict, CV results).
- [`doc/hpo_notes.md`](doc/hpo_notes.md) — search-space rationale, which dimensions are intentionally frozen, optimizer choice, and sizing guidance for tuning runs.
- [`CLAUDE.md`](CLAUDE.md) — repo-map and conventions cheat sheet, primarily aimed at Claude-assisted edits.
