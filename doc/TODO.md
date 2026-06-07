# TODO — known limitations and future work

This file tracks deliberate limitations of the current backtesting pipeline and the work needed to address them. It exists so results from this repo can be quoted honestly without re-deriving the asterisks each time.

## 1. Survivorship bias in the SP500 universe (active limitation)

### What we do today

`data/constituents.csv` holds the **current** S&P 500 constituents, regenerated from Wikipedia via `exp/refresh_constituents.py`. The backtester loads price history for every name in this CSV and runs the strategy against that universe at every date.

### Why this is biased

The S&P 500 churns. A name that was in the index in 2008 and collapsed (Lehman, Bear Stearns, AIG-pre-bailout, WorldCom, Enron, Sears, JCPenney, Kodak, GE during 2017–19) is **no longer in our universe**, so the backtest never has the chance to hold it through the collapse. The strategy is implicitly tested on names that survived to today.

Concrete impact for the weekly-rotation strategy in this repo:

- The strategy ranks by `ROC(lookback)` and holds the top-N. High-momentum names that *then* collapsed (Enron 2000, Lehman 2007, GE 2017) are exactly the cases the strategy would have bought into and held through the crash — and they're the cases hidden by survivorship bias.
- Empirical guesstimate from the literature: ~1–2% annualized yield overstatement and a materially understated max drawdown / left-tail. The Sharpe ratio reported by the current pipeline is best read with that caveat in mind.

### Why we accept the bias for now

Free historical price data sources (yfinance, Stooq, AlphaVantage free, Tiingo free) all drop delisted tickers from their index within a year or two of delisting. There is no free way to get a full survivor-bias-free SP500 history.

### Future work, in priority order

1. **Universe layer — extended membership** (cheap, partial fix).
   - Pull Wikipedia's "Selected changes to the list of S&P 500 components" table; union with the current list. Gives ~800 distinct tickers.
   - Many of the added tickers will fail to resolve on yfinance (delistings/mergers) — that's expected; log + skip.
   - Captures the *universe* part of the bias even though prices remain unavailable for the truly-delisted names.

2. **Famous-failures sidecar** (cheap, captures tail risk).
   - Hand-curate ~20–30 known crashes-while-in-SP500 (Enron, WorldCom, Lehman, Bear Stearns, AIG '08, Kodak, Sears, JCPenney, Circuit City, Washington Mutual, etc.).
   - Try Stooq as a fallback price source for these — its coverage of delisted US tickers is patchy but non-zero (estimated 30–60% hit rate). The `yf_query` fallback could be a `stooq_query` helper.
   - Even partial coverage of the famous tail events is a meaningful correction.

3. **Strategy layer — point-in-time membership filtering** (the rigorous fix).
   - Existing backlog item — see the `# TODO: Need the joiner and leaver data` line in `exp/strategy/weekly_rotation.py`.
   - For each backtest date `d`, the strategy should only consider names that were actually in the SP500 on `d` (using the effective-date intervals from Wikipedia's changes table).
   - Without this, even an extended universe over-counts: a 1990 add that was dropped in 2005 should not be tradeable in 2010.

4. **Paid data source for full rigor** (when the project graduates from research to "thinking about real money").
   - EOD Historical Data (eodhd.com) — ~$20/month, US only, explicit survivor-bias-free SP500 historical constituents dataset. Cheapest credible option.
   - Norgate Data — ~$50/month, the retail-quant favorite, delistings back to 1950. Windows desktop app + `norgatedata` Python package.
   - Polygon.io — $30–200/month, broader API surface (intraday, options) if needed beyond EOD daily.
   - Sharadar via Nasdaq Data Link — ~$50/month, fundamentals + prices, quant-grade.

### How to read current results until then

When quoting yields/Sharpe/Sortino from this pipeline: assume an ~1–2%/year optimistic yield bias and a more significant understatement of left-tail drawdowns. Anything that requires a tighter error bar (e.g., paper, live-trading decision) needs at least step 1 — ideally step 4 — before being publishable or actionable.

## 2. Related backlog items in code

Several deliberate corner-cuts are flagged in the strategy and backtester. Listed here so they don't get lost; the original `# TODO:` markers stay in the source as the ground truth.

- `exp/strategy/weekly_rotation.py` — ticker changes, mergers, acquisitions, bankruptcies are currently sold at the buy price (forced zero-P&L unwind) to prevent HPO from gaming the NaN-on-sell path. Real handling needs corporate-action data.
- `exp/strategy/weekly_rotation.py` — RSI/EMA indicators reimplementation noted as TODO.
- `exp/strategy/weekly_rotation.py` — forward-pad NaN imputation is unbounded; should cap padded-value count to avoid stale data masquerading as active.
- `exp/optimization.py` / `WeeklyRotationRunner.dimensions` — frozen dimensions (`sma_tol`, `volume_lookback`, `volume_threshold`, `price_min`, `n_positions`) are intentionally not tuned. See `doc/hpo_notes.md` for the rationale; revisit if those defaults stop holding up.

## 3. Refreshing the universe

Re-run `python -m exp.refresh_constituents` to pull the latest Wikipedia snapshot of current SP500 constituents into `data/constituents.csv`. This is the *current* membership only — it does not fix the survivorship bias above.
