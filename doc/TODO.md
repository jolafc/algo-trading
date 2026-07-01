# TODO — known limitations and future work

This file tracks deliberate limitations of the current backtesting pipeline. It exists so results from this repo can be quoted honestly without re-deriving the asterisks each time.

## How to read this list

Items are ranked by their estimated impact on the believability of reported backtest metrics. The first three concerns can each erase or invert the apparent edge of a tuned strategy; the items below them shift the numbers but don't typically flip the sign. Survivorship bias, despite being the most visible "data quality" gap, sits below the statistical-methodology issues — those bite harder.

## 1. HPO selection bias — reported metrics are inflated by multiple testing (highest impact)

`cv_opt_driver` evaluates hundreds of hyperparameter trials per fold and reports the best. Bailey, Borwein, López de Prado & Zhu (2014)[^pseudo-math] proved that with `N` trials over `T` observations, the maximum in-sample Sharpe on **pure noise** approaches `sqrt(2 · log N / T)`. For the default `n_iters=20 × n_calls=10 × n_folds=4 = 800` trials per fold against a ~52-bar weekly validation window, that noise ceiling sits around **0.6–0.8 Sharpe**. Any reported Sharpe below that floor is consistent with no edge at all — selection bias alone explains it.

This is the dominant limitation of the current pipeline. Until it's addressed, optimization-driven Sharpe/Sortino numbers from `cv_opt_driver` should not be taken at face value.

### Action items
- **Deflated Sharpe Ratio (DSR)**[^dsr]: implement alongside the raw Sharpe in `exp/metrics.py`. Inputs are the trial return distribution from skopt's history, the trial count, and the return skew/kurtosis. Report it in `make_backtesting_report` and on the convergence plot.
- **Probability of Backtest Overfitting (PBO)**[^pbo]: estimate from the full trial set in `cv_opt_driver`. Useful summary statistic per run.
- **Confidence intervals on Sharpe/Sortino**: currently we report point estimates. Bootstrap CIs over fold returns give a sense of the uncertainty even before DSR corrections.

## 2. Cross-validation methodology — autocorrelation leakage and gap-free folds

The walk-forward CV in `cross_validate_strategy` avoids look-ahead bias (validation is strictly later than training) but does nothing about the fact that adjacent windows share autocorrelated returns and overlapping label horizons. López de Prado (2018, ch. 7)[^afml] argues this is enough to inflate apparent generalization on its own, separately from #1.

### Action items
- **Embargo + purge** between each fold's `train_end` and `val_start` — at minimum `max_lookback` bars, since indicators consume that much past data. Drop-in change to `train_strategy_driver`.
- **Combinatorial Purged CV (CPCV)**[^afml] as the rigorous pass — keep walk-forward as the fast baseline.

## 3. Transaction-cost realism

`metrics.get_ib_fees` covers IB commissions but ignores:

- **Bid-ask spread** — ~1–5 bps round-trip for SP500 large caps; wider for the tail.
- **Market impact / slippage** — small at low AUM, ~5–10 bps round-trip approaching \$1M+.
- **Borrow cost** — n/a, strategy is long-only.

Frazzini, Israel & Moskowitz (2018)[^trading-costs] estimate momentum strategies lose roughly **40–60% of paper returns** to realistic trading costs at non-trivial size. The weekly rotation here turns over substantially every week, so this is not a small adjustment. The current backtest numbers should be read as "what an idealized zero-spread, zero-impact trader would have captured."

### Action items
- Add a configurable spread cost (basis points) deducted on both legs in `Backtesting.fit`.
- Surface the cost assumptions in `make_backtesting_report` output and reference them when quoting results.

## 4. Survivorship bias in the universe

`data/constituents.csv` is the **current** S&P 500 (regenerated via `python -m exp.refresh_constituents`). Names that were in the index then collapsed (Lehman, Bear Stearns, AIG '08, WorldCom, Enron, Sears, Kodak, GE during 2017–19, etc.) are not in our universe, so the backtest never holds them through the crash. For a ROC-ranked rotation strategy specifically, high-momentum-then-collapse names are exactly the cases hidden by the bias — estimated ~1–2%/yr yield overstatement and a materially understated left tail.

No free data source provides survivor-bias-free historical S&P 500 prices (yfinance, Stooq, AV free, Tiingo free all drop delisted tickers within a year or two).

### Future work, ranked
1. **Extended membership**: union current SP500 with Wikipedia's historical changes table (~800 tickers). Many won't resolve on yfinance — log + skip. Captures the *universe* part of the bias.
2. **Famous-failures sidecar**: hand-curated ~20–30 known crashes with Stooq as a fallback price source (~30–60% hit rate). Cheap, captures tail risk.
3. **Point-in-time membership filtering**: at each backtest date, only consider names actually in the SP500 then. Uses effective-date intervals from the same Wikipedia table. Existing backlog item in `exp/strategy/weekly_rotation.py`.
4. **Paid data**: EOD Historical Data (~\$20/mo) is the cheapest survivor-bias-free SP500 dataset; Norgate (~\$50/mo) is the retail-quant favorite; Polygon, Sharadar at \$30–50/mo are also credible.

## 5. Out-of-universe / out-of-sample robustness

The strategy is tuned and tested on SP500 weekly data only. To distinguish genuine momentum capture from overfitting to a specific market, the same strategy with the same parameters should be run on different markets (FTSE 100, Russell 2000, Nikkei 225, STOXX 600) and rebalance frequencies. Asness, Moskowitz & Pedersen (2013)[^value-momentum] document momentum effects across all major asset classes — if our strategy only "works" on SP500 weekly, that's evidence of overfitting, not edge.

### Action items
- Make `data_getter` universe-agnostic (drop the SP500-specific naming) so an alternate constituents CSV plugs in cleanly.
- Parameterize the rebalance period in `WeeklyRotationStrategy` — currently the day-of-week + weekly cadence are baked in.

## 6. Randomization baselines

Does the strategy beat a random portfolio of the same breadth, drawn from the same universe, with the same rebalance cadence? If not, the apparent "edge" is just exposure to the universe's drift, not the signal. This is a cheap and very informative diagnostic.

### Action items
- Add a `RandomStrategy` matching the existing strategy contract (`fit`/`predict`/`get_dates`/`price_min`).
- Add a reporting hook that runs both side-by-side over the same window and reports the gap in yield / Sharpe / Sortino.

## 7. Related backlog items in code (lower magnitude)

Deliberate corner-cuts flagged inline. Listed so they don't get lost; the source markers stay authoritative.

- `exp/strategy/weekly_rotation.py` — ticker changes, mergers, acquisitions, bankruptcies are sold at the buy price (forced zero-P&L) to prevent HPO from gaming the NaN-on-sell path. Real handling needs corporate-action data.
- `exp/strategy/weekly_rotation.py` — RSI/EMA indicator reimplementation.
- `exp/strategy/weekly_rotation.py` — forward-pad NaN imputation is unbounded; should cap padded-value count to avoid stale data masquerading as active.
- `exp/optimization.py` / `WeeklyRotationRunner.dimensions` — frozen dimensions (`sma_tol`, `volume_lookback`, `volume_threshold`, `price_min`, `n_positions`) are intentionally not tuned. See `doc/hpo_notes.md` for the rationale; revisit if those defaults stop holding up.

## 8. Refreshing the universe

`python -m exp.refresh_constituents` pulls the latest Wikipedia snapshot into `data/constituents.csv`. Current membership only — does not fix #4.

## References

[^pseudo-math]: Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2014). "Pseudo-mathematics and financial charlatanism: The effects of backtest overfitting on out-of-sample performance." *Notices of the AMS*, 61(5), 458–471 — <https://www.ams.org/notices/201405/rnoti-p458.pdf>

[^dsr]: Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio: correcting for selection bias, backtest overfitting, and non-normality." *Journal of Portfolio Management*, 40(5), 94–107 — <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551>

[^pbo]: Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2017). "The Probability of Backtest Overfitting." *Journal of Computational Finance*, 20(4), 39–69 — <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253>

[^afml]: López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. ISBN 978-1119482086. Chapters 7 (CV in finance), 11 (dangers of backtesting), 14 (backtest statistics) are the most directly relevant.

[^trading-costs]: Frazzini, A., Israel, R., & Moskowitz, T. J. (2018). "Trading Costs." AQR Working Paper — <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3229719>

[^value-momentum]: Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013). "Value and Momentum Everywhere." *Journal of Finance*, 68(3), 929–985 — <https://onlinelibrary.wiley.com/doi/10.1111/jofi.12021>
