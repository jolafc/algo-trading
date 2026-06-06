# Repo guide for Claude

Backtesting framework for long-only US equity trading strategies. Historical daily prices come from the Alpha Vantage API; the implemented strategy is a weekly rotation on the S&P 500 ("§8: Weekly rotation" from *The 30-Minute Stock Trader*). Strategy hyperparameters are tuned via Bayesian optimization (`scikit-optimize`) with walk-forward cross-validation.

## Environment

- Python 3.8, conda env `trading` defined in `environment.yaml` (pandas, scikit-learn, scikit-optimize, matplotlib, requests, `ta`, tensorflow).
- Two env vars are required at import time of `exp/data_getter.py`:
  - `AV_KEY` — Alpha Vantage API key
  - `AV_RQM` — Alpha Vantage requests-per-minute quota (used as throttle limit)
- Run tests with `pytest` from the repo root. `test_av.py` hits the live API; `test_weekly_rotation.py` and `test_hpo.py` need `data/sp500.pkl` (built by `data_getter.py`) and assert on hard-coded reference yields/sharpe ratios.

## Top-level layout

```
algo-trading/
├── data/              # constituents.csv (S&P 500 universe) + sp500.pkl (cached price data, gitignored)
├── exp/               # All production code
│   ├── __init__.py        # Paths, seed, metric/column name constants
│   ├── data_getter.py     # Alpha Vantage API client + pickle cache for SP500 history
│   ├── data_loader.py     # Build per-feature DataFrames, NaN handling, dividend/split adjustment
│   ├── backtesting.py     # Backtesting class: replays a strategy's positions, tracks P&L
│   ├── metrics.py         # IB fees, notional sizing, P&L, annualized yield, Sharpe, Sortino
│   ├── reporting.py       # Backtesting report + CV convergence plots
│   ├── optimization.py    # train_strategy / cross_validate_strategy / cv_opt_driver (skopt + joblib)
│   ├── default_parameters.py  # AV column names, trade-record columns, benchmark ticker, defaults
│   └── strategy/
│       └── weekly_rotation.py  # WeelkyRotationStrategy [sic] + WeeklyRotationRunner
├── tests/             # pytest suite (test_av, test_weekly_rotation, test_hpo)
├── notebooks/         # Exploratory notebook for AV data fetching
├── deprecated/        # Old Zipline-based tutorial code; do NOT reference for current API
├── results/           # Run outputs (plots, logs, checkpoints) — gitignored content
└── environment.yaml   # Conda env spec
```

## Where to make changes (cheat sheet)

| Want to change... | Edit |
|---|---|
| How prices are downloaded / cached | `exp/data_getter.py` |
| How OHLCV / dividends / splits are turned into per-feature DataFrames | `exp/data_loader.py` |
| How the simulated portfolio executes buys/sells and tracks balance | `exp/backtesting.py` (`Backtesting.fit`) |
| Fee model, P&L formula, yield/Sharpe/Sortino definitions | `exp/metrics.py` |
| Plots and printed report at the end of a backtest | `exp/reporting.py` |
| Add a new strategy | New module under `exp/strategy/`, mirror the `Strategy` + `Runner` pair in `weekly_rotation.py` |
| The weekly rotation rules (SMA, RSI, volume filter, ROC ranking) | `WeelkyRotationStrategy.fit` in `exp/strategy/weekly_rotation.py` |
| HPO search space for weekly rotation | `WeeklyRotationRunner.__init__` (`self.dimensions`) |
| Optimizer choice / kwargs (forest, GBRT, GP) | `OPTIMIZER_FUNCTION` / `OPTIMIZER_KWARGS` in `exp/optimization.py` |
| Walk-forward CV: window sizes, fold count, start date | `cv_opt_driver` kwargs (top-level `__main__` block) |
| Constants (column names, ticker for benchmark, start balance) | `exp/default_parameters.py` |
| Result/log paths, metric name constants, default optimizer | `exp/__init__.py` |

## Core abstractions

### Data flow

1. `data_getter.get_sp500_pkl()` queries Alpha Vantage `TIME_SERIES_DAILY_FULL` for every symbol in `data/constituents.csv`, throttled by `AV_RQM`, and pickles a `dict[symbol -> DataFrame]` to `data/sp500.pkl`.
2. `data_loader.get_feature(prices_dict, column=...)` reshapes that into a wide `DataFrame` (index=dates, columns=tickers) per OHLCV/dividend/split feature.
3. `data_loader.slice_backtesting_window(features, start, end, lookback)` clips all feature frames to the requested window plus a lookback prefix.

### Strategy contract

A strategy class must implement:
- `fit(data_by_feature)` — precompute the position list for every trade date in the window.
- `predict(date) -> list[str]` — return the tickers to hold on `date`.
- `get_dates()` — iterable of trade dates.
- attribute `price_min` — used by `metrics.get_notional` for position sizing.

`Backtesting.fit(strategy, prices, dates, low=None, high=None)` then iterates dates, diffs against the previous holdings to derive buys/sells, and applies fills at `prices.loc[date, ticker]` (or `low`/`high` if provided for conservative/optimistic execution).

### Runner pattern

`WeeklyRotationRunner` wraps "load data → build strategy → run backtest → return metrics" so the same object can be:
- Called directly (`runner(**params)` returns the metrics dict or a single metric).
- Used as `skopt`'s objective via `runner.skopt_func(x)` (signed by `WeeklyRotationRunner.signs` so all metrics are minimized).
- Reconfigured between train/val windows by mutating `start_date_requested` / `end_date_requested` (see `train_strategy` in `optimization.py`).

New strategies should follow this two-class split.

### Optimization stack

- `train_strategy` — one Bayesian optimization run over `[train_start, train_end]`, then evaluates the best parameters on `[val_start, val_end]`. Checkpoints to a `.pkl` so resume picks up `x_iters` / `func_vals`.
- `cross_validate_strategy` — runs `train_strategy_driver` in parallel (joblib) across `n_folds` walk-forward windows.
- `cv_opt_driver` — top-level driver that repeats CV for `n_iters` rounds, accumulates results across iterations, and emits convergence plots per fold. `resume=True` continues the most recent matching `results/run_<metric>_*` dir; `resume="<dir_name>"` picks a specific one.

### Results dir conventions

Every `cv_opt_driver` invocation creates `results/run_<metric>_<timestamp>/` containing:
- `cv_opt.log` — full log stream
- `checkpoint_<train_start>_<train_end>_<metric>.pkl` — one skopt checkpoint per fold
- `results_iter.pkl` — list of per-iteration result DataFrames (used for resume + plotting)
- `fold_<NN>.png` — convergence plot per CV fold
- `unrealized_pl__*.png` — per-evaluation P&L plot (only when `verbose=True`)

## Conventions & gotchas

- **Typo in class name**: `WeelkyRotationStrategy` (not `Weekly...`). The runner spells it correctly. Don't "fix" silently — it's used as-is in `optimization.py`.
- **Metric sign convention**: all metrics in `WeeklyRotationRunner.signs` map to `-1` because `skopt` minimizes. Any new metric you add to a runner needs an entry there.
- **`exp/__init__.py` side effect**: importing anything from `exp` creates `data/` and `results/` if missing.
- **Verbose + parallel is forbidden**: `cv_opt_driver` asserts `n_jobs == 1` when `verbose=True` (joblib workers can't share the logger).
- **AV API quota**: `data_getter.py` enforces `AV_RQM` requests/minute by sleeping to fill out a 61-second window. The "10 QPS" constant `QPS` is actually used as a *save frequency* (pickle every N symbols), not a rate limit — naming is misleading.
- **NaN sell prices**: `Backtesting.fit` treats `NaN` on a sell as "delisted/missing" and unwinds the position at the buy price (zero P&L), recording it in `errors_df`. NaN on a buy raises an assertion.
- **Adjusted execution price**: `WeeklyRotationRunner` builds `EXECUTION_PRICE_COLUMN` as `(open * adj_close/close).shift(-1)` — i.e. trades signaled on day *t* execute at the adjusted open of day *t+1*. Falls back to adjusted close when next-day open is missing.
- **`deprecated/`** uses Zipline and Quandl — predates the current AV-based pipeline. Don't pattern-match new code on it.
- **Tests assert exact reference values** (`np.isclose(..., atol=1e-8)`). Changes to fees, sizing, execution-price logic, or NaN handling will break these — update the references intentionally.

## When extending the repo

- New strategy: drop a module in `exp/strategy/`, expose a `Strategy` (fit/predict/get_dates/price_min) and a `Runner` (callable + `dimensions` OrderedDict of `skopt.space.*` + `signs` dict + `skopt_func`). Then either swap `StrategyRunner=...` in `cv_opt_driver` or add a new test next to `test_weekly_rotation.py`.
- New metric: add the constant to `exp/__init__.py`, compute it in `exp/metrics.py`, plumb it through `reporting.make_backtesting_report`'s `results` dict, and add the sign to the runner's `signs`.
- New feature/indicator: add the AV column constant in `default_parameters.py`, load via `data_loader.get_feature` inside the runner's `__call__`, then consume in the strategy's `fit`.

## Companion documentation

When the user's task touches any of these, read the relevant doc first — they capture detail intentionally kept out of this file:

- **[`doc/ARCHITECTURE.md`](doc/ARCHITECTURE.md)** — sequence diagram of the full tuning-run control flow, walk-forward CV layout, resume semantics, results-dir contents.
- **[`doc/data_schema.md`](doc/data_schema.md)** — exact columns/dtypes for every DataFrame passed between modules. Read before changing `Backtesting`, adding a metric, or adding a feature/indicator.
- **[`doc/hpo_notes.md`](doc/hpo_notes.md)** — `WeeklyRotationRunner.dimensions` rationale (active + frozen), optimizer choice, sizing guidance for `cv_opt_driver`.
- **[`examples/run_end_to_end.py`](examples/run_end_to_end.py)** — minimal end-to-end script (download → single backtest → 2-fold CV). Use this as the reference for "how is this thing supposed to be invoked" rather than the `__main__` blocks scattered through `exp/`.
- **Inline docstrings** on `Backtesting`, `WeeklyRotationRunner`, `cv_opt_driver`, `train_strategy`, `get_feature`, `slice_backtesting_window` — the public surface is now documented at the signature level.
