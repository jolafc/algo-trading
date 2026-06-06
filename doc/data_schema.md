# Data schemas

The codebase passes several untyped pandas structures between modules. This document pins down the exact shape, index, columns, and dtype expected at each handoff. If you're touching `Backtesting`, a new strategy, or a new metric, read this first.

## 1. Alpha Vantage per-ticker DataFrame

Produced by `data_getter.av_query(symbol)` and stored as values in the `data/sp500.pkl` `dict[symbol -> DataFrame]`.

| Column | Dtype | Notes |
|---|---|---|
| (index) `timestamp` | `datetime64[ns]` | Daily, descending order as returned by AV; not reindexed to a business calendar |
| `open` | float64 | Raw |
| `high` | float64 | Raw |
| `low` | float64 | Raw |
| `close` | float64 | Raw |
| `adjusted_close` | float64 | Adjusted for splits + dividends |
| `volume` | int64 | Shares |
| `dividend_amount` | float64 | Per-share dividend on that date (0 otherwise) |
| `split_coefficient` | float64 | Split ratio on that date (1.0 otherwise) |

Column constants live in `exp/default_parameters.py` (`OPEN_COLUMN`, `CLOSE_COLUMN`, `ADJUSTED_CLOSE_COLUMN`, `VOLUME_COLUMN`, `DIVIDENT_COLUMN`, `SPLIT_COLUMN`).

The pickle is `{ticker: DataFrame or None}`. Failed queries store `None`; `data_loader.get_feature` filters them out.

## 2. Wide per-feature DataFrame

Produced by `data_loader.get_feature(prices_dict, column=...)`. This is the canonical shape consumed by strategies.

- **Index**: `DatetimeIndex` (ascending after pandas concat; union of all tickers' dates)
- **Columns**: ticker symbols (`str`)
- **Dtype**: `float64` (or `int64` for volume; downcast on concat)
- **NaNs**: present wherever a ticker had no data on a given date (delisting, IPO, suspension). Optionally forward-padded via `impute='pad'`.

Strategies typically build a `data_by_feature: dict[str, DataFrame]` keyed by column constant:

```python
data_by_feature = {
    ADJUSTED_CLOSE_COLUMN: get_feature(..., column=ADJUSTED_CLOSE_COLUMN),
    VOLUME_COLUMN:         get_feature(..., column=VOLUME_COLUMN),
    CLOSE_COLUMN:          get_feature(..., column=CLOSE_COLUMN),
    OPEN_COLUMN:           get_feature(..., column=OPEN_COLUMN),
}
```

`slice_backtesting_window` returns the same dict with each frame clipped to `[start - lookback, end]`. All frames in the dict must share an index — the helper asserts this.

### Derived feature: execution price

`WeeklyRotationRunner` builds two extra frames:

| Key | Definition | Why |
|---|---|---|
| `FACTOR_COLUMN` | `adjusted_close / close` | Per-day split+dividend adjustment factor |
| `EXECUTION_PRICE_COLUMN` | `(open * factor).shift(-1)` | Adjusted open of the *next* day — signals on day t execute on day t+1's open |

Missing next-day open falls back to that day's adjusted close.

## 3. `prices` argument to `Backtesting.fit`

A single wide DataFrame (not a dict). Same shape as §2:
- Index: trade dates (must be a superset of the strategy's `get_dates()`)
- Columns: tickers (must contain every ticker the strategy ever returns from `predict`, plus the benchmark `SPY` when used by reporting)
- Dtype: float64

`Backtesting.fit` accepts optional `low=` and `high=` frames with the same shape, used to apply conservative (sell at low / buy at high) fills.

## 4. `positions_df` — open positions

Mutable state inside `Backtesting`. One row per open position. The index is a monotonically increasing position counter (`pos_counter`).

| Column | Dtype | Notes |
|---|---|---|
| `date_buy` | Timestamp | Date the position was opened |
| `ticker` | str | Symbol |
| `position` | int | Number of shares (`notional // price`) |
| `price_buy` | float | Fill price at buy |
| `fees_buy` | float | IB fees on the buy (`metrics.get_ib_fees`) |

Constants: `POSITIONS_COLUMNS` in `default_parameters.py`.

## 5. `trades_df` — closed positions

Same index space as `positions_df` (the position id carries over when a position closes). Has all `positions_df` columns plus:

| Column | Dtype | Notes |
|---|---|---|
| `date_sell` | Timestamp | Close date |
| `price_sell` | float | Fill price at sell; equals `price_buy` if AV had NaN for the sell day (delisting → recorded in `errors_df`) |
| `fees_sell` | float | IB fees on the sell |
| `fees` | float | `fees_buy + fees_sell` |
| `P&L` | float | `position * (price_sell - price_buy) - fees` |

Constants: `TRADES_EXTRA_COLUMNS` and `TRADES_COLUMNS`.

## 6. `errors_df` — recorded NaN-on-sell events

Same column space as `positions_df`, plus a `date_sell` column appended ad-hoc. Populated when a position is closed but the sell-day price was NaN (typically delisting / merger / bankruptcy). The position is still unwound at the buy price (zero P&L) so HPO can't game these.

## 7. `unrealized_pl` series

Returned by `Backtesting.get_unrealized_pl()`.

- Index: every date the backtest visited (= `dates` argument)
- Name: `'Unrealized P&L'`
- Value: `realized_P&L + (current_value_of_holdings - invested - buy_fees)` (see `metrics.get_total_unrealized_p_and_l`)

Used by `metrics.get_annualized_yield / get_sharpe_ratio / get_sortino_ratio` and plotted by `make_backtesting_report`.

## 8. Metrics dict

Returned by `make_backtesting_report` and by `WeeklyRotationRunner.__call__` when `output_metric=None`. Keys are the constants in `exp/__init__.py`:

| Key | Constant | Meaning |
|---|---|---|
| `'pl'` | `PL` | Realized P&L (sum over closed trades) |
| `'yield'` | `YIELD` | Annualized continuously-compounded yield |
| `'sharpe'` | `SHARPE` | Annualized Sharpe |
| `'sortino'` | `SORTINO` | Annualized Sortino |
| `'Bpl'` | `BPL` | Same, for the SPY buy-and-hold benchmark |
| `'Byield'` | `BYIELD` | " |
| `'Bsharpe'` | `BSHARPE` | " |
| `'Bsortino'` | `BSORTINO` | " |

The `B`-prefixed values come from sizing the entire start balance into SPY at the first date and holding to the last (one fee on entry, none on exit).

## 9. CV results DataFrame

Returned per iteration by `cross_validate_strategy`; the full `results_iter` is `list[DataFrame]`. One row per fold, with columns:

```
train_start, train_end, val_start, val_end,
Tpl, Tyield, Tsharpe, Tsortino, TBpl, TByield, TBsharpe, TBsortino,
Vpl, Vyield, Vsharpe, Vsortino, VBpl, VByield, VBsharpe, VBsortino
```

Prefixes come from `TRAIN_PREFIX='T'` and `VAL_PREFIX='V'` in `exp/__init__.py`. Benchmark metrics keep the `B` prefix on the metric name, so the train-set benchmark Sharpe is `TBsharpe`.

## 10. Strategy contract

To plug into `Backtesting.fit`, a strategy must expose:

```python
strategy.fit(data_by_feature: dict[str, DataFrame]) -> self
strategy.predict(date: Timestamp) -> list[str]    # tickers to hold on `date`
strategy.get_dates() -> Iterable[Timestamp]       # trade dates, ascending
strategy.price_min: float                         # used by metrics.get_notional for position sizing
```

`Backtesting` diffs successive `predict(date)` outputs to derive buys (new minus old) and sells (old minus new), then sizes each buy as `notional // price` where `notional = (balance - budget_fees) / n_buys`.
