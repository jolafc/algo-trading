"""Verify `compute_adjusted_close` matches yfinance's native `Adj Close` on a small sample.

Since Alpha Vantage's bundled adjusted_close is no longer available on the free tier, we use
yfinance's `Adj Close` (computed by Yahoo) as the ground truth for our locally-computed
adjustment.

yfinance's `Close` (with auto_adjust=False) is already split-adjusted by Yahoo; only the
dividend adjustment is missing. `compute_adjusted_close` therefore applies dividends only —
adding splits on top would double-count and produce massive errors (verified pre-fix).

Hits the live yfinance API. Sample is intentionally small to keep the test fast and stable.
"""
import numpy as np
import pandas as pd
import yfinance as yf

from exp.data_getter import compute_adjusted_close, yf_query
from exp.default_parameters import ADJUSTED_CLOSE_COLUMN, CLOSE_COLUMN

# A short, stable, dividend+split-heavy sample. Pick blue chips with long histories.
SAMPLE_TICKERS = ['MSFT', 'AAPL', 'IBM']

# With splits no longer double-applied, agreement with Yahoo is essentially perfect (~1 ppm
# in spot-check). Tolerance is set well above observed noise so genuine bugs trip the test.
COMPARE_FROM = pd.Timestamp('2010-01-01')
RELATIVE_TOL = 1e-4  # 0.01%


def _yf_adj_close(symbol):
    raw = yf.Ticker(symbol).history(period='max', auto_adjust=False)
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    return raw['Adj Close'].astype(float)


def test_compute_adjusted_close_matches_yfinance():
    prices = {}
    yf_ground_truth = {}
    for symbol in SAMPLE_TICKERS:
        df, _ = yf_query(symbol=symbol)
        assert df is not None, f'yfinance query failed for {symbol}'
        prices[symbol] = df
        yf_ground_truth[symbol] = _yf_adj_close(symbol)

    compute_adjusted_close(prices)

    for symbol in SAMPLE_TICKERS:
        ours = prices[symbol][ADJUSTED_CLOSE_COLUMN]
        truth = yf_ground_truth[symbol]
        # Align on dates both have, restrict to comparison window
        common = ours.index.intersection(truth.index)
        common = common[common >= COMPARE_FROM]
        assert len(common) > 100, f'Not enough overlapping dates for {symbol}'

        ours_w = ours.loc[common]
        truth_w = truth.loc[common]
        rel_err = (ours_w - truth_w).abs() / truth_w.abs()
        max_rel_err = rel_err.max()
        mean_rel_err = rel_err.mean()

        assert max_rel_err < RELATIVE_TOL * 5, (
            f'{symbol}: max relative error {max_rel_err:.6f} exceeds {RELATIVE_TOL * 5:.6f}'
        )
        assert mean_rel_err < RELATIVE_TOL, (
            f'{symbol}: mean relative error {mean_rel_err:.6f} exceeds {RELATIVE_TOL:.6f}'
        )


def test_compute_adjusted_close_preserves_close_at_latest_date():
    """Sanity: the most recent close price needs no adjustment (no future divs/splits)."""
    prices = {}
    for symbol in SAMPLE_TICKERS:
        df, _ = yf_query(symbol=symbol)
        assert df is not None
        prices[symbol] = df
    compute_adjusted_close(prices)

    for symbol in SAMPLE_TICKERS:
        df = prices[symbol]
        latest_close = df[CLOSE_COLUMN].iloc[-1]
        latest_adj = df[ADJUSTED_CLOSE_COLUMN].iloc[-1]
        assert np.isclose(latest_close, latest_adj, rtol=1e-12), (
            f'{symbol}: latest close {latest_close} != latest adjusted_close {latest_adj}'
        )


if __name__ == '__main__':
    test_compute_adjusted_close_matches_yfinance()
    test_compute_adjusted_close_preserves_close_at_latest_date()
