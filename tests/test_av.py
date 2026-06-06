"""Smoke test for exp.data_getter.

Originally tested the Alpha Vantage `av_query` (premium endpoint). After the migration to
yfinance (see exp/data_getter.py docstring) this verifies that `yf_query` returns a DataFrame
shaped like the legacy AV per-ticker DF with all expected columns. Hits the live yfinance API.
"""
import os

from matplotlib import pyplot as plt

from exp import PLT_FILE_FORMAT, RESULTS_DIR
from exp.data_getter import yf_query
from exp.default_parameters import (
    ADJUSTED_CLOSE_COLUMN,
    CLOSE_COLUMN,
    DIVIDENT_COLUMN,
    HIGH_COLUMN,
    LOW_COLUMN,
    OPEN_COLUMN,
    SPLIT_COLUMN,
    VOLUME_COLUMN,
)

EXPECTED_COLUMNS = {
    OPEN_COLUMN, HIGH_COLUMN, LOW_COLUMN, CLOSE_COLUMN, VOLUME_COLUMN,
    DIVIDENT_COLUMN, SPLIT_COLUMN,
}


def test_yf_query(symbol='MSFT'):
    print(f'yfinance - querying daily prices for {symbol}.')
    price_df, run_time = yf_query(symbol=symbol)
    print(f'Query success, took: {run_time:.3f} s.')

    assert price_df is not None, f'yfinance returned None for {symbol}'
    assert price_df.shape[0] > 1000, f'Expected long history for {symbol}, got {price_df.shape[0]} rows'
    missing = EXPECTED_COLUMNS - set(price_df.columns)
    assert not missing, f'Missing expected columns: {missing}'
    # adjusted_close is populated by compute_adjusted_close in a second pass — not by yf_query.
    assert ADJUSTED_CLOSE_COLUMN not in price_df.columns

    print(f'Price DF shape: {price_df.shape}')
    price_df[CLOSE_COLUMN].plot()
    plotfile = os.path.join(RESULTS_DIR, f'{symbol}_data.{PLT_FILE_FORMAT}')
    plt.savefig(plotfile)
    print(f'Plotted price history data for {symbol} in {plotfile}')


if __name__ == '__main__':
    test_yf_query()
