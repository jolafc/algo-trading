"""Fetch historical daily price data from Yahoo Finance and cache it as a per-ticker pickle.

History: this module used to call Alpha Vantage's TIME_SERIES_DAILY_ADJUSTED (premium). When
that endpoint became premium-only and the project lost its key, we switched to yfinance — free,
no API key, single call per ticker returns OHLCV + dividends + splits. The per-ticker DataFrame
shape (column names, index) is preserved so exp.data_loader and the strategies are unchanged.
The `adjusted_close` column is computed locally from close + dividends via
`compute_adjusted_close`, which delegates the per-ticker math to `data_loader.dividend_adjust`.
"""
import logging
import os
import pickle
import sys
import time
from copy import deepcopy
from timeit import default_timer as timer

import pandas as pd
import yfinance as yf

from exp import CONST_CSV, SP500_PKL
from exp.data_loader import dividend_adjust
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

pd.set_option('display.width', 160)

# yfinance has no published hard rate limit, but Yahoo throttles aggressively on burst traffic.
# A small inter-call sleep keeps SP500-scale downloads reliable. Override via env if needed.
INTER_CALL_SLEEP_S = float(os.environ.get('YF_SLEEP', '0.2'))
SAVE_EVERY_N = int(os.environ.get('YF_SAVE_EVERY', '10'))


def _to_yf_symbol(symbol):
    # yfinance uses dashes for class shares (e.g. BRK.B → BRK-B).
    return symbol.replace('.', '-')


def yf_query(symbol):
    """Fetch full daily history for `symbol` from yfinance, shaped like the legacy AV per-ticker DF.

    Returns:
        (DataFrame or None, run_time_seconds). DataFrame columns (in order):
        open, high, low, close, volume, dividend_amount, split_coefficient.
        Index: `timestamp` DatetimeIndex (ascending, tz-naive).

        `adjusted_close` is NOT populated here — it's computed in a second pass by
        `compute_adjusted_close` so dividend/split logic stays colocated with the
        existing adjustment classes in exp.data_loader.
    """
    run_time = timer()
    df = None
    try:
        raw = yf.Ticker(_to_yf_symbol(symbol)).history(period='max', auto_adjust=False)
        if raw.empty:
            logging.info(f'ERROR - yfinance returned empty data for {symbol}')
        else:
            df = pd.DataFrame({
                OPEN_COLUMN:     raw['Open'].astype(float),
                HIGH_COLUMN:     raw['High'].astype(float),
                LOW_COLUMN:      raw['Low'].astype(float),
                CLOSE_COLUMN:    raw['Close'].astype(float),
                VOLUME_COLUMN:   raw['Volume'].astype(float),
                DIVIDENT_COLUMN: raw['Dividends'].astype(float),
                # yfinance's Stock Splits uses 0.0 for "no split"; the existing adjustment code
                # expects 1.0 (multiplicative neutral). Match the legacy AV convention.
                SPLIT_COLUMN:    raw['Stock Splits'].astype(float).replace(0., 1.),
            })
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = 'timestamp'
    except Exception as e:  # noqa: BLE001 — yfinance raises a wide variety of network errors
        logging.info(f'ERROR - Querying yfinance for {symbol} raised {type(e).__name__}: {e}')
    run_time = timer() - run_time
    return df, run_time


# Current source (CONST_CSV) was taken from the page: https://datahub.io/core/s-and-p-500-companies
def get_universe_symbols(universe_csv=CONST_CSV, symbols_column='Symbol'):
    const_df = pd.read_csv(universe_csv)
    symbols = const_df[symbols_column].astype(str).str.strip()
    return symbols


def get_universe_prices(symbols, prices={}, save_file=None, save_frequency=SAVE_EVERY_N):
    updated_prices = deepcopy(prices)
    missing_symbols = sorted(set(symbols).difference(set(updated_prices.keys())))
    logging.info(f'Querying yfinance for {len(missing_symbols)} symbols historical prices:')
    for i, symbol in enumerate(missing_symbols):
        updated_prices[symbol], run_time = yf_query(symbol=symbol)
        logging.info(f'{i:03d} - Queried data for {symbol}, time={run_time:.2f} s.')
        sys.stdout.flush()
        if INTER_CALL_SLEEP_S > 0:
            time.sleep(INTER_CALL_SLEEP_S)
        if save_file is not None and i % save_frequency == 0 and i > 0:
            save_prices_dict(updated_prices, pkl_file=save_file)
            logging.info(f'{i:03d} - Saved data to file={save_file}')
    return updated_prices


def compute_adjusted_close(prices):
    """Populate `adjusted_close` on every per-ticker DataFrame in `prices`.

    Per-ticker math lives in `data_loader.dividend_adjust`. This is just the dict-level
    orchestrator: extract close + dividend series, delegate, write the result back.
    yfinance's close is already split-adjusted by Yahoo, so dividend-only adjustment is
    sufficient — the `split_coefficient` column stays populated for reference but is
    intentionally not consumed (see `tests/test_adjustment.py`).

    Mutates `prices` in place. Tickers whose value is None are skipped.
    """
    for df in prices.values():
        if df is None:
            continue
        df[ADJUSTED_CLOSE_COLUMN] = dividend_adjust(df[CLOSE_COLUMN], df[DIVIDENT_COLUMN])
    return prices


def save_prices_dict(prices, pkl_file=SP500_PKL):
    with open(pkl_file, 'wb') as file:
        pickle.dump(prices, file)


def load_pickled_dict(pkl_file=SP500_PKL):
    if os.path.isfile(pkl_file):
        with open(pkl_file, 'rb') as file:
            prices = pickle.load(file)
    else:
        prices = {}
    return prices


def get_sp500_pkl(update=True):
    """Download (or resume) the full SP500 history into `data/sp500.pkl`.

    Pulls raw OHLCV + dividends + splits per ticker via yfinance, then computes
    `adjusted_close` locally and stores it on each DataFrame. The on-disk shape
    matches the legacy AV pickle so downstream code is unchanged.
    """
    symbols = get_universe_symbols(universe_csv=CONST_CSV)
    prices = load_pickled_dict(pkl_file=SP500_PKL) if update else {}
    updated_prices = get_universe_prices(symbols, prices=prices, save_file=SP500_PKL)
    compute_adjusted_close(updated_prices)
    save_prices_dict(updated_prices)


if __name__ == '__main__':
    get_sp500_pkl(update=True)
