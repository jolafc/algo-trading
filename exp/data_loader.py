import os
import logging

import pandas as pd
from matplotlib import pyplot as plt

from exp import PLT_FILE_FORMAT, RESULTS_DIR
from exp.default_parameters import ADJUSTED_CLOSE_COLUMN


def get_feature(prices_dict, column=ADJUSTED_CLOSE_COLUMN, start_idx=None, end_idx=None, debug=False, impute=None,
                verbose=True):
    """Pivot a `{ticker: per-ticker DataFrame}` dict into one wide feature DataFrame.

    Args:
        prices_dict: output of `data_getter.load_pickled_dict`. Values that are None are dropped.
        column: which AV column to extract (see exp/default_parameters.py).
        start_idx, end_idx: optional integer slice on the resulting index.
        debug: if True, write a Nans-per-day plot to results/ and log NaN counts per ticker.
        impute: 'pad' to forward-fill NaNs, else no imputation.
        verbose: log a one-line summary.

    Returns:
        Wide DataFrame: index=DatetimeIndex (ascending), columns=tickers, dtype=float64 (or int for
        volume). See doc/data_schema.md §2.
    """
    prices_dict_clean = {k: v for k, v in prices_dict.items() if v is not None}

    feature_df = [df[column].rename(k) for k, df in prices_dict_clean.items()]
    feature_df = pd.concat(feature_df, axis=1)

    first_date = feature_df.index[0]
    last_date = feature_df.index[-1]
    n_tickers = len(feature_df.columns)
    if verbose:
        logging.info(f'Loaded {column} for {n_tickers} tickers from {first_date.date()} to {last_date.date()}.')

    if start_idx is not None and end_idx is not None:
        feature_df = feature_df.iloc[start_idx:end_idx + 1, :]
    if debug:
        report_nans(feature_df, feature_name=column, verbose=verbose)
    if impute is not None:
        feature_df = impute_time_series(feature_df, method=impute)

    return feature_df


def report_nans(df, feature_name=ADJUSTED_CLOSE_COLUMN, verbose=True):
    n_nans = df.isna().sum(axis=0).sort_values(ascending=False)

    df.isna().sum(axis=1).plot()
    plotfile = os.path.join(RESULTS_DIR, f'Nans_{feature_name}.{PLT_FILE_FORMAT}')
    plt.savefig(plotfile)

    if verbose:
        logging.info(f'Number of Nans per ticker for {feature_name}:\n{n_nans}\n')
        logging.info(f'Number of Nans per day for {feature_name} plotted in file: {plotfile}\n')


def impute_time_series(df, method=None):
    if method == 'pad':
        df = df.fillna(method=method)
    return df


def slice_backtesting_window(features={}, start_date_requested=None, end_date_requested=None, lookback=None,
                             verbose=True):
    """Clip every feature DataFrame to `[start - lookback, end]` and return the clipped dict.

    The lookback prefix lets strategies compute rolling indicators starting on `start_date_requested`.
    All frames in `features` must share an index; the helper asserts this. The actual start/end may
    differ from the requested ones if those dates aren't in the trading calendar — the resolved
    bounds are returned as the second tuple element.

    Args:
        features: dict[feature_name -> wide DataFrame].
        start_date_requested, end_date_requested: bracketing dates (inclusive).
        lookback: number of additional rows to keep before `start_date_requested`.
        verbose: log the adjusted window.

    Returns:
        (features_sliced, (start_date_actual, end_date_actual))
    """
    assert all([isinstance(feature, pd.DataFrame) for feature in features.values()]), \
        f'Passed features must all be pd.DataFrames!'
    indexes = [df.index for df in features.values()]
    assert all([indexes[0].equals(index) for index in indexes[1:]]), \
        f'All features DataFrames must have the same indexing.'

    df = list(features.values())[0]
    valid_dates = df[start_date_requested:end_date_requested].index
    start_date = valid_dates[0]
    end_date = valid_dates[-1]
    if verbose:
        logging.info(f'Backtesting date range adjusted from {start_date_requested.date()} - {end_date_requested.date()}'
              f' to {start_date.date()} - {end_date.date()}')

    start_idx = df.index.get_loc(start_date) - (lookback - 1)
    end_idx = df.index.get_loc(end_date)
    assert start_idx >= 0, \
        f'Lookback window could not be fulfilled, need {abs(start_idx)} additional data points before {start_date.date()}'

    features_sliced = {k: df.iloc[start_idx:end_idx + 1] for k, df in features.items()}

    return features_sliced, (start_date, end_date)


def dividend_adjust(close, dividends):
    """Apply Yahoo-style dividend adjustment to a per-ticker close series.

    For each dividend ex-date d with amount D, all prices strictly before d are scaled by
    (close[d-1] - D) / close[d-1]. The factor uses the running (already-adjusted) close at
    d-1, but since dividends are spaced apart in practice the result is order-independent.

    With yfinance-sourced data, `close` is already split-adjusted by Yahoo, so dividend
    adjustment alone reconstructs Yahoo's `Adj Close` to ~1 ppm — verified by
    tests/test_adjustment.py. Adding split adjustment on top would double-count.

    Args:
        close: pd.Series, per-ticker close prices, ascending DatetimeIndex.
        dividends: pd.Series, per-share dividend on each ex-date (0 elsewhere), same index.

    Returns:
        pd.Series of adjusted prices (a copy; `close` is not mutated).
    """
    adjusted = close.astype(float).copy()
    for date in dividends[(dividends != 0.) & ~dividends.isna()].index:
        i = adjusted.index.get_loc(date)
        if i == 0:
            continue
        prev = adjusted.iloc[i - 1]
        if pd.isna(prev) or prev == 0:
            continue
        factor = (prev - dividends.loc[date]) / prev
        adjusted.iloc[:i] = adjusted.iloc[:i] * factor
    return adjusted
