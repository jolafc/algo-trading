import os

import pandas as pd
from matplotlib import pyplot as plt

from exp import PLT_FILE_FORMAT, RESULTS_DIR
from exp.default_parameters import ADJUSTED_CLOSE_COLUMN


def get_feature(prices_dict, column=ADJUSTED_CLOSE_COLUMN, start_idx=None, end_idx=None, debug=False, impute=None,
                verbose=True):
    prices_dict_clean = {k: v for k, v in prices_dict.items() if v is not None}

    feature_df = [df[column].rename(k) for k, df in prices_dict_clean.items()]
    feature_df = pd.concat(feature_df, axis=1)

    first_date = feature_df.index[0]
    last_date = feature_df.index[-1]
    n_tickers = len(feature_df.columns)
    if verbose:
        print(f'Loaded {column} for {n_tickers} tickers from {first_date.date()} to {last_date.date()}.')

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
        print(f'Number of Nans per ticker for {feature_name}:\n{n_nans}\n')
        print(f'Number of Nans per day for {feature_name} plotted in file: {plotfile}\n')


def impute_time_series(df, method=None):
    if method == 'pad':
        df = df.fillna(method=method)
    return df


def slice_backtesting_window(features={}, start_date_requested=None, end_date_requested=None, lookback=None,
                             verbose=True):
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
        print(f'Backtesting date range adjusted from {start_date_requested.date()} - {end_date_requested.date()}'
              f' to {start_date.date()} - {end_date.date()}')

    start_idx = df.index.get_loc(start_date) - (lookback - 1)
    end_idx = df.index.get_loc(end_date)
    assert start_idx >= 0, \
        f'Lookback window could not be fulfilled, need {abs(start_idx)} additional data points before {start_date.date()}'

    features_sliced = {k: df.iloc[start_idx:end_idx + 1] for k, df in features.items()}

    return features_sliced, (start_date, end_date)
