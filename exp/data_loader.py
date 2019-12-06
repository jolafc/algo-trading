import os
import logging

import pandas as pd
from matplotlib import pyplot as plt

from exp import PLT_FILE_FORMAT, RESULTS_DIR, SP500_PKL
from exp.data_getter import load_pickled_dict
from exp.default_parameters import ADJUSTED_CLOSE_COLUMN, CLOSE_COLUMN, DIVIDENT_COLUMN, SPLIT_COLUMN


def get_feature(prices_dict, column=ADJUSTED_CLOSE_COLUMN, start_idx=None, end_idx=None, debug=False, impute=None,
                verbose=True):
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


class DividentsAdjustment(object):
    def __init__(self, dividents, display_interval=.01):
        self.dividents = dividents
        self.mask = (dividents != 0.) & (~dividents.isna())

        self.n = dividents.shape[1]
        if type(display_interval) is float:
            self.display_interval = max(int(display_interval*self.n), 1)
        else:
            self.display_interval = display_interval

    def __call__(self, prices):
        ticker = prices.name
        tdividents = self.dividents[ticker][self.mask[ticker]]
        dates = tdividents.index
        for date in dates:
            i = prices.index.get_loc(date)
            factor = (prices.iloc[i-1] - tdividents[date]) / prices.iloc[i-1]
            prices.iloc[:i] = prices.iloc[:i] * factor

        i = list(self.dividents.columns).index(ticker)
        if (i+1) % self.display_interval == 0:
            logging.info(f'Adjusted ticker {i+1} / {self.n}')


class SplitsAdjustment(object):
    def __init__(self, splits, display_interval=.01):
        self.splits = splits
        self.mask = (splits != 1.) & (~splits.isna())

        self.n = splits.shape[1]
        if type(display_interval) is float:
            self.display_interval = max(int(display_interval*self.n), 1)
        else:
            self.display_interval = display_interval

    def __call__(self, prices):
        ticker = prices.name
        tsplits = self.splits[ticker][self.mask[ticker]]
        dates = tsplits.index
        for date in dates:
            i = prices.index.get_loc(date)
            prices.iloc[:i] = prices.iloc[:i] / tsplits[date]

        i = list(self.splits.columns).index(ticker)
        if (i+1)%self.display_interval == 0:
            logging.info(f'Adjusted ticker {i+1} / {self.n}')


def price_adj_experiment(n=None, verbose=True):
    data_by_ticker = load_pickled_dict(pkl_file=SP500_PKL)

    adj_close = get_feature(data_by_ticker, column=ADJUSTED_CLOSE_COLUMN, debug=False, impute=False, verbose=False)
    close = get_feature(data_by_ticker, column=CLOSE_COLUMN, debug=False, impute=False, verbose=False)
    dividents = get_feature(data_by_ticker, column=DIVIDENT_COLUMN, debug=False, impute=False, verbose=False)
    splits = get_feature(data_by_ticker, column=SPLIT_COLUMN, debug=False, impute=False, verbose=False)

    close = close.iloc[:, :n]
    adj_close = adj_close.iloc[:, :n]

    man_close = close.copy()

    dividents_adj = DividentsAdjustment(dividents=dividents)
    man_close.apply(dividents_adj)

    splits_adj = SplitsAdjustment(splits=splits)
    man_close.apply(splits_adj)

    if verbose:
        rel_error_before = (close - adj_close).abs().sum(axis=0) / adj_close.sum(axis=0)
        rel_error_before.name = 'Before'
        rel_error_after = (man_close - adj_close).abs().sum(axis=0) / adj_close.sum(axis=0)
        rel_error_after.name = 'After'
        rel_errors = pd.concat([rel_error_after, rel_error_before], axis=1)
        logging.info('')
        logging.info(f'Summed abs errors, relative to the summed prices: ')
        logging.info(rel_errors)
        logging.info('')
        logging.info(f'Totals abs. rel. errors (over all tickers): ')
        logging.info(rel_errors.sum(axis=0))

        resid = man_close - adj_close
        pd.plotting.register_matplotlib_converters()
        plt.figure()
        plt.plot(resid)
        plt.savefig(os.path.join(RESULTS_DIR, f'adj_prices_diff.png'))

    return man_close


if __name__ == '__main__':
    price_adj_experiment(n=3)
