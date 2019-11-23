import os

import pandas as pd
from matplotlib import pyplot as plt

import ta.momentum

from exp import DATA_DIR, PLT_FILE_FORMAT, SP500_PKL
from exp.data_getter import load_prices_dict

ADJUSTED_CLOSE_COLUMN = 'adjusted_close'
OPEN_COLUMN = 'open'
HIGH_COLUMN = 'high'
LOW_COLUMN = 'low'
CLOSE_COLUMN = 'close'
VOLUME_COLUMN = 'volume'
DIVIDENT_COLUMN = 'dividend_amount'
SPLIT_COLUMN = 'split_coefficient'


### TODO: Need the joiner and leaver data of the SP500 stocks and filtering functions

def get_feature(prices_dict, column=ADJUSTED_CLOSE_COLUMN, debug=False, impute=None):
    prices_dict_clean = {k: v for k, v in prices_dict.items() if v is not None}

    feature_df = [df[column].rename(k) for k, df in prices_dict_clean.items()]
    feature_df = pd.concat(feature_df, axis=1)

    if debug:
        report_nans(feature_df, feature_name=column)
    if impute is not None:
        feature_df = impute_time_series(feature_df, method=impute)

    return feature_df


def report_nans(df, feature_name=ADJUSTED_CLOSE_COLUMN):
    n_nans = df.isna().sum(axis=0).sort_values(ascending=False)
    print(f'Number of Nans per ticker for {feature_name}:\n{n_nans}\n')

    df.isna().sum(axis=1).plot()
    plotfile = os.path.join(DATA_DIR, f'Nans_{feature_name}.{PLT_FILE_FORMAT}')
    plt.savefig(plotfile)
    print(f'Number of Nans per day for {feature_name} plotted in file: {plotfile}\n')


def impute_time_series(df, method=None):
    if method == 'pad':
        df = df.fillna(method=method)
    return df


def get_rolling_window(df, date, lookback=None, dropna=True):
    rolling_window = df[:date]
    if lookback is not None:
        rolling_window = rolling_window.iloc[-lookback:]
    if dropna:
        rolling_window = rolling_window.dropna(axis=1)

    return rolling_window


if __name__ == '__main__':
    pkl_file = SP500_PKL  # PAR
    prices_dict = load_prices_dict(pkl_file=pkl_file)

    adj_close_prices = get_feature(prices_dict, column=ADJUSTED_CLOSE_COLUMN, debug=False, impute=False)
    volumes = get_feature(prices_dict, column=VOLUME_COLUMN, debug=False, impute=False)
    rsi_lookback = 3 # PAR
    rsis = adj_close_prices.apply(lambda x: ta.momentum.rsi(x, n=rsi_lookback))

    date = pd.to_datetime('2019-01-31')  # PAR
    lookback = 200  # PAR
    prices_window = get_rolling_window(adj_close_prices, date=date, lookback=lookback, dropna=True)
    volumes_window = get_rolling_window(volumes, date=date, lookback=lookback, dropna=True)

    # print(prices_window.shape, volumes_window.shape)

    # Filters for §8: Weekly rotation of the S&P 500 - The 30-Minute Stock Trader
    # 1 - SPY is above 0.98*SMA(200)
    spy_series = adj_close_prices['SPY']
    spy_window = get_rolling_window(spy_series, date=date, lookback=lookback, dropna=False)
    spy_sma = spy_window.mean()
    spy_today = spy_series[date]
    sma_tol = 0.02 # PAR
    spy_condition = spy_today > spy_sma * (1 - sma_tol)
    print(f'SPY condition to trade: {spy_today} > {spy_sma} to {sma_tol*100:.0f}% ==> {spy_condition}')
    if spy_condition:

        # 2 - avg. vol. (20) >= 1M
        volume_lookback = 20 # PAR
        volumes_window = get_rolling_window(volumes, date=date, lookback=volume_lookback, dropna=False)
        volumes_sma = volumes_window.mean(axis=0)
        volume_threshold = 1e6 # PAR
        volumes_mask = volumes_sma >= volume_threshold
        print(f'Volume filtering: {volumes_mask.sum()} / {volumes_mask.shape} passed.')

        # 3 - Min price 1$
        price_threshold = 1. # PAR
        price_mask = adj_close_prices.loc[date, :] > price_threshold
        print(f'Price filtering: {price_mask.sum()} / {price_mask.shape} passed.')

        # 4 - RSI(3) < 50
        rsi_threshold = 50. # PAR
        rsi_mask = rsis.loc[date, :] < rsi_threshold
        print(f'RSI filtering: {rsi_mask.sum()} / {rsi_mask.shape} passed.')

        # Positions - sorted(ROC(200), descending), Update positions to 0.1 for first 10 elements of the sorted list
        mask = volumes_mask & price_mask & rsi_mask
        print(f'Filtering: {mask.sum()} / {mask.shape} passed.')
        prices_window = get_rolling_window(adj_close_prices, date=date, lookback=lookback, dropna=False)
        roc = prices_window.iloc[-1, :] / prices_window.iloc[0, :]
        n_positions = 10
        roc_sorted = roc[mask].sort_values(ascending=False)
        positions = list(roc_sorted[:n_positions].index)
        print(positions)
