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
### TODO: Need to (re)-implement the RSI/EMA indicators.
### TODO: Forward pad: limit the number of padded values

def get_feature(prices_dict, column=ADJUSTED_CLOSE_COLUMN, start_idx=None, end_idx=None, debug=False, impute=None):
    prices_dict_clean = {k: v for k, v in prices_dict.items() if v is not None}

    feature_df = [df[column].rename(k) for k, df in prices_dict_clean.items()]
    feature_df = pd.concat(feature_df, axis=1)
    if start_idx is not None and end_idx is not None:
        feature_df = feature_df.iloc[start_idx:end_idx+1, :]

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

    start_date = pd.to_datetime('2019-01-31')  # PAR
    end_date = pd.to_datetime('2019-10-31')  # PAR
    lookback = 200  # PAR

    rsi_lookback = 3 # PAR
    date = pd.to_datetime('2019-01-31')  # PAR
    sma_tol = 0.02 # PAR
    volume_lookback = 20 # PAR
    volume_threshold = 1e6 # PAR
    price_threshold = 1. # PAR
    rsi_threshold = 50. # PAR
    n_positions = 10 # PAR

    prices_dict = load_prices_dict(pkl_file=pkl_file)
    adj_close_prices = get_feature(prices_dict, column=ADJUSTED_CLOSE_COLUMN, debug=False, impute=False)
    start_idx = adj_close_prices.index.get_loc(start_date) - (lookback - 1)
    end_idx = adj_close_prices.index.get_loc(end_date)
    adj_close_prices = adj_close_prices.iloc[start_idx:end_idx+1]

    volumes = get_feature(prices_dict, column=VOLUME_COLUMN, start_idx=start_idx, end_idx=end_idx, debug=False, impute=False)

    # Filters for §8: Weekly rotation of the S&P 500 - The 30-Minute Stock Trader
    # 1 - SPY is above 0.98*SMA(200)
    spy = adj_close_prices['SPY']
    spy_sma = adj_close_prices['SPY'].rolling(lookback).mean()
    spy_conditions = spy > spy_sma * (1 - sma_tol)

    # 2 - avg. vol. (20) >= 1M
    volumes_sma = volumes.rolling(volume_lookback).mean()
    volumes_masks = volumes_sma >= volume_threshold

    # 3 - Min price 1$
    price_masks = adj_close_prices > price_threshold

    # 4 - RSI(3) < 50
    prices_rsi = adj_close_prices.apply(lambda x: ta.momentum.rsi(x, n=rsi_lookback))
    rsi_masks = prices_rsi < rsi_threshold

    # 5 - Sort wrt the ROC(200)
    rocs = adj_close_prices / adj_close_prices.shift(lookback-1)

    print(f'SPY condition to trade: {spy[date]} > {spy_sma[date]} to {sma_tol*100:.0f}% ==> {spy_conditions[date]}')
    if spy_conditions[date]:
        volumes_mask = volumes_masks.loc[date, :]
        print(f'Volume filtering: {volumes_mask.sum()} / {volumes_mask.shape} passed.')

        price_mask = price_masks.loc[date, :]
        print(f'Price filtering: {price_mask.sum()} / {price_mask.shape} passed.')

        rsi_mask = rsi_masks.loc[date, :]
        print(f'RSI filtering: {rsi_mask.sum()} / {rsi_mask.shape} passed.')

        # Positions - sorted(ROC(200), descending), Update positions to 0.1 for first 10 elements of the sorted list
        mask = volumes_mask & price_mask & rsi_mask
        print(f'Filtering: {mask.sum()} / {mask.shape} passed.')

        roc = rocs.loc[date, :]
        roc_sorted = roc[mask].sort_values(ascending=False)
        positions = list(roc_sorted[:n_positions].index)
        print(positions)
