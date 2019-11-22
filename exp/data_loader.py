import os

import pandas as pd
from matplotlib import pyplot as plt

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


def get_rolling_window(df, end, length, dropna=True):
    rolling_window = df[:end].iloc[-length:]
    if dropna:
        rolling_window = rolling_window.dropna(axis=1)

    return rolling_window


if __name__ == '__main__':
    pkl_file = SP500_PKL  # par
    prices_dict = load_prices_dict(pkl_file=pkl_file)

    adj_close_prices = get_feature(prices_dict, column=ADJUSTED_CLOSE_COLUMN, debug=False, impute=False)
    volumes = get_feature(prices_dict, column=VOLUME_COLUMN, debug=False, impute=False)

    end = pd.to_datetime('2019-01-02')  # par
    length = 200  # par
    prices_window = get_rolling_window(adj_close_prices, end=end, length=length, dropna=True)
    volumes_window = get_rolling_window(volumes, end=end, length=length, dropna=True)

    print(prices_window.shape, volumes_window.shape)
