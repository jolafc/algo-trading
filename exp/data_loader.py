import os

import pandas as pd
from matplotlib import pyplot as plt

from exp import DATA_DIR, PLT_FILE_FORMAT, SP500_PKL
from exp.data_getter import load_prices_dict

ADJUSTED_CLOSE_COLUMN = 'adjusted_close'

def get_close_prices(pkl_file=SP500_PKL, debug=False):
    prices_dict = load_prices_dict(pkl_file=pkl_file)
    prices_dict = {k: v for k, v in prices_dict.items() if v is not None}

    adj_close_prices = [df[ADJUSTED_CLOSE_COLUMN].rename(k) for k, df in prices_dict.items()]
    adj_close_prices = pd.concat(adj_close_prices, axis=1)

    if debug:
        n_nans = adj_close_prices.isna().sum(axis=0).sort_values(ascending=False)
        print(f'Number of Nans per ticker:\n{n_nans}\n')

        adj_close_prices.isna().sum(axis=1).plot()
        plotfile = os.path.join(DATA_DIR, f'N_nans_in_time.{PLT_FILE_FORMAT}')
        plt.savefig(plotfile)
        print(f'Number of Nans per day plotted in file: {plotfile}\n')

    return adj_close_prices


def get_rolling_window(df, end, length):
    rolling_window = adj_close_prices[:end].iloc[-length:]

    return rolling_window


if __name__ == '__main__':
    adj_close_prices = get_close_prices(debug=True)

    end = pd.to_datetime('2019-01-02')
    length = 200
    rolling_window = get_rolling_window(adj_close_prices, end=end, length=length)
    print(rolling_window)
