# Module importing and config

import os
import sys
from copy import deepcopy

import requests
import io
import time
import pickle
import pandas as pd
from timeit import default_timer as timer
from exp import CONST_CSV, SP500_PKL

pd.set_option('display.width', 160)

API_URL = "https://www.alphavantage.co/query"
API_KEY = os.environ['AV_KEY']
WAIT_TIME = 61.
QPM = int(os.environ['AV_RQM'])
QPS = 10

AV_QUERY_DATA = {
    "function": "TIME_SERIES_DAILY_ADJUSTED",
    "symbol": "MSFT",
    "outputsize": "full",
    "datatype": "csv",
    "apikey": API_KEY,
}


def av_query(symbol):
    av_query_data = AV_QUERY_DATA.copy()
    av_query_data['symbol'] = symbol
    run_time = timer()
    response = requests.get(API_URL, av_query_data)
    file = io.StringIO(response.text)
    price_df = None
    try:
        price_df = pd.read_csv(file, index_col=0, parse_dates=['timestamp'])
    except ValueError:
        print(f'ERROR - Querying data for {symbol} did not contain a valid DF. Instead found:\n{response.text}')
    run_time = timer() - run_time

    return price_df, run_time


# Current source (CONST_CSV) was taken from the page: https://datahub.io/core/s-and-p-500-companies
def get_universe_symbols(universe_csv=CONST_CSV, symbols_column='Symbol'):
    const_df = pd.read_csv(universe_csv)
    symbols = const_df[symbols_column]
    return symbols


def get_universe_prices(symbols, prices={}, save_file=None, save_frequency=QPS):
    updated_prices = deepcopy(prices)
    missing_symbols = set(symbols).difference(set(updated_prices.keys()))
    tot_time = 0.
    print(f'Querying AV API for {len(missing_symbols)} symbols historical prices:\n')
    for i, symbol in enumerate(missing_symbols):
        if i % QPM == 0 and i > 0:
            wait_time_left = WAIT_TIME - tot_time
            tot_time = 0.
            if wait_time_left > 0:
                print(f'API quota reached, waiting {wait_time_left:.2f} s...')
                sys.stdout.flush()
                time.sleep(wait_time_left)

        updated_prices[symbol], run_time = av_query(symbol=symbol)
        print(f'{i:03d} - Queried data for {symbol}, time={run_time:.2f} s.')

        sys.stdout.flush()
        tot_time += run_time

        if save_file is not None and i % save_frequency == 0 and i > 0:
            save_prices_dict(updated_prices, pkl_file=save_file)
            print(f'{i:03d} - Saved data to file={save_file}')

    return updated_prices


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
    symbols = get_universe_symbols(universe_csv=CONST_CSV)
    prices = {} if not update else load_pickled_dict(pkl_file=SP500_PKL)
    updated_prices = get_universe_prices(symbols, prices=prices, save_file=SP500_PKL, save_frequency=QPS)
    save_prices_dict(updated_prices)


if __name__ == '__main__':
    get_sp500_pkl(update=True)
