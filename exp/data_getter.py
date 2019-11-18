# Module importing and config

import os
import sys
import requests
import io
import time
import pickle
import pandas as pd
from matplotlib import pyplot as plt
from timeit import default_timer as timer
from exp import DATA_DIR

pd.set_option('display.width', 160)

# Global constant declaration

PLT_FILE_FORMAT = '.png'
API_URL = "https://www.alphavantage.co/query"
API_KEY = os.environ['AV_KEY']
CONST_CSV = os.path.join(DATA_DIR, 'constituents.csv')
SP500_PKL = os.path.join(DATA_DIR, 'sp500.pkl')
PLOTFILE = os.path.join(DATA_DIR, f'sample_data'+PLT_FILE_FORMAT)
WAIT_TIME = 61.
QPM = 5

data = {
    "function": "TIME_SERIES_DAILY_ADJUSTED",
    "symbol": "MSFT",
    "outputsize": "full",
    "datatype": "csv",
    "apikey": API_KEY,
    }

# Sample AV API query

run_time = timer()
response = requests.get(API_URL, data)
run_time = timer() - run_time
print(f'Response: {response}, took: {run_time:.3f} s.')

# Sample parsing of AV API request into pd DF of historical prices.

file = io.StringIO(response.text)
price_df = pd.read_csv(file, index_col=0, parse_dates=['timestamp'])
print(price_df.head())
print(f'Price DF shape: {price_df.shape}')
price_df['adjusted_close'].plot()
plt.savefig(PLOTFILE)

# Load the reference (full) list of stock symbols that we wish to use for backtesting.
# Load from a git versionned file to avoid remote changes affecting this set.
# Current source was taken from the page: https://datahub.io/core/s-and-p-500-companies

const_df = pd.read_csv(CONST_CSV)
print(const_df.head())
print(f'S&P 500 constituents DF shape: {const_df.shape}')

# Complete the data necessary for backtesting, e.g.:
# - load currently pickled dict of historical prices DFs,
# - see which symbols from the reference list above are missing,
# - query historical prices for theses,
# - parse them to DFs and add them to the dict of historical prices DFs,
# - update the saved pickle.

symbols = const_df['Symbol']
prices = {}
if os.path.isfile(SP500_PKL):
    with open(SP500_PKL, 'rb') as file:
        prices = pickle.load(file)
missing_symbols = set(symbols).difference(set(prices.keys()))
tot_time = 0.
print(f'Querying AV API for {len(missing_symbols)} symbols historical prices:\n')
for i, symbol in enumerate(missing_symbols):
    if i%QPM == 0 and i>0:
        wait_time_left = WAIT_TIME - tot_time
        print(f'API quota reached, waiting {wait_time_left:.2f} s...')
        sys.stdout.flush()
        time.sleep(wait_time_left)
        tot_time = 0.
    run_time = timer()
    av_query = data.copy()
    av_query['symbol'] = symbol
    response = requests.get(API_URL, data)
    file = io.StringIO(response.text)
    try:
        price_df = pd.read_csv(file, index_col=0, parse_dates=['timestamp'])
        prices[symbol] = price_df
        run_time = timer() - run_time
        print(f'{i:03d} - Querying data for {symbol}, time={run_time:.2f} s.')
    except ValueError:
        run_time = timer() - run_time
        print(f'{i:03d} - ERROR - Querying data for {symbol} did not contain a valid DF, time={run_time:.2f} s.')
    sys.stdout.flush()
    tot_time += run_time
with open(SP500_PKL, 'wb') as file:
    pickle.dump(prices, file)
print(f'\nDict of price DFs saved as {SP500_PKL}')

# Load currently pickled dict of historical prices DFs

with open(SP500_PKL, 'rb') as file:
    prices = pickle.load(file)
print(f'\nLoaded pickled prices DF dict with {len(prices)} keys:')
print(sorted(prices.keys()))
