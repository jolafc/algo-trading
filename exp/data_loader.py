import os
from timeit import default_timer as timer

import pandas as pd
from matplotlib import pyplot as plt

import ta.momentum

from exp import DATA_DIR, PLT_FILE_FORMAT, SP500_PKL
from exp.data_getter import load_pickled_dict

ADJUSTED_CLOSE_COLUMN = 'adjusted_close'
OPEN_COLUMN = 'open'
HIGH_COLUMN = 'high'
LOW_COLUMN = 'low'
CLOSE_COLUMN = 'close'
VOLUME_COLUMN = 'volume'
DIVIDENT_COLUMN = 'dividend_amount'
SPLIT_COLUMN = 'split_coefficient'

POSITIONS_COLUMNS = ['date_buy', 'ticker', 'position', 'price_buy', 'fees_buy']
TRADES_EXTRA_COLUMNS = ['date_sell', 'price_sell', 'fees_sell']
TRADES_COLUMNS = POSITIONS_COLUMNS + TRADES_EXTRA_COLUMNS + ['fees', 'P&L']


### TODO: Need the joiner and leaver data of the SP500 stocks and filtering functions
### TODO: Need to (re)-implement the RSI/EMA indicators.
### TODO: Forward pad: limit the number of padded values
### TODO: The price for the execution is currently the price for the decision making; maybe the next day open instead?

def get_feature(prices_dict, column=ADJUSTED_CLOSE_COLUMN, start_idx=None, end_idx=None, debug=False, impute=None):
    prices_dict_clean = {k: v for k, v in prices_dict.items() if v is not None}

    feature_df = [df[column].rename(k) for k, df in prices_dict_clean.items()]
    feature_df = pd.concat(feature_df, axis=1)
    if start_idx is not None and end_idx is not None:
        feature_df = feature_df.iloc[start_idx:end_idx + 1, :]

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


def get_ib_fees(position, price):
    IB_FEE_PER_SHARE = 0.005
    IB_MAX_FEE_PERCENT = 0.01
    IB_MIN_FEE = 1.

    fee_min = IB_MIN_FEE
    fee_max = position * price * IB_MAX_FEE_PERCENT
    fee = IB_FEE_PER_SHARE * position
    fee = min(max(fee, fee_min), fee_max)

    return fee


def get_notional(balance, n_buys, price_min):
    notional_max = balance / n_buys
    position_max = notional_max / price_min
    max_buy_fees = get_ib_fees(position=position_max, price=price_min)
    budget_fees = n_buys * max_buy_fees
    notional = (balance - budget_fees) / n_buys

    return notional


if __name__ == '__main__':
    time = timer()

    pkl_file = SP500_PKL  # PAR data

    start_date = pd.to_datetime('2019-01-31')  # PAR backtesting
    end_date = pd.to_datetime('2019-10-31')  # PAR backtesting
    lookback = 200  # PAR backtesting

    sma_tol = 0.02  # PAR strategy
    volume_lookback = 20  # PAR strategy
    volume_threshold = 1e6  # PAR strategy
    price_min = 1.  # PAR strategy
    rsi_lookback = 3  # PAR strategy
    rsi_threshold = 50.  # PAR strategy
    day_of_trade = 4  # PAR strategy - pandas DatetimeIndex.dayofweek value for Friday
    n_positions = 10  # PAR strategy
    start_balance = 100000  # PAR strategy

    data_by_ticker = load_pickled_dict(pkl_file=pkl_file)

    data_by_feature = {}
    adj_close_prices = get_feature(data_by_ticker, column=ADJUSTED_CLOSE_COLUMN, debug=False, impute=False)
    start_idx = adj_close_prices.index.get_loc(start_date) - (lookback - 1)
    end_idx = adj_close_prices.index.get_loc(end_date)
    adj_close_prices = adj_close_prices.iloc[start_idx:end_idx + 1]

    volumes = get_feature(data_by_ticker, column=VOLUME_COLUMN, start_idx=start_idx, end_idx=end_idx, debug=False,
                          impute=False)

    # Filters for §8: Weekly rotation of the S&P 500 - The 30-Minute Stock Trader
    # 1 - SPY is above 0.98*SMA(200)
    spy = adj_close_prices['SPY']
    spy_sma = adj_close_prices['SPY'].rolling(lookback).mean()
    spy_masks = spy > spy_sma * (1 - sma_tol)

    # 2 - avg. vol. (20) >= 1M
    volumes_sma = volumes.rolling(volume_lookback).mean()
    volumes_masks = volumes_sma >= volume_threshold

    # 3 - Min price 1$
    price_masks = adj_close_prices > price_min

    # 4 - RSI(3) < 50
    prices_rsi = adj_close_prices.apply(lambda x: ta.momentum.rsi(x, n=rsi_lookback))
    rsi_masks = prices_rsi < rsi_threshold

    # Total masks 1-4
    # For pd.Series: .values[:, None] is to transform the series to a np.array and use its broadcasting capabilities,
    # specifying which axis needs to be repeated
    masks = volumes_masks & price_masks & rsi_masks & spy_masks.values[:, None]

    # 5a - Sort wrt the ROC(200)
    performances = adj_close_prices / adj_close_prices.shift(lookback - 1)

    # 5b - trade on Fridays in the backtesting window
    valid_dates = adj_close_prices.index
    day_of_week_triggers = valid_dates.dayofweek == day_of_trade
    backtesting_window_triggers = (valid_dates >= start_date) & (valid_dates <= end_date)
    triggers = day_of_week_triggers & backtesting_window_triggers
    dates = valid_dates[triggers]

    positions_list = pd.Series(index=dates, name='positions')
    for date in dates:
        mask = masks.loc[date]
        performance = performances.loc[date]
        positions_list[date] = performance[mask].sort_values(ascending=False)[:n_positions].index.tolist()

    # Backtesting loop.
    # V 1 - Positions dataframe with buy price+date
    # V 2 - Trades dataframe with positions df columns + sell price+date+fees.
    # V   - Sells are first removed from the positions and added to the trades. Then buys are added to positions.
    # 3 - Then, add P&L.
    # 4 - Balance: remove each transaction and its fees as they occur.
    #   - And assert that the balance - P&L.sum() + positions bought == start_balance

    positions_df = pd.DataFrame(columns=POSITIONS_COLUMNS)
    trades_df = pd.DataFrame(columns=TRADES_COLUMNS)
    pos_idx = 0
    balance = start_balance
    for i, date in enumerate(dates):
        new_position = set(positions_list.iloc[i])
        old_position = set(positions_list.iloc[i - 1]) if i - 1 >= 0 else set()
        buys = new_position.difference(old_position)
        sells = old_position.difference(new_position)

        for sell in sells:
            mask = positions_df['ticker'] == sell
            assert mask.sum() == 1, f'The number of current positions matching {sell} is {mask.sum()} but should be ==1'
            pos_id = positions_df[mask].index[0]

            price = adj_close_prices.loc[date, sell]
            position = positions_df.loc[pos_id, 'position']
            fee = get_ib_fees(position=position, price=price)

            trades_df.loc[pos_id, POSITIONS_COLUMNS] = positions_df.loc[pos_id, POSITIONS_COLUMNS]
            trades_df.loc[pos_id, TRADES_EXTRA_COLUMNS] = [date, price, fee]
            positions_df = positions_df.drop(index=pos_id)

        notional = get_notional(balance=balance, n_buys=len(buys), price_min=price_min)
        for buy in buys:
            price = adj_close_prices.loc[date, buy]
            position = notional // price
            fee = get_ib_fees(position=position, price=price)
            positions_df.loc[pos_idx, POSITIONS_COLUMNS] = [date, buy, position, price, fee]
            pos_idx += 1

    print(positions_df)
    print(trades_df)

    time = timer() - time
    print(f'Loop time: {time} s.')
