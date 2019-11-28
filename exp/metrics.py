import numpy as np

from exp.default_parameters import N_DAYS_IN_YEAR


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
    n_buys = max(1, n_buys)
    notional_max = balance / n_buys
    position_max = notional_max / price_min
    max_buy_fees = get_ib_fees(position=position_max, price=price_min)
    budget_fees = n_buys * max_buy_fees
    notional = (balance - budget_fees) / n_buys

    return notional


def get_p_and_l(row):
    fees = row['fees']
    position = row['position']
    price_buy = row['price_buy']
    price_sell = row['price_sell']
    p_and_l = position * (price_sell - price_buy) - fees
    return p_and_l


def get_balance(start_balance, trades_df, positions_df):
    p_and_l = trades_df['P&L'].sum()
    fees = positions_df['fees_buy'].sum()
    invested = (positions_df['position'] * positions_df['price_buy']).sum()
    balance = start_balance + p_and_l - invested - fees

    return balance


def get_total_unrealized_p_and_l(trades_df, positions_df, prices_series):
    realized_pl = trades_df['P&L'].sum()

    invested = (positions_df['position'] * positions_df['price_buy']).sum()
    prices_current = prices_series[positions_df['ticker']].values
    value_current = (positions_df['position'] * prices_current).sum()
    fees = positions_df['fees_buy'].sum()
    unrealized_pl = value_current - invested - fees

    p_and_l = realized_pl + unrealized_pl

    return p_and_l


def get_annualized_yield(unrealized_pl, start_balance=0.):
    dates = unrealized_pl.index
    backtesting_duration = (dates[-1] - dates[0]).days
    start_value = start_balance + unrealized_pl.iloc[0]
    end_value = start_balance + unrealized_pl.iloc[-1]
    annualized_yield = N_DAYS_IN_YEAR / backtesting_duration * np.log(end_value / start_value)

    return annualized_yield


def get_sharpe_ratio(unrealized_pl, start_balance=0.):
    normalized_returns = (unrealized_pl - unrealized_pl.shift(1)) / (unrealized_pl.shift(1) + start_balance)

    duration_in_years = (unrealized_pl.index[-1] - unrealized_pl.index[0]).days / N_DAYS_IN_YEAR
    mean_steps_per_year = unrealized_pl.size / duration_in_years

    norm_returns_mean_annualized = normalized_returns.mean() * mean_steps_per_year
    norm_returns_std_annualized = normalized_returns.std() * np.sqrt(
        mean_steps_per_year)  # Assuming a Weiner process / random walk; mean total distance = distance per step * sqrt(number of steps)
    sharpe_ratio = norm_returns_mean_annualized / norm_returns_std_annualized

    return sharpe_ratio


def get_sortino_ratio(unrealized_pl, start_balance=0.):
    normalized_returns = (unrealized_pl - unrealized_pl.shift(1)) / (unrealized_pl.shift(1) + start_balance)

    duration_in_years = (unrealized_pl.index[-1] - unrealized_pl.index[0]).days / N_DAYS_IN_YEAR
    mean_steps_per_year = unrealized_pl.size / duration_in_years

    norm_returns_mean_annualized = normalized_returns.mean() * mean_steps_per_year

    neg_returns_mask = normalized_returns < 0.
    neg_returns_std = normalized_returns[neg_returns_mask].std()
    norm_returns_std_annualized = neg_returns_std * np.sqrt(
        mean_steps_per_year)  # Assuming a Weiner process / random walk; mean total distance = distance per step * sqrt(number of steps)

    sharpe_ratio = norm_returns_mean_annualized / norm_returns_std_annualized

    return sharpe_ratio
