import numpy as np
import pandas as pd

from exp.metrics import get_ib_fees, get_notional, get_p_and_l, get_balance, get_total_unrealized_p_and_l
from exp.default_parameters import DEFAULT_START_BALANCE, POSITIONS_COLUMNS, TRADES_COLUMNS, TRADES_EXTRA_COLUMNS


class Backtesting(object):
    """Replays a strategy's positions through a price history and tracks cash, P&L, and trades.

    Usage:
        bt = Backtesting(start_balance=100_000).fit(strategy, prices, dates)
        trades_df, positions_df, errors_df = bt.get_trades()
        unrealized_pl = bt.get_unrealized_pl()

    See doc/data_schema.md for the exact shape of the DataFrames returned.
    """

    def __init__(self, start_balance=DEFAULT_START_BALANCE):
        self.start_balance = start_balance

    def fit(self, strategy, prices, dates=None, low=None, high=None):
        """Iterate over `dates`, ask the strategy for holdings, simulate fills, accumulate P&L.

        Args:
            strategy: object exposing `predict(date) -> list[ticker]` and `price_min` attribute.
                See doc/data_schema.md §10 for the full strategy contract.
            prices: wide DataFrame, index=dates, columns=tickers, values=execution price.
            dates: iterable of trade dates; defaults to `prices.index`.
            low: optional DataFrame, same shape as `prices`. When provided, sells fill at the low
                (conservative). NaN at a sell date means delisting — the position is unwound at
                the buy price (zero P&L) and recorded in `errors_df`.
            high: optional DataFrame, same shape as `prices`. When provided, buys fill at the high
                (conservative). NaN at a buy date raises an assertion — buys can't be modeled.

        Asserts two end-of-run invariants on the final balance (transactions tally and P&L-derived
        balance). Returns self.
        """
        if dates is None:
            dates = prices.index
        positions_df = pd.DataFrame(columns=POSITIONS_COLUMNS)
        errors_df = pd.DataFrame(columns=POSITIONS_COLUMNS)
        trades_df = pd.DataFrame(columns=TRADES_COLUMNS)
        unrealized_pl = pd.Series(index=dates, name=f'Unrealized P&L')
        pos_counter = 0
        balance = self.start_balance
        old_position = set()
        for i, date in enumerate(dates):
            new_position = set(strategy.predict(date))
            buys = new_position.difference(old_position)
            sells = old_position.difference(new_position)
            old_position = new_position

            for sell in sells:
                mask = positions_df['ticker'] == sell
                assert mask.sum() == 1, f'The number of current positions matching {sell} is {mask.sum()} but should be ==1'
                pos_id = positions_df[mask].index[0]

                price = prices.loc[date, sell] if low is None else low.loc[date, sell]
                position = positions_df.loc[pos_id, 'position']
                if pd.isna(price):
                    price = positions_df.loc[pos_id, 'price_buy']
                    errors_df.loc[pos_id, POSITIONS_COLUMNS] = positions_df.loc[pos_id, POSITIONS_COLUMNS]
                    errors_df.loc[pos_id, 'date_sell'] = date
                fee = get_ib_fees(position=position, price=price)

                balance += position * price - fee

                trades_df.loc[pos_id, POSITIONS_COLUMNS] = positions_df.loc[pos_id, POSITIONS_COLUMNS]
                positions_df = positions_df.drop(index=pos_id)

                trades_df.loc[pos_id, TRADES_EXTRA_COLUMNS] = [date, price, fee]
                trades_df.loc[pos_id, 'fees'] = trades_df.loc[pos_id, 'fees_buy'] + trades_df.loc[pos_id, 'fees_sell']
                trades_df.loc[pos_id, 'P&L'] = get_p_and_l(row=trades_df.loc[pos_id])

            notional = get_notional(balance=balance, n_buys=len(buys), price_min=strategy.price_min)
            for buy in buys:
                price = prices.loc[date, buy] if high is None else high.loc[date, buy]
                assert not pd.isna(price), f'For buy orders, Backtesting does not handle price=np.nan. Here, for {buy}@{date}, price={price}.'
                position = notional // price
                fee = get_ib_fees(position=position, price=price)

                balance += -position * price - fee

                positions_df.loc[pos_counter, POSITIONS_COLUMNS] = [date, buy, position, price, fee]
                pos_counter += 1

            unrealized_pl[date] = get_total_unrealized_p_and_l(trades_df=trades_df, positions_df=positions_df,
                                                               prices_series=prices.loc[date])

        balance_theoric = get_balance(start_balance=self.start_balance, trades_df=trades_df, positions_df=positions_df)
        assert np.isclose(balance, balance_theoric, atol=0.01), \
            f'Backtesting end balance does not check out against transactions: ' \
            f'end balance = {balance:.2f}$ VS theoric balance = {balance_theoric:.2f}$.'

        date = dates[-1]
        prices_current = prices.loc[date, positions_df['ticker']].values
        value_current = (positions_df['position'] * prices_current).sum()
        balance_pl = self.start_balance + unrealized_pl[date] - value_current
        assert np.isclose(balance, balance_pl, atol=0.01), \
            f'Backtesting end balance does not check out against balance deduced from (last P&L - investments): ' \
            f'end balance = {balance:.2f}$ VS P&L derived balance = {balance_pl:.2f}$.'

        self.trades_df = trades_df
        self.positions_df = positions_df
        self.errors_df = errors_df
        self.unrealized_pl = unrealized_pl

        return self

    def get_trades(self):
        """Return (trades_df, positions_df, errors_df). Only valid after `fit`."""
        return self.trades_df, self.positions_df, self.errors_df

    def get_unrealized_pl(self):
        """Return the per-date unrealized P&L Series. Only valid after `fit`."""
        return self.unrealized_pl