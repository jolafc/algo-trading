import os

import pandas as pd
from matplotlib import pyplot as plt

from exp import PLT_FILE_FORMAT, RESULTS_DIR
from exp.backtesting import Backtesting
from exp.default_parameters import BENCKMARK_TICKER
from exp.metrics import get_ib_fees, get_annualized_yield, get_sharpe_ratio, get_sortino_ratio


def make_backtesting_report(backtesting, prices, dates=None, verbose=True, plotting=True):
    assert isinstance(backtesting, Backtesting), \
        f'This function requires an instance of the Backtesting class to work.'
    trades_df, positions_df, errors_df = backtesting.get_trades()
    unrealized_pl = backtesting.get_unrealized_pl()
    start_balance = backtesting.start_balance

    if verbose > 1:
        print(f'\nTrades DataFrame:\n{trades_df}')
        print(f'\nPositions DataFrame:\n{positions_df}')
        print(f'\nErrors DataFrame:\n{errors_df}')
        print(f'\nUnrealized P&L Series:\n{unrealized_pl}')

    benckmark_prices = prices.loc[dates, BENCKMARK_TICKER]
    benckmark_position = start_balance / benckmark_prices.iloc[0]
    benckmark_fee = get_ib_fees(position=benckmark_position, price=benckmark_prices.iloc[0])
    benckmark_pl = benckmark_position * benckmark_prices - start_balance - benckmark_fee

    realized_pl = trades_df['P&L'].sum()
    annualized_yield = get_annualized_yield(unrealized_pl=unrealized_pl, start_balance=start_balance)
    sharpe_ratio = get_sharpe_ratio(unrealized_pl=unrealized_pl, start_balance=start_balance)
    sortino_ratio = get_sortino_ratio(unrealized_pl=unrealized_pl, start_balance=start_balance)
    benchmark_annualized_yield = get_annualized_yield(unrealized_pl=benckmark_pl, start_balance=start_balance)
    benchmark_sharpe_ratio = get_sharpe_ratio(unrealized_pl=benckmark_pl, start_balance=start_balance)
    benchmark_sortino_ratio = get_sortino_ratio(unrealized_pl=benckmark_pl, start_balance=start_balance)

    if verbose:
        print(f'\nRealized P&L: {realized_pl:.2f}$')
        print(f'Annualized yield = {100 * annualized_yield:.1f}%; benckmark = {100 * benchmark_annualized_yield:.1f}%')
        print(f'Sharpe ratio: {sharpe_ratio:.2f}; benckmark = {benchmark_sharpe_ratio:.2f}')
        print(f'Sortino ratio: {sortino_ratio:.2f}; benckmark = {benchmark_sortino_ratio:.2f}')

    if plotting:
        pd.plotting.register_matplotlib_converters()
        plt.figure(figsize=(10, 5))
        label = f'{unrealized_pl.name} - yield={100 * annualized_yield:.1f}%; sharpe={sharpe_ratio:.2f}; sortino={sortino_ratio:.2f}'
        plt.plot(unrealized_pl, '--r', label=label)
        benchmark_label = f'{benckmark_pl.name} - yield={100 * benchmark_annualized_yield:.1f}%; sharpe={benchmark_sharpe_ratio:.2f}; sortino={benchmark_sortino_ratio:.2f}'
        plt.plot(benckmark_pl, '--b', label=benchmark_label)
        xlim = plt.xlim()
        plt.xlim(xlim)
        plt.plot(xlim, [0.] * 2, '--k')
        plt.title(f'Strategy performance')
        plt.legend()
        plotfile = os.path.join(RESULTS_DIR, f'unrealized_pl.{PLT_FILE_FORMAT}')
        plt.savefig(plotfile)
        if verbose:
            print(f'\nPlotted unrealized P&L to file: {plotfile}')

    results = {'realized_pl': realized_pl,
               'annualized_yield': annualized_yield,
               'sharpe_ratio': sharpe_ratio,
               'sortino_ratio': sortino_ratio}

    return results
