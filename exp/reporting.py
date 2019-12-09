import logging
import os

import pandas as pd
from matplotlib import pyplot as plt

from exp import PL, YIELD, SHARPE, SORTINO, BPL, BYIELD, BSHARPE, BSORTINO, PLOT_DEFAULT_FILE, RESULTS_DIR, \
    CONVERGENCE_PLOT_DEFAULT_BASENAME, PLT_FILE_FORMAT, TRAIN_PREFIX, BENCH_PREFIX, VAL_PREFIX
from exp.backtesting import Backtesting
from exp.default_parameters import BENCKMARK_TICKER
from exp.metrics import get_ib_fees, get_annualized_yield, get_sharpe_ratio, get_sortino_ratio


def make_backtesting_report(backtesting, prices, dates=None, verbose=True, plotting=True, plotfile=PLOT_DEFAULT_FILE):
    assert isinstance(backtesting, Backtesting), \
        f'This function requires an instance of the Backtesting class to work.'
    trades_df, positions_df, errors_df = backtesting.get_trades()
    unrealized_pl = backtesting.get_unrealized_pl()
    start_balance = backtesting.start_balance

    if verbose > 1:
        logging.info('')
        logging.debug(f'Trades DataFrame:\n{trades_df}')
        logging.info('')
        logging.debug(f'Positions DataFrame:\n{positions_df}')
        logging.info('')
        logging.debug(f'Errors DataFrame:\n{errors_df}')
        logging.info('')
        logging.debug(f'Unrealized P&L Series:\n{unrealized_pl}')

    benckmark_prices = prices.loc[dates, BENCKMARK_TICKER]
    benckmark_position = start_balance / benckmark_prices.iloc[0]
    benckmark_fee = get_ib_fees(position=benckmark_position, price=benckmark_prices.iloc[0])
    benckmark_pl = benckmark_position * benckmark_prices - start_balance - benckmark_fee

    realized_pl = trades_df['P&L'].sum()
    annualized_yield = get_annualized_yield(unrealized_pl=unrealized_pl, start_balance=start_balance)
    sharpe_ratio = get_sharpe_ratio(unrealized_pl=unrealized_pl, start_balance=start_balance)
    sortino_ratio = get_sortino_ratio(unrealized_pl=unrealized_pl, start_balance=start_balance)
    benchmark_realized_pl = benckmark_pl.sum()
    benchmark_annualized_yield = get_annualized_yield(unrealized_pl=benckmark_pl, start_balance=start_balance)
    benchmark_sharpe_ratio = get_sharpe_ratio(unrealized_pl=benckmark_pl, start_balance=start_balance)
    benchmark_sortino_ratio = get_sortino_ratio(unrealized_pl=benckmark_pl, start_balance=start_balance)

    if verbose:
        logging.info('')
        logging.info(f'Realized P&L: {realized_pl:.2f}$')
        logging.info(f'Annualized yield = {100 * annualized_yield:.1f}%; benckmark = {100 * benchmark_annualized_yield:.1f}%')
        logging.info(f'Sharpe ratio: {sharpe_ratio:.2f}; benckmark = {benchmark_sharpe_ratio:.2f}')
        logging.info(f'Sortino ratio: {sortino_ratio:.2f}; benckmark = {benchmark_sortino_ratio:.2f}')

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
        plt.savefig(plotfile)
        if verbose:
            logging.info('')
            logging.info(f'Plotted unrealized P&L to file: {plotfile}')

    results = {PL: realized_pl,
               YIELD: annualized_yield,
               SHARPE: sharpe_ratio,
               SORTINO: sortino_ratio,
               BPL: benchmark_realized_pl,
               BYIELD: benchmark_annualized_yield,
               BSHARPE: benchmark_sharpe_ratio,
               BSORTINO: benchmark_sortino_ratio,
               }

    return results


def plot_strategy_cv_convergence(results_iter, run_dir=RESULTS_DIR, basename=CONVERGENCE_PLOT_DEFAULT_BASENAME):
    cv_index = results_iter[0].index
    results_cvs = []
    logging.info(f'')
    for icv in cv_index:
        results_fold = [df[icv:icv + 1] for df in results_iter]
        results_fold = pd.concat(results_fold, axis=0, ignore_index=True)
        results_cvs.append(results_fold)

        train_start = results_fold.loc[0, 'train_start']
        train_end = results_fold.loc[0, 'train_end']
        val_start = results_fold.loc[0, 'val_start']
        val_end = results_fold.loc[0, 'val_end']

        plotfile = os.path.join(run_dir, f'{basename}_{icv:02d}.{PLT_FILE_FORMAT}')
        plt.figure()
        plt.title(f'Convergence for CV window {icv+1}/{cv_index[-1]+1}:\n{train_start} to {train_end} and {val_start} to {val_end}')
        plt.plot(results_fold[TRAIN_PREFIX + YIELD], '-r', label=f'Train yield')
        plt.plot(results_fold[TRAIN_PREFIX + BENCH_PREFIX + YIELD], '-k', label=f'Train bench yield')
        plt.plot(results_fold[VAL_PREFIX + YIELD], '--r', label=f'Val yield')
        plt.plot(results_fold[VAL_PREFIX + BENCH_PREFIX + YIELD], '--k', label=f'Val bench yield')
        x = results_fold.index
        y = [0.]*len(x)
        plt.plot(x, y, ':k')
        plt.legend()
        plt.savefig(plotfile)

    plotfiles = os.path.join(run_dir, f'{basename}_{cv_index[0]:02d}.{PLT_FILE_FORMAT}')
    plotfiles += f' -- {basename}_{cv_index[-1]:02d}.{PLT_FILE_FORMAT}'
    logging.info(f'Plotted convergence for fold {cv_index[0]}-{cv_index[-1]} in files: {plotfiles}')