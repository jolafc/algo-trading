"""End-to-end example: download SP500 history, run a single backtest, then a 2-fold CV tuning run.

Prerequisites:
    - conda env `trading` is active (see environment.yaml)
    - run from the repo root: `python examples/run_end_to_end.py`

No API key required (yfinance). This script is intentionally small and prints what it's doing
at each step.
"""
import logging
import os
import sys

import pandas as pd

from exp import SP500_PKL, YIELD
from exp.data_getter import get_sp500_pkl
from exp.optimization import cv_opt_driver
from exp.strategy.weekly_rotation import WeeklyRotationRunner

logging.basicConfig(level=logging.INFO, stream=sys.stdout)


def step_1_download_data():
    """Pull historical daily prices for every S&P 500 constituent into data/sp500.pkl.

    Skips the download if the pickle already exists. Throttled by `YF_SLEEP` seconds between
    yfinance calls (default 0.2s); a fresh full SP500 run takes ~30–60 minutes.
    """
    if os.path.exists(SP500_PKL):
        print(f'Found existing price cache at {SP500_PKL} — skipping download.')
        return
    print(f'Downloading S&P 500 history from yfinance to {SP500_PKL}...')
    get_sp500_pkl(update=True)


def step_2_single_backtest():
    """Run one backtest with the published rule-set defaults over 2019."""
    print('\nRunning single backtest: weekly rotation, 2019-01-31 to 2019-10-31, default params.')
    runner = WeeklyRotationRunner(
        start_date_requested=pd.to_datetime('2019-01-31'),
        end_date_requested=pd.to_datetime('2019-10-31'),
        max_lookback=200,
        verbose=True,
        output_metric=None,
    )
    results = runner(
        lookback=200,
        sma_tol=0.02,
        volume_lookback=20,
        volume_threshold=1e6,
        price_min=1.0,
        rsi_lookback=3,
        rsi_threshold=50.0,
        day_of_trade=4,
        n_positions=10,
    )
    print(f'Backtest metrics: {results}')


def step_3_cv_tuning():
    """Run a tiny walk-forward CV tuning sweep (2 folds, 2 iters, 2 calls each).

    This is sized to finish in a few minutes. Real tuning runs use n_iters~20 and
    n_calls~10 per fold — see doc/hpo_notes.md.
    """
    print('\nRunning 2-fold CV tuning (toy size — see doc/hpo_notes.md for real sizing).')
    results_iter = cv_opt_driver(
        train_window_size=pd.to_timedelta('26w'),
        val_window_size=pd.to_timedelta('26w'),
        val_start_date=pd.to_datetime('2018-01-01'),
        n_folds=2,
        StrategyRunner=WeeklyRotationRunner,
        optimizer='GBRT',
        output_metric=YIELD,
        max_lookback=200,
        n_iters=2,
        n_calls=2,
        n_rand=2,
        resume=False,
        verbose=False,
        n_jobs=-1,
    )
    print(f'\nFinal CV results (last iteration):\n{results_iter[-1]}')


if __name__ == '__main__':
    step_1_download_data()
    step_2_single_backtest()
    step_3_cv_tuning()
