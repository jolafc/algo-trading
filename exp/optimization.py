import os

import pandas as pd
import skopt

from exp import CHKPT_DEFAULT_FILE, SEED
from exp.strategy.weekly_rotation import WeeklyRotationRunner


def optimize_strategy(
        StrategyRunner=WeeklyRotationRunner,
        train_start=pd.to_datetime('2019-01-31'),
        train_end=pd.to_datetime('2019-10-31'),
        val_start=pd.to_datetime('2018-01-01'),
        val_end=pd.to_datetime('2018-12-31'),
        max_lookback=200,
        n_calls=100,
        n_random_starts=5,
        output_metric='annualized_yield',
        restart_from_chkpt=False,
        chkpt_file=CHKPT_DEFAULT_FILE,
):
    runner = StrategyRunner(start_date_requested=train_start,
                            end_date_requested=train_end,
                            max_lookback=max_lookback,
                            verbose=False,
                            output_metric=output_metric)

    dimensions_list = list(runner.dimensions.values())

    checkpoint_saver = skopt.callbacks.CheckpointSaver(chkpt_file, compress=3)
    if os.path.exists(chkpt_file) and restart_from_chkpt:
        res = skopt.load(chkpt_file)
        x0 = res.x_iters
        y0 = res.func_vals
        n_random_starts = 0
    else:
        x0 = None
        y0 = None
        n_random_starts = n_random_starts

    hpo_results = skopt.forest_minimize(func=runner.skopt_func, dimensions=dimensions_list,
                                        n_calls=n_calls, n_random_starts=n_random_starts,
                                        base_estimator='ET', acq_func='EI',
                                        x0=x0, y0=y0, random_state=SEED, verbose=True, callback=[checkpoint_saver],
                                        n_points=10000, xi=0.01, kappa=1.96, n_jobs=1)

    print(f'Optimal yield is: {hpo_results.fun} at parameters {hpo_results.x}')

    runner.verbose = True
    runner.output_metric = None
    kwargs = {k: v for k, v in zip(runner.dimensions.keys(), hpo_results.x)}
    train_metrics = runner(**kwargs)

    runner.start_date_requested = val_start
    runner.end_date_requested = val_end
    val_metrics = runner(**kwargs)

    return train_metrics, val_metrics


if __name__ == '__main__':
    optimize_strategy(
        StrategyRunner=WeeklyRotationRunner,
        train_start=pd.to_datetime('2019-01-31'),
        train_end=pd.to_datetime('2019-10-31'),
        val_start=pd.to_datetime('2018-01-01'),
        val_end=pd.to_datetime('2018-12-31'),
        max_lookback=200,
        n_calls=0,
        n_random_starts=5,
        output_metric='annualized_yield',
        restart_from_chkpt=True,
        chkpt_file=CHKPT_DEFAULT_FILE,
    )
