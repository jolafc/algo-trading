import os
import sys
from timeit import default_timer as timer
import logging
from datetime import datetime

import pandas as pd
import skopt

from exp import CHKPT_DEFAULT_FILE, SEED, RESULTS_DIR, YIELD, SHARPE, SORTINO, BYIELD, BSHARPE, BSORTINO, TRAIN_PREFIX, \
    VAL_PREFIX
from exp.strategy.weekly_rotation import WeeklyRotationRunner


def train_strategy(
        StrategyRunner=WeeklyRotationRunner,
        train_start=pd.to_datetime('2019-01-31'),
        train_end=pd.to_datetime('2019-10-31'),
        val_start=pd.to_datetime('2018-01-01'),
        val_end=pd.to_datetime('2018-12-31'),
        max_lookback=200,
        n_calls=100,
        n_random_starts=5,
        output_metric=YIELD,
        restart_from_chkpt=False,
        chkpt_file=CHKPT_DEFAULT_FILE,
        verbose=True,
):
    runner = StrategyRunner(start_date_requested=train_start,
                            end_date_requested=train_end,
                            max_lookback=max_lookback,
                            verbose=False,
                            res_dir=os.path.dirname(chkpt_file),
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
                                        x0=x0, y0=y0, random_state=SEED, verbose=verbose, callback=[checkpoint_saver],
                                        n_points=10000, xi=0.01, kappa=1.96, n_jobs=1)

    if verbose:
        logging.info(f'Optimal yield is: {hpo_results.fun} at parameters {hpo_results.x}')

    runner.verbose = verbose
    runner.output_metric = None
    optimized_parameters = {k: v for k, v in zip(runner.dimensions.keys(), hpo_results.x)}
    train_metrics = runner(**optimized_parameters)

    runner.start_date_requested = val_start
    runner.end_date_requested = val_end
    val_metrics = runner(**optimized_parameters)

    return train_metrics, val_metrics, optimized_parameters


def cross_validate_strategy(
        StrategyRunner=WeeklyRotationRunner,
        max_lookback=200,
        n_calls=10,
        n_random_starts=5,
        output_metric=YIELD,
        restart_from_chkpt=False,
        verbose=False,
        run_dir=RESULTS_DIR,
        train_window_size=pd.to_timedelta('52w'),
        val_window_size=pd.to_timedelta('52w'),
        start_date=pd.to_datetime('2001-01-01'),
        end_date=pd.to_datetime('2004-01-01'),
):
    kwargs = dict(
        StrategyRunner=StrategyRunner,
        max_lookback=max_lookback,
        n_calls=n_calls,
        n_random_starts=n_random_starts,
        output_metric=output_metric,
        restart_from_chkpt=restart_from_chkpt,
        verbose=verbose,
    )

    n_windows = (end_date - start_date - train_window_size) // val_window_size

    results = []
    for i in range(n_windows):
        train_start = start_date + i * val_window_size
        train_end = train_start + train_window_size
        val_start = train_end
        val_end = val_start + val_window_size
        chkpt_file = os.path.join(run_dir,
                                  f'checkpoint_{train_start.date()}_{train_end.date()}_{output_metric}.pkl')

        runtime = timer()
        train_metrics, val_metrics, optimized_parameters = train_strategy(
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            chkpt_file=chkpt_file,
            **kwargs
        )
        runtime = timer() - runtime

        logging.info('')
        logging.info(f'CV WINDOW {i + 1} / {n_windows} ({runtime:.1f}s runtime):')
        formatted_pars = " ".join([f"{k}={v:.1f}" for k, v in optimized_parameters.items()])
        logging.info(f'OPTIMIZED wrt {output_metric}; opt. parameters: {formatted_pars}')
        logging.info(f'TEST results for {train_start.date()} to {train_end.date()} '
                     f'are {100 * train_metrics[YIELD]:.1f}% / {train_metrics[SHARPE]:.3f} sharpe'
                     f' / {train_metrics[SORTINO]:.3f} sortino '
                     f'VS benchmark {100 * train_metrics[BYIELD]:.1f}% / {train_metrics[BSHARPE]:.3f} sharpe'
                     f' / {train_metrics[BSORTINO]:.3f} sortino ')
        logging.info(f'VAL  results for {val_start.date()} to {val_end.date()} '
                     f'are {100 * val_metrics[YIELD]:.1f}% / {val_metrics[SHARPE]:.3f} sharpe'
                     f' / {val_metrics[SORTINO]:.3f} sortino '
                     f'VS benchmark {100 * val_metrics[BYIELD]:.1f}% / {val_metrics[BSHARPE]:.3f} sharpe'
                     f' / {val_metrics[BSORTINO]:.3f} sortino ')

        metrics = {TRAIN_PREFIX + k: v for k, v in train_metrics.items()}
        metrics.update({VAL_PREFIX + k: v for k, v in val_metrics.items()})

        results.append(metrics)

    results = pd.DataFrame(results)

    return results


if __name__ == '__main__':
    output_metric = YIELD  # YIELD, SHARPE, SORTINO
    n_iters = 10
    n_calls = 10
    n_rand = 5
    from_chkpt = True

    run_dir = os.path.join(RESULTS_DIR, f'run_{output_metric}_{datetime.now()}')
    os.makedirs(run_dir, exist_ok=False)
    log_file = os.path.join(run_dir, f'cv_opt.log')
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(filename=log_file)]
    logging.basicConfig(handlers=handlers, level=logging.INFO)
    for i in range(n_iters):
        logging.info('')
        logging.info('')
        logging.info(f'ITERATION {i + 1} / {n_iters}')
        results = cross_validate_strategy(
            StrategyRunner=WeeklyRotationRunner,
            max_lookback=200,
            n_calls=n_calls,
            n_random_starts=n_rand if i == 0 else 0,
            output_metric=output_metric,
            restart_from_chkpt=from_chkpt if i == 0 else True,
            verbose=False,
            run_dir=run_dir,
            train_window_size=pd.to_timedelta('52w'),
            val_window_size=pd.to_timedelta('52w'),
            start_date=pd.to_datetime('2001-01-01'),
            end_date=pd.to_datetime('2004-01-01'),
        )

    # train_strategy(
    #     StrategyRunner=WeeklyRotationRunner,
    #     train_start=pd.to_datetime('2019-01-31'),
    #     train_end=pd.to_datetime('2019-10-31'),
    #     val_start=pd.to_datetime('2018-01-01'),
    #     val_end=pd.to_datetime('2018-12-31'),
    #     max_lookback=200,
    #     n_calls=0,
    #     n_random_starts=5,
    #     output_metric='annualized_yield',
    #     restart_from_chkpt=True,
    #     chkpt_file=CHKPT_DEFAULT_FILE,
    #     verbose=True,
    # )
