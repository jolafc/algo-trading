import os
from timeit import default_timer as timer

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
        print(f'Optimal yield is: {hpo_results.fun} at parameters {hpo_results.x}')

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
        chkpt_file = os.path.join(RESULTS_DIR,
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

        print(f'\nCV WINDOW {i+1} / {n_windows} ({runtime:.1f}s runtime):')
        formatted_pars = " ".join([f"{k}={v:.1f}" for k, v in optimized_parameters.items()])
        print(f'OPTIMIZED wrt {output_metric}; opt. parameters: {formatted_pars}')
        print(f'TEST results for {train_start.date()} to {train_end.date()} '
              f'are {100 * train_metrics[YIELD]:.1f}% / {train_metrics[SHARPE]:.3f} sharpe / {train_metrics[SORTINO]:.3f} sortino '
              f'VS benchmark {100 * train_metrics[BYIELD]:.1f}% / {train_metrics[BSHARPE]:.3f} sharpe / {train_metrics[BSORTINO]:.3f} sortino ')
        print(f'VAL  results for {val_start.date()} to {val_end.date()} '
              f'are {100 * val_metrics[YIELD]:.1f}% / {val_metrics[SHARPE]:.3f} sharpe / {val_metrics[SORTINO]:.3f} sortino '
              f'VS benchmark {100 * val_metrics[BYIELD]:.1f}% / {val_metrics[BSHARPE]:.3f} sharpe / {val_metrics[BSORTINO]:.3f} sortino ')

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
    for i in range(n_iters):
        print(f'\n\nITERATION {i+1} / {n_iters}')
        results = cross_validate_strategy(
            StrategyRunner=WeeklyRotationRunner,
            max_lookback=200,
            n_calls=n_calls,
            n_random_starts=n_rand if i == 0 else 0,
            output_metric=output_metric,
            restart_from_chkpt=True,
            verbose=False,
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
