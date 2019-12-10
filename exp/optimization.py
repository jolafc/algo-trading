import os
import pickle
import sys
from functools import partial
from timeit import default_timer as timer
import logging
from datetime import datetime

import pandas as pd
import skopt
from joblib import Parallel, delayed

from exp import CHKPT_DEFAULT_FILE, SEED, RESULTS_DIR, YIELD, SHARPE, SORTINO, BYIELD, BSHARPE, BSORTINO, TRAIN_PREFIX, \
    VAL_PREFIX, RESULTS_DEFAULT_FILENAME, LOG_DEFAULT_FILENAME
from exp.reporting import plot_strategy_cv_convergence
from exp.strategy.weekly_rotation import WeeklyRotationRunner


def train_strategy(
        StrategyRunner=WeeklyRotationRunner,
        train_start=pd.to_datetime('2019-01-31'),
        train_end=pd.to_datetime('2019-10-31'),
        val_start=pd.to_datetime('2018-01-01'),
        val_end=pd.to_datetime('2018-12-31'),
        max_lookback=200,
        n_calls=100,
        n_rand=5,
        output_metric=YIELD,
        resume=False,
        chkpt_file=CHKPT_DEFAULT_FILE,
        verbose=True,
):
    run_dir = os.path.dirname(chkpt_file)
    runner = StrategyRunner(start_date_requested=train_start,
                            end_date_requested=train_end,
                            max_lookback=max_lookback,
                            verbose=False,
                            res_dir=run_dir,
                            output_metric=output_metric)

    dimensions_list = list(runner.dimensions.values())

    checkpoint_saver = skopt.callbacks.CheckpointSaver(chkpt_file, compress=3)
    if os.path.exists(chkpt_file) and resume:
        res = skopt.load(chkpt_file)
        x0 = res.x_iters
        y0 = res.func_vals
        n_random_starts = 0
    else:
        x0 = None
        y0 = None
        n_random_starts = n_rand

    # hpo_results = skopt.forest_minimize(
    # hpo_results = skopt.gbrt_minimize(
    hpo_results = skopt.gp_minimize(
        func=runner.skopt_func, dimensions=dimensions_list,
        n_calls=n_calls, n_random_starts=n_random_starts,
        # base_estimator='ET', acq_func='LCB,
        # base_estimator=None, acq_func='LCB', acq_optimizer='auto',
        base_estimator=None, acq_func='gp_hedge', acq_optimizer='auto',
        x0=x0, y0=y0, random_state=SEED, verbose=verbose,
        callback=[checkpoint_saver],
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


def train_strategy_driver(i, n_windows, kwargs,
                          run_dir=RESULTS_DIR,
                          train_window_size=pd.to_timedelta('52w'),
                          val_window_size=pd.to_timedelta('52w'),
                          start_date=pd.to_datetime('2001-01-01')):
    output_metric = kwargs['output_metric']
    train_start = start_date + i * val_window_size
    train_end = train_start + train_window_size
    val_start = train_end
    val_end = val_start + val_window_size
    chkpt_file = os.path.join(run_dir, f'checkpoint_{train_start.date()}_{train_end.date()}_{output_metric}.pkl')

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

    msg_list = []
    msg_list.append('')
    msg_list.append(f'CV WINDOW {i + 1} / {n_windows} ({runtime:.1f}s runtime):')
    formatted_pars = " ".join([f"{k}={v:.1f}" for k, v in optimized_parameters.items()])
    msg_list.append(f'OPTIMIZED wrt {output_metric}; opt. parameters: {formatted_pars}')
    msg_list.append(f'TEST results for {train_start.date()} to {train_end.date()} '
                    f'are {100 * train_metrics[YIELD]:.1f}% / {train_metrics[SHARPE]:.3f} sharpe'
                    f' / {train_metrics[SORTINO]:.3f} sortino '
                    f'VS benchmark {100 * train_metrics[BYIELD]:.1f}% / {train_metrics[BSHARPE]:.3f} sharpe'
                    f' / {train_metrics[BSORTINO]:.3f} sortino ')
    msg_list.append(f'VAL  results for {val_start.date()} to {val_end.date()} '
                    f'are {100 * val_metrics[YIELD]:.1f}% / {val_metrics[SHARPE]:.3f} sharpe'
                    f' / {val_metrics[SORTINO]:.3f} sortino '
                    f'VS benchmark {100 * val_metrics[BYIELD]:.1f}% / {val_metrics[BSHARPE]:.3f} sharpe'
                    f' / {val_metrics[BSORTINO]:.3f} sortino ')

    metrics = dict(
        train_start=train_start.date(),
        train_end=train_end.date(),
        val_start=val_start.date(),
        val_end=val_end.date(),
    )
    metrics.update({TRAIN_PREFIX + k: v for k, v in train_metrics.items()})
    metrics.update({VAL_PREFIX + k: v for k, v in val_metrics.items()})

    return metrics, msg_list


def cross_validate_strategy(
        StrategyRunner=WeeklyRotationRunner,
        max_lookback=200,
        n_calls=10,
        n_rand=5,
        output_metric=YIELD,
        resume=False,
        verbose=False,
        n_jobs=-1,
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
        n_rand=n_rand,
        output_metric=output_metric,
        resume=resume,
        verbose=verbose)
    n_windows = (end_date - start_date - train_window_size) // val_window_size
    f = partial(train_strategy_driver,
                n_windows=n_windows,
                kwargs=kwargs,
                run_dir=run_dir,
                train_window_size=train_window_size,
                val_window_size=val_window_size,
                start_date=start_date)

    # outputs = [f(i) for i in range(n_windows)]
    outputs = Parallel(n_jobs=n_jobs)(delayed(f)(i) for i in range(n_windows))

    results = [result for result, msg_list in outputs]
    msg_lists = [msg_list for result, msg_list in outputs]

    for msg_list in msg_lists:
        for msg in msg_list:
            logging.info(msg)
    results = pd.DataFrame(results)

    return results


def cv_opt_driver(train_window_size=pd.to_timedelta('52w'),
                  val_window_size=pd.to_timedelta('52w'),
                  start_date=pd.to_datetime('2001-01-01'),
                  end_date=pd.to_datetime('2004-01-01'),
                  StrategyRunner=WeeklyRotationRunner,
                  output_metric=YIELD,  # YIELD, SHARPE, SORTINO
                  max_lookback=200,
                  n_iters=2,
                  n_calls=1,
                  n_rand=1,
                  resume=False,
                  verbose=False,
                  n_jobs=-1):
    assert (not verbose) or (n_jobs == 1), \
        f'Verbose logging only available for sequential runs; please either set verbose=False or n_jobs=1'

    if resume:
        run_dirs = os.listdir(RESULTS_DIR)
        run_dirs = [run_dir for run_dir in run_dirs if f'run_{output_metric}' in run_dir]
        run_dirs = sorted(run_dirs)
        run_dir = os.path.join(RESULTS_DIR, run_dirs[-1])
        start_msg = f'RUN RESUMING: {run_dir}'

        results_filename = os.path.join(run_dir, RESULTS_DEFAULT_FILENAME)
        with open(results_filename, mode='rb') as results_file:
            results_iter = pickle.load(results_file)

    else:
        now = datetime.now()
        run_dir = os.path.join(RESULTS_DIR, f'run_{output_metric}_{now}')
        os.makedirs(run_dir, exist_ok=False)
        start_msg = f'RUN BEGIN: {now}'

        results_filename = os.path.join(run_dir, RESULTS_DEFAULT_FILENAME)
        results_iter = []
    results_len = len(results_iter)

    log_file = os.path.join(run_dir, LOG_DEFAULT_FILENAME)
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(filename=log_file)]
    logging.basicConfig(handlers=handlers, level=logging.INFO)
    logging.info(f'')
    logging.info(f'-' * 80)
    logging.info(start_msg)
    logging.info(f'-' * 80)
    logging.info(f'')

    dimensions = StrategyRunner(max_lookback=max_lookback).dimensions
    logging.info(f'OPTIMIZATION SPACE:')
    for name, dimension in dimensions.items():
        logging.info(f'{name}: {dimension}')
    runtime_total = timer()

    for i in range(results_len, results_len + n_iters):
        logging.info('')
        logging.info('')
        logging.info(f'ITERATION {i + 1} / {results_len + n_iters}')
        runtime_iter = timer()

        results = cross_validate_strategy(
            StrategyRunner=StrategyRunner,
            max_lookback=max_lookback,
            n_calls=n_calls,
            n_rand=n_rand if i == 0 else 0,
            output_metric=output_metric,
            resume=resume if i == 0 else True,
            verbose=verbose,
            n_jobs=n_jobs,
            run_dir=run_dir,
            train_window_size=train_window_size,
            val_window_size=val_window_size,
            start_date=start_date,
            end_date=end_date,
        )
        results_iter.append(results)
        runtime_iter = timer() - runtime_iter
        logging.info(f'Iteration {i + 1}/{results_len + n_iters}: {runtime_iter:.1f}s runtime')

        with open(results_filename, mode='wb') as results_file:
            pickle.dump(results_iter, results_file)

        plot_strategy_cv_convergence(results_iter, run_dir=run_dir)

    runtime_total = timer() - runtime_total
    logging.info('')
    logging.info(f'TOTAL RUNTIME: {runtime_total:.1f}s for run: {run_dir}')

    return results_iter


if __name__ == '__main__':
    results_iter = cv_opt_driver(train_window_size=pd.to_timedelta('260w'),
                                 val_window_size=pd.to_timedelta('26w'),
                                 start_date=pd.to_datetime('2001-01-01'),
                                 end_date=pd.to_datetime('2008-01-01'),
                                 StrategyRunner=WeeklyRotationRunner,
                                 output_metric=YIELD,  # YIELD, SHARPE, SORTINO
                                 max_lookback=200,
                                 n_iters=10,
                                 n_calls=10,
                                 n_rand=10,
                                 resume=False,
                                 verbose=False,
                                 n_jobs=-1)
