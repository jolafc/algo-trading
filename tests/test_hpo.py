import os

import numpy as np
import pandas as pd

from exp import RESULTS_DIR, YIELD, VAL_PREFIX
from exp.optimization import train_strategy, cv_opt_driver
from exp.strategy.weekly_rotation import WeeklyRotationRunner

CHKPT_TEST_FILE = os.path.join(RESULTS_DIR, 'checkpoint_test.pkl')
REFERENCE_TRAIN_YIELD = 0.46299178593687473
REFERENCE_VAL_YIELD = -0.011870411339345141
REFERENCE_VYIELD = 0.22689941969756694


def test_cv_opt_driver():
    results_iter = cv_opt_driver(train_window_size=pd.to_timedelta('26w'),
                                 val_window_size=pd.to_timedelta('26w'),
                                 val_start_date=pd.to_datetime('2011-01-01'),
                                 n_folds=2,
                                 StrategyRunner=WeeklyRotationRunner,
                                 optimizer='GBRT',
                                 output_metric=YIELD,
                                 max_lookback=200,
                                 n_iters=2,
                                 n_calls=1,
                                 n_rand=1,
                                 resume=False,
                                 verbose=False,
                                 n_jobs=-1)

    vyield = results_iter[-1].loc[0, VAL_PREFIX+YIELD]  # Last iteration, first fold, validation yield.

    assert np.isclose(vyield, REFERENCE_VYIELD, atol=1e-8), \
        f'Train yield is {vyield} but should be {REFERENCE_VYIELD}'


def test_strategy_tainer():
    train_metrics, val_metrics, optimized_parameters = train_strategy(
        StrategyRunner=WeeklyRotationRunner,
        train_start=pd.to_datetime('2019-01-31'),
        train_end=pd.to_datetime('2019-10-31'),
        val_start=pd.to_datetime('2018-01-01'),
        val_end=pd.to_datetime('2018-12-31'),
        max_lookback=200,
        n_calls=5,
        n_rand=1,
        output_metric=YIELD,
        resume=False,
        chkpt_file=CHKPT_TEST_FILE,
    )

    train_yield = train_metrics[YIELD]
    val_yield = val_metrics[YIELD]

    assert np.isclose(train_yield, REFERENCE_TRAIN_YIELD, atol=1e-8), \
        f'Train yield is {train_yield} but should be {REFERENCE_TRAIN_YIELD}'
    assert np.isclose(val_yield, REFERENCE_VAL_YIELD, atol=1e-8), \
        f'Train yield is {val_yield} but should be {REFERENCE_VAL_YIELD}'


if __name__ == '__main__':
    test_strategy_tainer()
    test_cv_opt_driver()
