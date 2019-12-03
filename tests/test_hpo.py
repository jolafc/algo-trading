import os

import numpy as np
import pandas as pd

from exp import RESULTS_DIR, YIELD
from exp.optimization import train_strategy
from exp.strategy.weekly_rotation import WeeklyRotationRunner

CHKPT_TEST_FILE = os.path.join(RESULTS_DIR, 'checkpoint_test.pkl')
REFERENCE_TRAIN_YIELD = 0.2104730320069125
REFERENCE_VAL_YIELD = -0.13707755424376059


def test_optimization():
    train_metrics, val_metrics, optimized_parameters = train_strategy(
        StrategyRunner=WeeklyRotationRunner,
        train_start=pd.to_datetime('2019-01-31'),
        train_end=pd.to_datetime('2019-10-31'),
        val_start=pd.to_datetime('2018-01-01'),
        val_end=pd.to_datetime('2018-12-31'),
        max_lookback=200,
        n_calls=5,
        n_random_starts=1,
        output_metric=YIELD,
        restart_from_chkpt=False,
        chkpt_file=CHKPT_TEST_FILE,
    )

    train_yield = train_metrics[YIELD]
    val_yield = val_metrics[YIELD]

    assert np.isclose(train_yield, REFERENCE_TRAIN_YIELD, atol=1e-8), \
        f'Train yield is {train_yield} but should be {REFERENCE_TRAIN_YIELD}'
    assert np.isclose(val_yield, REFERENCE_VAL_YIELD, atol=1e-8), \
        f'Train yield is {val_yield} but should be {REFERENCE_VAL_YIELD}'


if __name__ == '__main__':
    test_optimization()
