import os

import numpy as np
import pandas as pd

from exp import RESULTS_DIR
from exp.optimization import train_strategy
from exp.strategy.weekly_rotation import WeeklyRotationRunner

CHKPT_TEST_FILE = os.path.join(RESULTS_DIR, 'checkpoint_test.pkl')
REFERENCE_TRAIN_YIELD = 0.2182188801650903
REFERENCE_VAL_YIELD = -0.12216407120430788


def test_optimization():
    train_metrics, val_metrics = train_strategy(
        StrategyRunner=WeeklyRotationRunner,
        train_start=pd.to_datetime('2019-01-31'),
        train_end=pd.to_datetime('2019-10-31'),
        val_start=pd.to_datetime('2018-01-01'),
        val_end=pd.to_datetime('2018-12-31'),
        max_lookback=200,
        n_calls=5,
        n_random_starts=1,
        output_metric='annualized_yield',
        restart_from_chkpt=False,
        chkpt_file=CHKPT_TEST_FILE,
    )

    train_yield = train_metrics['annualized_yield']
    val_yield = val_metrics['annualized_yield']

    assert np.isclose(train_yield, REFERENCE_TRAIN_YIELD, atol=1e-8), \
        f'Train yield is {train_yield} but should be {REFERENCE_TRAIN_YIELD}'
    assert np.isclose(val_yield, REFERENCE_VAL_YIELD, atol=1e-8), \
        f'Train yield is {val_yield} but should be {REFERENCE_VAL_YIELD}'


if __name__ == '__main__':
    test_optimization()
