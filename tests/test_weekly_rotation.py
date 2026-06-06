import numpy as np
import pandas as pd

from exp import YIELD, SHARPE
from exp.strategy.weekly_rotation import WeeklyRotationRunner

# NOTE: these references were captured against the legacy Alpha Vantage adjusted-close data.
# After the migration to yfinance + locally-computed adjusted_close (see exp/data_getter.py),
# the underlying prices differ slightly and these constants must be re-baselined by running
# this test, capturing the actual results[YIELD] / results[SHARPE] values, and updating below.
REFERENCE_YIELD = 0.17253136849657844
REFERENCE_SHARPE = 1.1567442855142187


def test_weekly_rotation():
    runner = WeeklyRotationRunner(start_date_requested=pd.to_datetime('2019-01-31'),
                                  end_date_requested=pd.to_datetime('2019-10-31'),
                                  max_lookback=200,
                                  verbose=False)
    results = runner(lookback=200,
                     sma_tol=0.02,
                     volume_lookback=20,
                     volume_threshold=1e6,
                     price_min=1.,
                     rsi_lookback=3,
                     rsi_threshold=50.,
                     day_of_trade=4,
                     n_positions=10)

    assert np.isclose(results[YIELD], REFERENCE_YIELD, atol=1e-8), \
        f'Yield is {results[YIELD]}  but should be {REFERENCE_YIELD}'
    assert np.isclose(results[SHARPE], REFERENCE_SHARPE, atol=1e-8), \
        f'Sharpe ratio is {results[SHARPE]}  but should be {REFERENCE_SHARPE}'


if __name__ == '__main__':
    test_weekly_rotation()
