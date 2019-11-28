import numpy as np
import pandas as pd

from exp import SP500_PKL
from exp.data_getter import load_pickled_dict
from exp.data_loader import WeeklyRotationRunner

REFERENCE_YIELD = 0.13514657590506485
REFERENCE_SHARPE = 0.8992035532340469


def test_weekly_rotation():
    data_by_ticker = load_pickled_dict(pkl_file=SP500_PKL)

    runner = WeeklyRotationRunner(start_date_requested=pd.to_datetime('2019-01-31'),
                                  end_date_requested=pd.to_datetime('2019-10-31'),
                                  lookback=200,
                                  verbose=False)
    runner.fit(data_by_ticker=data_by_ticker)
    results = runner.results

    assert np.isclose(results["annualized_yield"], REFERENCE_YIELD, atol=1e-8), \
        f'Yield is {results["annualized_yield"]}  but should be {REFERENCE_YIELD}'
    assert np.isclose(results["sharpe_ratio"], REFERENCE_SHARPE, atol=1e-8), \
        f'Sharpe ratio is {results["sharpe_ratio"]}  but should be {REFERENCE_SHARPE}'


if __name__ == '__main__':
    test_weekly_rotation()
