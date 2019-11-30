import numpy as np
import pandas as pd

from exp.strategy.weekly_rotation import weekly_rotation_runner_facotry

REFERENCE_YIELD = 0.13514657590506485
REFERENCE_SHARPE = 0.8992035532340469


def test_weekly_rotation():
    runner = weekly_rotation_runner_facotry(start_date_requested=pd.to_datetime('2019-01-31'),
                                            end_date_requested=pd.to_datetime('2019-10-31'),
                                            max_lookback=200,
                                            verbose=False)
    results = runner()

    assert np.isclose(results["annualized_yield"], REFERENCE_YIELD, atol=1e-8), \
        f'Yield is {results["annualized_yield"]}  but should be {REFERENCE_YIELD}'
    assert np.isclose(results["sharpe_ratio"], REFERENCE_SHARPE, atol=1e-8), \
        f'Sharpe ratio is {results["sharpe_ratio"]}  but should be {REFERENCE_SHARPE}'


if __name__ == '__main__':
    test_weekly_rotation()
