import os

from matplotlib import pyplot as plt

from exp import PLT_FILE_FORMAT, RESULTS_DIR
from exp.data_getter import av_query


def test_av_query(symbol='MSFT'):
    print(f'AV - querying daily prices for {symbol}.')
    price_df, run_time = av_query(symbol=symbol)
    print(f'Query success, took: {run_time:.3f} s.')

    print(f'Price DF shape: {price_df.shape}')
    price_df['adjusted_close'].plot()
    plotfile = os.path.join(RESULTS_DIR, f'{symbol}_data.{PLT_FILE_FORMAT}')
    plt.savefig(plotfile)
    print(f'Plotted price history data for {symbol} in {plotfile}')

    return price_df


if __name__ == '__main__':
    test_av_query()
