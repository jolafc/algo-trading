from zipline.api import symbol, order, record


def initialize(context):
    pass


def handle_data(context, data):
    order(symbol('AAPL'), 10)
    record(AAPL=data[symbol('AAPL')].price)
