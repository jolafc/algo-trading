ADJUSTED_CLOSE_COLUMN = 'adjusted_close'
OPEN_COLUMN = 'open'
HIGH_COLUMN = 'high'
LOW_COLUMN = 'low'
CLOSE_COLUMN = 'close'
VOLUME_COLUMN = 'volume'
DIVIDENT_COLUMN = 'dividend_amount'
SPLIT_COLUMN = 'split_coefficient'

EXECUTION_PRICE_COLUMN = 'execution_price'
FACTOR_COLUMN = 'adjustment_factor'

POSITIONS_COLUMNS = ['date_buy', 'ticker', 'position', 'price_buy', 'fees_buy']
TRADES_EXTRA_COLUMNS = ['date_sell', 'price_sell', 'fees_sell']
TRADES_COLUMNS = POSITIONS_COLUMNS + TRADES_EXTRA_COLUMNS + ['fees', 'P&L']

BENCKMARK_TICKER = 'SPY'
N_DAYS_IN_YEAR = 365
DEFAULT_START_BALANCE = 100000