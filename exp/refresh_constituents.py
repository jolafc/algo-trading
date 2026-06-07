"""Regenerate data/constituents.csv from the current Wikipedia S&P 500 list.

This is the *current* membership snapshot — using it for backtesting introduces survivorship
bias. See doc/TODO.md for the caveat and the planned remediation (extended universe + delisted
price data + point-in-time membership filtering).

Run with: `python -m exp.refresh_constituents`
"""
import io

import pandas as pd
import requests

from exp import CONST_CSV

WIKI_URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
USER_AGENT = 'algo-trading-research/1.0'

# SPY (the SPDR S&P 500 ETF) is not an SP500 *constituent* — it's the index itself, used by the
# strategy as the regime filter and by the reporter as the buy-and-hold benchmark. Keep it at
# row 0 to match the legacy CSV's convention.
BENCHMARK_ROW = {'Symbol': 'SPY', 'Name': 'S&P 500', 'Sector': 'ETF'}


def fetch_sp500_constituents():
    response = requests.get(WIKI_URL, headers={'User-Agent': USER_AGENT})
    response.raise_for_status()
    tables = pd.read_html(io.StringIO(response.text))
    df = tables[0][['Symbol', 'Security', 'GICS Sector']].copy()
    df.columns = ['Symbol', 'Name', 'Sector']
    df['Symbol'] = df['Symbol'].astype(str).str.strip()
    df = df.sort_values('Symbol').reset_index(drop=True)
    return df


def write_constituents_csv(path=CONST_CSV):
    constituents = fetch_sp500_constituents()
    out = pd.concat([pd.DataFrame([BENCHMARK_ROW]), constituents], ignore_index=True)
    out.to_csv(path, index=False)
    print(f'Wrote {len(out)} rows to {path} ({len(constituents)} constituents + 1 benchmark).')


if __name__ == '__main__':
    write_constituents_csv()
