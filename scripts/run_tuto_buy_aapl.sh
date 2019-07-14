#!/usr/bin/env bash

conda activate exp
# zipline ingest -b quandl
zipline run -f exp/tuto_buy_aapl.py -b quandl --start 2017-1-1 --end 2017-1-31 -o data/tuto_buy_aalp.pkl