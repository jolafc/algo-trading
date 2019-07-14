#!/usr/bin/env bash

# edit ~/.bashrc or ~/.bash_profile to include QUANDL_API_KEY, IEX_KEY, and PYTHONPATH=~/Workspace/experiments
# cd ~/Workspace/experiments
# conda env create -f environment.yaml
conda activate exp
# For zipline 1.3.0: make edits specified in the experiments/environment.yaml file at the zipline line.
# zipline ingest -b quandl
zipline run -f exp/tuto_buy_aapl.py -b quandl --start 2017-1-1 --end 2017-1-31 -o data/tuto_buy_aalp.pkl