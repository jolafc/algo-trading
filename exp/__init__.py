import os

EXP_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(EXP_DIR, '..', 'data'))
PLT_FILE_FORMAT = 'png'
CONST_CSV = os.path.join(DATA_DIR, 'constituents.csv')
SP500_PKL = os.path.join(DATA_DIR, 'sp500.pkl')
PLOTFILE = os.path.join(DATA_DIR, f'sample_data'+PLT_FILE_FORMAT)