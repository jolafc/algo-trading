import os

SEED = 42

EXP_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(EXP_DIR, '..', 'data'))
os.makedirs(DATA_DIR, exist_ok=True)
RESULTS_DIR = os.path.abspath(os.path.join(EXP_DIR, '..', 'results'))
os.makedirs(RESULTS_DIR, exist_ok=True)

PLT_FILE_FORMAT = 'png'
CONST_CSV = os.path.join(DATA_DIR, 'constituents.csv')
SP500_PKL = os.path.join(DATA_DIR, 'sp500.pkl')
PLOTFILE = os.path.join(RESULTS_DIR, f'sample_data'+PLT_FILE_FORMAT)
CHKPT_DEFAULT_FILE = os.path.join(RESULTS_DIR, f'checkpoint.pkl')
