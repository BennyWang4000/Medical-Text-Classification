from utils.df_processor import DfProcessor
from utils.config import *

def preprocessing():
    pass


def main():
    df_p= DfProcessor(i_data_dir=I_DATA_DIR, s_data_dir=S_DATA_DIR)
    df_p.df_filt(DEP_MIN_AMOUNT)
    

if '__name__'== '__main__':
    preprocessing()
    main()