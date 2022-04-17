# %%
from utils.df_processor import DfProcessor
import utils.word_segment as ws
from utils.config import *

# %%


def preprocessing():
    pass


def main():
    pass


# %%
df_p = DfProcessor(i_data_dir=I_DATA_DIR, s_data_dir=S_DATA_DIR)
df_p.df_filt(DEP_MIN_AMOUNT)
# %%
df_p.all_df['ask_clean'] = df_p.all_df['ask'].apply(
    lambda x: ws.word_segment(x))
# %%


# %%


if '__name__' == '__main__':
    main()
