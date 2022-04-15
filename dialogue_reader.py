# %%
from operator import index
import pandas as pd
import glob
import os

i_data_dir = 'D:\\CodeRepositories\\py_project\\data_mining\\Chinese-medical-dialogue-data\\clean_data\\IM.csv'
s_data_dir = 'D:\\CodeRepositories\\py_project\\data_mining\\Chinese-medical-dialogue-data\\clean_data\\Surgical.csv'

i_df = pd.read_csv(i_data_dir, encoding='utf-8')
s_df = pd.read_csv(s_data_dir, encoding='utf-8')

# %%
#* print all department
print(i_df[~i_df.duplicated('department')]['department'].values.tolist())
print(s_df[~s_df.duplicated('department')]['department'].values.tolist())
# %%
# * filter > 1000
df = s_df[s_df['department'].map(s_df['department'].value_counts()) > 1000]
# %%
# * count
print(df['department'].value_counts())
# %%
