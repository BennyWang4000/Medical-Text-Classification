# %%
import pandas as pd
import glob
import os


class DfProcessor():
    def __init__(self, i_data_dir, s_data_dir, encoding='utf-8'):
        self.i_data_dir = i_data_dir
        self.s_data_dir = s_data_dir

        self.i_df = pd.read_csv(self.i_data_dir, encoding=encoding)
        self.s_df = pd.read_csv(self.s_data_dir, encoding=encoding)

    def df_filt(self, min):
        self.i_df = self.i_df[self.i_df['department'].map(
            self.i_df['department'].value_counts()) > min]
        self.s_df = self.s_df[self.s_df['department'].map(
            self.s_df['department'].value_counts()) > min]

    def get_df(self, type):
        '''
        get dataframe
        @param:  str; assert in i or s or is
        @return: dataframe, dataframe; i_df, s_df
        '''
        assert type in ['i', 's', 'is']
        if type == 'i':
            return self.i_df
        if type == 's':
            return self.s_df
        if type == 'is':
            return self.i_df, self.s_df

    def get_dep_counts(self, type):
        assert type in ['i', 's', 'is']
        if type == 'i':
            return self.i_df['department'].value_counts()
        if type == 's':
            return self.s_df['department'].value_counts()
        if type == 'is':
            return self.i_df['department'].value_counts(), self.s_df['department'].value_counts()

    def save_df(self, saving_dir):
        pass

# # %%
# #* print all department
# print(i_df[~i_df.duplicated('department')]['department'].values.tolist())
# print(s_df[~s_df.duplicated('department')]['department'].values.tolist())
# # %%
# # * filter > 1000
# s_df = s_df[s_df['department'].map(s_df['department'].value_counts()) > 1000]
# i_df = i_df[i_df['department'].map(i_df['department'].value_counts()) > 1000]
# # %%
# # * count
# print(s_df['department'].value_counts())
# print(i_df['department'].value_counts())
# # %%

# %%
