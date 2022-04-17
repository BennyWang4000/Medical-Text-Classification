# %%
import pandas as pd
import glob
import os


def df_filt(df, min):
    return df[df['department'].map(df['department'].value_counts()) > min]


def get_dep_counts(df):
    return df['department'].value_counts()


# %%
