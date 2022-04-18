# %%
import pandas as pd
import glob
import os


def df_filt(df, min):
    return df[df['cat_dep'].map(df['cat_dep'].value_counts()) > min]


def get_dep_counts(df):
    return df['cat_dep'].value_counts()


# %%
