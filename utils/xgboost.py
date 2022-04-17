import pandas as pd
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline


# xgb = XGBClassifier(learning_rate=0.01, n_estimators=100, n_jobs=-1)
# w2v_xgb = Pipeline([
#     ('w2v', gensim_word2vec_tr), 
#     ('xgb', xgb)
# ])
# w2v_xgb