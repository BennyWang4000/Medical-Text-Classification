# %%
import pandas as pd
from sklearn.pipeline import Pipeline
import numpy as np

# from sklearn_api.w2vmodel import W2VTransformer
# from ..sklearn_api.gensim_word2vec import GensimWord2VecVectorizer
from sklearn_api.gensim_word2vec import GensimWord2VecVectorizer
from xgboost import XGBClassifier
from xgboost import plot_tree
import matplotlib.pyplot as plt
import joblib
# %%


class TextModel(Pipeline):
    def __init__(
        # w2v params
        layers,
        size=100,
        min_count=3,
        workers=4,
        window=5,

        # xgb params
        learning_rate=0.01,
        objective='muli:softmax',
        eval_metric='mlogloss',
        scale_pos_weight=1,
        silent=1,
        nthread=-1,
        max_depth=10,
        colsample_btree=0.8,
        subsample=0.8,
    ):
        all_layers = ['w2v', 'tfidf', 'svm', 'xgb']
        assert all([layer in all_layers for layer in layers]
                   ), 'InputError, unavaliable layer'

        layers = []
        for layer_name in layers:
            if layer_name == 'w2v':
                layer = (layer_name,  GensimWord2VecVectorizer(
                    size=size, min_count=min_count, workers=workers, window=window))
            elif layer_name == 'tfidf':
                pass
            elif layer_name == 'svm':
                pass
            elif layer_name == 'xgb':
                layer = (layer_name, XGBClassifier(
                    learning_rate=learning_rate,
                    objective=objective,
                    eval_metric=eval_metric,
                    scale_pos_weight=scale_pos_weight,
                    silent=silent,
                    nthread=nthread,
                    max_depth=max_depth,
                    colsample_btree=colsample_btree,
                    subsample=subsample))
            layers.append(layer_name, layer)

        super(layers)


def _createBalancedSampleWeights(y_train, largest_class_weight_coef):
    classes = np.unique(y_train, axis=0)
    classes.sort()
    class_samples = np.bincount(y_train)
    total_samples = class_samples.sum()
    n_classes = len(class_samples)
    weights = total_samples / (n_classes * class_samples * 1.0)
    class_weight_dict = {key: value for (key, value) in zip(classes, weights)}
    class_weight_dict[classes[1]] = class_weight_dict[classes[1]
                                                      ] * largest_class_weight_coef
    sample_weights = [class_weight_dict[y] for y in y_train]
    return sample_weights


def getWeights(df, y_train):
    df_copy = df
    largest_class_weight_coef = max(
        df_copy['dep_id'].value_counts().values)/df.shape[0]

    # pass y_train as numpy array
    weight = _createBalancedSampleWeights(y_train, largest_class_weight_coef)
    return weight


# %%

# # %%
# fig, ax = plt.subplots(figsize=(50, 50))
# plot_tree(xgb_layer, num_trees=0, rankdir='LR', ax=ax)
# plt.plot()
# plt.savefig('fig.pdf')
# # %%
# plot_tree(xgb_layer, num_trees=0, rankdir='LR')
# fig = plt.gcf()
# fig.set_size_inches(150, 100)
# fig.savefig('tree.pdf')
# # %%

#%%
# xgb = XGBClassifier(learning_rate=0.01, n_estimators=100, n_jobs=-1)
# w2v_xgb = Pipeline([
#     ('w2v', gensim_word2vec_tr),
#     ('xgb', xgb)
# ])
# w2v_xgb
