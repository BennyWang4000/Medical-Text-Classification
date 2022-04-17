from sklearn.feature_extraction.text import TfidfTransformer
from config import *
from df_processor import DfProcessor

df_p = DfProcessor(i_data_dir=I_DATA_DIR, s_data_dir=S_DATA_DIR)
df_p.df_filt(1000)
i_df, s_df = df_p.get_df('is')
# %%

# # vectorizer = CountVectorizer(stop_words=manual_stop_list)
# transformer = TfidfTransformer()
# X = vectorizer.fit_transform(df_article.sent_jieba)
# tfidf = transformer.fit_transform(X)
# weight = tfidf.toarray()

# tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
# X_train_tf = tf_transformer.transform(X_train_counts)
# X_train_tf.shape
