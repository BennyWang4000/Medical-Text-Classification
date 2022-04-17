import sklearn.svm as svm
import numpy as np
from sklearn.model_selection import train_test_split

# def get_word_vector(path):
#     ip = open(path, 'r', encoding='utf-8')
#     content = ip.readlines()
#     vecs = []

#     for words in content:
#         # vec = np.zeros(2).reshape((1, 2))
#         vec = np.zeros(50).reshape((1, 50))
#         count = 0
#         words = remove_some(words)
#         for word in words[1:]:
#             try:
#                 count += 1
#                 # vec += model[word].reshape((1, 2))
#                 vec += model[word].reshape((1, 50))
#                 # print(vec)
#             except KeyError:
#                 continue
#         vec /= count
#         vecs.append(vec)
#     return vecs
# #%%


# normal_tag = np.ones((len(normal)))
# spam_tag = np.zeros((len(spam)))

# X_train, X_test, y_train, y_test = train_test_split(np.array(train, dtype='float64'),
#                                                         np.array(train_tag, dtype='float64'), test_size=0.30,
#                                                         random_state=0)

# clf = svm.SVC()
# clf_res = clf.fit(X_train, y_train)
#     #  train_pred = clf_res.predict(X_train)
#     test_pred = clf_res.predict(X_test)
# print(classification_report(y_test, test_pred))
