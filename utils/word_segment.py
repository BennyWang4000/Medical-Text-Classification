# %%
from harvesttext.resources import get_baidu_stopwords
import jieba
from tqdm import tqdm
import re
# %%

# def _get_stop_words(self, add_set):
#     stopwords = get_baidu_stopwords()
#     return stopwords.union(add_set)


def _remove(text):
    remove_chars = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~～１２３４５６７８９０]+'
    return re.sub(remove_chars, '', text)


def _remove_stop_words(sentence):
    result = ''
    stopwords = get_baidu_stopwords()
    for word in sentence:
        if word not in stopwords:
            result += word + ' '
    return _remove(result)


def word_segment(line):
    seg_line = jieba.cut(line)
    seg_line = _remove_stop_words(seg_line)
    return seg_line

# def save_segment_


def save_segment_txt(saving_path, data_path, len_lines):
    '''
    data: 'D:\\CodeRepositories\\py_project\\data_mining\\DataMiningMid_Classification\\data\\i_s_ask_ans.txt'
    saving: 'D:\\CodeRepositories\\py_project\\data_mining\\DataMiningMid_Classification\\data\\i_s_ask_ans_ws.txt'
    '''
    lines_bar = tqdm(total=len_lines, position=0, leave=True)
    with open(data_path, 'r') as text:
        with open(saving_path, 'a+') as ws_text:
            lines = text.readlines()
            for line in lines:
                lines_bar.update()
                seg_context = jieba.cut(line)
                seg_context = _remove_stop_words(seg_context)
                ws_text.writelines(seg_context + '\n')


# %%
# * create csv

# df_p = DfProcessor(i_data_dir=I_DATA_DIR, s_data_dir=S_DATA_DIR)
# df_p.df_filt(1000)
# # %%
# cols = ['title', 'ask', 'answer']
# for col in cols:
#     print(type(df_p.i_df.loc[:, col]))
#     # seg_context = jieba.cut(df_p.i_df.loc[:, col])
#     # seg_context = remove_stop_words(
#     #     get_stop_words({'？', '\n'}), seg_context)
#     # df_p.i_df.loc[:, cols]= seg_context

# df_p.i_df.to_csv(
#     'D:\\CodeRepositories\\py_project\\data_mining\\DataMiningMid_Classification\\data\\i_ask_ans.csv')


# %%
# baidu_stopwords = open('./baidu_stopwords.txt', 'a+')
# for word in stopwords:
#     baidu_stopwords.write(word + '\n')
# baidu_stopwords.close()
# * get stop words


# %%
# * word embedding


# # Settings
# seed = 555
# sg = 0
# window_size = 12
# vector_size = 100
# min_count = 3
# workers = 4
# epochs = 5
# batch_words = 10000

# train_data = word2vec.LineSentence(
#     'D:\\CodeRepositories\\py_project\\data_mining\\DataMiningMid_Classification\\data\\i_s_ask_ans_ws.txt')
# model = word2vec.Word2Vec(
#     train_data,
#     min_count=min_count,
#     vector_size=vector_size,
#     workers=workers,
#     epochs=epochs,
#     window=window_size,
#     sg=sg,
#     seed=seed,
#     batch_words=batch_words
# )

# model.save('word2vec.model')


# model = word2vec.Word2Vec.load('word2vec.model')
# print(model.wv['口腔'].shape)

# for item in model.wv.most_similar('減輕'):
#     print(item)

# %%
