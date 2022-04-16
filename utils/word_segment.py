# %%
import gensim
from harvesttext.resources import get_baidu_stopwords
import jieba
from df_processor import DfProcessor
from config import *
from tqdm import tqdm

df_p = DfProcessor(i_data_dir=I_DATA_DIR, s_data_dir=S_DATA_DIR)
df_p.df_filt(1000)
i_df, s_df = df_p.get_df('is')

# %%


def get_stop_words(add_set):
    stopwords = get_baidu_stopwords()
    return stopwords.union(add_set)


def remove_stop_words(stopwords, sentence):
    result = ''
    for word in sentence:
        if word not in stopwords:
            if word != '\n':
                result += word + ' '
    return result


# %%
with open('D:\\CodeRepositories\\py_project\\data_mining\\DataMiningMid_Classification\\data\\i_s_ask_ans.txt', 'r') as text:
    with open('D:\CodeRepositories\py_project\data_mining\DataMiningMid_Classification\data\i_s_ask_ans_ws.txt', 'a+') as ws_text:
        for lines in text:
            lines_bar = tqdm(range(670862))
            for line in lines.split('\n'):
                lines_bar.update()
                seg_context = jieba.cut(line)
                seg_context = remove_stop_words(
                    get_stop_words(('？')), seg_context)
                ws_text.write(seg_context)

    # baidu_stopwords = open('./baidu_stopwords.txt', 'a+')
    # for word in stopwords:
    #     baidu_stopwords.write(word + '\n')
    # baidu_stopwords.close()
# * get stop words


# %%
