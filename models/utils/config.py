
# * magic number =============
LEN_LINES = 670862


# * data config =============
I_DATA_PATH = 'D:\\CodeRepositories\\py_project\\data_mining\\Chinese-medical-dialogue-data\\clean_data\\IM.csv'
S_DATA_PATH = 'D:\\CodeRepositories\\py_project\\data_mining\\Chinese-medical-dialogue-data\\clean_data\\Surgical.csv'
SAVING_DIR = 'D:\\CodeRepositories\\py_project\\data_mining\\DataMiningMid_Classification\\data'
ALL_DATA_PATH = 'D:\\CodeRepositories\\py_project\\data_mining\\DataMiningMid_Classification\\data\\df.csv'
STOP_WORDS_PATH = 'D:\\CodeRepositories\\py_project\\data_mining\\DataMiningMid_Classification\\data\\stopwords.txt'
RUNS_PATH = 'D:\\CodeRepositories\\py_project\\data_mining\\DataMiningMid_Classification\\runs'
ENCODING = 'utf-8'

DEP_MIN_AMOUNT = 1000

# * train config =============
# w2v params
VECTOR_SIZE = 100
MIN_COUNT = 3
WORKERS = 8
WINDOW = 7

# xgb params
LEARNING_RATE = 0.01,
OBJECTIVE = 'multi:softmax',
EVAL_METRIC = 'mlogloss',
SCALE_POS_WEIGHT = 1,
COLSAMPLE_BTREE = 0.8,
SUBSAMPLE = 0.8
