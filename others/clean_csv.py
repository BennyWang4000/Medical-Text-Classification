# %%
import os
import pandas as pd
import glob
import csv
import tqdm


data_dir = 'D:\\CodeRepositories\\py_project\\data_mining\\Chinese-medical-dialogue-data\\Data'
saving_dir = 'D:\\CodeRepositories\\py_project\\data_mining\\Chinese-medical-dialogue-data\\clean_data'
# data_dir = 'D:\\CodeRepositories\\py_project\\data_mining\\test.txt'
maximum_size_of_dep = 7
dep_lst = ['IM']

'''
外科    
    ├ 普通外科
    ├ 肛肠
    ├ 神经脑外科
    ├ 肝胆科
    ├ 泌尿科
    ├ 乳腺科
    ├ 心外科
    ├ 血管科
    └ 胸外科
     

复杂先心病
精神心理
传染病
美容
'''

# %%
df = pd.DataFrame
for csv_path in glob.glob(os.path.join(data_dir, '*', '*.csv'), recursive=True):
    isStart = True
    last_para = ''

    csv_name = csv_path.split('\\')[-2]
    print(csv_name)

    if csv_name in dep_lst:
        with open(csv_path, 'r', encoding='gb2312', errors='ignore') as ori_file:
            with open(os.path.join(saving_dir, csv_name + '.csv'), 'w+') as cln_file:
                for lines in ori_file:
                    for para in lines.split('\n'):
                        # print(last_para)
                        # print(',' in para[:10 if len(para) > 10 else len(para)])
                        front = para[:maximum_size_of_dep if len(
                            para) > maximum_size_of_dep else len(para)]
                        if ('精神疾病' in para or '计划生育' in front or '体检' in front or '减肥' in front or '生活疾病' in front or '结核病' in front or '美容' in front or '复杂先心病' in front or '精神心理' in front or '传染病' in front or '健身' in front or '动脉导管未闭' in front or '皮肤顽症' in front or '肛肠' in front or '科' in front) and ',' in front:
                            if isStart:
                                last_para = para
                                isStart = False
                                continue
                            cln_file.write(last_para + '\n')
                            last_para = para
                        else:
                            last_para = last_para + para

# %%
