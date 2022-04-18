# **DataMiningMid_Classification**
Data Mining Mid-term Report

Dataset repo: https://github.com/Toyhom/Chinese-medical-dialogue-data \
Ref: http://ethen8181.github.io/machine-learning/keras/text_classification/word2vec_text_classification.html

still building...\
kinda mess right now :(

1. [x] format csv file and turn to utf-8 encode
2. [x] filt the data in which amounts of department < 1000 
3. [x] word segment and remove stop words
4. [ ] bow or word2vec or tfidf
5. [ ] cluster or classification in some algorithm
6. [ ] compare to each algorithms and hyperparameters

## **Installation**


## **Usage**


# **More Detail**
## **Table of Contents**
1. [Introduce](#introduce)
2. [Clean Data](#clean-data)
3. [Clean Text](#clean-text)
4. 


## **Introduce**
以科別分類，

## **Clean Data**
本專案使用的資料是github上的簡體中文資料集

> Dataset Toyhom Chinese-medical-dialogue-data 
> repo: https://github.com/Toyhom/Chinese-medical-dialogue-data 

裡面有各科的問題，預計是要以科別分類，，在資料選擇上僅挑了內科與外科

另外提醒，這篇資料集是簡體字用的gb2312編碼

### **刪除多餘換行**
在這資料集中有非常多地方多了換行等等不符合csv的格式，通常是發生在answer的欄位中，有出現標號格式的，都會有過多換行的問題
查了一下，沒有看到有類似的工具座使用，所以我採取了字串暴力法將這些多出來的換行去除
如果在前七個字內，沒有出現「科」字，代表他不應該是一筆新的資料，而是上一筆多了換行，如此一來，便能將大部分的換行去除。
但有些department欄位不是以「科」結尾的，那我也沒想到更聰明的方法(qq)，是以窮舉的方式將所有欄位做區別

```python
dep_lst=['Surgical', 'IM']
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
                    # 依行分隔
                    for para in lines.split('\n'): 
                        # 取出前七個字, maximum_size_of_dep=7
                        front = para[:maximum_size_of_dep if len(para) > maximum_size_of_dep else len(para)]
                        
                        # 前七個字是否有科別名稱
                        if ('精神疾病' in para or '计划生育' in front or '体检' in front or '减肥' in front or '生活疾病' in front or '结核病' in front or '美容' in front or '复杂先心病' in front or '精神心理' in front or '传染病' in front or '健身' in front or '动脉导管未闭' in front or '皮肤顽症' in front or '肛肠' in front or '科' in front) and ',' in front:
                           # 為了讓第一行也能拿到用的
                            if isStart:
                                last_para = para
                                isStart = False
                                continue
                            # 如果有科別名稱的話，把前面那段寫進text，並開始新的一行
                            cln_file.write(last_para + '\n')
                            last_para = para
                            
                        else:
                            # 若不是，則把這次的內容加到上一段內
                            last_para = last_para + para
```


在預設上，也會存為習慣的utf-8編碼

### **資料選擇**
但有些類別在這兩個資料集中出現次數明顯過少，將他們過濾掉，我這裡篩選掉出現次數小於1000的資料

過濾後的結果
cat_dep|amounts
---|---
內科神经科    | 46844
內科消化科    | 32245
內科呼吸科    | 27931
外科肛肠      |24016
外科神经脑外科|   23620
內科心血管科  |  22841
內科内分泌科  |  21745
外科普通外科  |  21179
內科肝病科    | 20888
外科泌尿科    | 18422
內科肾内科    | 14010
內科普通内科  |  13447
內科血液科    |  9968
外科肝胆科    |  8831
外科乳腺科    |  8823
外科血管科    |  6404
內科风湿免疫科|    5486
內科感染科    |  4035
外科胸外科    |  2913
外科心外科    |  1777


## **Text Preprocessing**
在NLP任務時，還有許多準備工作要做
流程

```mermaid
graph LR
raw((Raw Text)) --> seg[Text Segment] --> rm[Remove Stopwords] --> w2v[Word2Vec] --> xgb[Xgboost Classifier] --> cls((Class))
​```

### **Word Segment**
因為資料集是簡體中文的原因，採用jieba做分詞

### **Remove Stopwords**
harvesttext是學長推薦的中文NLP工具包
停用字的選擇是用harvesttext裡包好的get_baidu_stop_words
除此之外，在數字、標點符號的地方沒有清乾淨
再另外做一個方法清理

上面分詞跟刪除停用字我將一起做，每次刪完記得回去看一下資料是不是如預期一樣分詞!


## **Tokenize**
為了要讓機器看得懂自然語言

### **Word Embedding**
https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4
gensimi在4.x.x版本以後不再支援第三方wrapper
將3.8.13複製過來或者降低gensim版本

我這裡是選擇繼續使用gensim 4.1.2 配合別人寫的 Word2Vec Vectorizer
解釋一下會什麼是用別人的，而不是從3.8.13複製。有試過，但會跳出 Key error，原因是gesim原本 sklearn_api沒有支援，
換用別人的就能解決了。
如下圖

且根據上面的說明，有兩處參數名稱需要更改
iter => epochs, size => vector_size

### **Bag of Words**

### **TF-IDF**
用sklearn裡面的套件做的tokenization，

## **Classification**
### **SVM**
### **Logistic**

### **XGBoost**

## **Experimental Result**
光是xgboost就有超多種參數可以做調整
<!-- 
 可以改由Skorch 或 Pytorch 架構改寫
還能加上wandb 使用 sweep 找到更好的hyper parameters -->

還可以用RandmizedCV調整參數

## **Package**
打包帶走

## **Discussion**

