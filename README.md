# **DataMiningMid_Classification**
Data Mining Mid-term Report

>Dataset repo: https://github.com/Toyhom/Chinese-medical-dialogue-data \
Ref: http://ethen8181.github.io/machine-learning/keras/text_classification/word2vec_text_classification.html

still building...\
kinda mess right now :(
# **TODO**
- [x] format csv file and turn to utf-8 encode
- [x] filt the data in which amounts of department < 1000
- [x] word segment and remove stop words
- [ ] bow or word2vec or tfidf
- [ ] cluster or classification in some algorithm
- [ ] compare to each algorithms and hyperparameters
- [ ] refactor
- [ ] pack it 

## **Installation**
not done yet :(
```bash
git clone https://github.com/BennyWang4000/DataMiningMid_Classification.git
```
```bash
python setup.py install
```
## **Usage**
not done either :(
```python
doc= '連我爸都沒有打過我'
foo(doc, lang='t')
```

# **More Detail**
## **Table of Contents**
1. [Introduce](#introduce)
2. [Clean Data](#clean-data)
3. [Text Pre-processing](#text-processing)
4. [Tokenization](#tokenization)
5. [Classification](#classification)
6. [Result](#result)
7. [Discussion](#discussion)

## **Introduce**
想利用已標記的大量文本，做分類。




## **Clean Data**
使用github上的簡體中文資料集

> Dataset from Toyhom Chinese-medical-dialogue-data \
> repo: https://github.com/Toyhom/Chinese-medical-dialogue-data 

裡面有各科的問題，預計是要以科別分類，在資料選擇上僅挑了內科與外科\
另外提醒，這篇資料集是簡體字用的gb2312編碼
在收入

### **刪除多餘換行**
在這資料集中有非常多地方多了換行等等不符合csv的格式，通常是發生在answer的欄位中，有出現標號格式的，都會有過多換行的問題。查了一下，沒有看到有類似的工具能使用，所以我依資料特性採取了字串暴力法將這些多出來的換行去除。

如果在前七個字內，沒有出現「科」字，代表他不應該是一筆新的資料，而是上一筆多了換行，如此一來，便能將大部分的換行去除。

但有些department欄位不是以「科」結尾的，沒想到更聰明的方法(qq)，以窮舉的方式將所有欄位做區別，

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
為了區別內科跟外科，加了一個欄位cat_dep，是內科或外加+department。再將它們加上id，以後訓練會用到的


但有些科別出現次數明顯過少，原本內科跟外科加起來總共有69種，將他們過濾掉，我這裡篩選掉出現次數小於1000的資料。最後剩下20種。

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


## **Text Process**
在NLP任務時，還有許多準備工作要做
在這部分流程圖在以下
::: mermaid
graph LR
raw((Raw Ask Text)) -->seg[Text Segment] 
seg --> rm[Remove Stopwords]
rm --> w2v[Word2Vec]
subgraph Pipeline
w2v --> cls[Classifier]
end
cls --> out((Output Class))
:::

### **Word Segment**
文本進來時，會是一長串的字串，必須要先經過分詞、
資料集是簡體中文，採用簡單的 jieba 做分詞
::: mermaid
graph LR
raw['人類總要重複同樣的錯誤']  -->|Word Segment|re['人類', '總要', '重複', '同樣', '的','錯誤']
:::
這部分沒有改任何的參數，原本的分詞效果就很不錯了。
```python
import jieba

def word_segment(sentence, stopwords_path):
    '''Word segment and remove stopwords
    Parameters
    ----------
        sentence: str
            Raw text
        stopwords_path: str
            Path of stopwords text file

    Returns
    -------
        list<str>
            A list that after segment and remove stopwords
    '''
    words = jieba.cut(sentence)
    words = _remove_stop_words(words, stopwords_path)
    return words
```
分完詞以後，再接著做去除停用詞
### **Remove Stopwords**
::: mermaid
graph LR
re['人類', '總要', '重複', '同樣', '的','錯誤']-->|Remove Stopwords|rmstp['人類', '重複', '同樣','錯誤']  
:::

在一段文字的組成中，介係詞、代名詞等等不具有關鍵意義的詞，會被視為是停用詞(stopwords)。再經過斷詞以後，這些詞同樣也會被切出，但它們不太能表達句子意思，故在這裡的NLP任務去除

並非所有的NLP任務都需要去除停用詞，比如翻譯，停用詞可以表達文句因果關係等等非常重要。

停用詞字典採用 HarvestText 裡包好的 get_baidu_stop_words()，可以取得 set of 百度停用詞字典。
HarvestText是學長推薦的中文NLP工具包
> **HarvestText**\
> repo: https://github.com/blmoistawinde/HarvestText

不過，除了使用裡面的停用字以外，我發現在原有的資料中，還有很多符號沒有被清除，所以在裡面新增了一些字，包含各類數字、標點符號、單位、全行空白等。期望能讓資料更乾淨。
```
0123456789’!．"＂#＃$＄%％&＆\'()（）*＊+＋×,-./:;<=>?@＠，。★、…【】《》？“”‘’！[]^︿_＿`{｛|｜}｝~～１２３４５６７８９０⓵⓶⓷⓸⓹⓺⓻⓼⓽⓾⒈⒉⒊⒋⒌⒍⒎⒏⒐⒑⒒⒓⒔⒕⒖⒗⒘⒙⒚⒛⓪①②③④⑤⑥⑦⑧⑨⑩➀➁➂➃➄➅➆➇➈➉⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳⓿❶❷❸❹❺❻❼❽❾❿➊➋➌➍➎➏➐➑➒➓⓫⓬⓭⓮⓯⓰⓱⓲⓳⓴⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽⑾⑿⒀⒁⒂⒃⒄⒅⒆⒇－℃mg MG kg KG um UM mm MM cm CM nm NM km KM ml ML abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ
```

停用字的選擇是用harvesttext裡包好的get_baidu_stop_words
除此之外，在數字、標點符號的地方沒有清乾淨
再另外做一個方法清理



```python
def _remove_stop_words(words, stopwords_path):
    '''Remove stopwords
    Parameters
    ----------
        words: list<dir>
            list of word segmentation
        stopwords_path: str
            Path of stopwords text file

    Returns
    -------
        list<str>
            A list that after remove stopwords
    '''
    result = []
    stopwords = set(line.strip() for line in open(stopwords_path))
    for word in words:
        if word not in stopwords:
            result.append(word)
    return result
```

分詞跟刪除停用字一起做，每次刪完記得回去看一下資料是不是如預期一樣分詞!
## **Pipeline**
後面的流程會由 sklearn 包，pipeline 會搞定一切
## **Tokenize**
為了要讓機器看得懂自然語言，會將文字以矩陣的方式表達。
目前我知道有以下作法
### **Bag of Words**

### **TF-IDF**
用sklearn裡面的套件做的tokenization，

### **Word Embedding**
在這篇文章將使用 Gensim Word2Vec 完成 word embedding

不過，gensimi在4.0.0以後的版本不再支援第三方wrapper
可以選擇將3.8.13複製過來或者降低gensim版本

我這裡是選擇繼續使用gensim 4.1.2 配合別人寫的 Word2Vec Vectorizer
解釋一下會什麼是用別人的，而不是從3.8.13複製。有試過，但會跳出 Key error，原因是gesim原本 sklearn_api 沒有支援新增文字，
換用別人的就能解決了。
如下圖

> **Migrating from Gensim 3.x to 4** \
> link: https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4

且根據上面連結的說明，有兩處參數名稱需要更改
iter => epochs, size => vector_size
## **Classification**
### **SVM**
### **Logistic**

### **Xgboost**
第一次選用的參數，跑了一個半小時，在測試資料集僅能到達 56% 左右的準確率qq

## **Experimental Result**
光是xgboost就有超多種參數可以做調整
<!-- 
 可以改由Skorch 或 Pytorch 架構改寫
還能加上wandb 使用 sweep 找到更好的hyper parameters -->

還可以用RandmizedCV調整參數

## **Package**
上面的 sklearn pipeline 能夠用 joblib 儲存。

## **Discussion**

