# DataMiningMid_Classification
Data Mining Mid-term Report

Dataset repo: https://github.com/Toyhom/Chinese-medical-dialogue-data \\
Ref: http://ethen8181.github.io/machine-learning/keras/text_classification/word2vec_text_classification.html

still building...
kinda mess right now :(

1. [x] format csv file and turn to utf-8 encode
2. [x] filt the data in which amounts of department < 1000 
3. [x] word segment and remove stop words
4. [ ] bow or word2vec
5. [ ] svm cluster or classification in 10 dimention or other algorithm
6. [ ] compare to other algorithms 

## Installation


## Usage


# More Detail
## Table of Contents
1. [Introduce](#introduce)
2. [Clean Data](#clean-data)
3. [Clean Text](#clean-text)
4. 


## Introduce
以科別分類，

## Clean Data
本專案使用的資料是github上的開源資料集

### 刪除多餘換行
在這資料集中有多處地方多了換行等等不符合csv的格式
就用了暴力法將這些多出來的換行去除

### 資料選擇
僅挑了內科與外科
預計是要以科別分類，但有些類別在這兩個資料集中出現次數明顯過少，將他們篩選掉
在這裡會用到一些pandas dataframe的操作。

## Text Preprocessing
在NLP任務時，還有許多準備工作要做
### 分詞
因為資料集是簡體中文的原因，採用jieba做分詞

### 刪除停用字
harvesttext是我們實驗室學長推薦的中文NLP工具包，前一周才知道的，馬上用看看。
停用字的選擇是用harvesttext裡包好的get_baidu_stop_words
除此之外，在數字、標點符號的地方沒有清乾淨
再另外做一個方法清理

上面分詞跟刪除停用字我將一起做，每次刪完記得回去看一下資料是不是如預期一樣分詞!

## Tokenize
為了要讓機器看得懂自然語言

### Bag of Words


### Word Embedding
https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4
gensimi在4.x.x版本以後不再支援第三方wrapper
將3.8.13複製過來或者降低gensim版本


### TF-IDF
這個比較特別，是用sklearn裡面的套件做的tokenization，
## Classification
### SVM
### Logistic

### XGBoost

## PipeLine


## Experimental Result
光是xgboost就有超多種參數可以做調整

 可以改由Skorch 或 Pytorch 架構改寫
還能加上wandb 使用 sweep 找到更好的hyper parameters

## Package
打包帶走
## Discussion

