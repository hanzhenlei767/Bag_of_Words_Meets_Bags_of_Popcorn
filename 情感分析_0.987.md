
* Kaggle比赛情感分析题目：Bag of Words Meets Bags of Popcorn
* Kaggle比赛地址：https://www.kaggle.com/c/word2vec-nlp-tutorial#description

# 数据特征处理

## 一、数据预处理

### 1、加载工具包


```python
import pandas as pd
import numpy as np
```

### 2、读取并查看数据


```python
root_dir = "data"
# 载入数据集
train = pd.read_csv('%s/%s' % (root_dir, 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=1)
unlabel_train = pd.read_csv('%s/%s' % (root_dir, 'unlabeledTrainData.tsv'), header=0, delimiter="\t", quoting=3)   
test = pd.read_csv('%s/%s' % (root_dir, 'testData.tsv'), header=0, delimiter="\t", quoting=1)

print(train.shape)
print(train.columns.values)
print(unlabel_train.shape)
print(unlabel_train.columns.values)
print(test.shape)
print(test.columns.values)

print(train.head(3))
print(unlabel_train.head(3))
print(test.head(3))
```

    (25000, 3)
    ['id' 'sentiment' 'review']
    (50000, 2)
    ['id' 'review']
    (25000, 2)
    ['id' 'review']
           id  sentiment                                             review
    0  5814_8          1  With all this stuff going down at the moment w...
    1  2381_9          1  \The Classic War of the Worlds\" by Timothy Hi...
    2  7759_3          0  The film starts with a manager (Nicholas Bell)...
              id                                             review
    0   "9999_0"  "Watching Time Chasers, it obvious that it was...
    1  "45057_0"  "I saw this film about 20 years ago and rememb...
    2  "15561_0"  "Minor Spoilers<br /><br />In New York, Joan B...
             id                                             review
    0  12311_10  Naturally in a film who's main themes are of m...
    1    8348_2  This movie is a disaster within a disaster fil...
    2    5828_4  All in all, this is a movie for kids. We saw i...
    

从原始数据中可以看出：
* 1.labeledTrainData数据用于模型训练；unlabeledTrainData数据用于Word2vec提取特征；testData数据用于提交结果预测。
* 2.文本数据来自网络爬虫数据，带有html格式

### 3.去除HTML标签+数字+全部小写


```python
def review_to_wordlist(review):
    '''
    把IMDB的评论转成词序列
    '''
    from bs4 import BeautifulSoup
    # 1.去掉HTML标签，拿到内容
    review_text = BeautifulSoup(review, "html.parser").get_text()
    
    import re
    # 用正则表达式取出符合规范的部分
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    
    # 小写化所有的词，并转成词list
    words_list = review_text.lower().split()
    
    #去除停用词。需要下载nltk库，并且下载stopwords。
    from nltk.corpus import stopwords
    stopwords = set(stopwords.words("english"))
    words = [word for word in words_list if word not in stopwords]
    
    # 返回words
    return words

# 预处理数据
label = train['sentiment']
train_data = []
for i in range(len(train['review'])):
    train_data.append(' '.join(review_to_wordlist(train['review'][i])))
    
unlable_data = []    
for i in range(len(unlabel_train['review'])):
    unlable_data.append(' '.join(review_to_wordlist(unlabel_train['review'][i])))   
    
test_data = []
for i in range(len(test['review'])):
    test_data.append(' '.join(review_to_wordlist(test['review'][i])))

# 预览数据
print(train_data[0], '\n')
print(unlable_data[0], '\n')
print(test_data[0])
```

    stuff going moment mj started listening music watching odd documentary watched wiz watched moonwalker maybe want get certain insight guy thought really cool eighties maybe make mind whether guilty innocent moonwalker part biography part feature film remember going see cinema originally released subtle messages mj feeling towards press also obvious message drugs bad kay visually impressive course michael jackson unless remotely like mj anyway going hate find boring may call mj egotist consenting making movie mj fans would say made fans true really nice actual feature film bit finally starts minutes excluding smooth criminal sequence joe pesci convincing psychopathic powerful drug lord wants mj dead bad beyond mj overheard plans nah joe pesci character ranted wanted people know supplying drugs etc dunno maybe hates mj music lots cool things like mj turning car robot whole speed demon sequence also director must patience saint came filming kiddy bad sequence usually directors hate working one kid let alone whole bunch performing complex dance scene bottom line movie people like mj one level another think people stay away try give wholesome message ironically mj bestest buddy movie girl michael jackson truly one talented people ever grace planet guilty well attention gave subject hmmm well know people different behind closed doors know fact either extremely nice stupid guy one sickest liars hope latter 
    
    watching time chasers obvious made bunch friends maybe sitting around one day film school said hey let pool money together make really bad movie something like ever said still ended making really bad movie dull story bad script lame acting poor cinematography bottom barrel stock music etc corners cut except one would prevented film release life like 
    
    naturally film main themes mortality nostalgia loss innocence perhaps surprising rated highly older viewers younger ones however craftsmanship completeness film anyone enjoy pace steady constant characters full engaging relationships interactions natural showing need floods tears show emotion screams show fear shouting show dispute violence show anger naturally joyce short story lends film ready made structure perfect polished diamond small changes huston makes inclusion poem fit neatly truly masterpiece tact subtlety overwhelming beauty
    

* 查看数据量


```python
print(len(train_data),len(unlable_data),len(test_data))
```

    25000 50000 25000
    

## 二、特征工程

把文本转换为向量，有几种常见的文本向量处理方法，比如：

* 1.单词计数
* 2.TF-IDF向量
* 3.Word2vec向量


```python
from sklearn.feature_extraction.text import CountVectorizer
X_test = ['I sed about sed the lack','of any Actually']

count_vec=CountVectorizer(stop_words=None)
print (count_vec.fit_transform(X_test).toarray())
print ('\nvocabulary list:\n\n',count_vec.vocabulary_)
```

    [[1 0 0 1 0 2 1]
     [0 1 1 0 1 0 0]]
    
    vocabulary list:
    
     {'sed': 5, 'about': 0, 'the': 6, 'lack': 3, 'of': 4, 'any': 2, 'actually': 1}
    


```python
news[0]
```


```python
data_all = [["naturally", "film", "main", "themes"],["you", "film", "main", "themes"]]
data_all[0]
```




    ['naturally', 'film', 'main', 'themes']




```python
#help(CountVectorizer)
```


```python
from sklearn.feature_extraction.text import CountVectorizer

count_vec = CountVectorizer(
    max_features=3,#过滤掉低频
    analyzer='char_wb', # tokenise by character ngrams
    ngram_range=None,  # 二元n-gram模型
    stop_words = 'english')
data_all = [["naturally", "film", "main", "themes"],["you", "film", "main", "themes"]]
count_vec.fit(data_all)
data_all = count_vec.transform(data_all)

```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-18-01f86357982d> in <module>()
          7     stop_words = 'english')
          8 data_all = [["naturally", "film", "main", "themes"],["you", "film", "main", "themes"]]
    ----> 9 count_vec.fit(data_all)
         10 data_all = count_vec.transform(data_all)
    

    ~\Anaconda3\lib\site-packages\sklearn\feature_extraction\text.py in fit(self, raw_documents, y)
        834         self
        835         """
    --> 836         self.fit_transform(raw_documents)
        837         return self
        838 
    

    ~\Anaconda3\lib\site-packages\sklearn\feature_extraction\text.py in fit_transform(self, raw_documents, y)
        867 
        868         vocabulary, X = self._count_vocab(raw_documents,
    --> 869                                           self.fixed_vocabulary_)
        870 
        871         if self.binary:
    

    ~\Anaconda3\lib\site-packages\sklearn\feature_extraction\text.py in _count_vocab(self, raw_documents, fixed_vocab)
        790         for doc in raw_documents:
        791             feature_counter = {}
    --> 792             for feature in analyze(doc):
        793                 try:
        794                     feature_idx = vocabulary[feature]
    

    ~\Anaconda3\lib\site-packages\sklearn\feature_extraction\text.py in <lambda>(doc)
        257         elif self.analyzer == 'char_wb':
        258             return lambda doc: self._char_wb_ngrams(
    --> 259                 preprocess(self.decode(doc)))
        260 
        261         elif self.analyzer == 'word':
    

    ~\Anaconda3\lib\site-packages\sklearn\feature_extraction\text.py in <lambda>(x)
        230 
        231         if self.lowercase:
    --> 232             return lambda x: strip_accents(x.lower())
        233         else:
        234             return strip_accents
    

    AttributeError: 'list' object has no attribute 'lower'


### 1.Count词向量


```python
from sklearn.feature_extraction.text import CountVectorizer

count_vec = CountVectorizer(
    max_features=4000,#过滤掉低频
    analyzer='word', # tokenise by character ngrams
    ngram_range=(1,2),  # 二元n-gram模型
    stop_words = 'english')

# 合并训练和测试集以便进行TFIDF向量化操作
data_all = train_data + test_data
len_train = len(train_data)

count_vec.fit(data_all)
data_all = count_vec.transform(data_all)

# 恢复成训练集和测试集部分
count_train_x = data_all[:len_train]
count_test_x = data_all[len_train:]

print('count处理结束.')

print("train: \n", np.shape(count_train_x[0]))
print("test: \n", np.shape(count_test_x[0]))
```

    count处理结束.
    train: 
     (1, 4000)
    test: 
     (1, 4000)
    


```python
count_train_x.shape
```




    (25000, 4000)




```python
count_test_x.shape
```




    (25000, 4000)



* Count特征
* count_train_x
* count_test_x

### 2.TF-IDF词向量


```python
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
"""
min_df: 最小支持度为2（词汇出现的最小次数）
max_features: 默认为None，可设为int，对所有关键词的term frequency进行降序排序，只取前max_features个作为关键词集
strip_accents: 将使用ascii或unicode编码在预处理步骤去除raw document中的重音符号
analyzer: 设置返回类型
token_pattern: 表示token的正则表达式，需要设置analyzer == 'word'，默认的正则表达式选择2个及以上的字母或数字作为token，标点符号默认当作token分隔符，而不会被当作token
ngram_range: 词组切分的长度范围
use_idf: 启用逆文档频率重新加权
use_idf：默认为True，权值是tf*idf，如果设为False，将不使用idf，就是只使用tf，相当于CountVectorizer了。
smooth_idf: idf平滑参数，默认为True，idf=ln((文档总数+1)/(包含该词的文档数+1))+1，如果设为False，idf=ln(文档总数/包含该词的文档数)+1
sublinear_tf: 默认为False，如果设为True，则替换tf为1 + log(tf)
stop_words: 设置停用词，设为english将使用内置的英语停用词，设为一个list可自定义停用词，设为None不使用停用词，设为None且max_df∈[0.7, 1.0)将自动根据当前的语料库建立停用词表
"""
tfidf = TFIDF(min_df=2,
           max_features=4000,#过滤掉低频
           strip_accents='unicode',
           analyzer='word',
           token_pattern=r'\w{1,}',
           ngram_range=(1,2),  # 二元n-gram模型
           use_idf=1,
           smooth_idf=1,
           sublinear_tf=1,
           stop_words = 'english') # 去掉英文停用词

# 合并训练和测试集以便进行TFIDF向量化操作
data_all = train_data + test_data
len_train = len(train_data)

tfidf.fit(data_all)
data_all = tfidf.transform(data_all)
# 恢复成训练集和测试集部分
tfidf_train_x = data_all[:len_train]
tfidf_test_x = data_all[len_train:]
print('TF-IDF处理结束.')

print("train: \n", np.shape(tfidf_train_x[0]))
print("test: \n", np.shape(tfidf_test_x[0]))
```

    TF-IDF处理结束.
    train: 
     (1, 4000)
    test: 
     (1, 4000)
    


```python
tfidf_train_x.shape
```




    (25000, 4000)




```python
tfidf_test_x.shape
```




    (25000, 4000)



### 3.Word2vec词向量

* gensim.models.word2vec.Word2Vec 输入数据是字符的list格式，所以需要对数据进行预处理

#### 3.1 输入数据预处理


```python
#预处理训练数据
train_words = []
for i in train_data:
    train_words.append(i.split())
    
#预处理特征数据
unlable_words = []
for i in unlable_data:
    unlable_words.append(i.split())

#预处理测试数据
test_words = []
for i in test_data:
    test_words.append(i.split())

#合并数据
all_words = train_words + unlable_words + test_words

len(all_words)
```




    100000



#### 3.2数据预览


```python
# 预览数据
print(all_words[0])
```

    ['stuff', 'going', 'moment', 'mj', 'started', 'listening', 'music', 'watching', 'odd', 'documentary', 'watched', 'wiz', 'watched', 'moonwalker', 'maybe', 'want', 'get', 'certain', 'insight', 'guy', 'thought', 'really', 'cool', 'eighties', 'maybe', 'make', 'mind', 'whether', 'guilty', 'innocent', 'moonwalker', 'part', 'biography', 'part', 'feature', 'film', 'remember', 'going', 'see', 'cinema', 'originally', 'released', 'subtle', 'messages', 'mj', 'feeling', 'towards', 'press', 'also', 'obvious', 'message', 'drugs', 'bad', 'kay', 'visually', 'impressive', 'course', 'michael', 'jackson', 'unless', 'remotely', 'like', 'mj', 'anyway', 'going', 'hate', 'find', 'boring', 'may', 'call', 'mj', 'egotist', 'consenting', 'making', 'movie', 'mj', 'fans', 'would', 'say', 'made', 'fans', 'true', 'really', 'nice', 'actual', 'feature', 'film', 'bit', 'finally', 'starts', 'minutes', 'excluding', 'smooth', 'criminal', 'sequence', 'joe', 'pesci', 'convincing', 'psychopathic', 'powerful', 'drug', 'lord', 'wants', 'mj', 'dead', 'bad', 'beyond', 'mj', 'overheard', 'plans', 'nah', 'joe', 'pesci', 'character', 'ranted', 'wanted', 'people', 'know', 'supplying', 'drugs', 'etc', 'dunno', 'maybe', 'hates', 'mj', 'music', 'lots', 'cool', 'things', 'like', 'mj', 'turning', 'car', 'robot', 'whole', 'speed', 'demon', 'sequence', 'also', 'director', 'must', 'patience', 'saint', 'came', 'filming', 'kiddy', 'bad', 'sequence', 'usually', 'directors', 'hate', 'working', 'one', 'kid', 'let', 'alone', 'whole', 'bunch', 'performing', 'complex', 'dance', 'scene', 'bottom', 'line', 'movie', 'people', 'like', 'mj', 'one', 'level', 'another', 'think', 'people', 'stay', 'away', 'try', 'give', 'wholesome', 'message', 'ironically', 'mj', 'bestest', 'buddy', 'movie', 'girl', 'michael', 'jackson', 'truly', 'one', 'talented', 'people', 'ever', 'grace', 'planet', 'guilty', 'well', 'attention', 'gave', 'subject', 'hmmm', 'well', 'know', 'people', 'different', 'behind', 'closed', 'doors', 'know', 'fact', 'either', 'extremely', 'nice', 'stupid', 'guy', 'one', 'sickest', 'liars', 'hope', 'latter']
    

#### 3.3 Word2vec模型训练保存


```python
from gensim.models.word2vec import Word2Vec
import os

# 设定词向量训练的参数
size = 100      # Word vector dimensionality
min_count = 3   # Minimum word count
num_workers = 4 # Number of threads to run in parallel
window = 10     # Context window size
model_name = '{}size_{}min_count_{}window.model'.format(size, min_count, window)

wv_model = Word2Vec(all_words, workers=num_workers, size=size, min_count = min_count,window = window)
wv_model.init_sims(replace=True)#模型训练好后，锁定模型

wv_model.save(model_name)#保存模型
```

    C:\Users\xuliang\Anaconda3\lib\site-packages\gensim\utils.py:1212: UserWarning: detected Windows; aliasing chunkize to chunkize_serial
      warnings.warn("detected Windows; aliasing chunkize to chunkize_serial")
    

#### 3.4 Word2vec模型加载


```python
from gensim.models.word2vec import Word2Vec
wv_model = Word2Vec.load("100size_3min_count_10window.model")
```

#### 3.5 word2vec特征处理

此处画风比较奇特:
* 将一个句子对应的词向量求和取平均，做为机器学习的特征，但是效果还不错。


```python
def to_review_vector(review):
    global word_vec
    word_vec = np.zeros((1,100))
    for word in review:
        if word in wv_model:
            word_vec += np.array([wv_model[word]])
    return pd.Series(word_vec.mean(axis = 0))

train_data_features = []

for i in train_words:
    train_data_features.append(to_review_vector(i))

test_data_features = []
for i in test_words:
    test_data_features.append(to_review_vector(i))
```

    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel\__main__.py:5: DeprecationWarning: Call to deprecated `__contains__` (Method will be removed in 4.0.0, use self.wv.__contains__() instead).
    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel\__main__.py:6: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).
    

# 机器学习方法

## 一、机器学习建模

### 1. TF-IDF+朴素贝叶斯模型+交叉验证


```python
# 朴素贝叶斯训练
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.naive_bayes import MultinomialNB as MNB

model_MNB = MNB() # (alpha=1.0, class_prior=None, fit_prior=True)
# 为了在预测的时候使用
model_MNB.fit(tfidf_train_x, label)

print("多项式贝叶斯分类器5折交叉验证得分: ", cross_val_score(model_MNB, tfidf_train_x, label, cv=5, scoring='roc_auc'))
print("多项式贝叶斯分类器5折交叉验证得分: ", np.mean(cross_val_score(model_MNB, tfidf_train_x, label, cv=5, scoring='roc_auc')))


test_predicted = np.array(model_MNB.predict(tfidf_test_x))

print('保存结果...')

submission_df = pd.DataFrame({'id': test['id'].values, 'sentiment': test_predicted})
print(submission_df.head())
submission_df.to_csv('submission_mnb_tfidf.csv', index = False)

print('结束.')
```

    多项式贝叶斯分类器5折交叉验证得分:  [0.92969136 0.93232992 0.93167488 0.93575024 0.92634304]
    多项式贝叶斯分类器5折交叉验证得分:  0.9311578880000001
    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          1
    3    7186_2          1
    4   12128_7          1
    结束.
    

![image.png](attachment:image.png)

![image.png](attachment:image.png)

### 2. TF-IDF+逻辑回归模型+网格搜索交叉验证


```python
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import GridSearchCV

# 设定grid search的参数
grid_values = {'C': [0.1, 1, 10]}  
# 设定打分为roc_auc
"""
penalty: l1 or l2, 用于指定惩罚中使用的标准。
"""
model_LR = GridSearchCV(LR(penalty='l2', dual=True, random_state=0), grid_values, scoring='roc_auc', cv=5)
model_LR.fit(tfidf_train_x, label)

# 输出结果
print("最好的参数：")
print( model_LR.best_params_)

print("最好的得分：")
print(model_LR.best_score_)


print("网格搜索参数及得分：")
print(model_LR.grid_scores_)

print("网格搜索结果：")
print(model_LR.cv_results_)

model_LR = LR(penalty='l2', dual=True, random_state=0)
model_LR.fit(tfidf_train_x, label)

test_predicted = np.array(model_LR.predict(tfidf_test_x))

print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_lr_tfidf.csv',columns = ['id','sentiment'], index = False)
print('结束.')
```

    最好的参数：
    {'C': 1}
    最好的得分：
    0.9507756479999999
    网格搜索参数及得分：
    [mean: 0.93949, std: 0.00324, params: {'C': 0.1}, mean: 0.95078, std: 0.00276, params: {'C': 1}, mean: 0.94398, std: 0.00275, params: {'C': 10}]
    网格搜索结果：
    {'mean_fit_time': array([0.12460012, 0.17419996, 0.4506001 ]), 'std_fit_time': array([0.01323034, 0.02166458, 0.01783922]), 'mean_score_time': array([0.0026    , 0.00319996, 0.00260005]), 'std_score_time': array([0.00048982, 0.00040011, 0.00048996]), 'param_C': masked_array(data=[0.1, 1, 10],
                 mask=[False, False, False],
           fill_value='?',
                dtype=object), 'params': [{'C': 0.1}, {'C': 1}, {'C': 10}], 'split0_test_score': array([0.93795456, 0.95020352, 0.9436768 ]), 'split1_test_score': array([0.94196192, 0.9542512 , 0.94876688]), 'split2_test_score': array([0.93759728, 0.9471152 , 0.9402704 ]), 'split3_test_score': array([0.9444336 , 0.95359056, 0.94413712]), 'split4_test_score': array([0.93548752, 0.94871776, 0.94303536]), 'mean_test_score': array([0.93948698, 0.95077565, 0.94397731]), 'std_test_score': array([0.00324066, 0.00275551, 0.00274533]), 'rank_test_score': array([3, 1, 2]), 'split0_train_score': array([0.9499337 , 0.97234608, 0.9860402 ]), 'split1_train_score': array([0.94941395, 0.97180812, 0.98526184]), 'split2_train_score': array([0.95103983, 0.97310123, 0.98656722]), 'split3_train_score': array([0.94978255, 0.97232675, 0.98596239]), 'split4_train_score': array([0.95038084, 0.97259464, 0.98615951]), 'mean_train_score': array([0.95011017, 0.97243536, 0.98599823]), 'std_train_score': array([0.0005587 , 0.00041999, 0.0004231 ])}
    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          1
    3    7186_2          1
    4   12128_7          1
    

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_search.py:761: DeprecationWarning: The grid_scores_ attribute was deprecated in version 0.18 in favor of the more elaborate cv_results_ attribute. The grid_scores_ attribute will not be available from 0.20
      DeprecationWarning)
    

    结束.
    

![image.png](attachment:image.png)

![image.png](attachment:image.png)

### 3. TF-IDF+SVM模型+网格搜索交叉验证

* SVM模型训练太耗时，尤其是使用网格搜索训练
* 参数调优时，使用param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}+5折交叉验证，连续训练时长将近48小时。


```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

'''
线性的SVM只需要，只需要调优正则化参数C
基于RBF核的SVM，需要调优gamma参数和C
'''
param_grid = {'C': [10],'gamma': [1]}

model_SVM = GridSearchCV(SVC(), param_grid, scoring='roc_auc', cv=5)
model_SVM.fit(tfidf_train_x, label)

# 输出结果
print("最好的参数：")
print( model_SVM.best_params_)

print("最好的得分：")
print(model_SVM.best_score_)

print("网格搜索参数及得分：")
print(model_SVM.grid_scores_)

```

    最好的参数：
    {'C': 10, 'gamma': 1}
    最好的得分：
    0.9517343039999999
    网格搜索参数及得分：
    [mean: 0.95173, std: 0.00264, params: {'C': 10, 'gamma': 1}]
    


```python
from sklearn.externals import joblib
joblib.dump(model_SVM, "model_SVM")
model_SVM = joblib.load("model_SVM")
```


```python
from sklearn.svm import SVC
svm = SVC(kernel='linear',C=10,gamma = 1)
svm.fit(tfidf_train_x, label)

test_predicted = np.array(svm.predict(tfidf_test_x))
print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_svm_tfidf.csv',columns = ['id','sentiment'], index = False)
print('结束.')
```

    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          0
    3    7186_2          0
    4   12128_7          1
    结束.
    

![image.png](attachment:image.png)

### 4. TF-IDF+MLP（多层感知机模型）


```python
#None 维度在这里是一个 batch size 的占位符

import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split

#train test split
mlp_train_x, mlp_test_x, mlp_train_y, mlp_test_y = train_test_split(tfidf_train_x, label, test_size=0.3, random_state=123)

model_MLP = Sequential()
#model.add(Dense(3, activation='relu', input_shape=(18,)))

model_MLP.add(Dense(10, input_shape=(4000,), activation='relu'))#

model_MLP.add(Dropout(0.2))
model_MLP.add(Dense(2, activation='softmax'))

model_MLP.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model_MLP.summary()

keras_y_train = np.array(keras.utils.to_categorical(mlp_train_y, 2))
keras_y_test = np.array(keras.utils.to_categorical(mlp_test_y, 2))

#仅保存最好的模型
filepath="model_MLP/weights.best.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,mode='max')#验证集准确率比之前效果好就保存权重

callbacks_list = [checkpoint]

model_MLP.fit(mlp_train_x, keras_y_train, validation_data=(mlp_test_x, keras_y_test), epochs=500, batch_size=5000,callbacks=callbacks_list)
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_9 (Dense)              (None, 10)                40010     
    _________________________________________________________________
    dropout_5 (Dropout)          (None, 10)                0         
    _________________________________________________________________
    dense_10 (Dense)             (None, 2)                 22        
    =================================================================
    Total params: 40,032
    Trainable params: 40,032
    Non-trainable params: 0
    _________________________________________________________________
    Train on 17500 samples, validate on 7500 samples
    Epoch 1/500
    17500/17500 [==============================] - 1s 61us/step - loss: 0.6908 - acc: 0.5581 - val_loss: 0.6856 - val_acc: 0.6764
    
    Epoch 00001: val_acc improved from -inf to 0.67640, saving model to model_MLP/weights.best.hdf5
    Epoch 2/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.6817 - acc: 0.6987 - val_loss: 0.6745 - val_acc: 0.7712
    
    Epoch 00002: val_acc improved from 0.67640 to 0.77120, saving model to model_MLP/weights.best.hdf5
    Epoch 3/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.6692 - acc: 0.7624 - val_loss: 0.6611 - val_acc: 0.8071
    
    Epoch 00003: val_acc improved from 0.77120 to 0.80707, saving model to model_MLP/weights.best.hdf5
    Epoch 4/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.6550 - acc: 0.7938 - val_loss: 0.6477 - val_acc: 0.8277
    
    Epoch 00004: val_acc improved from 0.80707 to 0.82773, saving model to model_MLP/weights.best.hdf5
    Epoch 5/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.6418 - acc: 0.8066 - val_loss: 0.6346 - val_acc: 0.8359
    
    Epoch 00005: val_acc improved from 0.82773 to 0.83587, saving model to model_MLP/weights.best.hdf5
    Epoch 6/500
    17500/17500 [==============================] - 1s 45us/step - loss: 0.6281 - acc: 0.8171 - val_loss: 0.6217 - val_acc: 0.8369
    
    Epoch 00006: val_acc improved from 0.83587 to 0.83693, saving model to model_MLP/weights.best.hdf5
    Epoch 7/500
    17500/17500 [==============================] - 1s 46us/step - loss: 0.6149 - acc: 0.8209 - val_loss: 0.6089 - val_acc: 0.8401
    
    Epoch 00007: val_acc improved from 0.83693 to 0.84013, saving model to model_MLP/weights.best.hdf5
    Epoch 8/500
    17500/17500 [==============================] - 1s 46us/step - loss: 0.6011 - acc: 0.8323 - val_loss: 0.5965 - val_acc: 0.8415
    
    Epoch 00008: val_acc improved from 0.84013 to 0.84147, saving model to model_MLP/weights.best.hdf5
    Epoch 9/500
    17500/17500 [==============================] - 1s 45us/step - loss: 0.5881 - acc: 0.8338 - val_loss: 0.5843 - val_acc: 0.8412
    
    Epoch 00009: val_acc did not improve from 0.84147
    Epoch 10/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.5755 - acc: 0.8350 - val_loss: 0.5725 - val_acc: 0.8419
    
    Epoch 00010: val_acc improved from 0.84147 to 0.84187, saving model to model_MLP/weights.best.hdf5
    Epoch 11/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.5631 - acc: 0.8406 - val_loss: 0.5609 - val_acc: 0.8424
    
    Epoch 00011: val_acc improved from 0.84187 to 0.84240, saving model to model_MLP/weights.best.hdf5
    Epoch 12/500
    17500/17500 [==============================] - 1s 45us/step - loss: 0.5516 - acc: 0.8417 - val_loss: 0.5496 - val_acc: 0.8435
    
    Epoch 00012: val_acc improved from 0.84240 to 0.84347, saving model to model_MLP/weights.best.hdf5
    Epoch 13/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.5403 - acc: 0.8417 - val_loss: 0.5386 - val_acc: 0.8452
    
    Epoch 00013: val_acc improved from 0.84347 to 0.84520, saving model to model_MLP/weights.best.hdf5
    Epoch 14/500
    17500/17500 [==============================] - 1s 44us/step - loss: 0.5287 - acc: 0.8470 - val_loss: 0.5279 - val_acc: 0.8463
    
    Epoch 00014: val_acc improved from 0.84520 to 0.84627, saving model to model_MLP/weights.best.hdf5
    Epoch 15/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.5177 - acc: 0.8456 - val_loss: 0.5176 - val_acc: 0.8475
    
    Epoch 00015: val_acc improved from 0.84627 to 0.84747, saving model to model_MLP/weights.best.hdf5
    Epoch 16/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.5073 - acc: 0.8466 - val_loss: 0.5076 - val_acc: 0.8477
    
    Epoch 00016: val_acc improved from 0.84747 to 0.84773, saving model to model_MLP/weights.best.hdf5
    Epoch 17/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.4961 - acc: 0.8530 - val_loss: 0.4979 - val_acc: 0.8485
    
    Epoch 00017: val_acc improved from 0.84773 to 0.84853, saving model to model_MLP/weights.best.hdf5
    Epoch 18/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.4866 - acc: 0.8547 - val_loss: 0.4886 - val_acc: 0.8496
    
    Epoch 00018: val_acc improved from 0.84853 to 0.84960, saving model to model_MLP/weights.best.hdf5
    Epoch 19/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.4771 - acc: 0.8526 - val_loss: 0.4797 - val_acc: 0.8508
    
    Epoch 00019: val_acc improved from 0.84960 to 0.85080, saving model to model_MLP/weights.best.hdf5
    Epoch 20/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.4684 - acc: 0.8599 - val_loss: 0.4711 - val_acc: 0.8516
    
    Epoch 00020: val_acc improved from 0.85080 to 0.85160, saving model to model_MLP/weights.best.hdf5
    Epoch 21/500
    17500/17500 [==============================] - 1s 44us/step - loss: 0.4575 - acc: 0.8591 - val_loss: 0.4629 - val_acc: 0.8523
    
    Epoch 00021: val_acc improved from 0.85160 to 0.85227, saving model to model_MLP/weights.best.hdf5
    Epoch 22/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.4497 - acc: 0.8609 - val_loss: 0.4549 - val_acc: 0.8535
    
    Epoch 00022: val_acc improved from 0.85227 to 0.85347, saving model to model_MLP/weights.best.hdf5
    Epoch 23/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.4417 - acc: 0.8625 - val_loss: 0.4473 - val_acc: 0.8547
    
    Epoch 00023: val_acc improved from 0.85347 to 0.85467, saving model to model_MLP/weights.best.hdf5
    Epoch 24/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.4333 - acc: 0.8638 - val_loss: 0.4401 - val_acc: 0.8556
    
    Epoch 00024: val_acc improved from 0.85467 to 0.85560, saving model to model_MLP/weights.best.hdf5
    Epoch 25/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.4257 - acc: 0.8643 - val_loss: 0.4332 - val_acc: 0.8571
    
    Epoch 00025: val_acc improved from 0.85560 to 0.85707, saving model to model_MLP/weights.best.hdf5
    Epoch 26/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.4180 - acc: 0.8694 - val_loss: 0.4266 - val_acc: 0.8584
    
    Epoch 00026: val_acc improved from 0.85707 to 0.85840, saving model to model_MLP/weights.best.hdf5
    Epoch 27/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.4109 - acc: 0.8686 - val_loss: 0.4203 - val_acc: 0.8585
    
    Epoch 00027: val_acc improved from 0.85840 to 0.85853, saving model to model_MLP/weights.best.hdf5
    Epoch 28/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.4055 - acc: 0.8694 - val_loss: 0.4143 - val_acc: 0.8588
    
    Epoch 00028: val_acc improved from 0.85853 to 0.85880, saving model to model_MLP/weights.best.hdf5
    Epoch 29/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.3995 - acc: 0.8695 - val_loss: 0.4086 - val_acc: 0.8592
    
    Epoch 00029: val_acc improved from 0.85880 to 0.85920, saving model to model_MLP/weights.best.hdf5
    Epoch 30/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.3913 - acc: 0.8742 - val_loss: 0.4031 - val_acc: 0.8600
    
    Epoch 00030: val_acc improved from 0.85920 to 0.86000, saving model to model_MLP/weights.best.hdf5
    Epoch 31/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.3854 - acc: 0.8743 - val_loss: 0.3979 - val_acc: 0.8607
    
    Epoch 00031: val_acc improved from 0.86000 to 0.86067, saving model to model_MLP/weights.best.hdf5
    Epoch 32/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.3807 - acc: 0.8782 - val_loss: 0.3929 - val_acc: 0.8615
    
    Epoch 00032: val_acc improved from 0.86067 to 0.86147, saving model to model_MLP/weights.best.hdf5
    Epoch 33/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.3745 - acc: 0.8783 - val_loss: 0.3881 - val_acc: 0.8625
    
    Epoch 00033: val_acc improved from 0.86147 to 0.86253, saving model to model_MLP/weights.best.hdf5
    Epoch 34/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.3700 - acc: 0.8789 - val_loss: 0.3836 - val_acc: 0.8627
    
    Epoch 00034: val_acc improved from 0.86253 to 0.86267, saving model to model_MLP/weights.best.hdf5
    Epoch 35/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.3636 - acc: 0.8813 - val_loss: 0.3793 - val_acc: 0.8631
    
    Epoch 00035: val_acc improved from 0.86267 to 0.86307, saving model to model_MLP/weights.best.hdf5
    Epoch 36/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.3589 - acc: 0.8809 - val_loss: 0.3752 - val_acc: 0.8633
    
    Epoch 00036: val_acc improved from 0.86307 to 0.86333, saving model to model_MLP/weights.best.hdf5
    Epoch 37/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.3551 - acc: 0.8802 - val_loss: 0.3713 - val_acc: 0.8641
    
    Epoch 00037: val_acc improved from 0.86333 to 0.86413, saving model to model_MLP/weights.best.hdf5
    Epoch 38/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.3497 - acc: 0.8815 - val_loss: 0.3675 - val_acc: 0.8651
    
    Epoch 00038: val_acc improved from 0.86413 to 0.86507, saving model to model_MLP/weights.best.hdf5
    Epoch 39/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.3448 - acc: 0.8849 - val_loss: 0.3640 - val_acc: 0.8656
    
    Epoch 00039: val_acc improved from 0.86507 to 0.86560, saving model to model_MLP/weights.best.hdf5
    Epoch 40/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.3405 - acc: 0.8866 - val_loss: 0.3606 - val_acc: 0.8661
    
    Epoch 00040: val_acc improved from 0.86560 to 0.86613, saving model to model_MLP/weights.best.hdf5
    Epoch 41/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.3371 - acc: 0.8853 - val_loss: 0.3574 - val_acc: 0.8667
    
    Epoch 00041: val_acc improved from 0.86613 to 0.86667, saving model to model_MLP/weights.best.hdf5
    Epoch 42/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.3325 - acc: 0.8859 - val_loss: 0.3543 - val_acc: 0.8668
    
    Epoch 00042: val_acc improved from 0.86667 to 0.86680, saving model to model_MLP/weights.best.hdf5
    Epoch 43/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.3284 - acc: 0.8891 - val_loss: 0.3513 - val_acc: 0.8675
    
    Epoch 00043: val_acc improved from 0.86680 to 0.86747, saving model to model_MLP/weights.best.hdf5
    Epoch 44/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.3252 - acc: 0.8885 - val_loss: 0.3485 - val_acc: 0.8672
    
    Epoch 00044: val_acc did not improve from 0.86747
    Epoch 45/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.3209 - acc: 0.8937 - val_loss: 0.3459 - val_acc: 0.8675
    
    Epoch 00045: val_acc did not improve from 0.86747
    Epoch 46/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.3178 - acc: 0.8927 - val_loss: 0.3433 - val_acc: 0.8681
    
    Epoch 00046: val_acc improved from 0.86747 to 0.86813, saving model to model_MLP/weights.best.hdf5
    Epoch 47/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.3149 - acc: 0.8923 - val_loss: 0.3409 - val_acc: 0.8681
    
    Epoch 00047: val_acc did not improve from 0.86813
    Epoch 48/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.3095 - acc: 0.8950 - val_loss: 0.3386 - val_acc: 0.8691
    
    Epoch 00048: val_acc improved from 0.86813 to 0.86907, saving model to model_MLP/weights.best.hdf5
    Epoch 49/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.3087 - acc: 0.8928 - val_loss: 0.3364 - val_acc: 0.8695
    
    Epoch 00049: val_acc improved from 0.86907 to 0.86947, saving model to model_MLP/weights.best.hdf5
    Epoch 50/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.3048 - acc: 0.8954 - val_loss: 0.3343 - val_acc: 0.8699
    
    Epoch 00050: val_acc improved from 0.86947 to 0.86987, saving model to model_MLP/weights.best.hdf5
    Epoch 51/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.3035 - acc: 0.8936 - val_loss: 0.3323 - val_acc: 0.8709
    
    Epoch 00051: val_acc improved from 0.86987 to 0.87093, saving model to model_MLP/weights.best.hdf5
    Epoch 52/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2987 - acc: 0.8949 - val_loss: 0.3304 - val_acc: 0.8701
    
    Epoch 00052: val_acc did not improve from 0.87093
    Epoch 53/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2956 - acc: 0.8969 - val_loss: 0.3286 - val_acc: 0.8711
    
    Epoch 00053: val_acc improved from 0.87093 to 0.87107, saving model to model_MLP/weights.best.hdf5
    Epoch 54/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2938 - acc: 0.8994 - val_loss: 0.3268 - val_acc: 0.8709
    
    Epoch 00054: val_acc did not improve from 0.87107
    Epoch 55/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2897 - acc: 0.8982 - val_loss: 0.3251 - val_acc: 0.8708
    
    Epoch 00055: val_acc did not improve from 0.87107
    Epoch 56/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2885 - acc: 0.8987 - val_loss: 0.3235 - val_acc: 0.8705
    
    Epoch 00056: val_acc did not improve from 0.87107
    Epoch 57/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2864 - acc: 0.8997 - val_loss: 0.3220 - val_acc: 0.8709
    
    Epoch 00057: val_acc did not improve from 0.87107
    Epoch 58/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2836 - acc: 0.9011 - val_loss: 0.3205 - val_acc: 0.8709
    
    Epoch 00058: val_acc did not improve from 0.87107
    Epoch 59/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2805 - acc: 0.9025 - val_loss: 0.3191 - val_acc: 0.8712
    
    Epoch 00059: val_acc improved from 0.87107 to 0.87120, saving model to model_MLP/weights.best.hdf5
    Epoch 60/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2791 - acc: 0.9005 - val_loss: 0.3178 - val_acc: 0.8713
    
    Epoch 00060: val_acc improved from 0.87120 to 0.87133, saving model to model_MLP/weights.best.hdf5
    Epoch 61/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2769 - acc: 0.9037 - val_loss: 0.3165 - val_acc: 0.8716
    
    Epoch 00061: val_acc improved from 0.87133 to 0.87160, saving model to model_MLP/weights.best.hdf5
    Epoch 62/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2749 - acc: 0.9048 - val_loss: 0.3153 - val_acc: 0.8719
    
    Epoch 00062: val_acc improved from 0.87160 to 0.87187, saving model to model_MLP/weights.best.hdf5
    Epoch 63/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2714 - acc: 0.9063 - val_loss: 0.3142 - val_acc: 0.8720
    
    Epoch 00063: val_acc improved from 0.87187 to 0.87200, saving model to model_MLP/weights.best.hdf5
    Epoch 64/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2698 - acc: 0.9057 - val_loss: 0.3130 - val_acc: 0.8724
    
    Epoch 00064: val_acc improved from 0.87200 to 0.87240, saving model to model_MLP/weights.best.hdf5
    Epoch 65/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2683 - acc: 0.9051 - val_loss: 0.3120 - val_acc: 0.8724
    
    Epoch 00065: val_acc did not improve from 0.87240
    Epoch 66/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2668 - acc: 0.9073 - val_loss: 0.3109 - val_acc: 0.8732
    
    Epoch 00066: val_acc improved from 0.87240 to 0.87320, saving model to model_MLP/weights.best.hdf5
    Epoch 67/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2649 - acc: 0.9071 - val_loss: 0.3100 - val_acc: 0.8740
    
    Epoch 00067: val_acc improved from 0.87320 to 0.87400, saving model to model_MLP/weights.best.hdf5
    Epoch 68/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.2633 - acc: 0.9078 - val_loss: 0.3091 - val_acc: 0.8743
    
    Epoch 00068: val_acc improved from 0.87400 to 0.87427, saving model to model_MLP/weights.best.hdf5
    Epoch 69/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2610 - acc: 0.9079 - val_loss: 0.3083 - val_acc: 0.8736
    
    Epoch 00069: val_acc did not improve from 0.87427
    Epoch 70/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2589 - acc: 0.9078 - val_loss: 0.3075 - val_acc: 0.8741
    
    Epoch 00070: val_acc did not improve from 0.87427
    Epoch 71/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.2557 - acc: 0.9075 - val_loss: 0.3067 - val_acc: 0.8736
    
    Epoch 00071: val_acc did not improve from 0.87427
    Epoch 72/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2549 - acc: 0.9093 - val_loss: 0.3059 - val_acc: 0.8739
    
    Epoch 00072: val_acc did not improve from 0.87427
    Epoch 73/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2541 - acc: 0.9108 - val_loss: 0.3053 - val_acc: 0.8744
    
    Epoch 00073: val_acc improved from 0.87427 to 0.87440, saving model to model_MLP/weights.best.hdf5
    Epoch 74/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2530 - acc: 0.9099 - val_loss: 0.3046 - val_acc: 0.8744
    
    Epoch 00074: val_acc did not improve from 0.87440
    Epoch 75/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2520 - acc: 0.9086 - val_loss: 0.3040 - val_acc: 0.8744
    
    Epoch 00075: val_acc improved from 0.87440 to 0.87440, saving model to model_MLP/weights.best.hdf5
    Epoch 76/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2503 - acc: 0.9101 - val_loss: 0.3035 - val_acc: 0.8743
    
    Epoch 00076: val_acc did not improve from 0.87440
    Epoch 77/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.2467 - acc: 0.9131 - val_loss: 0.3030 - val_acc: 0.8745
    
    Epoch 00077: val_acc improved from 0.87440 to 0.87453, saving model to model_MLP/weights.best.hdf5
    Epoch 78/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2445 - acc: 0.9133 - val_loss: 0.3024 - val_acc: 0.8743
    
    Epoch 00078: val_acc did not improve from 0.87453
    Epoch 79/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2436 - acc: 0.9149 - val_loss: 0.3019 - val_acc: 0.8743
    
    Epoch 00079: val_acc did not improve from 0.87453
    Epoch 80/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2428 - acc: 0.9138 - val_loss: 0.3015 - val_acc: 0.8741
    
    Epoch 00080: val_acc did not improve from 0.87453
    Epoch 81/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2411 - acc: 0.9147 - val_loss: 0.3011 - val_acc: 0.8741
    
    Epoch 00081: val_acc did not improve from 0.87453
    Epoch 82/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2404 - acc: 0.9144 - val_loss: 0.3006 - val_acc: 0.8733
    
    Epoch 00082: val_acc did not improve from 0.87453
    Epoch 83/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2376 - acc: 0.9177 - val_loss: 0.3002 - val_acc: 0.8736
    
    Epoch 00083: val_acc did not improve from 0.87453
    Epoch 84/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2375 - acc: 0.9154 - val_loss: 0.2999 - val_acc: 0.8741
    
    Epoch 00084: val_acc did not improve from 0.87453
    Epoch 85/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2351 - acc: 0.9141 - val_loss: 0.2996 - val_acc: 0.8743
    
    Epoch 00085: val_acc did not improve from 0.87453
    Epoch 86/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2350 - acc: 0.9166 - val_loss: 0.2993 - val_acc: 0.8741
    
    Epoch 00086: val_acc did not improve from 0.87453
    Epoch 87/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2306 - acc: 0.9203 - val_loss: 0.2990 - val_acc: 0.8743
    
    Epoch 00087: val_acc did not improve from 0.87453
    Epoch 88/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2303 - acc: 0.9187 - val_loss: 0.2988 - val_acc: 0.8744
    
    Epoch 00088: val_acc did not improve from 0.87453
    Epoch 89/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2293 - acc: 0.9183 - val_loss: 0.2985 - val_acc: 0.8741
    
    Epoch 00089: val_acc did not improve from 0.87453
    Epoch 90/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2288 - acc: 0.9179 - val_loss: 0.2983 - val_acc: 0.8745
    
    Epoch 00090: val_acc did not improve from 0.87453
    Epoch 91/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2275 - acc: 0.9187 - val_loss: 0.2982 - val_acc: 0.8743
    
    Epoch 00091: val_acc did not improve from 0.87453
    Epoch 92/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2287 - acc: 0.9174 - val_loss: 0.2980 - val_acc: 0.8747
    
    Epoch 00092: val_acc improved from 0.87453 to 0.87467, saving model to model_MLP/weights.best.hdf5
    Epoch 93/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2255 - acc: 0.9207 - val_loss: 0.2979 - val_acc: 0.8745
    
    Epoch 00093: val_acc did not improve from 0.87467
    Epoch 94/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2233 - acc: 0.9187 - val_loss: 0.2978 - val_acc: 0.8736
    
    Epoch 00094: val_acc did not improve from 0.87467
    Epoch 95/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2221 - acc: 0.9197 - val_loss: 0.2976 - val_acc: 0.8739
    
    Epoch 00095: val_acc did not improve from 0.87467
    Epoch 96/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2215 - acc: 0.9206 - val_loss: 0.2975 - val_acc: 0.8737
    
    Epoch 00096: val_acc did not improve from 0.87467
    Epoch 97/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2196 - acc: 0.9231 - val_loss: 0.2974 - val_acc: 0.8736
    
    Epoch 00097: val_acc did not improve from 0.87467
    Epoch 98/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2208 - acc: 0.9213 - val_loss: 0.2974 - val_acc: 0.8736
    
    Epoch 00098: val_acc did not improve from 0.87467
    Epoch 99/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2205 - acc: 0.9223 - val_loss: 0.2973 - val_acc: 0.8741
    
    Epoch 00099: val_acc did not improve from 0.87467
    Epoch 100/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2171 - acc: 0.9238 - val_loss: 0.2974 - val_acc: 0.8736
    
    Epoch 00100: val_acc did not improve from 0.87467
    Epoch 101/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2158 - acc: 0.9245 - val_loss: 0.2974 - val_acc: 0.8735
    
    Epoch 00101: val_acc did not improve from 0.87467
    Epoch 102/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2156 - acc: 0.9246 - val_loss: 0.2974 - val_acc: 0.8729
    
    Epoch 00102: val_acc did not improve from 0.87467
    Epoch 103/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2149 - acc: 0.9249 - val_loss: 0.2974 - val_acc: 0.8731
    
    Epoch 00103: val_acc did not improve from 0.87467
    Epoch 104/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2126 - acc: 0.9246 - val_loss: 0.2974 - val_acc: 0.8737
    
    Epoch 00104: val_acc did not improve from 0.87467
    Epoch 105/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2112 - acc: 0.9265 - val_loss: 0.2975 - val_acc: 0.8736
    
    Epoch 00105: val_acc did not improve from 0.87467
    Epoch 106/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2133 - acc: 0.9234 - val_loss: 0.2975 - val_acc: 0.8736
    
    Epoch 00106: val_acc did not improve from 0.87467
    Epoch 107/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2100 - acc: 0.9274 - val_loss: 0.2976 - val_acc: 0.8733
    
    Epoch 00107: val_acc did not improve from 0.87467
    Epoch 108/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2089 - acc: 0.9280 - val_loss: 0.2977 - val_acc: 0.8739
    
    Epoch 00108: val_acc did not improve from 0.87467
    Epoch 109/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2072 - acc: 0.9271 - val_loss: 0.2978 - val_acc: 0.8739
    
    Epoch 00109: val_acc did not improve from 0.87467
    Epoch 110/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2094 - acc: 0.9256 - val_loss: 0.2979 - val_acc: 0.8735
    
    Epoch 00110: val_acc did not improve from 0.87467
    Epoch 111/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2078 - acc: 0.9255 - val_loss: 0.2980 - val_acc: 0.8735
    
    Epoch 00111: val_acc did not improve from 0.87467
    Epoch 112/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2054 - acc: 0.9269 - val_loss: 0.2981 - val_acc: 0.8735
    
    Epoch 00112: val_acc did not improve from 0.87467
    Epoch 113/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2042 - acc: 0.9279 - val_loss: 0.2983 - val_acc: 0.8732
    
    Epoch 00113: val_acc did not improve from 0.87467
    Epoch 114/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2053 - acc: 0.9266 - val_loss: 0.2984 - val_acc: 0.8735
    
    Epoch 00114: val_acc did not improve from 0.87467
    Epoch 115/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2023 - acc: 0.9285 - val_loss: 0.2986 - val_acc: 0.8727
    
    Epoch 00115: val_acc did not improve from 0.87467
    Epoch 116/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2015 - acc: 0.9292 - val_loss: 0.2988 - val_acc: 0.8731
    
    Epoch 00116: val_acc did not improve from 0.87467
    Epoch 117/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2012 - acc: 0.9282 - val_loss: 0.2989 - val_acc: 0.8728
    
    Epoch 00117: val_acc did not improve from 0.87467
    Epoch 118/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2008 - acc: 0.9280 - val_loss: 0.2991 - val_acc: 0.8725
    
    Epoch 00118: val_acc did not improve from 0.87467
    Epoch 119/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.2006 - acc: 0.9296 - val_loss: 0.2993 - val_acc: 0.8724
    
    Epoch 00119: val_acc did not improve from 0.87467
    Epoch 120/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1995 - acc: 0.9309 - val_loss: 0.2995 - val_acc: 0.8728
    
    Epoch 00120: val_acc did not improve from 0.87467
    Epoch 121/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1969 - acc: 0.9300 - val_loss: 0.2997 - val_acc: 0.8725
    
    Epoch 00121: val_acc did not improve from 0.87467
    Epoch 122/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1975 - acc: 0.9303 - val_loss: 0.2999 - val_acc: 0.8725
    
    Epoch 00122: val_acc did not improve from 0.87467
    Epoch 123/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1981 - acc: 0.9289 - val_loss: 0.3001 - val_acc: 0.8725
    
    Epoch 00123: val_acc did not improve from 0.87467
    Epoch 124/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1952 - acc: 0.9300 - val_loss: 0.3004 - val_acc: 0.8725
    
    Epoch 00124: val_acc did not improve from 0.87467
    Epoch 125/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1956 - acc: 0.9297 - val_loss: 0.3006 - val_acc: 0.8724
    
    Epoch 00125: val_acc did not improve from 0.87467
    Epoch 126/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1955 - acc: 0.9307 - val_loss: 0.3009 - val_acc: 0.8724
    
    Epoch 00126: val_acc did not improve from 0.87467
    Epoch 127/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1921 - acc: 0.9331 - val_loss: 0.3011 - val_acc: 0.8723
    
    Epoch 00127: val_acc did not improve from 0.87467
    Epoch 128/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1927 - acc: 0.9311 - val_loss: 0.3014 - val_acc: 0.8728
    
    Epoch 00128: val_acc did not improve from 0.87467
    Epoch 129/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1918 - acc: 0.9331 - val_loss: 0.3017 - val_acc: 0.8723
    
    Epoch 00129: val_acc did not improve from 0.87467
    Epoch 130/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.1918 - acc: 0.9325 - val_loss: 0.3020 - val_acc: 0.8721
    
    Epoch 00130: val_acc did not improve from 0.87467
    Epoch 131/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1917 - acc: 0.9310 - val_loss: 0.3023 - val_acc: 0.8720
    
    Epoch 00131: val_acc did not improve from 0.87467
    Epoch 132/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1896 - acc: 0.9332 - val_loss: 0.3026 - val_acc: 0.8720
    
    Epoch 00132: val_acc did not improve from 0.87467
    Epoch 133/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1879 - acc: 0.9346 - val_loss: 0.3029 - val_acc: 0.8719
    
    Epoch 00133: val_acc did not improve from 0.87467
    Epoch 134/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1882 - acc: 0.9350 - val_loss: 0.3032 - val_acc: 0.8719
    
    Epoch 00134: val_acc did not improve from 0.87467
    Epoch 135/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1865 - acc: 0.9362 - val_loss: 0.3035 - val_acc: 0.8713
    
    Epoch 00135: val_acc did not improve from 0.87467
    Epoch 136/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1866 - acc: 0.9323 - val_loss: 0.3038 - val_acc: 0.8712
    
    Epoch 00136: val_acc did not improve from 0.87467
    Epoch 137/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1859 - acc: 0.9342 - val_loss: 0.3042 - val_acc: 0.8712
    
    Epoch 00137: val_acc did not improve from 0.87467
    Epoch 138/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1851 - acc: 0.9339 - val_loss: 0.3045 - val_acc: 0.8712
    
    Epoch 00138: val_acc did not improve from 0.87467
    Epoch 139/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1863 - acc: 0.9349 - val_loss: 0.3048 - val_acc: 0.8711
    
    Epoch 00139: val_acc did not improve from 0.87467
    Epoch 140/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1843 - acc: 0.9351 - val_loss: 0.3052 - val_acc: 0.8712
    
    Epoch 00140: val_acc did not improve from 0.87467
    Epoch 141/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1832 - acc: 0.9342 - val_loss: 0.3056 - val_acc: 0.8712
    
    Epoch 00141: val_acc did not improve from 0.87467
    Epoch 142/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1820 - acc: 0.9376 - val_loss: 0.3059 - val_acc: 0.8712
    
    Epoch 00142: val_acc did not improve from 0.87467
    Epoch 143/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1818 - acc: 0.9364 - val_loss: 0.3063 - val_acc: 0.8709
    
    Epoch 00143: val_acc did not improve from 0.87467
    Epoch 144/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1801 - acc: 0.9383 - val_loss: 0.3067 - val_acc: 0.8709
    
    Epoch 00144: val_acc did not improve from 0.87467
    Epoch 145/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1818 - acc: 0.9366 - val_loss: 0.3072 - val_acc: 0.8708
    
    Epoch 00145: val_acc did not improve from 0.87467
    Epoch 146/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1812 - acc: 0.9358 - val_loss: 0.3076 - val_acc: 0.8699
    
    Epoch 00146: val_acc did not improve from 0.87467
    Epoch 147/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1788 - acc: 0.9364 - val_loss: 0.3080 - val_acc: 0.8703
    
    Epoch 00147: val_acc did not improve from 0.87467
    Epoch 148/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1782 - acc: 0.9392 - val_loss: 0.3084 - val_acc: 0.8700
    
    Epoch 00148: val_acc did not improve from 0.87467
    Epoch 149/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1774 - acc: 0.9393 - val_loss: 0.3088 - val_acc: 0.8696
    
    Epoch 00149: val_acc did not improve from 0.87467
    Epoch 150/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1776 - acc: 0.9367 - val_loss: 0.3093 - val_acc: 0.8696
    
    Epoch 00150: val_acc did not improve from 0.87467
    Epoch 151/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1774 - acc: 0.9371 - val_loss: 0.3097 - val_acc: 0.8692
    
    Epoch 00151: val_acc did not improve from 0.87467
    Epoch 152/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1753 - acc: 0.9399 - val_loss: 0.3101 - val_acc: 0.8688
    
    Epoch 00152: val_acc did not improve from 0.87467
    Epoch 153/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1766 - acc: 0.9366 - val_loss: 0.3106 - val_acc: 0.8693
    
    Epoch 00153: val_acc did not improve from 0.87467
    Epoch 154/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1768 - acc: 0.9385 - val_loss: 0.3109 - val_acc: 0.8691
    
    Epoch 00154: val_acc did not improve from 0.87467
    Epoch 155/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1743 - acc: 0.9376 - val_loss: 0.3114 - val_acc: 0.8692
    
    Epoch 00155: val_acc did not improve from 0.87467
    Epoch 156/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1742 - acc: 0.9410 - val_loss: 0.3118 - val_acc: 0.8692
    
    Epoch 00156: val_acc did not improve from 0.87467
    Epoch 157/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1729 - acc: 0.9406 - val_loss: 0.3122 - val_acc: 0.8687
    
    Epoch 00157: val_acc did not improve from 0.87467
    Epoch 158/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1735 - acc: 0.9409 - val_loss: 0.3127 - val_acc: 0.8684
    
    Epoch 00158: val_acc did not improve from 0.87467
    Epoch 159/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1712 - acc: 0.9399 - val_loss: 0.3131 - val_acc: 0.8685
    
    Epoch 00159: val_acc did not improve from 0.87467
    Epoch 160/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1696 - acc: 0.9403 - val_loss: 0.3136 - val_acc: 0.8687
    
    Epoch 00160: val_acc did not improve from 0.87467
    Epoch 161/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1734 - acc: 0.9401 - val_loss: 0.3141 - val_acc: 0.8687
    
    Epoch 00161: val_acc did not improve from 0.87467
    Epoch 162/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1691 - acc: 0.9390 - val_loss: 0.3145 - val_acc: 0.8683
    
    Epoch 00162: val_acc did not improve from 0.87467
    Epoch 163/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1708 - acc: 0.9396 - val_loss: 0.3149 - val_acc: 0.8677
    
    Epoch 00163: val_acc did not improve from 0.87467
    Epoch 164/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1688 - acc: 0.9405 - val_loss: 0.3154 - val_acc: 0.8673
    
    Epoch 00164: val_acc did not improve from 0.87467
    Epoch 165/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1683 - acc: 0.9417 - val_loss: 0.3159 - val_acc: 0.8672
    
    Epoch 00165: val_acc did not improve from 0.87467
    Epoch 166/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1683 - acc: 0.9414 - val_loss: 0.3163 - val_acc: 0.8675
    
    Epoch 00166: val_acc did not improve from 0.87467
    Epoch 167/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1668 - acc: 0.9424 - val_loss: 0.3168 - val_acc: 0.8672
    
    Epoch 00167: val_acc did not improve from 0.87467
    Epoch 168/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1682 - acc: 0.9424 - val_loss: 0.3172 - val_acc: 0.8673
    
    Epoch 00168: val_acc did not improve from 0.87467
    Epoch 169/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.1654 - acc: 0.9434 - val_loss: 0.3177 - val_acc: 0.8673
    
    Epoch 00169: val_acc did not improve from 0.87467
    Epoch 170/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1644 - acc: 0.9444 - val_loss: 0.3182 - val_acc: 0.8676
    
    Epoch 00170: val_acc did not improve from 0.87467
    Epoch 171/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1653 - acc: 0.9438 - val_loss: 0.3188 - val_acc: 0.8672
    
    Epoch 00171: val_acc did not improve from 0.87467
    Epoch 172/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.1645 - acc: 0.9427 - val_loss: 0.3193 - val_acc: 0.8672
    
    Epoch 00172: val_acc did not improve from 0.87467
    Epoch 173/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1639 - acc: 0.9429 - val_loss: 0.3199 - val_acc: 0.8677
    
    Epoch 00173: val_acc did not improve from 0.87467
    Epoch 174/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1649 - acc: 0.9425 - val_loss: 0.3204 - val_acc: 0.8673
    
    Epoch 00174: val_acc did not improve from 0.87467
    Epoch 175/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1633 - acc: 0.9424 - val_loss: 0.3209 - val_acc: 0.8672
    
    Epoch 00175: val_acc did not improve from 0.87467
    Epoch 176/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1637 - acc: 0.9437 - val_loss: 0.3214 - val_acc: 0.8668
    
    Epoch 00176: val_acc did not improve from 0.87467
    Epoch 177/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1644 - acc: 0.9436 - val_loss: 0.3219 - val_acc: 0.8669
    
    Epoch 00177: val_acc did not improve from 0.87467
    Epoch 178/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1638 - acc: 0.9439 - val_loss: 0.3223 - val_acc: 0.8668
    
    Epoch 00178: val_acc did not improve from 0.87467
    Epoch 179/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1620 - acc: 0.9429 - val_loss: 0.3228 - val_acc: 0.8667
    
    Epoch 00179: val_acc did not improve from 0.87467
    Epoch 180/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1620 - acc: 0.9443 - val_loss: 0.3233 - val_acc: 0.8663
    
    Epoch 00180: val_acc did not improve from 0.87467
    Epoch 181/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1617 - acc: 0.9434 - val_loss: 0.3238 - val_acc: 0.8664
    
    Epoch 00181: val_acc did not improve from 0.87467
    Epoch 182/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1601 - acc: 0.9433 - val_loss: 0.3243 - val_acc: 0.8663
    
    Epoch 00182: val_acc did not improve from 0.87467
    Epoch 183/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1596 - acc: 0.9462 - val_loss: 0.3249 - val_acc: 0.8663
    
    Epoch 00183: val_acc did not improve from 0.87467
    Epoch 184/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1603 - acc: 0.9440 - val_loss: 0.3254 - val_acc: 0.8664
    
    Epoch 00184: val_acc did not improve from 0.87467
    Epoch 185/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1586 - acc: 0.9446 - val_loss: 0.3259 - val_acc: 0.8659
    
    Epoch 00185: val_acc did not improve from 0.87467
    Epoch 186/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.1573 - acc: 0.9458 - val_loss: 0.3265 - val_acc: 0.8653
    
    Epoch 00186: val_acc did not improve from 0.87467
    Epoch 187/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1570 - acc: 0.9452 - val_loss: 0.3271 - val_acc: 0.8655
    
    Epoch 00187: val_acc did not improve from 0.87467
    Epoch 188/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1585 - acc: 0.9450 - val_loss: 0.3276 - val_acc: 0.8651
    
    Epoch 00188: val_acc did not improve from 0.87467
    Epoch 189/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1572 - acc: 0.9464 - val_loss: 0.3281 - val_acc: 0.8644
    
    Epoch 00189: val_acc did not improve from 0.87467
    Epoch 190/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1565 - acc: 0.9462 - val_loss: 0.3287 - val_acc: 0.8648
    
    Epoch 00190: val_acc did not improve from 0.87467
    Epoch 191/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1560 - acc: 0.9453 - val_loss: 0.3292 - val_acc: 0.8645
    
    Epoch 00191: val_acc did not improve from 0.87467
    Epoch 192/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1536 - acc: 0.9478 - val_loss: 0.3299 - val_acc: 0.8640
    
    Epoch 00192: val_acc did not improve from 0.87467
    Epoch 193/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1539 - acc: 0.9464 - val_loss: 0.3305 - val_acc: 0.8640
    
    Epoch 00193: val_acc did not improve from 0.87467
    Epoch 194/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1541 - acc: 0.9461 - val_loss: 0.3311 - val_acc: 0.8644
    
    Epoch 00194: val_acc did not improve from 0.87467
    Epoch 195/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1538 - acc: 0.9482 - val_loss: 0.3317 - val_acc: 0.8649
    
    Epoch 00195: val_acc did not improve from 0.87467
    Epoch 196/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1528 - acc: 0.9471 - val_loss: 0.3323 - val_acc: 0.8645
    
    Epoch 00196: val_acc did not improve from 0.87467
    Epoch 197/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.1519 - acc: 0.9484 - val_loss: 0.3328 - val_acc: 0.8640
    
    Epoch 00197: val_acc did not improve from 0.87467
    Epoch 198/500
    17500/17500 [==============================] - 1s 44us/step - loss: 0.1530 - acc: 0.9458 - val_loss: 0.3334 - val_acc: 0.8639
    
    Epoch 00198: val_acc did not improve from 0.87467
    Epoch 199/500
    17500/17500 [==============================] - 1s 45us/step - loss: 0.1519 - acc: 0.9485 - val_loss: 0.3340 - val_acc: 0.8636
    
    Epoch 00199: val_acc did not improve from 0.87467
    Epoch 200/500
    17500/17500 [==============================] - 1s 47us/step - loss: 0.1518 - acc: 0.9472 - val_loss: 0.3346 - val_acc: 0.8640
    
    Epoch 00200: val_acc did not improve from 0.87467
    Epoch 201/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.1508 - acc: 0.9468 - val_loss: 0.3351 - val_acc: 0.8643
    
    Epoch 00201: val_acc did not improve from 0.87467
    Epoch 202/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.1512 - acc: 0.9481 - val_loss: 0.3356 - val_acc: 0.8637
    
    Epoch 00202: val_acc did not improve from 0.87467
    Epoch 203/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.1505 - acc: 0.9483 - val_loss: 0.3362 - val_acc: 0.8631
    
    Epoch 00203: val_acc did not improve from 0.87467
    Epoch 204/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1519 - acc: 0.9483 - val_loss: 0.3366 - val_acc: 0.8627
    
    Epoch 00204: val_acc did not improve from 0.87467
    Epoch 205/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.1509 - acc: 0.9474 - val_loss: 0.3371 - val_acc: 0.8637
    
    Epoch 00205: val_acc did not improve from 0.87467
    Epoch 206/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1500 - acc: 0.9489 - val_loss: 0.3376 - val_acc: 0.8633
    
    Epoch 00206: val_acc did not improve from 0.87467
    Epoch 207/500
    17500/17500 [==============================] - 1s 63us/step - loss: 0.1490 - acc: 0.9473 - val_loss: 0.3381 - val_acc: 0.8635
    
    Epoch 00207: val_acc did not improve from 0.87467
    Epoch 208/500
    17500/17500 [==============================] - 1s 66us/step - loss: 0.1484 - acc: 0.9494 - val_loss: 0.3386 - val_acc: 0.8633
    
    Epoch 00208: val_acc did not improve from 0.87467
    Epoch 209/500
    17500/17500 [==============================] - 1s 54us/step - loss: 0.1486 - acc: 0.9489 - val_loss: 0.3392 - val_acc: 0.8629
    
    Epoch 00209: val_acc did not improve from 0.87467
    Epoch 210/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.1486 - acc: 0.9483 - val_loss: 0.3399 - val_acc: 0.8620
    
    Epoch 00210: val_acc did not improve from 0.87467
    Epoch 211/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.1471 - acc: 0.9497 - val_loss: 0.3405 - val_acc: 0.8621
    
    Epoch 00211: val_acc did not improve from 0.87467
    Epoch 212/500
    17500/17500 [==============================] - 1s 46us/step - loss: 0.1465 - acc: 0.9484 - val_loss: 0.3410 - val_acc: 0.8616
    
    Epoch 00212: val_acc did not improve from 0.87467
    Epoch 213/500
    17500/17500 [==============================] - 1s 49us/step - loss: 0.1473 - acc: 0.9496 - val_loss: 0.3416 - val_acc: 0.8616
    
    Epoch 00213: val_acc did not improve from 0.87467
    Epoch 214/500
    17500/17500 [==============================] - 1s 49us/step - loss: 0.1445 - acc: 0.9489 - val_loss: 0.3422 - val_acc: 0.8617
    
    Epoch 00214: val_acc did not improve from 0.87467
    Epoch 215/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1459 - acc: 0.9498 - val_loss: 0.3428 - val_acc: 0.8616
    
    Epoch 00215: val_acc did not improve from 0.87467
    Epoch 216/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1460 - acc: 0.9493 - val_loss: 0.3433 - val_acc: 0.8611
    
    Epoch 00216: val_acc did not improve from 0.87467
    Epoch 217/500
    17500/17500 [==============================] - 1s 45us/step - loss: 0.1446 - acc: 0.9491 - val_loss: 0.3439 - val_acc: 0.8607
    
    Epoch 00217: val_acc did not improve from 0.87467
    Epoch 218/500
    17500/17500 [==============================] - 1s 57us/step - loss: 0.1454 - acc: 0.9483 - val_loss: 0.3445 - val_acc: 0.8605
    
    Epoch 00218: val_acc did not improve from 0.87467
    Epoch 219/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.1449 - acc: 0.9507 - val_loss: 0.3451 - val_acc: 0.8601
    
    Epoch 00219: val_acc did not improve from 0.87467
    Epoch 220/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1442 - acc: 0.9489 - val_loss: 0.3456 - val_acc: 0.8603
    
    Epoch 00220: val_acc did not improve from 0.87467
    Epoch 221/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.1430 - acc: 0.9515 - val_loss: 0.3462 - val_acc: 0.8601
    
    Epoch 00221: val_acc did not improve from 0.87467
    Epoch 222/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.1429 - acc: 0.9518 - val_loss: 0.3469 - val_acc: 0.8600
    
    Epoch 00222: val_acc did not improve from 0.87467
    Epoch 223/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.1430 - acc: 0.9502 - val_loss: 0.3476 - val_acc: 0.8601
    
    Epoch 00223: val_acc did not improve from 0.87467
    Epoch 224/500
    17500/17500 [==============================] - 1s 44us/step - loss: 0.1412 - acc: 0.9507 - val_loss: 0.3482 - val_acc: 0.8600
    
    Epoch 00224: val_acc did not improve from 0.87467
    Epoch 225/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.1425 - acc: 0.9511 - val_loss: 0.3489 - val_acc: 0.8597
    
    Epoch 00225: val_acc did not improve from 0.87467
    Epoch 226/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.1410 - acc: 0.9512 - val_loss: 0.3496 - val_acc: 0.8599
    
    Epoch 00226: val_acc did not improve from 0.87467
    Epoch 227/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.1414 - acc: 0.9498 - val_loss: 0.3502 - val_acc: 0.8600
    
    Epoch 00227: val_acc did not improve from 0.87467
    Epoch 228/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.1407 - acc: 0.9530 - val_loss: 0.3508 - val_acc: 0.8596
    
    Epoch 00228: val_acc did not improve from 0.87467
    Epoch 229/500
    17500/17500 [==============================] - 1s 44us/step - loss: 0.1402 - acc: 0.9509 - val_loss: 0.3513 - val_acc: 0.8595
    
    Epoch 00229: val_acc did not improve from 0.87467
    Epoch 230/500
    17500/17500 [==============================] - 1s 45us/step - loss: 0.1394 - acc: 0.9518 - val_loss: 0.3519 - val_acc: 0.8593
    
    Epoch 00230: val_acc did not improve from 0.87467
    Epoch 231/500
    17500/17500 [==============================] - 1s 45us/step - loss: 0.1389 - acc: 0.9529 - val_loss: 0.3526 - val_acc: 0.8591
    
    Epoch 00231: val_acc did not improve from 0.87467
    Epoch 232/500
    17500/17500 [==============================] - 1s 44us/step - loss: 0.1392 - acc: 0.9521 - val_loss: 0.3533 - val_acc: 0.8587
    
    Epoch 00232: val_acc did not improve from 0.87467
    Epoch 233/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.1374 - acc: 0.9539 - val_loss: 0.3540 - val_acc: 0.8591
    
    Epoch 00233: val_acc did not improve from 0.87467
    Epoch 234/500
    17500/17500 [==============================] - 1s 45us/step - loss: 0.1387 - acc: 0.9517 - val_loss: 0.3547 - val_acc: 0.8595
    
    Epoch 00234: val_acc did not improve from 0.87467
    Epoch 235/500
    17500/17500 [==============================] - 1s 45us/step - loss: 0.1382 - acc: 0.9526 - val_loss: 0.3554 - val_acc: 0.8589
    
    Epoch 00235: val_acc did not improve from 0.87467
    Epoch 236/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.1381 - acc: 0.9519 - val_loss: 0.3561 - val_acc: 0.8583
    
    Epoch 00236: val_acc did not improve from 0.87467
    Epoch 237/500
    17500/17500 [==============================] - 1s 46us/step - loss: 0.1382 - acc: 0.9530 - val_loss: 0.3567 - val_acc: 0.8583
    
    Epoch 00237: val_acc did not improve from 0.87467
    Epoch 238/500
    17500/17500 [==============================] - 1s 48us/step - loss: 0.1349 - acc: 0.9538 - val_loss: 0.3573 - val_acc: 0.8581
    
    Epoch 00238: val_acc did not improve from 0.87467
    Epoch 239/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1356 - acc: 0.9530 - val_loss: 0.3580 - val_acc: 0.8584
    
    Epoch 00239: val_acc did not improve from 0.87467
    Epoch 240/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1352 - acc: 0.9530 - val_loss: 0.3587 - val_acc: 0.8583
    
    Epoch 00240: val_acc did not improve from 0.87467
    Epoch 241/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.1358 - acc: 0.9533 - val_loss: 0.3594 - val_acc: 0.8579
    
    Epoch 00241: val_acc did not improve from 0.87467
    Epoch 242/500
    17500/17500 [==============================] - 1s 45us/step - loss: 0.1356 - acc: 0.9531 - val_loss: 0.3599 - val_acc: 0.8581
    
    Epoch 00242: val_acc did not improve from 0.87467
    Epoch 243/500
    17500/17500 [==============================] - 1s 44us/step - loss: 0.1334 - acc: 0.9538 - val_loss: 0.3606 - val_acc: 0.8579
    
    Epoch 00243: val_acc did not improve from 0.87467
    Epoch 244/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.1341 - acc: 0.9533 - val_loss: 0.3613 - val_acc: 0.8581
    
    Epoch 00244: val_acc did not improve from 0.87467
    Epoch 245/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.1333 - acc: 0.9535 - val_loss: 0.3620 - val_acc: 0.8583
    
    Epoch 00245: val_acc did not improve from 0.87467
    Epoch 246/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.1370 - acc: 0.9516 - val_loss: 0.3626 - val_acc: 0.8579
    
    Epoch 00246: val_acc did not improve from 0.87467
    Epoch 247/500
    17500/17500 [==============================] - 1s 51us/step - loss: 0.1325 - acc: 0.9545 - val_loss: 0.3632 - val_acc: 0.8575
    
    Epoch 00247: val_acc did not improve from 0.87467
    Epoch 248/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.1330 - acc: 0.9538 - val_loss: 0.3638 - val_acc: 0.8569
    
    Epoch 00248: val_acc did not improve from 0.87467
    Epoch 249/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.1326 - acc: 0.9532 - val_loss: 0.3644 - val_acc: 0.8577
    
    Epoch 00249: val_acc did not improve from 0.87467
    Epoch 250/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.1347 - acc: 0.9521 - val_loss: 0.3651 - val_acc: 0.8581
    
    Epoch 00250: val_acc did not improve from 0.87467
    Epoch 251/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.1313 - acc: 0.9545 - val_loss: 0.3657 - val_acc: 0.8577
    
    Epoch 00251: val_acc did not improve from 0.87467
    Epoch 252/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.1320 - acc: 0.9553 - val_loss: 0.3664 - val_acc: 0.8577
    
    Epoch 00252: val_acc did not improve from 0.87467
    Epoch 253/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.1316 - acc: 0.9552 - val_loss: 0.3670 - val_acc: 0.8573
    
    Epoch 00253: val_acc did not improve from 0.87467
    Epoch 254/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.1318 - acc: 0.9542 - val_loss: 0.3677 - val_acc: 0.8575
    
    Epoch 00254: val_acc did not improve from 0.87467
    Epoch 255/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.1318 - acc: 0.9549 - val_loss: 0.3684 - val_acc: 0.8572
    
    Epoch 00255: val_acc did not improve from 0.87467
    Epoch 256/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.1309 - acc: 0.9555 - val_loss: 0.3690 - val_acc: 0.8571
    
    Epoch 00256: val_acc did not improve from 0.87467
    Epoch 257/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.1298 - acc: 0.9543 - val_loss: 0.3696 - val_acc: 0.8573
    
    Epoch 00257: val_acc did not improve from 0.87467
    Epoch 258/500
    17500/17500 [==============================] - 1s 44us/step - loss: 0.1306 - acc: 0.9535 - val_loss: 0.3702 - val_acc: 0.8573
    
    Epoch 00258: val_acc did not improve from 0.87467
    Epoch 259/500
    17500/17500 [==============================] - 1s 50us/step - loss: 0.1286 - acc: 0.9566 - val_loss: 0.3709 - val_acc: 0.8573
    
    Epoch 00259: val_acc did not improve from 0.87467
    Epoch 260/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.1297 - acc: 0.9545 - val_loss: 0.3714 - val_acc: 0.8568
    
    Epoch 00260: val_acc did not improve from 0.87467
    Epoch 261/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.1288 - acc: 0.9545 - val_loss: 0.3721 - val_acc: 0.8568
    
    Epoch 00261: val_acc did not improve from 0.87467
    Epoch 262/500
    17500/17500 [==============================] - 1s 48us/step - loss: 0.1279 - acc: 0.9558 - val_loss: 0.3727 - val_acc: 0.8568
    
    Epoch 00262: val_acc did not improve from 0.87467
    Epoch 263/500
    17500/17500 [==============================] - 1s 59us/step - loss: 0.1284 - acc: 0.9560 - val_loss: 0.3734 - val_acc: 0.8565
    
    Epoch 00263: val_acc did not improve from 0.87467
    Epoch 264/500
    17500/17500 [==============================] - 1s 57us/step - loss: 0.1278 - acc: 0.9562 - val_loss: 0.3741 - val_acc: 0.8569
    
    Epoch 00264: val_acc did not improve from 0.87467
    Epoch 265/500
    17500/17500 [==============================] - 1s 56us/step - loss: 0.1273 - acc: 0.9558 - val_loss: 0.3747 - val_acc: 0.8567
    
    Epoch 00265: val_acc did not improve from 0.87467
    Epoch 266/500
    17500/17500 [==============================] - 1s 51us/step - loss: 0.1268 - acc: 0.9564 - val_loss: 0.3754 - val_acc: 0.8563
    
    Epoch 00266: val_acc did not improve from 0.87467
    Epoch 267/500
    17500/17500 [==============================] - 1s 49us/step - loss: 0.1257 - acc: 0.9574 - val_loss: 0.3761 - val_acc: 0.8563
    
    Epoch 00267: val_acc did not improve from 0.87467
    Epoch 268/500
    17500/17500 [==============================] - 1s 52us/step - loss: 0.1272 - acc: 0.9567 - val_loss: 0.3768 - val_acc: 0.8559
    
    Epoch 00268: val_acc did not improve from 0.87467
    Epoch 269/500
    17500/17500 [==============================] - 1s 45us/step - loss: 0.1270 - acc: 0.9559 - val_loss: 0.3775 - val_acc: 0.8559
    
    Epoch 00269: val_acc did not improve from 0.87467
    Epoch 270/500
    17500/17500 [==============================] - 1s 46us/step - loss: 0.1265 - acc: 0.9563 - val_loss: 0.3782 - val_acc: 0.8557
    
    Epoch 00270: val_acc did not improve from 0.87467
    Epoch 271/500
    17500/17500 [==============================] - 1s 45us/step - loss: 0.1263 - acc: 0.9570 - val_loss: 0.3787 - val_acc: 0.8564
    
    Epoch 00271: val_acc did not improve from 0.87467
    Epoch 272/500
    17500/17500 [==============================] - 1s 53us/step - loss: 0.1250 - acc: 0.9558 - val_loss: 0.3793 - val_acc: 0.8564
    
    Epoch 00272: val_acc did not improve from 0.87467
    Epoch 273/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.1265 - acc: 0.9573 - val_loss: 0.3800 - val_acc: 0.8555
    
    Epoch 00273: val_acc did not improve from 0.87467
    Epoch 274/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1251 - acc: 0.9570 - val_loss: 0.3807 - val_acc: 0.8552
    
    Epoch 00274: val_acc did not improve from 0.87467
    Epoch 275/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1249 - acc: 0.9557 - val_loss: 0.3814 - val_acc: 0.8557
    
    Epoch 00275: val_acc did not improve from 0.87467
    Epoch 276/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1239 - acc: 0.9577 - val_loss: 0.3822 - val_acc: 0.8559
    
    Epoch 00276: val_acc did not improve from 0.87467
    Epoch 277/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.1233 - acc: 0.9581 - val_loss: 0.3831 - val_acc: 0.8555
    
    Epoch 00277: val_acc did not improve from 0.87467
    Epoch 278/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1238 - acc: 0.9570 - val_loss: 0.3839 - val_acc: 0.8556
    
    Epoch 00278: val_acc did not improve from 0.87467
    Epoch 279/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.1223 - acc: 0.9574 - val_loss: 0.3845 - val_acc: 0.8552
    
    Epoch 00279: val_acc did not improve from 0.87467
    Epoch 280/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1223 - acc: 0.9578 - val_loss: 0.3852 - val_acc: 0.8549
    
    Epoch 00280: val_acc did not improve from 0.87467
    Epoch 281/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1236 - acc: 0.9562 - val_loss: 0.3858 - val_acc: 0.8543
    
    Epoch 00281: val_acc did not improve from 0.87467
    Epoch 282/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1230 - acc: 0.9581 - val_loss: 0.3863 - val_acc: 0.8539
    
    Epoch 00282: val_acc did not improve from 0.87467
    Epoch 283/500
    17500/17500 [==============================] - 1s 53us/step - loss: 0.1230 - acc: 0.9569 - val_loss: 0.3869 - val_acc: 0.8540
    
    Epoch 00283: val_acc did not improve from 0.87467
    Epoch 284/500
    17500/17500 [==============================] - 1s 51us/step - loss: 0.1223 - acc: 0.9561 - val_loss: 0.3875 - val_acc: 0.8545
    
    Epoch 00284: val_acc did not improve from 0.87467
    Epoch 285/500
    17500/17500 [==============================] - 1s 58us/step - loss: 0.1206 - acc: 0.9579 - val_loss: 0.3882 - val_acc: 0.8543
    
    Epoch 00285: val_acc did not improve from 0.87467
    Epoch 286/500
    17500/17500 [==============================] - 1s 54us/step - loss: 0.1211 - acc: 0.9587 - val_loss: 0.3889 - val_acc: 0.8536
    
    Epoch 00286: val_acc did not improve from 0.87467
    Epoch 287/500
    17500/17500 [==============================] - 1s 69us/step - loss: 0.1205 - acc: 0.9606 - val_loss: 0.3896 - val_acc: 0.8536
    
    Epoch 00287: val_acc did not improve from 0.87467
    Epoch 288/500
    17500/17500 [==============================] - 1s 62us/step - loss: 0.1200 - acc: 0.9586 - val_loss: 0.3904 - val_acc: 0.8533
    
    Epoch 00288: val_acc did not improve from 0.87467
    Epoch 289/500
    17500/17500 [==============================] - 1s 45us/step - loss: 0.1203 - acc: 0.9606 - val_loss: 0.3912 - val_acc: 0.8536
    
    Epoch 00289: val_acc did not improve from 0.87467
    Epoch 290/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.1204 - acc: 0.9581 - val_loss: 0.3918 - val_acc: 0.8540
    
    Epoch 00290: val_acc did not improve from 0.87467
    Epoch 291/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.1206 - acc: 0.9593 - val_loss: 0.3924 - val_acc: 0.8545
    
    Epoch 00291: val_acc did not improve from 0.87467
    Epoch 292/500
    17500/17500 [==============================] - 1s 44us/step - loss: 0.1198 - acc: 0.9594 - val_loss: 0.3930 - val_acc: 0.8548
    
    Epoch 00292: val_acc did not improve from 0.87467
    Epoch 293/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.1178 - acc: 0.9594 - val_loss: 0.3939 - val_acc: 0.8545
    
    Epoch 00293: val_acc did not improve from 0.87467
    Epoch 294/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1191 - acc: 0.9584 - val_loss: 0.3947 - val_acc: 0.8540
    
    Epoch 00294: val_acc did not improve from 0.87467
    Epoch 295/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.1179 - acc: 0.9602 - val_loss: 0.3954 - val_acc: 0.8543
    
    Epoch 00295: val_acc did not improve from 0.87467
    Epoch 296/500
    17500/17500 [==============================] - 1s 51us/step - loss: 0.1186 - acc: 0.9597 - val_loss: 0.3961 - val_acc: 0.8543
    
    Epoch 00296: val_acc did not improve from 0.87467
    Epoch 297/500
    17500/17500 [==============================] - 1s 49us/step - loss: 0.1177 - acc: 0.9591 - val_loss: 0.3968 - val_acc: 0.8540
    
    Epoch 00297: val_acc did not improve from 0.87467
    Epoch 298/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.1162 - acc: 0.9605 - val_loss: 0.3976 - val_acc: 0.8536
    
    Epoch 00298: val_acc did not improve from 0.87467
    Epoch 299/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.1172 - acc: 0.9602 - val_loss: 0.3984 - val_acc: 0.8537
    
    Epoch 00299: val_acc did not improve from 0.87467
    Epoch 300/500
    17500/17500 [==============================] - 1s 50us/step - loss: 0.1166 - acc: 0.9607 - val_loss: 0.3990 - val_acc: 0.8536
    
    Epoch 00300: val_acc did not improve from 0.87467
    Epoch 301/500
    17500/17500 [==============================] - 1s 52us/step - loss: 0.1171 - acc: 0.9607 - val_loss: 0.3997 - val_acc: 0.8532
    
    Epoch 00301: val_acc did not improve from 0.87467
    Epoch 302/500
    17500/17500 [==============================] - 1s 46us/step - loss: 0.1170 - acc: 0.9606 - val_loss: 0.4004 - val_acc: 0.8532
    
    Epoch 00302: val_acc did not improve from 0.87467
    Epoch 303/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.1150 - acc: 0.9622 - val_loss: 0.4013 - val_acc: 0.8532
    
    Epoch 00303: val_acc did not improve from 0.87467
    Epoch 304/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.1150 - acc: 0.9624 - val_loss: 0.4022 - val_acc: 0.8531
    
    Epoch 00304: val_acc did not improve from 0.87467
    Epoch 305/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.1160 - acc: 0.9617 - val_loss: 0.4030 - val_acc: 0.8528
    
    Epoch 00305: val_acc did not improve from 0.87467
    Epoch 306/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.1151 - acc: 0.9618 - val_loss: 0.4036 - val_acc: 0.8528
    
    Epoch 00306: val_acc did not improve from 0.87467
    Epoch 307/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.1146 - acc: 0.9607 - val_loss: 0.4042 - val_acc: 0.8532
    
    Epoch 00307: val_acc did not improve from 0.87467
    Epoch 308/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.1149 - acc: 0.9605 - val_loss: 0.4048 - val_acc: 0.8535
    
    Epoch 00308: val_acc did not improve from 0.87467
    Epoch 309/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.1155 - acc: 0.9599 - val_loss: 0.4054 - val_acc: 0.8536
    
    Epoch 00309: val_acc did not improve from 0.87467
    Epoch 310/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.1147 - acc: 0.9611 - val_loss: 0.4061 - val_acc: 0.8537
    
    Epoch 00310: val_acc did not improve from 0.87467
    Epoch 311/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.1139 - acc: 0.9628 - val_loss: 0.4068 - val_acc: 0.8537
    
    Epoch 00311: val_acc did not improve from 0.87467
    Epoch 312/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.1140 - acc: 0.9598 - val_loss: 0.4076 - val_acc: 0.8533
    
    Epoch 00312: val_acc did not improve from 0.87467
    Epoch 313/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.1133 - acc: 0.9625 - val_loss: 0.4084 - val_acc: 0.8533
    
    Epoch 00313: val_acc did not improve from 0.87467
    Epoch 314/500
    17500/17500 [==============================] - 1s 45us/step - loss: 0.1124 - acc: 0.9615 - val_loss: 0.4090 - val_acc: 0.8536
    
    Epoch 00314: val_acc did not improve from 0.87467
    Epoch 315/500
    17500/17500 [==============================] - 1s 44us/step - loss: 0.1124 - acc: 0.9620 - val_loss: 0.4097 - val_acc: 0.8528
    
    Epoch 00315: val_acc did not improve from 0.87467
    Epoch 316/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.1124 - acc: 0.9627 - val_loss: 0.4105 - val_acc: 0.8531
    
    Epoch 00316: val_acc did not improve from 0.87467
    Epoch 317/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.1125 - acc: 0.9620 - val_loss: 0.4112 - val_acc: 0.8532
    
    Epoch 00317: val_acc did not improve from 0.87467
    Epoch 318/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.1134 - acc: 0.9630 - val_loss: 0.4119 - val_acc: 0.8529
    
    Epoch 00318: val_acc did not improve from 0.87467
    Epoch 319/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.1132 - acc: 0.9618 - val_loss: 0.4125 - val_acc: 0.8525
    
    Epoch 00319: val_acc did not improve from 0.87467
    Epoch 320/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.1104 - acc: 0.9632 - val_loss: 0.4132 - val_acc: 0.8527
    
    Epoch 00320: val_acc did not improve from 0.87467
    Epoch 321/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.1125 - acc: 0.9602 - val_loss: 0.4138 - val_acc: 0.8525
    
    Epoch 00321: val_acc did not improve from 0.87467
    Epoch 322/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.1105 - acc: 0.9622 - val_loss: 0.4145 - val_acc: 0.8527
    
    Epoch 00322: val_acc did not improve from 0.87467
    Epoch 323/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.1103 - acc: 0.9640 - val_loss: 0.4152 - val_acc: 0.8524
    
    Epoch 00323: val_acc did not improve from 0.87467
    Epoch 324/500
    17500/17500 [==============================] - 1s 43us/step - loss: 0.1102 - acc: 0.9622 - val_loss: 0.4162 - val_acc: 0.8523
    
    Epoch 00324: val_acc did not improve from 0.87467
    Epoch 325/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.1091 - acc: 0.9626 - val_loss: 0.4171 - val_acc: 0.8527
    
    Epoch 00325: val_acc did not improve from 0.87467
    Epoch 326/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1103 - acc: 0.9637 - val_loss: 0.4180 - val_acc: 0.8523
    
    Epoch 00326: val_acc did not improve from 0.87467
    Epoch 327/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.1108 - acc: 0.9622 - val_loss: 0.4186 - val_acc: 0.8520
    
    Epoch 00327: val_acc did not improve from 0.87467
    Epoch 328/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1092 - acc: 0.9626 - val_loss: 0.4191 - val_acc: 0.8520
    
    Epoch 00328: val_acc did not improve from 0.87467
    Epoch 329/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.1085 - acc: 0.9634 - val_loss: 0.4198 - val_acc: 0.8520
    
    Epoch 00329: val_acc did not improve from 0.87467
    Epoch 330/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.1078 - acc: 0.9649 - val_loss: 0.4208 - val_acc: 0.8520
    
    Epoch 00330: val_acc did not improve from 0.87467
    Epoch 331/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.1072 - acc: 0.9647 - val_loss: 0.4217 - val_acc: 0.8516
    
    Epoch 00331: val_acc did not improve from 0.87467
    Epoch 332/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.1084 - acc: 0.9630 - val_loss: 0.4226 - val_acc: 0.8512
    
    Epoch 00332: val_acc did not improve from 0.87467
    Epoch 333/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.1100 - acc: 0.9634 - val_loss: 0.4231 - val_acc: 0.8512
    
    Epoch 00333: val_acc did not improve from 0.87467
    Epoch 334/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1083 - acc: 0.9623 - val_loss: 0.4234 - val_acc: 0.8519
    
    Epoch 00334: val_acc did not improve from 0.87467
    Epoch 335/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1072 - acc: 0.9643 - val_loss: 0.4239 - val_acc: 0.8513
    
    Epoch 00335: val_acc did not improve from 0.87467
    Epoch 336/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.1079 - acc: 0.9645 - val_loss: 0.4246 - val_acc: 0.8515
    
    Epoch 00336: val_acc did not improve from 0.87467
    Epoch 337/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.1076 - acc: 0.9632 - val_loss: 0.4253 - val_acc: 0.8515
    
    Epoch 00337: val_acc did not improve from 0.87467
    Epoch 338/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.1063 - acc: 0.9643 - val_loss: 0.4262 - val_acc: 0.8515
    
    Epoch 00338: val_acc did not improve from 0.87467
    Epoch 339/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.1053 - acc: 0.9647 - val_loss: 0.4272 - val_acc: 0.8512
    
    Epoch 00339: val_acc did not improve from 0.87467
    Epoch 340/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1057 - acc: 0.9653 - val_loss: 0.4281 - val_acc: 0.8511
    
    Epoch 00340: val_acc did not improve from 0.87467
    Epoch 341/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.1061 - acc: 0.9646 - val_loss: 0.4288 - val_acc: 0.8516
    
    Epoch 00341: val_acc did not improve from 0.87467
    Epoch 342/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1050 - acc: 0.9653 - val_loss: 0.4294 - val_acc: 0.8515
    
    Epoch 00342: val_acc did not improve from 0.87467
    Epoch 343/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.1045 - acc: 0.9644 - val_loss: 0.4300 - val_acc: 0.8515
    
    Epoch 00343: val_acc did not improve from 0.87467
    Epoch 344/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.1065 - acc: 0.9633 - val_loss: 0.4306 - val_acc: 0.8511
    
    Epoch 00344: val_acc did not improve from 0.87467
    Epoch 345/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1066 - acc: 0.9650 - val_loss: 0.4310 - val_acc: 0.8515
    
    Epoch 00345: val_acc did not improve from 0.87467
    Epoch 346/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.1045 - acc: 0.9652 - val_loss: 0.4317 - val_acc: 0.8515
    
    Epoch 00346: val_acc did not improve from 0.87467
    Epoch 347/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.1051 - acc: 0.9638 - val_loss: 0.4324 - val_acc: 0.8509
    
    Epoch 00347: val_acc did not improve from 0.87467
    Epoch 348/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1049 - acc: 0.9643 - val_loss: 0.4331 - val_acc: 0.8504
    
    Epoch 00348: val_acc did not improve from 0.87467
    Epoch 349/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1053 - acc: 0.9659 - val_loss: 0.4339 - val_acc: 0.8511
    
    Epoch 00349: val_acc did not improve from 0.87467
    Epoch 350/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1031 - acc: 0.9665 - val_loss: 0.4345 - val_acc: 0.8515
    
    Epoch 00350: val_acc did not improve from 0.87467
    Epoch 351/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1032 - acc: 0.9663 - val_loss: 0.4354 - val_acc: 0.8512
    
    Epoch 00351: val_acc did not improve from 0.87467
    Epoch 352/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.1019 - acc: 0.9670 - val_loss: 0.4364 - val_acc: 0.8509
    
    Epoch 00352: val_acc did not improve from 0.87467
    Epoch 353/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.1030 - acc: 0.9670 - val_loss: 0.4371 - val_acc: 0.8508
    
    Epoch 00353: val_acc did not improve from 0.87467
    Epoch 354/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1021 - acc: 0.9655 - val_loss: 0.4376 - val_acc: 0.8507
    
    Epoch 00354: val_acc did not improve from 0.87467
    Epoch 355/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1022 - acc: 0.9669 - val_loss: 0.4383 - val_acc: 0.8507
    
    Epoch 00355: val_acc did not improve from 0.87467
    Epoch 356/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.1028 - acc: 0.9653 - val_loss: 0.4389 - val_acc: 0.8509
    
    Epoch 00356: val_acc did not improve from 0.87467
    Epoch 357/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.1021 - acc: 0.9657 - val_loss: 0.4397 - val_acc: 0.8511
    
    Epoch 00357: val_acc did not improve from 0.87467
    Epoch 358/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1020 - acc: 0.9664 - val_loss: 0.4406 - val_acc: 0.8511
    
    Epoch 00358: val_acc did not improve from 0.87467
    Epoch 359/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.1006 - acc: 0.9668 - val_loss: 0.4413 - val_acc: 0.8508
    
    Epoch 00359: val_acc did not improve from 0.87467
    Epoch 360/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.1020 - acc: 0.9660 - val_loss: 0.4421 - val_acc: 0.8508
    
    Epoch 00360: val_acc did not improve from 0.87467
    Epoch 361/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.1002 - acc: 0.9676 - val_loss: 0.4429 - val_acc: 0.8511
    
    Epoch 00361: val_acc did not improve from 0.87467
    Epoch 362/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1018 - acc: 0.9655 - val_loss: 0.4436 - val_acc: 0.8509
    
    Epoch 00362: val_acc did not improve from 0.87467
    Epoch 363/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.1006 - acc: 0.9658 - val_loss: 0.4440 - val_acc: 0.8509
    
    Epoch 00363: val_acc did not improve from 0.87467
    Epoch 364/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0993 - acc: 0.9682 - val_loss: 0.4446 - val_acc: 0.8517
    
    Epoch 00364: val_acc did not improve from 0.87467
    Epoch 365/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0983 - acc: 0.9690 - val_loss: 0.4458 - val_acc: 0.8515
    
    Epoch 00365: val_acc did not improve from 0.87467
    Epoch 366/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.1002 - acc: 0.9671 - val_loss: 0.4468 - val_acc: 0.8516
    
    Epoch 00366: val_acc did not improve from 0.87467
    Epoch 367/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0987 - acc: 0.9677 - val_loss: 0.4478 - val_acc: 0.8512
    
    Epoch 00367: val_acc did not improve from 0.87467
    Epoch 368/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.1004 - acc: 0.9670 - val_loss: 0.4484 - val_acc: 0.8517
    
    Epoch 00368: val_acc did not improve from 0.87467
    Epoch 369/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0998 - acc: 0.9672 - val_loss: 0.4489 - val_acc: 0.8512
    
    Epoch 00369: val_acc did not improve from 0.87467
    Epoch 370/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0972 - acc: 0.9683 - val_loss: 0.4495 - val_acc: 0.8504
    
    Epoch 00370: val_acc did not improve from 0.87467
    Epoch 371/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0973 - acc: 0.9677 - val_loss: 0.4502 - val_acc: 0.8504
    
    Epoch 00371: val_acc did not improve from 0.87467
    Epoch 372/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0971 - acc: 0.9679 - val_loss: 0.4513 - val_acc: 0.8509
    
    Epoch 00372: val_acc did not improve from 0.87467
    Epoch 373/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.0985 - acc: 0.9678 - val_loss: 0.4524 - val_acc: 0.8512
    
    Epoch 00373: val_acc did not improve from 0.87467
    Epoch 374/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.0975 - acc: 0.9686 - val_loss: 0.4533 - val_acc: 0.8500
    
    Epoch 00374: val_acc did not improve from 0.87467
    Epoch 375/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0978 - acc: 0.9675 - val_loss: 0.4536 - val_acc: 0.8500
    
    Epoch 00375: val_acc did not improve from 0.87467
    Epoch 376/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0970 - acc: 0.9695 - val_loss: 0.4539 - val_acc: 0.8499
    
    Epoch 00376: val_acc did not improve from 0.87467
    Epoch 377/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0977 - acc: 0.9679 - val_loss: 0.4545 - val_acc: 0.8508
    
    Epoch 00377: val_acc did not improve from 0.87467
    Epoch 378/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0960 - acc: 0.9684 - val_loss: 0.4553 - val_acc: 0.8501
    
    Epoch 00378: val_acc did not improve from 0.87467
    Epoch 379/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0953 - acc: 0.9691 - val_loss: 0.4562 - val_acc: 0.8495
    
    Epoch 00379: val_acc did not improve from 0.87467
    Epoch 380/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0963 - acc: 0.9691 - val_loss: 0.4571 - val_acc: 0.8497
    
    Epoch 00380: val_acc did not improve from 0.87467
    Epoch 381/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0947 - acc: 0.9698 - val_loss: 0.4580 - val_acc: 0.8503
    
    Epoch 00381: val_acc did not improve from 0.87467
    Epoch 382/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0960 - acc: 0.9687 - val_loss: 0.4589 - val_acc: 0.8505
    
    Epoch 00382: val_acc did not improve from 0.87467
    Epoch 383/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0946 - acc: 0.9695 - val_loss: 0.4597 - val_acc: 0.8501
    
    Epoch 00383: val_acc did not improve from 0.87467
    Epoch 384/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0956 - acc: 0.9683 - val_loss: 0.4605 - val_acc: 0.8500
    
    Epoch 00384: val_acc did not improve from 0.87467
    Epoch 385/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0945 - acc: 0.9691 - val_loss: 0.4611 - val_acc: 0.8504
    
    Epoch 00385: val_acc did not improve from 0.87467
    Epoch 386/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0944 - acc: 0.9704 - val_loss: 0.4617 - val_acc: 0.8507
    
    Epoch 00386: val_acc did not improve from 0.87467
    Epoch 387/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0940 - acc: 0.9697 - val_loss: 0.4625 - val_acc: 0.8505
    
    Epoch 00387: val_acc did not improve from 0.87467
    Epoch 388/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0944 - acc: 0.9702 - val_loss: 0.4634 - val_acc: 0.8503
    
    Epoch 00388: val_acc did not improve from 0.87467
    Epoch 389/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.0938 - acc: 0.9689 - val_loss: 0.4640 - val_acc: 0.8511
    
    Epoch 00389: val_acc did not improve from 0.87467
    Epoch 390/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0941 - acc: 0.9692 - val_loss: 0.4646 - val_acc: 0.8509
    
    Epoch 00390: val_acc did not improve from 0.87467
    Epoch 391/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0925 - acc: 0.9704 - val_loss: 0.4654 - val_acc: 0.8503
    
    Epoch 00391: val_acc did not improve from 0.87467
    Epoch 392/500
    17500/17500 [==============================] - 1s 42us/step - loss: 0.0941 - acc: 0.9701 - val_loss: 0.4662 - val_acc: 0.8496
    
    Epoch 00392: val_acc did not improve from 0.87467
    Epoch 393/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0946 - acc: 0.9690 - val_loss: 0.4667 - val_acc: 0.8495
    
    Epoch 00393: val_acc did not improve from 0.87467
    Epoch 394/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0939 - acc: 0.9705 - val_loss: 0.4675 - val_acc: 0.8501
    
    Epoch 00394: val_acc did not improve from 0.87467
    Epoch 395/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0928 - acc: 0.9710 - val_loss: 0.4683 - val_acc: 0.8499
    
    Epoch 00395: val_acc did not improve from 0.87467
    Epoch 396/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0925 - acc: 0.9702 - val_loss: 0.4694 - val_acc: 0.8500
    
    Epoch 00396: val_acc did not improve from 0.87467
    Epoch 397/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0928 - acc: 0.9706 - val_loss: 0.4702 - val_acc: 0.8503
    
    Epoch 00397: val_acc did not improve from 0.87467
    Epoch 398/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0921 - acc: 0.9715 - val_loss: 0.4709 - val_acc: 0.8501
    
    Epoch 00398: val_acc did not improve from 0.87467
    Epoch 399/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0929 - acc: 0.9691 - val_loss: 0.4716 - val_acc: 0.8503
    
    Epoch 00399: val_acc did not improve from 0.87467
    Epoch 400/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0920 - acc: 0.9705 - val_loss: 0.4721 - val_acc: 0.8499
    
    Epoch 00400: val_acc did not improve from 0.87467
    Epoch 401/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0911 - acc: 0.9718 - val_loss: 0.4730 - val_acc: 0.8500
    
    Epoch 00401: val_acc did not improve from 0.87467
    Epoch 402/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0924 - acc: 0.9703 - val_loss: 0.4737 - val_acc: 0.8496
    
    Epoch 00402: val_acc did not improve from 0.87467
    Epoch 403/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0912 - acc: 0.9704 - val_loss: 0.4746 - val_acc: 0.8496
    
    Epoch 00403: val_acc did not improve from 0.87467
    Epoch 404/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0917 - acc: 0.9713 - val_loss: 0.4755 - val_acc: 0.8499
    
    Epoch 00404: val_acc did not improve from 0.87467
    Epoch 405/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0902 - acc: 0.9708 - val_loss: 0.4763 - val_acc: 0.8493
    
    Epoch 00405: val_acc did not improve from 0.87467
    Epoch 406/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0901 - acc: 0.9711 - val_loss: 0.4772 - val_acc: 0.8491
    
    Epoch 00406: val_acc did not improve from 0.87467
    Epoch 407/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0908 - acc: 0.9707 - val_loss: 0.4780 - val_acc: 0.8492
    
    Epoch 00407: val_acc did not improve from 0.87467
    Epoch 408/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0903 - acc: 0.9699 - val_loss: 0.4784 - val_acc: 0.8491
    
    Epoch 00408: val_acc did not improve from 0.87467
    Epoch 409/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0897 - acc: 0.9724 - val_loss: 0.4791 - val_acc: 0.8496
    
    Epoch 00409: val_acc did not improve from 0.87467
    Epoch 410/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0902 - acc: 0.9716 - val_loss: 0.4800 - val_acc: 0.8491
    
    Epoch 00410: val_acc did not improve from 0.87467
    Epoch 411/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0893 - acc: 0.9715 - val_loss: 0.4807 - val_acc: 0.8493
    
    Epoch 00411: val_acc did not improve from 0.87467
    Epoch 412/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0890 - acc: 0.9725 - val_loss: 0.4817 - val_acc: 0.8493
    
    Epoch 00412: val_acc did not improve from 0.87467
    Epoch 413/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0885 - acc: 0.9730 - val_loss: 0.4826 - val_acc: 0.8491
    
    Epoch 00413: val_acc did not improve from 0.87467
    Epoch 414/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0884 - acc: 0.9729 - val_loss: 0.4831 - val_acc: 0.8491
    
    Epoch 00414: val_acc did not improve from 0.87467
    Epoch 415/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0890 - acc: 0.9722 - val_loss: 0.4838 - val_acc: 0.8491
    
    Epoch 00415: val_acc did not improve from 0.87467
    Epoch 416/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0871 - acc: 0.9723 - val_loss: 0.4849 - val_acc: 0.8492
    
    Epoch 00416: val_acc did not improve from 0.87467
    Epoch 417/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0884 - acc: 0.9717 - val_loss: 0.4859 - val_acc: 0.8488
    
    Epoch 00417: val_acc did not improve from 0.87467
    Epoch 418/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0875 - acc: 0.9731 - val_loss: 0.4868 - val_acc: 0.8488
    
    Epoch 00418: val_acc did not improve from 0.87467
    Epoch 419/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0876 - acc: 0.9711 - val_loss: 0.4875 - val_acc: 0.8489
    
    Epoch 00419: val_acc did not improve from 0.87467
    Epoch 420/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0876 - acc: 0.9726 - val_loss: 0.4882 - val_acc: 0.8489
    
    Epoch 00420: val_acc did not improve from 0.87467
    Epoch 421/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0883 - acc: 0.9726 - val_loss: 0.4886 - val_acc: 0.8491
    
    Epoch 00421: val_acc did not improve from 0.87467
    Epoch 422/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0867 - acc: 0.9738 - val_loss: 0.4891 - val_acc: 0.8491
    
    Epoch 00422: val_acc did not improve from 0.87467
    Epoch 423/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0858 - acc: 0.9730 - val_loss: 0.4901 - val_acc: 0.8489
    
    Epoch 00423: val_acc did not improve from 0.87467
    Epoch 424/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0862 - acc: 0.9737 - val_loss: 0.4915 - val_acc: 0.8489
    
    Epoch 00424: val_acc did not improve from 0.87467
    Epoch 425/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0859 - acc: 0.9742 - val_loss: 0.4924 - val_acc: 0.8491
    
    Epoch 00425: val_acc did not improve from 0.87467
    Epoch 426/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0860 - acc: 0.9744 - val_loss: 0.4929 - val_acc: 0.8491
    
    Epoch 00426: val_acc did not improve from 0.87467
    Epoch 427/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0861 - acc: 0.9737 - val_loss: 0.4936 - val_acc: 0.8492
    
    Epoch 00427: val_acc did not improve from 0.87467
    Epoch 428/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0858 - acc: 0.9730 - val_loss: 0.4941 - val_acc: 0.8491
    
    Epoch 00428: val_acc did not improve from 0.87467
    Epoch 429/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0853 - acc: 0.9745 - val_loss: 0.4948 - val_acc: 0.8491
    
    Epoch 00429: val_acc did not improve from 0.87467
    Epoch 430/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0842 - acc: 0.9743 - val_loss: 0.4957 - val_acc: 0.8488
    
    Epoch 00430: val_acc did not improve from 0.87467
    Epoch 431/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0851 - acc: 0.9736 - val_loss: 0.4967 - val_acc: 0.8488
    
    Epoch 00431: val_acc did not improve from 0.87467
    Epoch 432/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0852 - acc: 0.9741 - val_loss: 0.4975 - val_acc: 0.8484
    
    Epoch 00432: val_acc did not improve from 0.87467
    Epoch 433/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0853 - acc: 0.9737 - val_loss: 0.4979 - val_acc: 0.8485
    
    Epoch 00433: val_acc did not improve from 0.87467
    Epoch 434/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0844 - acc: 0.9754 - val_loss: 0.4986 - val_acc: 0.8488
    
    Epoch 00434: val_acc did not improve from 0.87467
    Epoch 435/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0827 - acc: 0.9756 - val_loss: 0.4997 - val_acc: 0.8489
    
    Epoch 00435: val_acc did not improve from 0.87467
    Epoch 436/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0841 - acc: 0.9742 - val_loss: 0.5007 - val_acc: 0.8488
    
    Epoch 00436: val_acc did not improve from 0.87467
    Epoch 437/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0832 - acc: 0.9739 - val_loss: 0.5012 - val_acc: 0.8488
    
    Epoch 00437: val_acc did not improve from 0.87467
    Epoch 438/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0831 - acc: 0.9737 - val_loss: 0.5014 - val_acc: 0.8487
    
    Epoch 00438: val_acc did not improve from 0.87467
    Epoch 439/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0837 - acc: 0.9744 - val_loss: 0.5019 - val_acc: 0.8487
    
    Epoch 00439: val_acc did not improve from 0.87467
    Epoch 440/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0823 - acc: 0.9758 - val_loss: 0.5030 - val_acc: 0.8488
    
    Epoch 00440: val_acc did not improve from 0.87467
    Epoch 441/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0841 - acc: 0.9738 - val_loss: 0.5040 - val_acc: 0.8489
    
    Epoch 00441: val_acc did not improve from 0.87467
    Epoch 442/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0836 - acc: 0.9745 - val_loss: 0.5045 - val_acc: 0.8493
    
    Epoch 00442: val_acc did not improve from 0.87467
    Epoch 443/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0832 - acc: 0.9741 - val_loss: 0.5049 - val_acc: 0.8489
    
    Epoch 00443: val_acc did not improve from 0.87467
    Epoch 444/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0825 - acc: 0.9745 - val_loss: 0.5054 - val_acc: 0.8488
    
    Epoch 00444: val_acc did not improve from 0.87467
    Epoch 445/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0823 - acc: 0.9754 - val_loss: 0.5064 - val_acc: 0.8492
    
    Epoch 00445: val_acc did not improve from 0.87467
    Epoch 446/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0815 - acc: 0.9753 - val_loss: 0.5075 - val_acc: 0.8491
    
    Epoch 00446: val_acc did not improve from 0.87467
    Epoch 447/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0825 - acc: 0.9738 - val_loss: 0.5083 - val_acc: 0.8491
    
    Epoch 00447: val_acc did not improve from 0.87467
    Epoch 448/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0818 - acc: 0.9751 - val_loss: 0.5089 - val_acc: 0.8487
    
    Epoch 00448: val_acc did not improve from 0.87467
    Epoch 449/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0811 - acc: 0.9758 - val_loss: 0.5096 - val_acc: 0.8484
    
    Epoch 00449: val_acc did not improve from 0.87467
    Epoch 450/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0825 - acc: 0.9747 - val_loss: 0.5105 - val_acc: 0.8484
    
    Epoch 00450: val_acc did not improve from 0.87467
    Epoch 451/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0805 - acc: 0.9757 - val_loss: 0.5114 - val_acc: 0.8484
    
    Epoch 00451: val_acc did not improve from 0.87467
    Epoch 452/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0816 - acc: 0.9747 - val_loss: 0.5122 - val_acc: 0.8485
    
    Epoch 00452: val_acc did not improve from 0.87467
    Epoch 453/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0807 - acc: 0.9750 - val_loss: 0.5128 - val_acc: 0.8479
    
    Epoch 00453: val_acc did not improve from 0.87467
    Epoch 454/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0816 - acc: 0.9754 - val_loss: 0.5133 - val_acc: 0.8481
    
    Epoch 00454: val_acc did not improve from 0.87467
    Epoch 455/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0801 - acc: 0.9758 - val_loss: 0.5141 - val_acc: 0.8484
    
    Epoch 00455: val_acc did not improve from 0.87467
    Epoch 456/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0789 - acc: 0.9767 - val_loss: 0.5153 - val_acc: 0.8484
    
    Epoch 00456: val_acc did not improve from 0.87467
    Epoch 457/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0796 - acc: 0.9766 - val_loss: 0.5164 - val_acc: 0.8484
    
    Epoch 00457: val_acc did not improve from 0.87467
    Epoch 458/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0791 - acc: 0.9774 - val_loss: 0.5171 - val_acc: 0.8483
    
    Epoch 00458: val_acc did not improve from 0.87467
    Epoch 459/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0799 - acc: 0.9764 - val_loss: 0.5173 - val_acc: 0.8485
    
    Epoch 00459: val_acc did not improve from 0.87467
    Epoch 460/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0782 - acc: 0.9778 - val_loss: 0.5182 - val_acc: 0.8485
    
    Epoch 00460: val_acc did not improve from 0.87467
    Epoch 461/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0783 - acc: 0.9774 - val_loss: 0.5195 - val_acc: 0.8484
    
    Epoch 00461: val_acc did not improve from 0.87467
    Epoch 462/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0790 - acc: 0.9774 - val_loss: 0.5206 - val_acc: 0.8485
    
    Epoch 00462: val_acc did not improve from 0.87467
    Epoch 463/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0781 - acc: 0.9767 - val_loss: 0.5214 - val_acc: 0.8485
    
    Epoch 00463: val_acc did not improve from 0.87467
    Epoch 464/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0786 - acc: 0.9758 - val_loss: 0.5219 - val_acc: 0.8487
    
    Epoch 00464: val_acc did not improve from 0.87467
    Epoch 465/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0785 - acc: 0.9757 - val_loss: 0.5225 - val_acc: 0.8485
    
    Epoch 00465: val_acc did not improve from 0.87467
    Epoch 466/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0774 - acc: 0.9770 - val_loss: 0.5233 - val_acc: 0.8484
    
    Epoch 00466: val_acc did not improve from 0.87467
    Epoch 467/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0786 - acc: 0.9766 - val_loss: 0.5241 - val_acc: 0.8491
    
    Epoch 00467: val_acc did not improve from 0.87467
    Epoch 468/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0768 - acc: 0.9782 - val_loss: 0.5254 - val_acc: 0.8491
    
    Epoch 00468: val_acc did not improve from 0.87467
    Epoch 469/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0771 - acc: 0.9775 - val_loss: 0.5267 - val_acc: 0.8488
    
    Epoch 00469: val_acc did not improve from 0.87467
    Epoch 470/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0766 - acc: 0.9773 - val_loss: 0.5274 - val_acc: 0.8489
    
    Epoch 00470: val_acc did not improve from 0.87467
    Epoch 471/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0782 - acc: 0.9779 - val_loss: 0.5275 - val_acc: 0.8491
    
    Epoch 00471: val_acc did not improve from 0.87467
    Epoch 472/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0761 - acc: 0.9785 - val_loss: 0.5279 - val_acc: 0.8491
    
    Epoch 00472: val_acc did not improve from 0.87467
    Epoch 473/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0768 - acc: 0.9774 - val_loss: 0.5286 - val_acc: 0.8488
    
    Epoch 00473: val_acc did not improve from 0.87467
    Epoch 474/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0752 - acc: 0.9793 - val_loss: 0.5299 - val_acc: 0.8488
    
    Epoch 00474: val_acc did not improve from 0.87467
    Epoch 475/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0763 - acc: 0.9792 - val_loss: 0.5312 - val_acc: 0.8487
    
    Epoch 00475: val_acc did not improve from 0.87467
    Epoch 476/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0760 - acc: 0.9791 - val_loss: 0.5320 - val_acc: 0.8488
    
    Epoch 00476: val_acc did not improve from 0.87467
    Epoch 477/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0777 - acc: 0.9767 - val_loss: 0.5326 - val_acc: 0.8485
    
    Epoch 00477: val_acc did not improve from 0.87467
    Epoch 478/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0758 - acc: 0.9786 - val_loss: 0.5330 - val_acc: 0.8488
    
    Epoch 00478: val_acc did not improve from 0.87467
    Epoch 479/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0750 - acc: 0.9784 - val_loss: 0.5337 - val_acc: 0.8485
    
    Epoch 00479: val_acc did not improve from 0.87467
    Epoch 480/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0768 - acc: 0.9783 - val_loss: 0.5345 - val_acc: 0.8487
    
    Epoch 00480: val_acc did not improve from 0.87467
    Epoch 481/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0751 - acc: 0.9788 - val_loss: 0.5352 - val_acc: 0.8488
    
    Epoch 00481: val_acc did not improve from 0.87467
    Epoch 482/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0746 - acc: 0.9787 - val_loss: 0.5361 - val_acc: 0.8487
    
    Epoch 00482: val_acc did not improve from 0.87467
    Epoch 483/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0747 - acc: 0.9795 - val_loss: 0.5370 - val_acc: 0.8488
    
    Epoch 00483: val_acc did not improve from 0.87467
    Epoch 484/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0743 - acc: 0.9790 - val_loss: 0.5377 - val_acc: 0.8484
    
    Epoch 00484: val_acc did not improve from 0.87467
    Epoch 485/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0759 - acc: 0.9784 - val_loss: 0.5383 - val_acc: 0.8483
    
    Epoch 00485: val_acc did not improve from 0.87467
    Epoch 486/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0747 - acc: 0.9786 - val_loss: 0.5391 - val_acc: 0.8489
    
    Epoch 00486: val_acc did not improve from 0.87467
    Epoch 487/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0738 - acc: 0.9793 - val_loss: 0.5402 - val_acc: 0.8485
    
    Epoch 00487: val_acc did not improve from 0.87467
    Epoch 488/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0735 - acc: 0.9797 - val_loss: 0.5415 - val_acc: 0.8484
    
    Epoch 00488: val_acc did not improve from 0.87467
    Epoch 489/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0735 - acc: 0.9803 - val_loss: 0.5421 - val_acc: 0.8481
    
    Epoch 00489: val_acc did not improve from 0.87467
    Epoch 490/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0729 - acc: 0.9801 - val_loss: 0.5424 - val_acc: 0.8484
    
    Epoch 00490: val_acc did not improve from 0.87467
    Epoch 491/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0731 - acc: 0.9795 - val_loss: 0.5434 - val_acc: 0.8488
    
    Epoch 00491: val_acc did not improve from 0.87467
    Epoch 492/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0740 - acc: 0.9787 - val_loss: 0.5446 - val_acc: 0.8485
    
    Epoch 00492: val_acc did not improve from 0.87467
    Epoch 493/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0732 - acc: 0.9795 - val_loss: 0.5456 - val_acc: 0.8487
    
    Epoch 00493: val_acc did not improve from 0.87467
    Epoch 494/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0729 - acc: 0.9798 - val_loss: 0.5464 - val_acc: 0.8485
    
    Epoch 00494: val_acc did not improve from 0.87467
    Epoch 495/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0719 - acc: 0.9805 - val_loss: 0.5470 - val_acc: 0.8483
    
    Epoch 00495: val_acc did not improve from 0.87467
    Epoch 496/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0717 - acc: 0.9816 - val_loss: 0.5477 - val_acc: 0.8481
    
    Epoch 00496: val_acc did not improve from 0.87467
    Epoch 497/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0724 - acc: 0.9802 - val_loss: 0.5484 - val_acc: 0.8479
    
    Epoch 00497: val_acc did not improve from 0.87467
    Epoch 498/500
    17500/17500 [==============================] - 1s 41us/step - loss: 0.0713 - acc: 0.9809 - val_loss: 0.5495 - val_acc: 0.8479
    
    Epoch 00498: val_acc did not improve from 0.87467
    Epoch 499/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0721 - acc: 0.9797 - val_loss: 0.5502 - val_acc: 0.8480
    
    Epoch 00499: val_acc did not improve from 0.87467
    Epoch 500/500
    17500/17500 [==============================] - 1s 40us/step - loss: 0.0710 - acc: 0.9811 - val_loss: 0.5509 - val_acc: 0.8480
    
    Epoch 00500: val_acc did not improve from 0.87467
    




    <keras.callbacks.History at 0x47b50828>




```python
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import LSTM, Embedding
from keras.models import Sequential
from keras.utils import plot_model

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
import gensim
from gensim.models.word2vec import Word2Vec
from keras.callbacks import ModelCheckpoint


model_MLP = Sequential()

model_MLP.add(Dense(10, input_shape=(4000,), activation='relu'))#

model_MLP.add(Dropout(0.2))
model_MLP.add(Dense(2, activation='softmax'))

model_MLP.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model_MLP.summary()

# load weights
model_MLP.load_weights("model_MLP/weights.best.hdf5")

```

    C:\ProgramData\Anaconda3\lib\site-packages\gensim\utils.py:1197: UserWarning: detected Windows; aliasing chunkize to chunkize_serial
      warnings.warn("detected Windows; aliasing chunkize to chunkize_serial")
    

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_11 (Dense)             (None, 10)                40010     
    _________________________________________________________________
    dropout_6 (Dropout)          (None, 10)                0         
    _________________________________________________________________
    dense_12 (Dense)             (None, 2)                 22        
    =================================================================
    Total params: 40,032
    Trainable params: 40,032
    Non-trainable params: 0
    _________________________________________________________________
    


```python
test_predicted = np.array(model_MLP.predict_classes(tfidf_test_x))
print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_mlp_tfidf.csv',columns = ['id','sentiment'], index = False)
print('结束.')
```

    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          0
    3    7186_2          1
    4   12128_7          1
    结束.
    

### 5. TF-IDF+DT


```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

model_DT = DecisionTreeClassifier()
model_DT.fit(tfidf_train_x, label)

scores = cross_val_score(model_DT, tfidf_train_x, label, cv=5, scoring='roc_auc')

print("决策树 5折交叉验证得分: ", scores)
print("决策树 5折交叉验证平均得分: ", np.mean(scores))

test_predicted = np.array(model_DT.predict(tfidf_test_x))

print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_dt_tfidf.csv',columns = ['id','sentiment'], index = False)
print('结束.')
```

    决策树 5折交叉验证得分: 
     [0.7128 0.705  0.7102 0.7062 0.7096]
    决策树 5折交叉验证平均得分: 
     0.7087600000000001
    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          1
    3    7186_2          1
    4   12128_7          1
    结束.
    

### 6. TF-IDF+xgboost


```python
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import numpy as np
import pandas as pd

model_XGB = XGBClassifier(n_estimators=150, min_samples_leaf=3, max_depth=6)
"""
AttributeError: 'list' object has no attribute 'shape'
list => np.array
"""
model_XGB.fit(tfidf_train_x, label)

scores = cross_val_score(model_XGB, tfidf_train_x, label, cv=5, scoring='roc_auc')

print("XGB 5折交叉验证得分: ", scores)
print("XGB 5折交叉验证平均得分: ", np.mean(scores))

test_predicted = np.array(model_XGB.predict(tfidf_test_x))

print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_xgb_tfidf.csv',columns = ['id','sentiment'], index = False)
print('结束.')
```

    XGB 5折交叉验证得分:  [0.91607008 0.91845504 0.91138704 0.922242   0.9142976 ]
    XGB 5折交叉验证平均得分:  0.916490352
    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          1
    3    7186_2          1
    4   12128_7          1
    结束.
    

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\preprocessing\label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:
    

### 7. TF-IDF+GBDT


```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

model_GBDT = GradientBoostingClassifier()
model_GBDT.fit(tfidf_train_x, label)

scores = cross_val_score(model_GBDT, tfidf_train_x, label, cv=5, scoring='roc_auc')

print("GBDT 5折交叉验证得分: \n", scores)
print("GBDT 折交叉验证平均得分: \n", np.mean(scores))

test_predicted = np.array(model_GBDT.predict(tfidf_test_x))

print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_gbdt_tfidf.csv',columns = ['id','sentiment'], index = False)
print('结束.')
```

    GBDT 5折交叉验证得分: 
     [0.88863632 0.88981456 0.88274104 0.89644848 0.88723104]
    GBDT 折交叉验证平均得分: 
     0.888974288
    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          1
    3    7186_2          1
    4   12128_7          1
    结束.
    

### 8. TF-IDF+RF


```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

model_RF = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=0)
model_RF.fit(tfidf_train_x, label)

scores = cross_val_score(model_RF, tfidf_train_x, label, cv=5, scoring='roc_auc')

print("随机森林 5折交叉验证得分: ", scores)
print("随机森林 5折交叉验证平均得分: ", np.mean(scores))

test_predicted = np.array(model_RF.predict(tfidf_test_x))

print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_rfc_tfidf.csv',columns = ['id','sentiment'], index = False)
print('结束.')
```

    随机森林 5折交叉验证得分:  [0.89770248 0.90205984 0.89408512 0.9035528  0.89733304]
    随机森林 5折交叉验证平均得分:  0.898946656
    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          1
    3    7186_2          1
    4   12128_7          1
    结束.
    

### 9. TF-IDF+Voting


```python
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score

model_VOT = VotingClassifier(estimators=[('lr', model_LR), ('xgb', model_XGB), ('rf', model_RF)],voting='hard')
model_VOT.fit(tfidf_train_x, np.array(label))

scores = cross_val_score(model_VOT, tfidf_train_x,label, cv=5, scoring=None,n_jobs = -1)

print("VotingClassifier 5折交叉验证得分: \n", scores)
print("VotingClassifier 5折交叉验证平均得分: \n", np.mean(scores))

test_predicted = np.array(model_VOT.predict(tfidf_test_x))

print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_vot_tfidf.csv',columns = ['id','sentiment'], index = False)
print('结束.')
```

    VotingClassifier 5折交叉验证得分: 
     [0.8394 0.8568 0.8388 0.854  0.8464]
    VotingClassifier 5折交叉验证平均得分: 
     0.84708
    

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\preprocessing\label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:
    

    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          1
    3    7186_2          1
    4   12128_7          1
    结束.
    

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\preprocessing\label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:
    

### 10. TF-IDF+Stacking


```python
'''模型融合中使用到的各个单模型'''
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
#from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc

# 划分train数据集,调用代码,把数据集名字转成和代码一样
X = tfidf_train_x
y = np.array(label)

X_test_features = tfidf_test_x

stacking_LR = LR(penalty='l2', dual=True, random_state=0)

stacking_xgb = XGBClassifier(n_estimators=150, min_samples_leaf=3, max_depth=6)

stacking_rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=0)


clfs = [stacking_LR,stacking_xgb,stacking_rf]#模型

# 创建n_folds
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=1)#K-Flod数据

# 创建零矩阵（存储第一层的预测结果）
dataset_blend_train = np.zeros((X.shape[0], len(clfs)))#行数：训练集的行数，列数：模型的个数

dataset_blend_test = np.zeros((X_test_features.shape[0], len(clfs)))#行数：测试集数量，列数：模型的个数
dataset_blend_test_j = np.zeros((X_test_features.shape[0], n_folds))#行数：测试集数量，列数：K折

# 建立第一层模型
for j, clf in enumerate(clfs):#枚举分类器
    i = 0
    for train_index, test_index in skf.split(X, y):#K折数据
        X_1_train, y_1_train, X_1_test, y_1_test = X[train_index], y[train_index], X[test_index], y[test_index]
        
        clf.fit(X_1_train, y_1_train)#第j个模型预测第k折数据
        
        y_submission = clf.predict_proba(X_1_test)[:, 1]#第j个模型预测剩下的1折数据，去除答案是1的概率列
        
        dataset_blend_train[test_index, j] = y_submission#第j个模型预测的第k折数据的答案写到预测结果里
        
        dataset_blend_test_j[:, i] = clf.predict_proba(X_test_features)[:, 1]#对测试集进行预测
        
        i = i + 1 #第i折
        
    '''对于测试集，直接用这k个模型的预测值均值作为新的特征。'''
    dataset_blend_test[:, j] = dataset_blend_test_j.mean(1) #每个模型的K折的预测值取平均做为第j个分类器的预测值
    
# 用建立第二层模型

C = [0.01,0.1,1,10]

for i in C:
    stacking_model_lr = LR(C=i, max_iter=100)
    print(i)
    aucs = []
    for train_index, test_index in skf.split(dataset_blend_train, y):#K折数据
        X_2_train, y_2_train, X_2_test, y_2_test = dataset_blend_train[train_index], y[train_index], dataset_blend_train[test_index], y[test_index]
        stacking_model_lr.fit(X_2_train, y_2_train)
        test_predict_proba = stacking_model_lr.predict_proba(X_2_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_2_test, test_predict_proba, pos_label=1)
        print("stacking auc",auc(fpr, tpr))
        aucs.append(auc(fpr, tpr))
    print(np.average(aucs))

    stacking_model_lr = LR(C=10, max_iter=100)

stacking_model_lr.fit(dataset_blend_train, y)

test_predict = stacking_model_lr.predict(dataset_blend_test)

test_predicted = np.array(test_predict)

print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_stacking_tfidf.csv',columns = ['id','sentiment'], index = False)
print('结束.')
```

    0.01
    stacking auc 0.9520484800000001
    stacking auc 0.94503472
    stacking auc 0.94511664
    stacking auc 0.9553464
    stacking auc 0.94973888
    0.949457024
    0.1
    stacking auc 0.95298912
    stacking auc 0.9471843200000001
    stacking auc 0.9471433599999999
    stacking auc 0.9565649600000001
    stacking auc 0.9505606400000002
    0.95088848
    1
    stacking auc 0.9529368000000001
    stacking auc 0.9477648
    stacking auc 0.94768992
    stacking auc 0.9566342399999999
    stacking auc 0.95043744
    0.9510926399999999
    10
    stacking auc 0.9528793600000001
    stacking auc 0.9478297600000001
    stacking auc 0.94773648
    stacking auc 0.9566390399999998
    stacking auc 0.9503939199999999
    0.9510957120000001
    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          1
    3    7186_2          1
    4   12128_7          1
    结束.
    

## 二、Word2vec+机器学习建模

### 1.Word2vec+LR


```python
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import cross_val_score

wv_model_LR = LR(penalty='l2', dual=True, random_state=0)
wv_model_LR.fit(train_data_features, label)

scores = cross_val_score(wv_model_LR, train_data_features, label, cv=10, scoring='roc_auc')

print("LR分类器 10折交叉验证得分: \n", scores)
print("LR分类器 10折交叉验证平均得分: \n", np.mean(scores))

test_predicted = np.array(wv_model_LR.predict(test_data_features))
print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_lr_wv.csv',columns = ['id','sentiment'], index = False)
print('结束.')
```

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)
    

    LR分类器 10折交叉验证得分: 
     [0.9412576  0.93986304 0.94739904 0.93963392 0.93683136 0.93972416
     0.94117184 0.94543232 0.93660928 0.94397184]
    LR分类器 10折交叉验证平均得分: 
     0.9411894399999999
    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          1
    3    7186_2          0
    4   12128_7          1
    结束.
    

![image.png](attachment:image.png)

### 2.Word2vec+GNB


```python
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.cross_validation import cross_val_score

wv_gnb_model = GNB()
wv_gnb_model.fit(train_data_features, label)


scores = cross_val_score(wv_gnb_model, train_data_features, label, cv=10, scoring='roc_auc')
print("\n高斯贝叶斯分类器 10折交叉验证得分: \n", scores)
print("\n高斯贝叶斯分类器 10折交叉验证平均得分: \n", np.mean(scores))

test_predicted = np.array(wv_gnb_model.predict(test_data_features))
print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_gnb_wv.csv',columns = ['id','sentiment'], index = False)
print('结束.')
```

    
    高斯贝叶斯分类器 10折交叉验证得分: 
     [0.80043648 0.78343712 0.78856768 0.79913536 0.7964928  0.78470464
     0.79811424 0.80671424 0.7928432  0.80504768]
    
    高斯贝叶斯分类器 10折交叉验证平均得分: 
     0.7955493440000001
    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          0
    3    7186_2          0
    4   12128_7          0
    结束.
    

![image.png](attachment:image.png)

### 3.Word2vec+Knn


```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

wv_knn_model = KNeighborsClassifier(n_neighbors=5)
wv_knn_model.fit(train_data_features, label)

scores = cross_val_score(wv_knn_model, train_data_features, label, cv=10, scoring='roc_auc')

print("\nknn算法 10折交叉验证得分: \n", scores)
print("\nknn算法 10折交叉验证平均得分: \n", np.mean(scores))

test_predicted = np.array(wv_knn_model.predict(test_data_features))
print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_knn_wv.csv',columns = ['id','sentiment'], index = False)
print('结束.')
```

    
    knn算法 10折交叉验证得分: 
     [0.89709856 0.89423168 0.90665152 0.8890768  0.89263712 0.89179264
     0.88568    0.90213216 0.89707968 0.88941376]
    
    knn算法 10折交叉验证平均得分: 
     0.894579392
    

    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel\__main__.py:5: DeprecationWarning: Call to deprecated `__contains__` (Method will be removed in 4.0.0, use self.wv.__contains__() instead).
    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel\__main__.py:6: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).
    

    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          0
    3    7186_2          0
    4   12128_7          1
    结束.
    

### 4.Word2vec+SVM


```python
from sklearn.svm import SVC

'''
线性的SVM只需要，只需要调优正则化参数C
基于RBF核的SVM，需要调优gamma参数和C
'''

wv_svm_model = SVC(kernel='linear',C=10,gamma = 1)
wv_svm_model.fit(train_data_features, label)

```

    最好的参数：
    


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-60-5591ac426418> in <module>()
         14 # 输出结果
         15 print("最好的参数：")
    ---> 16 print( wv_svm_model.best_params_)
         17 
         18 print("最好的得分：")
    

    AttributeError: 'SVC' object has no attribute 'best_params_'



```python
test_predicted = np.array(wv_svm_model.predict(test_data_features))
print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_svm_wv.csv',columns = ['id','sentiment'], index = False)
print('结束.')
```

    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          1
    3    7186_2          0
    4   12128_7          1
    结束.
    

### 5.Word2vec+DT


```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

wv_tree_model = DecisionTreeClassifier()
wv_tree_model.fit(train_data_features, label)

scores = cross_val_score(wv_tree_model, train_data_features, label, cv=5, scoring='roc_auc')

print("决策树 10折交叉验证得分: \n", scores)
print("决策树 10折交叉验证平均得分: \n", np.mean(scores))

test_predicted = np.array(wv_tree_model.predict(test_data_features))

print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_dtc_wv.csv',columns = ['id','sentiment'], index = False)
print('结束.')
```

    
    决策树 10折交叉验证得分: 
     [0.7498 0.7506 0.7572 0.7648 0.736 ]
    
    决策树 10折交叉验证平均得分: 
     0.75168
    

    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel\__main__.py:5: DeprecationWarning: Call to deprecated `__contains__` (Method will be removed in 4.0.0, use self.wv.__contains__() instead).
    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel\__main__.py:6: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).
    

    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          0
    3    7186_2          0
    4   12128_7          1
    结束.
    

### 6.Word2vec+xgboost


```python
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import numpy as np
import pandas as pd

wv_xgb_model = XGBClassifier(n_estimators=150, min_samples_leaf=3, max_depth=6)
"""
AttributeError: 'list' object has no attribute 'shape'
list => np.array
"""
wv_xgb_model.fit(pd.DataFrame(train_data_features), label)

scores = cross_val_score(wv_xgb_model, pd.DataFrame(train_data_features), label, cv=5, scoring='roc_auc')

print("XGB 5折交叉验证得分: \n", scores)
print("XGB 5折交叉验证平均得分: \n", np.mean(scores))

test_predicted = np.array(wv_xgb_model.predict(pd.DataFrame(test_data_features)))

print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_xgb_wv.csv',columns = ['id','sentiment'], index = False)
print('结束.')
```

    XGB 5折交叉验证得分: 
     [0.94458    0.94526208 0.94390896 0.9463752  0.94129232]
    XGB 5折交叉验证平均得分: 
     0.944283712
    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          1
    3    7186_2          0
    4   12128_7          1
    结束.
    

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\preprocessing\label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:
    

### 7.Word2vec+RF


```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

wv_rf_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=0)
wv_rf_model.fit(train_data_features, label)

scores = cross_val_score(wv_rf_model, train_data_features, label, cv=5, scoring='roc_auc')

print("\n随机森林 5折交叉验证得分: \n", scores)
print("\n随机森林 5折交叉验证平均得分: \n", np.mean(scores))

test_predicted = np.array(wv_rf_model.predict(test_data_features))
print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_rf_wv.csv',columns = ['id','sentiment'], index = False)
print('结束.')
```

    
    随机森林 5折交叉验证得分: 
     [0.9182096  0.91855744 0.91677872 0.92213184 0.91022928]
    
    随机森林 5折交叉验证平均得分: 
     0.9171813759999999
    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          0
    3    7186_2          0
    4   12128_7          1
    结束.
    

### 8.Word2vec+GBDT


```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

wv_gbdt_model = GradientBoostingClassifier()
wv_gbdt_model.fit(train_data_features, label)

scores = cross_val_score(wv_gbdt_model, train_data_features, label, cv=5, scoring='roc_auc')

print("GBDT 5折交叉验证得分: \n", scores)
print("GBDT 折交叉验证平均得分: \n", np.mean(scores))

test_predicted = np.array(wv_gbdt_model.predict(test_data_features))

print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_gbdt_wv.csv',columns = ['id','sentiment'], index = False)
print('结束.')
```

    GBDT 5折交叉验证得分: 
     [0.9328144  0.93499104 0.93023024 0.93560096 0.92776608]
    GBDT 折交叉验证平均得分: 
     0.9322805439999999
    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          1
    3    7186_2          0
    4   12128_7          1
    结束.
    

### 9.Word2vec+Adaboost

* adaboost模型训练太耗时,没跑完


```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score

wv_ab_model = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=600,
    learning_rate=1)

wv_ab_model.fit(train_data_features, label)

scores = cross_val_score(wv_ab_model, train_data_features, label, cv=3, scoring='roc_auc')

print("AdaBoost 3折交叉验证得分: \n", scores)
print("AdaBoost 3折交叉验证平均得分: \n", np.mean(scores))

test_predicted = np.array(wv_ab_model.predict(test_data_features))
print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_adaboost_wv.csv',columns = ['id','sentiment'], index = False)
print('结束.')
```

### 10.Word2vec+Voting


```python
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score

wv_vot_model = VotingClassifier(estimators=[('lr', wv_model_LR), ('xgb', wv_xgb_model), ('gbdt', wv_gbdt_model),('rf', wv_rf_model)],voting='hard')
wv_vot_model.fit(pd.DataFrame(train_data_features), np.array(label))

scores = cross_val_score(wv_vot_model, pd.DataFrame(train_data_features),label, cv=5, scoring=None,n_jobs = -1)

print("VotingClassifier 5折交叉验证得分: \n", scores)
print("VotingClassifier 5折交叉验证平均得分: \n", np.mean(scores))

test_predicted = np.array(wv_vot_model.predict(pd.DataFrame(test_data_features)))
print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_vot_wv.csv',columns = ['id','sentiment'], index = False)
print('结束.')
```

    VotingClassifier 5折交叉验证得分: 
     [0.8656 0.8702 0.8644 0.8674 0.8572]
    VotingClassifier 5折交叉验证平均得分: 
     0.86496
    

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\preprocessing\label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:
    

    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          1
    3    7186_2          0
    4   12128_7          1
    结束.
    

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\preprocessing\label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:
    

### 11.Word2vec+Stacking


```python
#K折数据切分
from sklearn.model_selection import StratifiedKFold
import numpy as np
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [5, 6],[7, 8]])
y = np.array([0, 0, 0, 1, 1, 1])#1的个数和0的个数要大于3，3也就是n_splits
skf = StratifiedKFold(n_splits=3,shuffle=True,random_state=1)

for train_index, test_index in skf.split(X, y):
   print("TRAIN:", train_index, "TEST:", test_index)

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

print(X_train)
print(X_test)

print(y_train)
print(y_test)
```

    TRAIN: [1 2 4 5] TEST: [0 3]
    TRAIN: [0 1 3 4] TEST: [2 5]
    TRAIN: [0 2 3 5] TEST: [1 4]
    [[1 2]
     [1 2]
     [3 4]
     [7 8]]
    [[3 4]
     [5 6]]
    [0 0 1 1]
    [0 1]
    


```python
'''模型融合中使用到的各个单模型'''
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
#from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc

# 划分train数据集,调用代码,把数据集名字转成和代码一样
X = np.array(train_data_features)
y = np.array(label)

X_test_features = np.array(test_data_features)

stacking_LR = LR(penalty='l2', dual=True, random_state=0)

stacking_xgb = XGBClassifier(n_estimators=150, min_samples_leaf=3, max_depth=6)

stacking_rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=0)


clfs = [stacking_LR,stacking_xgb,stacking_rf]#模型

# 创建n_folds
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=1)#K-Flod数据

# 创建零矩阵（存储第一层的预测结果）
dataset_blend_train = np.zeros((X.shape[0], len(clfs)))#行数：训练集的行数，列数：模型的个数

dataset_blend_test = np.zeros((X_test_features.shape[0], len(clfs)))#行数：测试集数量，列数：模型的个数
dataset_blend_test_j = np.zeros((X_test_features.shape[0], n_folds))#行数：测试集数量，列数：K折

# 建立第一层模型
for j, clf in enumerate(clfs):#枚举分类器
    i = 0
    for train_index, test_index in skf.split(X, y):#K折数据
        X_1_train, y_1_train, X_1_test, y_1_test = X[train_index], y[train_index], X[test_index], y[test_index]
        
        clf.fit(X_1_train, y_1_train)#第j个模型预测第k折数据
        
        y_submission = clf.predict_proba(X_1_test)[:, 1]#第j个模型预测剩下的1折数据，去除答案是1的概率列
        
        dataset_blend_train[test_index, j] = y_submission#第j个模型预测的第k折数据的答案写到预测结果里
        
        dataset_blend_test_j[:, i] = clf.predict_proba(X_test_features)[:, 1]#对测试集进行预测
        
        i = i + 1 #第i折
        
    '''对于测试集，直接用这k个模型的预测值均值作为新的特征。'''
    dataset_blend_test[:, j] = dataset_blend_test_j.mean(1) #每个模型的K折的预测值取平均做为第j个分类器的预测值
```


```python
# 用建立第二层模型

C = [0.01,0.1,1,10]

for i in C:
    stacking_model_lr = LR(C=i, max_iter=100)
    print(i)
    aucs = []
    for train_index, test_index in skf.split(dataset_blend_train, y):#K折数据
        X_2_train, y_2_train, X_2_test, y_2_test = dataset_blend_train[train_index], y[train_index], dataset_blend_train[test_index], y[test_index]
        stacking_model_lr.fit(X_2_train, y_2_train)
        test_predict_proba = stacking_model_lr.predict_proba(X_2_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_2_test, test_predict_proba, pos_label=1)
        print("stacking auc",auc(fpr, tpr))
        aucs.append(auc(fpr, tpr))
    print(np.average(aucs))

```

    0.01
    stacking auc 0.9506499199999999
    stacking auc 0.94645408
    stacking auc 0.9462804800000002
    stacking auc 0.9520553599999999
    stacking auc 0.9493892799999999
    0.9489658240000001
    0.1
    stacking auc 0.9507424000000001
    stacking auc 0.94657616
    stacking auc 0.9463376
    stacking auc 0.95219488
    stacking auc 0.9494544
    0.949061088
    1
    stacking auc 0.95079408
    stacking auc 0.9466462400000001
    stacking auc 0.9463564799999999
    stacking auc 0.9522491200000001
    stacking auc 0.94948288
    0.9491057599999999
    10
    stacking auc 0.9508049599999999
    stacking auc 0.94665712
    stacking auc 0.94635328
    stacking auc 0.9522579200000001
    stacking auc 0.94948464
    0.949111584
    


```python
#stacking 预测
stacking_model_lr = LR(C=10, max_iter=100)

stacking_model_lr.fit(dataset_blend_train, y)

test_predict = stacking_model_lr.predict(dataset_blend_test)

test_predicted = np.array(test_predict)

print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_stacking_wv.csv',columns = ['id','sentiment'], index = False)
print('结束.')
```

    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          1
    3    7186_2          0
    4   12128_7          1
    结束.
    

# 深度学习方法

## 一、深度学习建模

### 1.LSTM


```python
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import LSTM, Embedding
from keras.models import Sequential

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
from keras.callbacks import ModelCheckpoint

MAX_SEQUENCE_LENGTH = 100 # 每条新闻最大长度
EMBEDDING_DIM = 100       # 词向量空间维度

all_data = train_data+test_data
#Tokenizer是一个用于向量化文本，或将文本转换为序列（即单词在字典中的下标构成的列表，从1算起）的类
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_data)
sequences = tokenizer.texts_to_sequences(all_data)

#总共词数(word_index：key:词，value:索引)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
#print(word_index)

#将整篇文章根据向量化文本序列都退少补生成文章矩阵
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data.shape)

x_train,x_test = data[:len(train_data)],data[len(train_data):]

#将标签独热向量处理
labels = to_categorical(np.asarray(label))
print('Shape of label tensor:', labels.shape)


model = Sequential()
model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))#LSTM参数：LSTM的输出向量的维度
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
model.summary()

model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])

VALIDATION_SPLIT = 0.16 # 验证集比例
TEST_SPLIT = 0.2 # 测试集比例

p1 = int(len(x_train)*(1-VALIDATION_SPLIT-TEST_SPLIT))
p2 = int(len(x_train)*(1-TEST_SPLIT))

train_x = x_train[:p1]
train_y = labels[:p1]
val_x = x_train[p1:p2]
val_y = labels[p1:p2]
test_x = x_train[p2:]
test_y = labels[p2:]

print ('train docs: '+str(len(train_x)))
print ('val docs: '+str(len(val_x)))
print ('test docs: '+str(len(test_x)))

filepath="lstm_model/lstm_weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,mode='max')
callbacks_list = [checkpoint]

# Fit the model
#model.fit(X, Y, validation_split=0.33, nb_epoch=150, batch_size=10,callbacks=callbacks_list, verbose=0)

model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=12, batch_size=5000,callbacks=callbacks_list)

#model.save('word_vector_cnn.h5')
print (model.evaluate(test_x, test_y))

test_predicted = np.array(model.predict_classes(x_test))

print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_lstm.csv',columns = ['id','sentiment'], index = False)
print('结束.')

```

    Found 101245 unique tokens.
    Shape of data tensor: (50000, 100)
    Shape of label tensor: (25000, 2)
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_4 (Embedding)      (None, 100, 100)          10124600  
    _________________________________________________________________
    lstm_5 (LSTM)                (None, 100)               80400     
    _________________________________________________________________
    dropout_6 (Dropout)          (None, 100)               0         
    _________________________________________________________________
    dense_7 (Dense)              (None, 2)                 202       
    =================================================================
    Total params: 10,205,202
    Trainable params: 10,205,202
    Non-trainable params: 0
    _________________________________________________________________
    train docs: 15999
    val docs: 4001
    test docs: 5000
    Train on 15999 samples, validate on 4001 samples
    Epoch 1/12
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.6922 - acc: 0.5401 - val_loss: 0.6898 - val_acc: 0.6151
    
    Epoch 00001: val_acc improved from -inf to 0.61510, saving model to lstm_model/lstm_weights-improvement-01-0.62.hdf5
    Epoch 2/12
    15999/15999 [==============================] - 40s 2ms/step - loss: 0.6865 - acc: 0.6664 - val_loss: 0.6823 - val_acc: 0.6776
    
    Epoch 00002: val_acc improved from 0.61510 to 0.67758, saving model to lstm_model/lstm_weights-improvement-02-0.68.hdf5
    Epoch 3/12
    15999/15999 [==============================] - 38s 2ms/step - loss: 0.6728 - acc: 0.7438 - val_loss: 0.6590 - val_acc: 0.7423
    
    Epoch 00003: val_acc improved from 0.67758 to 0.74231, saving model to lstm_model/lstm_weights-improvement-03-0.74.hdf5
    Epoch 4/12
    15999/15999 [==============================] - 47s 3ms/step - loss: 0.6261 - acc: 0.8066 - val_loss: 0.5257 - val_acc: 0.7908
    
    Epoch 00004: val_acc improved from 0.74231 to 0.79080, saving model to lstm_model/lstm_weights-improvement-04-0.79.hdf5
    Epoch 5/12
    15999/15999 [==============================] - 46s 3ms/step - loss: 0.5113 - acc: 0.7992 - val_loss: 0.4856 - val_acc: 0.8180
    
    Epoch 00005: val_acc improved from 0.79080 to 0.81805, saving model to lstm_model/lstm_weights-improvement-05-0.82.hdf5
    Epoch 6/12
    15999/15999 [==============================] - 40s 2ms/step - loss: 0.4427 - acc: 0.8469 - val_loss: 0.3895 - val_acc: 0.8460
    
    Epoch 00006: val_acc improved from 0.81805 to 0.84604, saving model to lstm_model/lstm_weights-improvement-06-0.85.hdf5
    Epoch 7/12
    15999/15999 [==============================] - 44s 3ms/step - loss: 0.3473 - acc: 0.8820 - val_loss: 0.3471 - val_acc: 0.8583
    
    Epoch 00007: val_acc improved from 0.84604 to 0.85829, saving model to lstm_model/lstm_weights-improvement-07-0.86.hdf5
    Epoch 8/12
    15999/15999 [==============================] - 46s 3ms/step - loss: 0.2877 - acc: 0.9022 - val_loss: 0.3206 - val_acc: 0.8670
    
    Epoch 00008: val_acc improved from 0.85829 to 0.86703, saving model to lstm_model/lstm_weights-improvement-08-0.87.hdf5
    Epoch 9/12
    15999/15999 [==============================] - 46s 3ms/step - loss: 0.2308 - acc: 0.9202 - val_loss: 0.3152 - val_acc: 0.8693
    
    Epoch 00009: val_acc improved from 0.86703 to 0.86928, saving model to lstm_model/lstm_weights-improvement-09-0.87.hdf5
    Epoch 10/12
    15999/15999 [==============================] - 46s 3ms/step - loss: 0.1895 - acc: 0.9388 - val_loss: 0.3132 - val_acc: 0.8735
    
    Epoch 00010: val_acc improved from 0.86928 to 0.87353, saving model to lstm_model/lstm_weights-improvement-10-0.87.hdf5
    Epoch 11/12
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.1540 - acc: 0.9499 - val_loss: 0.3206 - val_acc: 0.8725
    
    Epoch 00011: val_acc did not improve from 0.87353
    Epoch 12/12
    15999/15999 [==============================] - 38s 2ms/step - loss: 0.1284 - acc: 0.9604 - val_loss: 0.3267 - val_acc: 0.8750
    
    Epoch 00012: val_acc improved from 0.87353 to 0.87503, saving model to lstm_model/lstm_weights-improvement-12-0.88.hdf5
    5000/5000 [==============================] - 2s 496us/step
    [0.3418441625118256, 0.866]
    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          0
    3    7186_2          1
    4   12128_7          1
    结束.
    

### 2.CNN


```python
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Sequential
from keras.utils import plot_model

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import numpy as np


MAX_SEQUENCE_LENGTH = 100 # 每条新闻最大长度
EMBEDDING_DIM = 100 # 词向量空间维度



#合并训练集和测试集
all_data = train_data+test_data

#Tokenizer是一个用于向量化文本，或将文本转换为序列（即单词在字典中的下标构成的列表，从1算起）的类
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_data)
sequences = tokenizer.texts_to_sequences(all_data)

#总共词数(word_index：key:词，value:词频)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
#print(word_index)

#将整篇文章根据向量化文本序列多退少补生成文章矩阵
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data.shape)

x_train,x_test = data[:len(train_data)],data[len(train_data):]

#将标签独热向量处理
labels = to_categorical(np.asarray(label))
print('Shape of label tensor:', labels.shape)

VALIDATION_SPLIT = 0.16 # 验证集比例
TEST_SPLIT = 0.2 # 测试集比例

p1 = int(len(x_train)*(1-VALIDATION_SPLIT-TEST_SPLIT))
p2 = int(len(x_train)*(1-TEST_SPLIT))

train_x = x_train[:p1]
train_y = labels[:p1]
val_x = x_train[p1:p2]
val_y = labels[p1:p2]
test_x = x_train[p2:]
test_y = labels[p2:]

print ('train docs: '+str(len(train_x)))
print ('val docs: '+str(len(val_x)))
print ('test docs: '+str(len(test_x)))

model = Sequential()
model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(Dropout(0.2))
model.add(Conv1D(250, 3, padding='valid', activation='relu', strides=1))
model.add(MaxPooling1D(3))
model.add(Flatten())
model.add(Dense(EMBEDDING_DIM, activation='relu'))
model.add(Dense(labels.shape[1], activation='softmax'))
model.summary()

#plot_model(model, to_file='model.png',show_shapes=True)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])


filepath="cnn_weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,mode='max')
callbacks_list = [checkpoint]

# Fit the model
#model.fit(X, Y, validation_split=0.33, nb_epoch=150, batch_size=10,callbacks=callbacks_list, verbose=0)


model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=12, batch_size=5000,callbacks=callbacks_list)

#model.save('word_vector_cnn.h5')
print (model.evaluate(test_x, test_y))

test_predicted = np.array(model.predict_classes(x_test))

print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_cnn.csv',columns = ['id','sentiment'], index = False)
print('结束.')

```

    C:\ProgramData\Anaconda3\lib\site-packages\h5py\__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.
    

    Found 101245 unique tokens.
    Shape of data tensor: (50000, 100)
    Shape of label tensor: (25000, 2)
    train docs: 15999
    val docs: 4001
    test docs: 5000
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 100, 100)          10124600  
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 100, 100)          0         
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 98, 250)           75250     
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 32, 250)           0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 8000)              0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 100)               800100    
    _________________________________________________________________
    dense_2 (Dense)              (None, 2)                 202       
    =================================================================
    Total params: 11,000,152
    Trainable params: 11,000,152
    Non-trainable params: 0
    _________________________________________________________________
    Train on 15999 samples, validate on 4001 samples
    Epoch 1/6
    15000/15999 [===========================>..] - ETA: 21s - loss: 1.0420 - acc: 0.5028 

    C:\ProgramData\Anaconda3\lib\site-packages\keras\callbacks.py:122: UserWarning: Method on_batch_end() is slow compared to the batch update (5.411809). Check your callbacks.
      % delta_t_median)
    

    15999/15999 [==============================] - 360s 22ms/step - loss: 1.0201 - acc: 0.5049 - val_loss: 0.6928 - val_acc: 0.4989
    
    Epoch 00001: val_acc improved from -inf to 0.49888, saving model to weights-improvement-01-0.50.hdf5
    Epoch 2/6
    15999/15999 [==============================] - 125s 8ms/step - loss: 0.6877 - acc: 0.5836 - val_loss: 0.6910 - val_acc: 0.4986
    
    Epoch 00002: val_acc did not improve from 0.49888
    Epoch 3/6
    15999/15999 [==============================] - 36s 2ms/step - loss: 0.6810 - acc: 0.5130 - val_loss: 0.6879 - val_acc: 0.4986
    
    Epoch 00003: val_acc did not improve from 0.49888
    Epoch 4/6
    15999/15999 [==============================] - 33s 2ms/step - loss: 0.6656 - acc: 0.5419 - val_loss: 0.6849 - val_acc: 0.4994
    
    Epoch 00004: val_acc improved from 0.49888 to 0.49938, saving model to weights-improvement-04-0.50.hdf5
    Epoch 5/6
    15999/15999 [==============================] - 56s 4ms/step - loss: 0.6498 - acc: 0.5272 - val_loss: 0.6014 - val_acc: 0.7183
    
    Epoch 00005: val_acc improved from 0.49938 to 0.71832, saving model to weights-improvement-05-0.72.hdf5
    Epoch 6/6
    15999/15999 [==============================] - 34s 2ms/step - loss: 0.5499 - acc: 0.7414 - val_loss: 0.5173 - val_acc: 0.7471
    
    Epoch 00006: val_acc improved from 0.71832 to 0.74706, saving model to weights-improvement-06-0.75.hdf5
    5000/5000 [==============================] - 4s 865us/step
    [0.5218041631698609, 0.7492]
    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          1
    3    7186_2          1
    4   12128_7          1
    结束.
    

## 二、Word2vec+深度学习建模

### 1.LSTM+Word2Vec


```python
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import LSTM, Embedding
from keras.models import Sequential
from keras.utils import plot_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
import gensim
from gensim.models.word2vec import Word2Vec
from keras.callbacks import ModelCheckpoint


all_data = train_data+test_data
#Tokenizer是一个用于向量化文本，或将文本转换为序列（即单词在字典中的下标构成的列表，从1算起）的类
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_data)
sequences = tokenizer.texts_to_sequences(all_data)

#总共词数(word_index：key:词，value:索引)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
#print(word_index)

#将整篇文章根据向量化文本序列都退少补生成文章矩阵
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data.shape)

x_train,x_test = data[:len(train_data)],data[len(train_data):]

#将标签独热向量处理
labels = to_categorical(np.asarray(label))
print('Shape of label tensor:', labels.shape)

wv_model = Word2Vec.load("100size_3min_count_10window.model")

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items(): 
    if word in wv_model:
        embedding_matrix[i] = np.asarray(wv_model[word],dtype='float32')
        
embedding_layer = Embedding(len(word_index) + 1,             #input_dim：词向量矩阵的维度
                            EMBEDDING_DIM,                   #output_dim:词向量的长度
                            weights=[embedding_matrix],      #weights：词向量矩阵
                            input_length=MAX_SEQUENCE_LENGTH,#input_length：句子的最大长度
                            trainable=False)                 #trainable：是否冻结嵌入层      

model = Sequential()
model.add(embedding_layer)
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))#LSTM参数：LSTM的输出向量的维度
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
model.summary()

model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])


VALIDATION_SPLIT = 0.16 # 验证集比例
TEST_SPLIT = 0.2 # 测试集比例

p1 = int(len(x_train)*(1-VALIDATION_SPLIT-TEST_SPLIT))
p2 = int(len(x_train)*(1-TEST_SPLIT))

train_x = x_train[:p1]
train_y = labels[:p1]
val_x = x_train[p1:p2]
val_y = labels[p1:p2]
test_x = x_train[p2:]
test_y = labels[p2:]

print ('train docs: '+str(len(train_x)))
print ('val docs: '+str(len(val_x)))
print ('test docs: '+str(len(test_x)))

filepath="lstm_word2vec_model/lstm_word2vec_weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
#仅保存最好的模型
#filepath="weights.best.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,mode='max')#验证集准确率比之前效果好就保存权重
callbacks_list = [checkpoint]


model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=40, batch_size=5000,callbacks=callbacks_list)

# Fit the model
#model.fit(X, Y, validation_split=0.33, nb_epoch=150, batch_size=10,callbacks=callbacks_list, verbose=0)


#model.save('word_vector_cnn.h5')
print (model.evaluate(test_x, test_y))


test_predicted = np.array(model.predict_classes(x_test))

print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_lstm_word2vec.csv',columns = ['id','sentiment'], index = False)
print('结束.')

```

    Found 101245 unique tokens.
    Shape of data tensor: (50000, 100)
    Shape of label tensor: (25000, 2)
    

    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel\__main__.py:39: DeprecationWarning: Call to deprecated `__contains__` (Method will be removed in 4.0.0, use self.wv.__contains__() instead).
    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel\__main__.py:40: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).
    

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_2 (Embedding)      (None, 100, 100)          10124600  
    _________________________________________________________________
    lstm_2 (LSTM)                (None, 100)               80400     
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 100)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 2)                 202       
    =================================================================
    Total params: 10,205,202
    Trainable params: 80,602
    Non-trainable params: 10,124,600
    _________________________________________________________________
    train docs: 15999
    val docs: 4001
    test docs: 5000
    Train on 15999 samples, validate on 4001 samples
    Epoch 1/40
    15999/15999 [==============================] - 36s 2ms/step - loss: 0.6852 - acc: 0.5797 - val_loss: 0.6684 - val_acc: 0.7096
    
    Epoch 00001: val_acc improved from -inf to 0.70957, saving model to lstm_word2vec_weights-improvement-01-0.71.hdf5
    Epoch 2/40
    15999/15999 [==============================] - 27s 2ms/step - loss: 0.6619 - acc: 0.6885 - val_loss: 0.6328 - val_acc: 0.7546
    
    Epoch 00002: val_acc improved from 0.70957 to 0.75456, saving model to lstm_word2vec_weights-improvement-02-0.75.hdf5
    Epoch 3/40
    15999/15999 [==============================] - 29s 2ms/step - loss: 0.6155 - acc: 0.7386 - val_loss: 0.5217 - val_acc: 0.7928
    
    Epoch 00003: val_acc improved from 0.75456 to 0.79280, saving model to lstm_word2vec_weights-improvement-03-0.79.hdf5
    Epoch 4/40
    15999/15999 [==============================] - 30s 2ms/step - loss: 0.4992 - acc: 0.7850 - val_loss: 0.4086 - val_acc: 0.8230
    
    Epoch 00004: val_acc improved from 0.79280 to 0.82304, saving model to lstm_word2vec_weights-improvement-04-0.82.hdf5
    Epoch 5/40
    15999/15999 [==============================] - 30s 2ms/step - loss: 0.4629 - acc: 0.8022 - val_loss: 0.4273 - val_acc: 0.8295
    
    Epoch 00005: val_acc improved from 0.82304 to 0.82954, saving model to lstm_word2vec_weights-improvement-05-0.83.hdf5
    Epoch 6/40
    15999/15999 [==============================] - 29s 2ms/step - loss: 0.4421 - acc: 0.8135 - val_loss: 0.3680 - val_acc: 0.8515
    
    Epoch 00006: val_acc improved from 0.82954 to 0.85154, saving model to lstm_word2vec_weights-improvement-06-0.85.hdf5
    Epoch 7/40
    15999/15999 [==============================] - 30s 2ms/step - loss: 0.4188 - acc: 0.8216 - val_loss: 0.3469 - val_acc: 0.8575
    
    Epoch 00007: val_acc improved from 0.85154 to 0.85754, saving model to lstm_word2vec_weights-improvement-07-0.86.hdf5
    Epoch 8/40
    15999/15999 [==============================] - 29s 2ms/step - loss: 0.4013 - acc: 0.8312 - val_loss: 0.3650 - val_acc: 0.8568
    
    Epoch 00008: val_acc did not improve from 0.85754
    Epoch 9/40
    15999/15999 [==============================] - 29s 2ms/step - loss: 0.3988 - acc: 0.8309 - val_loss: 0.3352 - val_acc: 0.8628
    
    Epoch 00009: val_acc improved from 0.85754 to 0.86278, saving model to lstm_word2vec_weights-improvement-09-0.86.hdf5
    Epoch 10/40
    15999/15999 [==============================] - 29s 2ms/step - loss: 0.3877 - acc: 0.8385 - val_loss: 0.3276 - val_acc: 0.8660
    
    Epoch 00010: val_acc improved from 0.86278 to 0.86603, saving model to lstm_word2vec_weights-improvement-10-0.87.hdf5
    Epoch 11/40
    15999/15999 [==============================] - 29s 2ms/step - loss: 0.3876 - acc: 0.8387 - val_loss: 0.3492 - val_acc: 0.8590
    
    Epoch 00011: val_acc did not improve from 0.86603
    Epoch 12/40
    15999/15999 [==============================] - 32s 2ms/step - loss: 0.3873 - acc: 0.8377 - val_loss: 0.3324 - val_acc: 0.8635
    
    Epoch 00012: val_acc did not improve from 0.86603
    Epoch 13/40
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.3783 - acc: 0.8371 - val_loss: 0.3253 - val_acc: 0.8640
    
    Epoch 00013: val_acc did not improve from 0.86603
    Epoch 14/40
    15999/15999 [==============================] - 40s 2ms/step - loss: 0.3739 - acc: 0.8438 - val_loss: 0.3292 - val_acc: 0.8613
    
    Epoch 00014: val_acc did not improve from 0.86603
    Epoch 15/40
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.3735 - acc: 0.8406 - val_loss: 0.3234 - val_acc: 0.8665
    
    Epoch 00015: val_acc improved from 0.86603 to 0.86653, saving model to lstm_word2vec_weights-improvement-15-0.87.hdf5
    Epoch 16/40
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.3714 - acc: 0.8422 - val_loss: 0.3225 - val_acc: 0.8668
    
    Epoch 00016: val_acc improved from 0.86653 to 0.86678, saving model to lstm_word2vec_weights-improvement-16-0.87.hdf5
    Epoch 17/40
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.3662 - acc: 0.8457 - val_loss: 0.3213 - val_acc: 0.8673
    
    Epoch 00017: val_acc improved from 0.86678 to 0.86728, saving model to lstm_word2vec_weights-improvement-17-0.87.hdf5
    Epoch 18/40
    15999/15999 [==============================] - 38s 2ms/step - loss: 0.3721 - acc: 0.8429 - val_loss: 0.3255 - val_acc: 0.8645
    
    Epoch 00018: val_acc did not improve from 0.86728
    Epoch 19/40
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.3736 - acc: 0.8409 - val_loss: 0.3223 - val_acc: 0.8653
    
    Epoch 00019: val_acc did not improve from 0.86728
    Epoch 20/40
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.3659 - acc: 0.8434 - val_loss: 0.3215 - val_acc: 0.8668
    
    Epoch 00020: val_acc did not improve from 0.86728
    Epoch 21/40
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.3648 - acc: 0.8464 - val_loss: 0.3167 - val_acc: 0.8693
    
    Epoch 00021: val_acc improved from 0.86728 to 0.86928, saving model to lstm_word2vec_weights-improvement-21-0.87.hdf5
    Epoch 22/40
    15999/15999 [==============================] - 33s 2ms/step - loss: 0.3711 - acc: 0.8407 - val_loss: 0.3228 - val_acc: 0.8653
    
    Epoch 00022: val_acc did not improve from 0.86928
    Epoch 23/40
    15999/15999 [==============================] - 32s 2ms/step - loss: 0.3659 - acc: 0.8459 - val_loss: 0.3155 - val_acc: 0.8685
    
    Epoch 00023: val_acc did not improve from 0.86928
    Epoch 24/40
    15999/15999 [==============================] - 37s 2ms/step - loss: 0.3635 - acc: 0.8479 - val_loss: 0.3183 - val_acc: 0.8673
    
    Epoch 00024: val_acc did not improve from 0.86928
    Epoch 25/40
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.3599 - acc: 0.8458 - val_loss: 0.3164 - val_acc: 0.8698
    
    Epoch 00025: val_acc improved from 0.86928 to 0.86978, saving model to lstm_word2vec_weights-improvement-25-0.87.hdf5
    Epoch 26/40
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.3630 - acc: 0.8477 - val_loss: 0.3216 - val_acc: 0.8650
    
    Epoch 00026: val_acc did not improve from 0.86978
    Epoch 27/40
    15999/15999 [==============================] - 40s 2ms/step - loss: 0.3627 - acc: 0.8472 - val_loss: 0.3142 - val_acc: 0.8715
    
    Epoch 00027: val_acc improved from 0.86978 to 0.87153, saving model to lstm_word2vec_weights-improvement-27-0.87.hdf5
    Epoch 28/40
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.3618 - acc: 0.8476 - val_loss: 0.3188 - val_acc: 0.8660
    
    Epoch 00028: val_acc did not improve from 0.87153
    Epoch 29/40
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.3612 - acc: 0.8477 - val_loss: 0.3150 - val_acc: 0.8688
    
    Epoch 00029: val_acc did not improve from 0.87153
    Epoch 30/40
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.3506 - acc: 0.8524 - val_loss: 0.3130 - val_acc: 0.8715
    
    Epoch 00030: val_acc did not improve from 0.87153
    Epoch 31/40
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.3590 - acc: 0.8481 - val_loss: 0.3235 - val_acc: 0.8640
    
    Epoch 00031: val_acc did not improve from 0.87153
    Epoch 32/40
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.3596 - acc: 0.8474 - val_loss: 0.3114 - val_acc: 0.8718
    
    Epoch 00032: val_acc improved from 0.87153 to 0.87178, saving model to lstm_word2vec_weights-improvement-32-0.87.hdf5
    Epoch 33/40
    15999/15999 [==============================] - 38s 2ms/step - loss: 0.3595 - acc: 0.8480 - val_loss: 0.3313 - val_acc: 0.8583
    
    Epoch 00033: val_acc did not improve from 0.87178
    Epoch 34/40
    15999/15999 [==============================] - 34s 2ms/step - loss: 0.3598 - acc: 0.8461 - val_loss: 0.3076 - val_acc: 0.8733
    
    Epoch 00034: val_acc improved from 0.87178 to 0.87328, saving model to lstm_word2vec_weights-improvement-34-0.87.hdf5
    Epoch 35/40
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.3564 - acc: 0.8486 - val_loss: 0.3217 - val_acc: 0.8643
    
    Epoch 00035: val_acc did not improve from 0.87328
    Epoch 36/40
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.3522 - acc: 0.8522 - val_loss: 0.3051 - val_acc: 0.8740
    
    Epoch 00036: val_acc improved from 0.87328 to 0.87403, saving model to lstm_word2vec_weights-improvement-36-0.87.hdf5
    Epoch 37/40
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.3558 - acc: 0.8502 - val_loss: 0.3213 - val_acc: 0.8663
    
    Epoch 00037: val_acc did not improve from 0.87403
    Epoch 38/40
    15999/15999 [==============================] - 36s 2ms/step - loss: 0.3533 - acc: 0.8494 - val_loss: 0.3094 - val_acc: 0.8713
    
    Epoch 00038: val_acc did not improve from 0.87403
    Epoch 39/40
    15999/15999 [==============================] - 35s 2ms/step - loss: 0.3554 - acc: 0.8499 - val_loss: 0.3129 - val_acc: 0.8710
    
    Epoch 00039: val_acc did not improve from 0.87403
    Epoch 40/40
    15999/15999 [==============================] - 39s 2ms/step - loss: 0.3502 - acc: 0.8516 - val_loss: 0.3079 - val_acc: 0.8710
    
    Epoch 00040: val_acc did not improve from 0.87403
    5000/5000 [==============================] - 3s 636us/step
    [0.31709566490650176, 0.868]
    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          1
    3    7186_2          0
    4   12128_7          1
    结束.
    

#### 加载模型（使用保存的模型评估或继续训练）


```python
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import LSTM, Embedding
from keras.models import Sequential
from keras.utils import plot_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
import gensim
from gensim.models.word2vec import Word2Vec
from keras.callbacks import ModelCheckpoint

model = Sequential()
model.add(embedding_layer)
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))#LSTM参数：LSTM的输出向量的维度
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# load weights
model.load_weights("lstm_word2vec_weights-improvement-36-0.87.hdf5")

#model.save('word_vector_cnn.h5')
print (model.evaluate(test_x, test_y))
```

    5000/5000 [==============================] - 3s 519us/step
    [0.31665992724895475, 0.8666]
    

### 2.CNN+Word2Vec


```python
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Sequential
from keras.utils import plot_model

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
from gensim.models.word2vec import Word2Vec

MAX_SEQUENCE_LENGTH = 100 # 每条新闻最大长度
EMBEDDING_DIM = 100       # 词向量空间维度

VALIDATION_SPLIT = 0.16 # 验证集比例
TEST_SPLIT = 0.2 # 测试集比例

#合并训练集和测试集
all_data = train_data+test_data

#Tokenizer是一个用于向量化文本，或将文本转换为序列（即单词在字典中的下标构成的列表，从1算起）的类
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_data)
sequences = tokenizer.texts_to_sequences(all_data)

#总共词数(word_index：key:词，value:索引)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
#print(word_index)

#将整篇文章根据向量化文本序列都退少补生成文章矩阵
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data.shape)

x_train,x_test = data[:len(train_data)],data[len(train_data):]

#将标签独热向量处理
labels = to_categorical(np.asarray(label))
print('Shape of label tensor:', labels.shape)


p1 = int(len(x_train)*(1-VALIDATION_SPLIT-TEST_SPLIT))
p2 = int(len(x_train)*(1-TEST_SPLIT))

train_x = x_train[:p1]
train_y = labels[:p1]
val_x = x_train[p1:p2]
val_y = labels[p1:p2]
test_x = x_train[p2:]
test_y = labels[p2:]

print ('train docs: '+str(len(train_x)))
print ('val docs: '+str(len(val_x)))
print ('test docs: '+str(len(test_x)))


wv_model = Word2Vec.load("100size_3min_count_10window.model")

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items(): 
    if word in wv_model:
        embedding_matrix[i] = np.asarray(wv_model[word],dtype='float32')
        
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)#冻结嵌入层


model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.2))
model.add(Conv1D(250, 3, padding='valid', activation='relu', strides=1))
model.add(MaxPooling1D(3))
model.add(Flatten())
model.add(Dense(EMBEDDING_DIM, activation='relu'))
model.add(Dense(labels.shape[1], activation='softmax'))
model.summary()

#plot_model(model, to_file='model.png',show_shapes=True)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])


filepath="cnn_model/cnn_word2vec_weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,mode='max')
callbacks_list = [checkpoint]

# Fit the model
#model.fit(X, Y, validation_split=0.33, nb_epoch=150, batch_size=10,callbacks=callbacks_list, verbose=0)


model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=20, batch_size=5000,callbacks=callbacks_list)

#model.save('word_vector_cnn.h5')
print (model.evaluate(test_x, test_y))

test_predicted = np.array(model.predict_classes(x_test))

print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_cnn_word2vec.csv',columns = ['id','sentiment'], index = False)
print('结束.')

```

    Found 101245 unique tokens.
    Shape of data tensor: (50000, 100)
    Shape of label tensor: (25000, 2)
    train docs: 15999
    val docs: 4001
    test docs: 5000
    

    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel\__main__.py:61: DeprecationWarning: Call to deprecated `__contains__` (Method will be removed in 4.0.0, use self.wv.__contains__() instead).
    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel\__main__.py:62: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).
    

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_3 (Embedding)      (None, 100, 100)          10124600  
    _________________________________________________________________
    dropout_5 (Dropout)          (None, 100, 100)          0         
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 98, 250)           75250     
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 32, 250)           0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 8000)              0         
    _________________________________________________________________
    dense_5 (Dense)              (None, 100)               800100    
    _________________________________________________________________
    dense_6 (Dense)              (None, 2)                 202       
    =================================================================
    Total params: 11,000,152
    Trainable params: 875,552
    Non-trainable params: 10,124,600
    _________________________________________________________________
    Train on 15999 samples, validate on 4001 samples
    Epoch 1/20
    15999/15999 [==============================] - 18s 1ms/step - loss: 1.8571 - acc: 0.5046 - val_loss: 0.6610 - val_acc: 0.6353
    
    Epoch 00001: val_acc improved from -inf to 0.63534, saving model to cnn_model/cnn_word2vec_weights-improvement-01-0.64.hdf5
    Epoch 2/20
    15999/15999 [==============================] - 19s 1ms/step - loss: 0.6381 - acc: 0.6811 - val_loss: 0.6190 - val_acc: 0.6473
    
    Epoch 00002: val_acc improved from 0.63534 to 0.64734, saving model to cnn_model/cnn_word2vec_weights-improvement-02-0.65.hdf5
    Epoch 3/20
    15999/15999 [==============================] - 19s 1ms/step - loss: 0.6863 - acc: 0.5880 - val_loss: 0.5875 - val_acc: 0.6901
    
    Epoch 00003: val_acc improved from 0.64734 to 0.69008, saving model to cnn_model/cnn_word2vec_weights-improvement-03-0.69.hdf5
    Epoch 4/20
    15999/15999 [==============================] - 20s 1ms/step - loss: 0.5728 - acc: 0.6988 - val_loss: 0.5574 - val_acc: 0.7083
    
    Epoch 00004: val_acc improved from 0.69008 to 0.70832, saving model to cnn_model/cnn_word2vec_weights-improvement-04-0.71.hdf5
    Epoch 5/20
    15999/15999 [==============================] - 19s 1ms/step - loss: 0.5576 - acc: 0.7026 - val_loss: 0.5020 - val_acc: 0.7538
    
    Epoch 00005: val_acc improved from 0.70832 to 0.75381, saving model to cnn_model/cnn_word2vec_weights-improvement-05-0.75.hdf5
    Epoch 6/20
    15999/15999 [==============================] - 19s 1ms/step - loss: 0.6535 - acc: 0.6495 - val_loss: 0.4529 - val_acc: 0.8268
    
    Epoch 00006: val_acc improved from 0.75381 to 0.82679, saving model to cnn_model/cnn_word2vec_weights-improvement-06-0.83.hdf5
    Epoch 7/20
    15999/15999 [==============================] - 19s 1ms/step - loss: 0.4425 - acc: 0.8086 - val_loss: 0.5880 - val_acc: 0.6948
    
    Epoch 00007: val_acc did not improve from 0.82679
    Epoch 8/20
    15999/15999 [==============================] - 19s 1ms/step - loss: 0.5308 - acc: 0.7394 - val_loss: 0.3934 - val_acc: 0.8383
    
    Epoch 00008: val_acc improved from 0.82679 to 0.83829, saving model to cnn_model/cnn_word2vec_weights-improvement-08-0.84.hdf5
    Epoch 9/20
    15999/15999 [==============================] - 19s 1ms/step - loss: 0.4318 - acc: 0.8035 - val_loss: 0.4658 - val_acc: 0.7731
    
    Epoch 00009: val_acc did not improve from 0.83829
    Epoch 10/20
    15999/15999 [==============================] - 17s 1ms/step - loss: 0.4140 - acc: 0.8127 - val_loss: 0.3626 - val_acc: 0.8423
    
    Epoch 00010: val_acc improved from 0.83829 to 0.84229, saving model to cnn_model/cnn_word2vec_weights-improvement-10-0.84.hdf5
    Epoch 11/20
    15999/15999 [==============================] - 16s 990us/step - loss: 0.3769 - acc: 0.8356 - val_loss: 0.4600 - val_acc: 0.7823
    
    Epoch 00011: val_acc did not improve from 0.84229
    Epoch 12/20
    15999/15999 [==============================] - 15s 943us/step - loss: 0.4385 - acc: 0.7902 - val_loss: 0.3469 - val_acc: 0.8540
    
    Epoch 00012: val_acc improved from 0.84229 to 0.85404, saving model to cnn_model/cnn_word2vec_weights-improvement-12-0.85.hdf5
    Epoch 13/20
    15999/15999 [==============================] - 15s 944us/step - loss: 0.3454 - acc: 0.8516 - val_loss: 0.3733 - val_acc: 0.8293
    
    Epoch 00013: val_acc did not improve from 0.85404
    Epoch 14/20
    15999/15999 [==============================] - 15s 944us/step - loss: 0.4367 - acc: 0.7958 - val_loss: 0.3613 - val_acc: 0.8423
    
    Epoch 00014: val_acc did not improve from 0.85404
    Epoch 15/20
    15999/15999 [==============================] - 15s 941us/step - loss: 0.3444 - acc: 0.8539 - val_loss: 0.3234 - val_acc: 0.8598
    
    Epoch 00015: val_acc improved from 0.85404 to 0.85979, saving model to cnn_model/cnn_word2vec_weights-improvement-15-0.86.hdf5
    Epoch 16/20
    15999/15999 [==============================] - 15s 944us/step - loss: 0.3219 - acc: 0.8628 - val_loss: 0.3491 - val_acc: 0.8450
    
    Epoch 00016: val_acc did not improve from 0.85979
    Epoch 17/20
    15999/15999 [==============================] - 15s 946us/step - loss: 0.4664 - acc: 0.7804 - val_loss: 0.3365 - val_acc: 0.8595
    
    Epoch 00017: val_acc did not improve from 0.85979
    Epoch 18/20
    15999/15999 [==============================] - 15s 953us/step - loss: 0.3263 - acc: 0.8640 - val_loss: 0.3366 - val_acc: 0.8535
    
    Epoch 00018: val_acc did not improve from 0.85979
    Epoch 19/20
    15999/15999 [==============================] - 15s 940us/step - loss: 0.3643 - acc: 0.8394 - val_loss: 0.3416 - val_acc: 0.8503
    
    Epoch 00019: val_acc did not improve from 0.85979
    Epoch 20/20
    15999/15999 [==============================] - 15s 956us/step - loss: 0.3400 - acc: 0.8516 - val_loss: 0.3439 - val_acc: 0.8485
    
    Epoch 00020: val_acc did not improve from 0.85979
    5000/5000 [==============================] - 2s 404us/step
    [0.35134012341499327, 0.8474]
    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          1
    3    7186_2          0
    4   12128_7          1
    结束.
    

# DSSM + attention


```python
from importlib import reload
import sys
from imp import reload
import warnings
warnings.filterwarnings('ignore')
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")
import pandas as pd

df1 = pd.read_csv('data/labeledTrainData.tsv', delimiter="\t")
df1 = df1.drop(['id'], axis=1)
df1.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sentiment</th>
      <th>review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>With all this stuff going down at the moment w...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>\The Classic War of the Worlds\" by Timothy Hi...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>The film starts with a manager (Nicholas Bell)...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>It must be assumed that those who praised this...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>Superbly trashy and wondrously unpretentious 8...</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df1.shape)
```

    (25000, 2)
    


```python
df2 = pd.read_csv('data/imdb_master.csv',encoding="latin-1")
df2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>type</th>
      <th>review</th>
      <th>label</th>
      <th>file</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>test</td>
      <td>Once again Mr. Costner has dragged out a movie...</td>
      <td>neg</td>
      <td>0_2.txt</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>test</td>
      <td>This is an example of why the majority of acti...</td>
      <td>neg</td>
      <td>10000_4.txt</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>test</td>
      <td>First of all I hate those moronic rappers, who...</td>
      <td>neg</td>
      <td>10001_1.txt</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>test</td>
      <td>Not even the Beatles could write songs everyon...</td>
      <td>neg</td>
      <td>10002_3.txt</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>test</td>
      <td>Brass pictures (movies is not a fitting word f...</td>
      <td>neg</td>
      <td>10003_3.txt</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df2.shape)
```

    (100000, 5)
    


```python
df2 = df2.drop(['Unnamed: 0','type','file'],axis=1)
df2.columns = ["review","sentiment"]
df2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Once again Mr. Costner has dragged out a movie...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>1</th>
      <td>This is an example of why the majority of acti...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>2</th>
      <td>First of all I hate those moronic rappers, who...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Not even the Beatles could write songs everyon...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brass pictures (movies is not a fitting word f...</td>
      <td>neg</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2 = df2[df2.sentiment != 'unsup']
df2['sentiment'] = df2['sentiment'].map({'pos': 1, 'neg': 0})
df2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Once again Mr. Costner has dragged out a movie...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>This is an example of why the majority of acti...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>First of all I hate those moronic rappers, who...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Not even the Beatles could write songs everyon...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brass pictures (movies is not a fitting word f...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = df1#pd.concat([df1, df2])
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sentiment</th>
      <th>review</th>
      <th>Processed_Reviews</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>With all this stuff going down at the moment w...</td>
      <td>stuff go moment mj ive start listen music watc...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>\The Classic War of the Worlds\" by Timothy Hi...</td>
      <td>classic war world timothy hines entertain film...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>The film starts with a manager (Nicholas Bell)...</td>
      <td>film start manager nicholas bell give welcome ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>It must be assumed that those who praised this...</td>
      <td>must assume praise film greatest film opera ev...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>Superbly trashy and wondrously unpretentious 8...</td>
      <td>superbly trashy wondrously unpretentious 80 ex...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (25000, 3)




```python
import nltk
nltk.download("wordnet")
```

    [nltk_data] Downloading package wordnet to
    [nltk_data]     C:\Users\admin\AppData\Roaming\nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    




    True




```python
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english")) 
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text

df['Processed_Reviews'] = df.review.apply(lambda x: clean_text(x))
```


```python
df.Processed_Reviews.apply(lambda x: len(x.split(" "))).mean()
```




    129.54916




```python
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
from keras import initializers, regularizers, constraints, optimizers, layers

max_features = 6000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(df['Processed_Reviews'])#词序列化
list_tokenized_train = tokenizer.texts_to_sequences(df['Processed_Reviews'])#生成文章序列

maxlen = 130
#将整篇文章根据向量化文本序列都退少补生成文章矩阵
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
y = df['sentiment']

```


```python
embed_size = 128
model = Sequential()
model.add(Embedding(max_features, embed_size))
model.add(Bidirectional(LSTM(32, return_sequences = True)))
model.add(GlobalMaxPool1D())
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.05))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_5 (Embedding)      (None, None, 128)         768000    
    _________________________________________________________________
    bidirectional_5 (Bidirection (None, None, 64)          41216     
    _________________________________________________________________
    global_max_pooling1d_2 (Glob (None, 64)                0         
    _________________________________________________________________
    dense_9 (Dense)              (None, 20)                1300      
    _________________________________________________________________
    dropout_5 (Dropout)          (None, 20)                0         
    _________________________________________________________________
    dense_10 (Dense)             (None, 1)                 21        
    =================================================================
    Total params: 810,537
    Trainable params: 810,537
    Non-trainable params: 0
    _________________________________________________________________
    


```python
batch_size = 100
epochs = 3
model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.2)
```

    Train on 20000 samples, validate on 5000 samples
    Epoch 1/3
    20000/20000 [==============================] - 49s 2ms/step - loss: 0.4530 - acc: 0.7857 - val_loss: 0.3215 - val_acc: 0.8672
    Epoch 2/3
    20000/20000 [==============================] - 47s 2ms/step - loss: 0.2471 - acc: 0.9030 - val_loss: 0.3053 - val_acc: 0.8702
    Epoch 3/3
    20000/20000 [==============================] - 48s 2ms/step - loss: 0.1829 - acc: 0.9311 - val_loss: 0.3511 - val_acc: 0.8638
    




    <keras.callbacks.History at 0x3d6c4b38>




```python
df_test=pd.read_csv("data/testData.tsv",header=0, delimiter="\t", quoting=3)
df_test.head()
df_test["review"]=df_test.review.apply(lambda x: clean_text(x))
df_test["sentiment"] = df_test["id"].map(lambda x: 1 if int(x.strip('"').split("_")[1]) >= 5 else 0)
y_test = df_test["sentiment"]
list_sentences_test = df_test["review"]
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

prediction = model.predict(X_te)
y_pred = (prediction > 0.5)

from sklearn.metrics import f1_score, confusion_matrix

print('F1-score: {0}'.format(f1_score(y_pred, y_test)))
print('Confusion matrix:')
confusion_matrix(y_pred, y_test)
```

    F1-score: 0.8606749612433915
    Confusion matrix:
    




    array([[10669,  1674],
           [ 1831, 10826]], dtype=int64)




```python
import numpy as np
df_test=pd.read_csv("data/testData.tsv",header=0, delimiter="\t", quoting=1)
test_predicted = model.predict_classes(X_te).reshape(1,df_test.shape[0])[0]
print('保存结果...')
submission_df = pd.DataFrame(data ={'id': df_test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_lstm_imdb_master.csv',columns = ['id','sentiment'], index = False)
print('结束.')
```

    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          0
    3    7186_2          0
    4   12128_7          1
    结束.
    

![image.png](attachment:image.png)


```python
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.utils import to_categorical
from keras.models import load_model
from keras import backend as K#返回当前后端
from keras.models import Sequential,Model
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Embedding,LSTM,Layer,initializers,regularizers,constraints,Input,Dropout,concatenate,BatchNormalization
from keras.layers import Dense,Bidirectional,Concatenate,Multiply,Maximum,Subtract,Lambda,dot,Flatten,Reshape
import gc


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
#静态Attention Model
class AttentionLayer(Layer):
    def __init__(self,step_dim,W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)	#正则化
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)		#约束、限制
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias										#偏置
        self.step_dim = step_dim								#step维度
        self.features_dim = 0									#特征维度

        super(AttentionLayer,self).__init__(**kwargs)#用于调用父类(超类)的一个方法。

    #设置self.supports_masking = True后需要复写该方法
    def compute_mask(self, inputs, mask=None):
        return None

    #参数设置，必须实现
    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight((input_shape[-1],),#词向量维度，神经元个数
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]#词向量维度
        if self.bias:
            self.b = self.add_weight((input_shape[1],),#词的个数，神经元个数
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,#正则化
                                     constraint=self.b_constraint)	#约束和限制
        else:
            self.b = None
        self.built = True

    # input (None,sentence_length,embedding_size)
    def call(self, x, mask = None):
        # 计算输出
        features_dim = self.features_dim#词向量维度
        step_dim = self.step_dim		#词的个数

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        # print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim


    def get_config(self):
        config = {'step_dim': self.step_dim}
        base_config = super(AttentionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


```


```python

embed_size = 128

left_input = Input(shape=(130,), dtype='int32')
# 定义需要使用的网络层
embedding_layer1 = Embedding(
    input_dim=6000,
    output_dim=128,
    trainable=True,
    input_length=130
)
att_layer1 = AttentionLayer(130)

#bi_lstm_layer =Bidirectional(LSTM(32, return_sequences = True))
bi_lstm_layer =Bidirectional(LSTM(32))

#bi_lstm_layer = globalMaxp(bi_lstm_layer)

s1 = embedding_layer1(left_input)
s1_bi = bi_lstm_layer(s1)
s1_att = att_layer1(s1)
s1_last = Concatenate(axis=1)([s1_att,s1_bi])#横着拼接
dense_layer1 = Dense(20,activation='relu')(s1_last)
dropout1 = Dropout(0.05)(dense_layer1)
dense_layer2 = Dense(1,activation='sigmoid')(dropout1)

model = Model(inputs=left_input,outputs=[dense_layer2], name="simaese_lstm_attention")

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```

    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_3 (InputLayer)            (None, 130)          0                                            
    __________________________________________________________________________________________________
    embedding_4 (Embedding)         (None, 130, 128)     768000      input_3[0][0]                    
    __________________________________________________________________________________________________
    attention_layer_3 (AttentionLay (None, 128)          258         embedding_4[0][0]                
    __________________________________________________________________________________________________
    bidirectional_4 (Bidirectional) (None, 64)           41216       embedding_4[0][0]                
    __________________________________________________________________________________________________
    concatenate_3 (Concatenate)     (None, 192)          0           attention_layer_3[0][0]          
                                                                     bidirectional_4[0][0]            
    __________________________________________________________________________________________________
    dense_7 (Dense)                 (None, 20)           3860        concatenate_3[0][0]              
    __________________________________________________________________________________________________
    dropout_4 (Dropout)             (None, 20)           0           dense_7[0][0]                    
    __________________________________________________________________________________________________
    dense_8 (Dense)                 (None, 1)            21          dropout_4[0][0]                  
    ==================================================================================================
    Total params: 813,355
    Trainable params: 813,355
    Non-trainable params: 0
    __________________________________________________________________________________________________
    


```python
batch_size = 100
epochs = 3
model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.2)
```

    Train on 20000 samples, validate on 5000 samples
    Epoch 1/3
    20000/20000 [==============================] - 45s 2ms/step - loss: 0.0522 - acc: 0.9840 - val_loss: 0.5606 - val_acc: 0.8504
    Epoch 2/3
    20000/20000 [==============================] - 45s 2ms/step - loss: 0.0473 - acc: 0.9850 - val_loss: 0.6668 - val_acc: 0.8500
    Epoch 3/3
    20000/20000 [==============================] - 44s 2ms/step - loss: 0.0272 - acc: 0.9921 - val_loss: 0.7165 - val_acc: 0.8512
    




    <keras.callbacks.History at 0x44882518>




```python
df_test=pd.read_csv("data/testData.tsv",header=0, delimiter="\t", quoting=3)
df_test.head()
df_test["review"]=df_test.review.apply(lambda x: clean_text(x))
df_test["sentiment"] = df_test["id"].map(lambda x: 1 if int(x.strip('"').split("_")[1]) >= 5 else 0)
y_test = df_test["sentiment"]
list_sentences_test = df_test["review"]
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

prediction = model.predict(X_te)
y_pred = (prediction > 0.5)

from sklearn.metrics import f1_score, confusion_matrix

print('F1-score: {0}'.format(f1_score(y_pred, y_test)))
print('Confusion matrix:')
confusion_matrix(y_pred, y_test)
```

    F1-score: 0.842258181671442
    Confusion matrix:
    




    array([[10655,  2064],
           [ 1845, 10436]], dtype=int64)




```python
import numpy as np

test_predicted = np.array(model.predict(X_te))
test_predicted =list(map(lambda x:1 if x >0.5 else 0 ,test_predicted))

df_test=pd.read_csv("data/testData.tsv",header=0, delimiter="\t", quoting=1)

print('保存结果...')
submission_df = pd.DataFrame(data ={'id': df_test['id'], 'sentiment': test_predicted})
print(submission_df.head(5))
submission_df.to_csv('submission_DSSM_bi-lstm_attention_imdb_master.csv',columns = ['id','sentiment'], index = False)
print('结束.')

```

    保存结果...
             id  sentiment
    0  12311_10          1
    1    8348_2          0
    2    5828_4          0
    3    7186_2          0
    4   12128_7          1
    结束.
    

![image.png](attachment:image.png)

## Top 3 共578组

# MaxPooling1D和GlobalMaxPooling1D的区别


```python
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, GlobalMaxPooling1D,MaxPooling1D

D = np.random.rand(10, 6, 10)

model = Sequential()
model.add(LSTM(16, input_shape=(6, 10), return_sequences=True))
model.add(MaxPooling1D(pool_size=2, strides=1))
model.add(LSTM(10))
model.add(Dense(1))
model.compile(loss='binary_crossentropy', optimizer='sgd')

# print the summary to see how the dimension change after the layers are 
# applied

print(model.summary())
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lstm_4 (LSTM)                (None, 6, 16)             1728      
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 5, 16)             0         
    _________________________________________________________________
    lstm_5 (LSTM)                (None, 10)                1080      
    _________________________________________________________________
    dense_5 (Dense)              (None, 1)                 11        
    =================================================================
    Total params: 2,819
    Trainable params: 2,819
    Non-trainable params: 0
    _________________________________________________________________
    None
    


```python
# try a model with GlobalMaxPooling1D now

model = Sequential()
model.add(LSTM(16, input_shape=(6, 10), return_sequences=True))
model.add(GlobalMaxPooling1D())
model.add(Dense(1))
model.compile(loss='binary_crossentropy', optimizer='sgd')

print(model.summary())
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lstm_6 (LSTM)                (None, 6, 16)             1728      
    _________________________________________________________________
    global_max_pooling1d_3 (Glob (None, 16)                0         
    _________________________________________________________________
    dense_6 (Dense)              (None, 1)                 17        
    =================================================================
    Total params: 1,745
    Trainable params: 1,745
    Non-trainable params: 0
    _________________________________________________________________
    None
    
