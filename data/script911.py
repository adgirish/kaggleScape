
# coding: utf-8

# There are some really cool Kernels sitting out there for this competition and they are far slicker and efficient than this. This is just a brief data dive and some features using TFIDF and Truncated SVD on them. 

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
import re
import gc
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


import os
cdir = os.getcwd()
tVars = pd.read_csv('../input/training_variants')
vVars = pd.read_csv('../input/test_variants')
tText = pd.read_csv('../input/training_text',sep='\|\|',
                    skiprows=1,engine='python',names=["ID","text"])
vText = pd.read_csv('../input/test_text',sep='\|\|',
                    skiprows=1,engine='python',names=["ID","text"])


# In[3]:


tVars.head()


# In[4]:


tText.head()


# In[5]:


print(tText['text'][0][:20], '   ', len(tText['text'][0]))


# It looks like each of these items in the text are going to have fairly lengthy descriptions and we can treat these as documents, or at least I am.

# In[6]:


print(len(tText), len(tVars), len(vText), len(vVars))


# In[7]:


from collections import Counter
varsGeneCount = Counter(tVars.Gene)
print(varsGeneCount, '\n', len(varsGeneCount))


# That will be giving us 264 different categories if we decide to encode the labels. At this point I am not sure I know what they mean either.  What about the classes?

# In[8]:


plt.figure(figsize=(12,8))
ax = sns.countplot(x="Class", data=tVars)
plt.ylabel('Frequency'); plt.xlabel('Class')
plt.title('Freq. of Classes in Training Variants')
plt.show()


# In[9]:


subfile = pd.read_csv('../input/submissionFile')
subfile.head(3)


# With so many "7" classes there could be an issue of predictions skewed that way which may be something to look at down the road.

# In[10]:


varsVariationCount = Counter(tVars.Variation)
print('Number of unique varations in trainVars.Variation: \n', len(varsVariationCount))


# In[11]:


fig, ax=plt.subplots(1,1,figsize=(12,8))
ax = sns.distplot(pd.factorize(tVars['Variation'])[0]/len(tVars), bins=150, color='r')


# Once again, many, many different categories here, and it appears there are a lot of unique ones with only 1 count. With so many in "7" it could be a slight disadvantage.

# Since there is not that much else to look at in the Training Variants, lets take a little closer look into the Training Text. _I kept getting errors importing the stop words so I just grabbed them._

# In[12]:


def textClean(text):
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = text.lower().split()
    stops = {'so', 'his', 't', 'y', 'ours', 'herself', 
             'your', 'all', 'some', 'they', 'i', 'of', 'didn', 
             'them', 'when', 'will', 'that', 'its', 'because', 
             'while', 'those', 'my', 'don', 'again', 'her', 'if',
             'further', 'now', 'does', 'against', 'won', 'same', 
             'a', 'during', 'who', 'here', 'have', 'in', 'being', 
             'it', 'other', 'once', 'itself', 'hers', 'after', 're',
             'just', 'their', 'himself', 'theirs', 'whom', 'then', 'd', 
             'out', 'm', 'mustn', 'where', 'below', 'about', 'isn',
             'shouldn', 'wouldn', 'these', 'me', 'to', 'doesn', 'into',
             'the', 'until', 'she', 'am', 'under', 'how', 'yourself',
             'couldn', 'ma', 'up', 'than', 'from', 'themselves', 'yourselves',
             'off', 'above', 'yours', 'having', 'mightn', 'needn', 'on', 
             'too', 'there', 'an', 'and', 'down', 'ourselves', 'each',
             'hadn', 'ain', 'such', 've', 'did', 'be', 'or', 'aren', 'he', 
             'should', 'for', 'both', 'doing', 'this', 'through', 'do', 'had',
             'own', 'but', 'were', 'over', 'not', 'are', 'few', 'by', 
             'been', 'most', 'no', 'as', 'was', 'what', 's', 'is', 'you', 
             'shan', 'between', 'wasn', 'has', 'more', 'him', 'nor',
             'can', 'why', 'any', 'at', 'myself', 'very', 'with', 'we', 
             'which', 'hasn', 'weren', 'haven', 'our', 'll', 'only',
             'o', 'before'}
    ## I ketp getting errors on importing the stopwords and I have no clue why
    #stops = set(stopwords.words("English"))
    text = [w for w in text if not w in stops]    
    text = " ".join(text)
    text = text.replace("."," ").replace(","," ")
    return(text)


# In[13]:


trainText = []
for it in tText['text']:
    newT = textClean(it)
    trainText.append(newT)
testText = []
for it in vText['text']:
    newT = textClean(it)
    testText.append(newT)


# After a little cleaning, I wonder how the text shapes up now? Are there any other additional words that we could cut out?

# In[14]:


trainText[0][:100]


# So each of the portions of the doc are cleaned and we can check out the most common of a few of them...

# In[15]:


for i in range(10):
    print('\n Doc ', str(i))
    stopCheck = Counter(trainText[i].split())
    print(stopCheck.most_common()[:10])


# There looks like a lot of similarity here just by a quick visual examination. 

# In[16]:


tops = Counter(str(trainText).split()).most_common()[:20]
labs, vals = zip(*tops)
idx = np.arange(len(labs))
wid=0.6
fig, ax=plt.subplots(1,1,figsize=(14,8))
ax=plt.bar(idx, vals, wid, color='g')
ax=plt.xticks(idx - wid/8, labs, rotation=25, size=14)
plt.title('Top Twenty Counts of Most-Common Words Among Text')


# In[17]:


gc.collect()


# There is some weight on the far left of this plot. Lets throw these words into another stop word check and re examine. If we really worked this into the mix, we could probably use the max_df setting in the TFIDF and Count Vectorizers below. 
# 
# A few time through this and I think we should drop some more words than just the top 20. Lets try 30.

# In[18]:


topsInc = Counter(str(trainText).split()).most_common()[:30]
labsInc, valsInc = zip(*topsInc)


# In[19]:


def stopCheck(text, stops):
    text = text.split()
#     stops = {'mutations', 'cancer'}
    text = [w for w in text if not w in stops]    
    text = " ".join(text)
    return text


# In[20]:


trainText2 = []
for it in trainText:
    newT = stopCheck(it,labsInc)
    trainText2.append(newT)
    
testText2 = []
for it in testText:
    newT = stopCheck(it,labsInc)
    testText2.append(newT)


# In[21]:


trainText2[2][:100]


# In[22]:


tops = Counter(str(trainText2).split()).most_common()[:20]
labs, vals = zip(*tops)
idx = np.arange(len(labs))
wid=0.6
fig, ax=plt.subplots(1,1,figsize=(14,8))
ax=plt.bar(idx, vals, wid, color='b')
ax=plt.xticks(idx - wid/8, labs, rotation=25, size=14)
plt.title('Top Twenty Counts of Most-Common Words Among Text')


# In[23]:


gc.collect()


# This looks a little more balanced so we will go with this set of texts.

# In[24]:


maxFeats = 500


# In[ ]:


tfidf = TfidfVectorizer(min_df=5, max_features=maxFeats, ngram_range=(1,3),
                        strip_accents='unicode',
                        lowercase =True, analyzer='word', token_pattern=r'\w+',
                        use_idf=True, smooth_idf=True, sublinear_tf=True, 
                        stop_words = 'english')
tfidf.fit(trainText2)


# In[ ]:


cvec = CountVectorizer(min_df=5, ngram_range=(1,3), max_features=maxFeats, 
                       strip_accents='unicode',
                       lowercase =True, analyzer='word', token_pattern=r'\w+',
                       stop_words = 'english')
cvec.fit(trainText2)


# _3321x8595819 sparse matrix of type 'class 'numpy.int64''
# 	with 38030516 stored elements in Compressed Sparse Row format_
#     
# This is the print output of just using nGram features 1-3. Its kind of a large item to pass around even if its sparse so limiting the maximum features helps in here.

# In[ ]:


## I played around with the componenets and 360-390 seemed to work best for me...
svdT = TruncatedSVD(n_components=390)
svdTFit = svdT.fit_transform(tfidf.transform(trainText2))


# This is something I did in other comps that helped out some in feature building. 

# In[ ]:


def buildFeats(texts, variations):
    temp = variations.copy()
    print('Encoding...')
    temp['Gene'] = pd.factorize(variations['Gene'])[0]
    temp['Variation'] = pd.factorize(variations['Variation'])[0]
    temp['Gene_to_Variation_Ratio'] = temp['Gene']/temp['Variation']
    
    print('Lengths...')
    temp['doc_len'] = [len(x) for x in texts]
    temp['unique_words'] = [len(set(x))  for x in texts]
    
    print('TFIDF...')
    temp_tfidf = tfidf.transform(texts)
    temp['tfidf_sum'] = temp_tfidf.sum(axis=1)
    temp['tfidf_mean'] = temp_tfidf.mean(axis=1)
    temp['tfidf_len'] =  (temp_tfidf != 0).sum(axis = 1)
    
    print('Count Vecs...')
    temp_cvec = cvec.transform(texts)
    temp['cvec_sum'] = temp_cvec.sum(axis=1)
    temp['cvec_mean'] = temp_cvec.mean(axis=1)
    temp['cvec_len'] =  (temp_cvec != 0).sum(axis = 1)
    
    print('Latent Semantic Analysis Cols...')
    tempc = list(temp.columns)
    temp_lsa = svdT.transform(temp_tfidf)
    
    for i in range(np.shape(temp_lsa)[1]):
        tempc.append('lsa'+str(i+1))
    temp = pd.concat([temp, pd.DataFrame(temp_lsa, index=temp.index)], axis=1)
    
    return temp, tempc


# In[ ]:


trainDf, traincol = buildFeats(trainText2, tVars)
testDf, testcol = buildFeats(testText2, vVars)


# In[ ]:


trainDf.columns = traincol
testDf.columns = testcol


# In order to get the right prediction we have to get the classes in the range of 0-8.

# In[ ]:


classes = tVars.Class - 1
print('Original:', Counter(tVars.Class), '\n ReHashed:', Counter(classes))


# In[ ]:


dft, dfv, yt, yv = train_test_split(trainDf.drop(['ID','Class'],axis=1),
                                    classes,
                                    test_size = 0.1,
                                    random_state=31415)
print(np.shape(dft))


# In[ ]:


import gc
print('Format a Train and Validation Set for LGB')
d_train = lgb.Dataset(dft, label=yt)
d_val = lgb.Dataset(dfv, label=yv)
               
gc.collect()


# In[ ]:


parms = {'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 9,
    'metric': {'multi_logloss'},
    'learning_rate': 0.05, 
    'max_depth': 5,
    'num_iterations': 400, 
    'num_leaves': 95, 
    'min_data_in_leaf': 60, 
    'lambda_l1': 1.0,
    'feature_fraction': 0.8, 
    'bagging_fraction': 0.8, 
    'bagging_freq': 5}

rnds = 260
mod = lgb.train(parms, train_set=d_train, num_boost_round=rnds,
               valid_sets=[d_val], valid_names=['dval'], verbose_eval=20,
               early_stopping_rounds=20)


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
lgb.plot_importance(mod, max_num_features=30, figsize=(14,10))


# In[ ]:


pred = mod.predict(testDf.drop(['ID'],axis=1))


# In[ ]:


sub = pd.DataFrame(pred, index=testDf.index)
sub.columns = subfile.columns[1:]
sub.index_name = subfile.columns[0]
sub['ID'] = testDf.index
sub.head()


# In[ ]:


import datetime
now = datetime.datetime.now()
sub.to_csv('lgb_'+str(now.strftime("%Y-%m-%d-%H-%M"))+'.csv', index=False)

