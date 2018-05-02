
# coding: utf-8

# Unusual meaning map: Treating question pairs as image / surface
# ---------------------------------------------------------------
# 

# Other people have already written really nice exploratory kernels which helped me to write the minimal code myself. 
# 
# In this kernel, I have tried to extract a different type of feature from which we can learn using any algorithm which can learn via image. The basic assumption behind this exercise is to capture non-sequential closeness between words.
# 
# For example:
# A Question pair has pointing arrows from each of the words of one sentence to each of the words from another sentence
# ![A Question pair has pointing arrows from each of the words of one sentence to each of the words from another sentence][1]
# 
#   [1]: http://image.prntscr.com/image/97e92b0357a843078b61eef5ad8a183b.png
# 
# To capture this we can create NxM matrix with Word2Vec distance between each word with other. and resize the matrix just like an image to a 10x10 matrix and use this as a feature to xgboost.

# In[ ]:


import csv
import pip
from gensim import corpora, models, similarities
import pandas as pd
import numpy as np
train_file = "../input/train.csv"
df = pd.read_csv(train_file, index_col="id")
df


# In[ ]:


import matplotlib.pylab as plt


# **Extracting unique questions**

# In[ ]:


questions = dict()

for row in df.iterrows():
    questions[row[1]['qid1']] = row[1]['question1']
    questions[row[1]['qid2']] = row[1]['question2']


# **Creating a simple tokenizer**

# In[ ]:


import re
import nltk
def basic_cleaning(string):
    string = str(string)
    try:
        string = string.decode('unicode-escape')
    except Exception:
        pass
    string = string.lower()
    string = re.sub(' +', ' ', string)
    return string
sentences = []
for i in questions:
    sentences.append(nltk.word_tokenize(basic_cleaning(questions[i])))


# **Creating a simple Word2Vec model from the question pair, we can use a pre-trained model instead to get better results**

# In[ ]:


import gensim
model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)


# **A very simple term frequency and document frequency extractor** 

# In[ ]:


tf = dict()
docf = dict()
total_docs = 0
for qid in questions:
    total_docs += 1
    toks = nltk.word_tokenize(basic_cleaning(questions[qid]))
    uniq_toks = set(toks)
    for i in toks:
        if i not in tf:
            tf[i] = 1
        else:
            tf[i] += 1
    for i in uniq_toks:
        if i not in docf:
            docf[i] = 1
        else:
            docf[i] += 1


# Mimic the IDF function but penalize the words which have fairly high score otherwise, and give a strong boost to the words which appear sporadically.

# In[ ]:


from __future__ import division
import math
def idf(word):
    return 1 - math.sqrt(docf[word]/total_docs)


# In[ ]:


print(idf("kenya"))


# A simple cleaning module for feature extraction

# In[ ]:


import re
import nltk
def basic_cleaning(string):
    string = str(string)
    string = string.lower()
    string = re.sub('[0-9\(\)\!\^\%\$\'\"\.;,-\?\{\}\[\]\\/]', ' ', string)
    string = ' '.join([i for i in string.split() if i not in ["a", "and", "of", "the", "to", "on", "in", "at", "is"]])
    string = re.sub(' +', ' ', string)
    return string


# In[ ]:


def w2v_sim(w1, w2):
    try:
        return model.similarity(w1, w2)*idf(w1)*idf(w2)
    except Exception:
        return 0.0


# **Visualizing features**
# 
# This function will create a 10x10 matrix using MxN word pairs among the words of question pair

# In[ ]:



from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.cm as cm
from scipy import *
df = df.sample(n=30000)
def imagify(row):
    s1 = row['question1']
    s2 = row['question2']
    t1 = list((basic_cleaning(s1)).split())
    t2 = list((basic_cleaning(s2)).split())
    print("Q1: "+ s1)
    print("Q2: "+ s2)
    print("Duplicate: " + str(row['is_duplicate']))
    
    img = [[w2v_sim(x, y) for x in t1] for y in t2] 
    a = np.array(img, order='C')
    img = np.resize(a,(10,10))
    # print img
    fig = plt.figure()
    # tell imshow about color map so that only set colors are used
    image = plt.imshow(img,interpolation='nearest')
    # make a color bar
    plt.colorbar(image)
    plt.show()
s = df.sample(n=3)
plt.close()
s.apply(imagify, axis=1, raw=True)


# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from numpy import *

plt.close()
def surface(row):
    s1 = row['question1']
    s2 = row['question2']
    t1 = list((basic_cleaning(s1)).split())
    t2 = list((basic_cleaning(s2)).split())
    print("Q1: "+ s1)
    print("Q2: "+ s2)
    print("Duplicate: " + str(row['is_duplicate']))
    
#     img = [[w2v_sim(x, y) for x in t1] for y in t2] 

    fig = plt.figure()
    ax = Axes3D(fig)
    X = linspace(0,10,10)
    Y = linspace(0,10,10)
    X, Y = meshgrid(X, Y)
    Z = [[w2v_sim(x, y) for x in t1] for y in t2] 
    a = np.array(Z, order='C')
    Z = np.resize(a,(10,10))
    
    ax.plot_surface(Y, X, Z, rstride=1, cstride=1, cmap=cm.jet)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    plt.show()
    
s = df.sample(n=3)
plt.close()
s.apply(surface, axis=1, raw=True)


# In[ ]:


def img_feature(row):
    s1 = row['question1']
    s2 = row['question2']
    t1 = list((basic_cleaning(s1)).split())
    t2 = list((basic_cleaning(s2)).split())
    Z = [[w2v_sim(x, y) for x in t1] for y in t2] 
    a = np.array(Z, order='C')
    return [np.resize(a,(10,10)).flatten()]
s = df

img = s.apply(img_feature, axis=1, raw=True)
pix_col = [[] for y in range(100)] 
for k in img.iteritems():
        for f in range(len(list(k[1][0]))):
           pix_col[f].append(k[1][0][f])


# **Extracting Features**

# In[ ]:


from nltk.corpus import stopwords
from __future__ import division
stops = set(stopwords.words("english"))

def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R

train_word_match = df.apply(word_match_share, axis=1, raw=True)


# In[ ]:


from __future__ import division
x_train = pd.DataFrame()

for g in range(len(pix_col)):
    x_train['img'+str(g)] = pix_col[g]

    
x_train['word_match'] = train_word_match

y_train = s['is_duplicate'].values
pos_train = x_train[y_train == 1]
neg_train = x_train[y_train == 0]
# Now we oversample the negative class
# There is likely a much more elegant way to do this...
p = 0.165
scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
while scale > 1:
    neg_train = pd.concat([neg_train, neg_train])
    scale -=1
neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
print(len(pos_train) / (len(pos_train) + len(neg_train)))

x_train = pd.concat([pos_train, neg_train])
y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
del pos_train, neg_train


# In[ ]:


from sklearn.cross_validation import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)


# In[ ]:


import xgboost as xgb

# Set our parameters for xgboost
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 7

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, 500, watchlist, early_stopping_rounds=100, verbose_eval=10)


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.rcParams['figure.figsize'] = (12.0, 30.0)
xgb.plot_importance(bst); plt.show()


# Using this technique and combining it with word match features I got log loss of **0.31858** on test dataset. 
# 
# I thought this feature can be of some help to others hence shared. Enjoy :)
