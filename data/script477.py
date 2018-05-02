
# coding: utf-8

# In[ ]:


# https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re, string
import time
from scipy.sparse import hstack, vstack

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


# In[ ]:


# Functions
def tokenize(s): return re_tok.sub(r' \1 ', s).split()


# In[ ]:


# read data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
subm = pd.read_csv('../input/sample_submission.csv')

id_train = train['id'].copy()
id_test = test['id'].copy()

# add empty label for None
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['none'] = 1-train[label_cols].max(axis=1)
# fill missing values
COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)


# In[ ]:


# Tf-idf

# prepare tokenizer
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

# create sparse matrices
n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,

                      smooth_idf=1, sublinear_tf=1 )
trn_term_doc = vec.fit_transform(train[COMMENT])
test_term_doc = vec.transform(test[COMMENT])


# In[ ]:


# combine
ytrain = np.zeros((trn_term_doc.shape[0],1)) + 1
ytest = np.zeros((test_term_doc.shape[0],1))
ydat = np.vstack((ytrain, ytest))

xdat = vstack([trn_term_doc, test_term_doc], format='csr')


# In[ ]:


nfolds = 5
xseed = 29
cval = 4

# stratified split
skf = StratifiedKFold(n_splits= nfolds, random_state= xseed)

score_vec = np.zeros((nfolds,1))


# In[ ]:


for (f, (train_index, test_index)) in enumerate(skf.split(xdat, ydat[:,0])):
    # split 
    x0, x1 = xdat[train_index], xdat[test_index]
    y0, y1 = ydat[train_index,0], ydat[test_index,0]    

    clf = LogisticRegression()
    clf.fit(x0,y0)
    prv = clf.predict_proba(x1)[:,1]
    score_vec[f,:] = roc_auc_score(y1,prv)
    print(score_vec[f,:])

