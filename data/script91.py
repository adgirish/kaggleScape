
# coding: utf-8

# # Donated to Cancer Treatment Too

# * Note: As this is my first public Kernel and I'm still in learning NLP, please feel free to comment if you have any questions.
# * This Kernel is modified from [the1owl: Redefining Treatment](https://www.kaggle.com/the1owl/redefining-treatment-0-57456), really thanks!
# * I will highlight the main differences from the original.
# * Finnally, donated to cancer treatment too.

# In[1]:


from sklearn import preprocessing, pipeline, feature_extraction, decomposition, model_selection, metrics, cross_validation, svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import normalize, Imputer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

import sklearn
import pandas as pd
import numpy as np
import xgboost as xgb

import datetime


# In[2]:


train = pd.read_csv('../input/training_variants')
test = pd.read_csv('../input/test_variants')
trainx = pd.read_csv('../input/training_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
testx = pd.read_csv('../input/test_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

train = pd.merge(train, trainx, how='left', on='ID').fillna('')
y = train['Class'].values
train = train.drop(['Class'], axis=1)

test = pd.merge(test, testx, how='left', on='ID').fillna('')
pid = test['ID'].values


# In[3]:


train.head()


# In[4]:


y


# In[5]:


test.head()


# In[6]:


pid


# ### 1. Not use the codes below.
# ***
# Not used in this Kernel.
# 
# ```python
# # commented for Kaggle Limits
# for i in range(56):
#     df_all['Gene_'+str(i)] = df_all['Gene'].map(lambda x: str(x[i]) if len(x)>i else '')
#     df_all['Variation'+str(i)] = df_all['Variation'].map(lambda x: str(x[i]) if len(x)>i else '')
# ```

# In[7]:


df_all = pd.concat((train, test), axis=0, ignore_index=True)
df_all['Gene_Share'] = df_all.apply(lambda r: sum([1 for w in r['Gene'].split(' ') if w in r['Text'].split(' ')]), axis=1)
df_all['Variation_Share'] = df_all.apply(lambda r: sum([1 for w in r['Variation'].split(' ') if w in r['Text'].split(' ')]), axis=1)


# In[8]:


df_all.head()


# In[9]:


gen_var_lst = sorted(list(train.Gene.unique()) + list(train.Variation.unique()))
print(len(gen_var_lst))


# In[10]:


gen_var_lst = [x for x in gen_var_lst if len(x.split(' '))==1]
print(len(gen_var_lst))
i_ = 0

#commented for Kaggle Limits
# for gen_var_lst_itm in gen_var_lst:
#     if i_ % 100 == 0: print(i_)
#     df_all['GV_'+str(gen_var_lst_itm)] = df_all['Text'].map(lambda x: str(x).count(str(gen_var_lst_itm)))
#     i_ += 1


# In[11]:


for c in df_all.columns:
    if df_all[c].dtype == 'object':
        if c in ['Gene','Variation']:
            lbl = preprocessing.LabelEncoder()
            df_all[c+'_lbl_enc'] = lbl.fit_transform(df_all[c].values)  
            df_all[c+'_len'] = df_all[c].map(lambda x: len(str(x)))
            df_all[c+'_words'] = df_all[c].map(lambda x: len(str(x).split(' ')))
        elif c != 'Text':
            lbl = preprocessing.LabelEncoder()
            df_all[c] = lbl.fit_transform(df_all[c].values)
        if c=='Text': 
            df_all[c+'_len'] = df_all[c].map(lambda x: len(str(x)))
            df_all[c+'_words'] = df_all[c].map(lambda x: len(str(x).split(' '))) 

train = df_all.iloc[:len(train)]
test = df_all.iloc[len(train):]


# In[12]:


train.head()


# In[13]:


test.head()


# In[14]:


train.shape


# In[15]:


test.shape


# In[ ]:


class cust_regression_vals(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        x = x.drop(['Gene', 'Variation','ID','Text'],axis=1).values
        return x

class cust_txt_col(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        return x[self.key].apply(str)


# ### 2. Main difference
# ***
# #### 1. Pipeline Changed
# 
# The original Kernel uses the pipeline with these codes below for 'Text' feature extraction:
# ```python
# #commented for Kaggle Limits
# ('pi3', pipeline.Pipeline([('Text', cust_txt_col('Text')), 
#                            ('tfidf_Text', feature_extraction.text.TfidfVectorizer(ngram_range=(1, 2))), 
#                            ('tsvd3', decomposition.TruncatedSVD(n_components=50, n_iter=25, random_state=12))]))
# ```
# Unfortunately, it can not fit my memory of 8GB + 2GB(swap). And without these features, I can only get nearly 0.7xxx on PL.
# 
# So, I try to use **HashingVectorizer + TfidfTransformer** instead of **TfidfVectorizer**. [Reference](http://scikit-learn.org/stable/modules/feature_extraction.html#vectorizing-a-large-text-corpus-with-the-hashing-trick)
# 
# #### 2. Parameter Tuning
# 
# For HashingVectorizer saved my memory, I try to use **ngram_range=(1, 3)** with HashingVectorizer. 
# 
# And **n_components=300** with **TruncatedSVD**.
# 
# #### 3. Batch Transform
# 
# With these codes, I can fit_transform the **train**, but still out of memory if transform **test**.
# 
# So, I try to use batch transform of test data step by step, and vstack all. 
# 
# And, it works!

# In[ ]:


print('Pipeline...')
fp = pipeline.Pipeline([
    ('union', pipeline.FeatureUnion(
        n_jobs = -1,
        transformer_list = [
            ('standard', cust_regression_vals()),
            ('pi1', pipeline.Pipeline([('Gene', cust_txt_col('Gene')), 
                                       ('count_Gene', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))), 
                                       ('tsvd1', decomposition.TruncatedSVD(n_components=20, n_iter=25, random_state=12))])),
            ('pi2', pipeline.Pipeline([('Variation', cust_txt_col('Variation')), 
                                       ('count_Variation', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))), 
                                       ('tsvd2', decomposition.TruncatedSVD(n_components=20, n_iter=25, random_state=12))])),
            #commented for Kaggle Limits
#             ('pi3', pipeline.Pipeline([('Text', cust_txt_col('Text')), 
#                                        ('hv', feature_extraction.text.HashingVectorizer(decode_error='ignore', n_features=2 ** 16, non_negative=True, ngram_range=(1, 3))),
#                                        ('tfidf_Text', feature_extraction.text.TfidfTransformer()), 
#                                        ('tsvd3', decomposition.TruncatedSVD(n_components=300, n_iter=25, random_state=12))]))

        
        ])
    )])


train = fp.fit_transform(train)
print (train.shape)

test_t = np.empty([0, train.shape[1]])
step = 200
for i in range(0, len(test), step):
    step_end = i+step
    step_end = step_end if step_end < len(test) else len(test)
    _test = fp.transform(test.iloc[i:step_end])
    test_t = np.vstack((test_t, _test))
test = test_t
print (test.shape)


# ### 3. Xgboost Parameter Tuning
# ***
# #### 1. eta 0.03333 -> 0.02 
# 
# I like small learning rate.
# 
# #### 2. max_depth 4 -> 6 
# 
# Bigger means higher risk of overfitting. But, since we got more features, maybe it could be better for this big data?! I'm not sure yet.
# 
# #### 3. test_size 0.18 -> 0.15
# 
# With slightly more data to train.

# In[ ]:


y = y - 1 #fix for zero bound array


# In[ ]:


file_pre = datetime.datetime.now().strftime('%m_%d_%H_%M_%S')

denom = 0
fold = 1 #Change to 5, 1 for Kaggle Limits
for i in range(fold):
    params = {
#         'eta': 0.03333,
        'eta': 0.02,
#         'max_depth': 4,
        'max_depth': 6,
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': 9,
        'seed': i,
        'silent': True
    }
    x1, x2, y1, y2 = model_selection.train_test_split(train, y, test_size=0.15, random_state=i)
    watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
    model = xgb.train(params, xgb.DMatrix(x1, y1), 1000,  watchlist, verbose_eval=50, early_stopping_rounds=100)
    score1 = metrics.log_loss(y2, model.predict(xgb.DMatrix(x2), ntree_limit=model.best_ntree_limit), labels = list(range(9)))
    print(score1)
    #if score < 0.9:
    if denom != 0:
        pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit+80)
        preds += pred
    else:
        pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit+80)
        preds = pred.copy()
    denom += 1
#     submission = pd.DataFrame(pred, columns=['class'+str(c+1) for c in range(9)])
#     submission['ID'] = pid
#     submission.to_csv('./result/submission_xgb_fold_'  + str(i) + '_' + file_pre + '.csv', index=False)
preds /= denom
submission = pd.DataFrame(preds, columns=['class'+str(c+1) for c in range(9)])
submission['ID'] = pid
submission.to_csv('./result/submission_xgb_' + file_pre + '.csv', index=False)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

plt.rcParams['figure.figsize'] = (7.0, 7.0)
xgb.plot_importance(booster=model,); plt.show()


# ### References
# * [Redefining Treatment](https://www.kaggle.com/the1owl/redefining-treatment-0-57456)
# * [Vectorizing a large text corpus with the hashing trick](http://scikit-learn.org/stable/modules/feature_extraction.html#vectorizing-a-large-text-corpus-with-the-hashing-trick)
# * [HashingVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html)
