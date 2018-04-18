
# coding: utf-8

#  This kernel is only a  simple lgbm model.  If you find it is useful, please give me an upvote.   The running finished within 1 hour.  I am sure the parameters can be further tuned to get better result.

# In[ ]:


import lightgbm as lgbm
from sklearn.metrics import mean_squared_error
from scipy import sparse as ssp
import random
import string
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import time
import re
import collections
import gc


t00 = time.time()
#  stop-word, can add any wording I want to replace
stopwords=set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
              'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 
              'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
              'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 
              'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
              'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
              'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 
              'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
              'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
              'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
              'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
              'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 
              've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 
              'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn',
               '&','brand new','new','\[rm\]','free ship.*?','free home',
               'rm','price firm','no description yet'               
              ])

pattern = re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*')
train = pd.read_csv('../input/train.tsv', sep="\t",encoding='utf-8',
                    converters={'item_description':lambda x:  pattern.sub('',x.lower()),
                               'name':lambda x:  pattern.sub('',x.lower())}
                   )
test = pd.read_csv('../input/test.tsv', sep="\t",encoding='utf-8',
                    converters={'item_description':lambda x:  pattern.sub('',x.lower()),
                               'name':lambda x:  pattern.sub('',x.lower())}
                    )
simulation=Fa

# ...then introduce random new words
def introduce_new_unseen_words(desc):
    desc = desc.split(' ')
    if random.randrange(0, 10) == 0: # 10% chance of adding an unseen word
        new_word = ''.join(random.sample(string.ascii_letters, random.randrange(3, 15)))
        desc.insert(0, new_word)
    return ' '.join(desc)

if (simulation==True):
    test = pd.concat([test.copy(), test.copy(), test.copy(), test.copy(), test.copy()], axis=0)
    test.item_description = test.item_description.apply(introduce_new_unseen_words)


train_label = np.log1p(train['price'])
train_texts = train['name'].tolist()
test_texts = test['name'].tolist()
print('load tsv completed')

# 
#  replace missing word
# 
train['category_name'].fillna('other', inplace=True)
test['category_name'].fillna('other', inplace=True)

train['brand_name'].fillna('missing', inplace=True)
test['brand_name'].fillna('missing', inplace=True)

test['item_description'].fillna('none', inplace=True)
train['item_description'].fillna('none', inplace=True)

test['nm_word_len']=list(map(lambda x: len(x.split()), test_texts))
train['nm_word_len']=list(map(lambda x: len(x.split()),train_texts))
test['desc_word_len']=list(map(lambda x: len(x.split()), test['item_description'].tolist()))
train['desc_word_len']=list(map(lambda x: len(x.split()), train['item_description'].tolist()))
test['nm_len']=list(map(lambda x: len(x),test_texts))
train['nm_len']=list(map(lambda x: len(x),train_texts))
test['desc_len']=list(map(lambda x: len(x), test['item_description'].tolist()))
train['desc_len']=list(map(lambda x: len(x), train['item_description'].tolist()))
nrow_train = train.shape[0]
test_id=test['test_id']


def split_cat(text):
    try:
        cat_nm=text.split("/")
        if len(cat_nm)>=3:
            return cat_nm[0],cat_nm[1],cat_nm[2]
        if len(cat_nm)==2:
            return cat_Nm[0],cat_nm[1],'missing'
        if len(cat_nm)==1:
            return cat_nm[0],'missing','missing'
    except: return ("missing", "missing", "missing")
train['subcat_0'], train['subcat_1'], train['subcat_2'] = zip(*train['category_name'].apply(lambda x: split_cat(x)))
test['subcat_0'], test['subcat_1'], test['subcat_2'] = zip(*test['category_name'].apply(lambda x: split_cat(x)))
                                 
NAME_MIN_DF=30
count = CountVectorizer(min_df=NAME_MIN_DF)
X_name_mix = count.fit_transform(train['name'].append(test['name']))
X_name=X_name_mix[:nrow_train]
X_t_name = X_name_mix[nrow_train:]


MAX_FEATURES_ITEM_DESCRIPTION=25000
tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION,
                         ngram_range=(1,3))
X_description_mix = tv.fit_transform(train['item_description'].append(test['item_description']))
X_description=X_description_mix[:nrow_train]
X_t_description = X_description_mix[nrow_train:]


print('make categorical features')
cat_features=['subcat_2','subcat_1','subcat_0','brand_name','category_name','item_condition_id','shipping']
for c in cat_features:
    newlist=train[c].append(test[c])
    le = LabelEncoder()
    le.fit(newlist)
    train[c] = le.transform(train[c])
    test[c] = le.transform(test[c])
enc = OneHotEncoder()
enc.fit(train[cat_features].append(test[cat_features]))
X_cat = enc.transform(train[cat_features])
X_t_cat = enc.transform(test[cat_features])
    
train_feature=['desc_word_len','nm_word_len','desc_len','nm_len']
train_list = [train[train_feature].values,X_description,X_name,X_cat]
test_list = [test[train_feature].values,X_t_description,X_t_name,X_t_cat]
X = ssp.hstack(train_list).tocsr()
X_test = ssp.hstack(test_list).tocsr()

print (' finish feature for training')

NFOLDS = 3
kfold =KFold(n_splits=NFOLDS, shuffle=True, random_state=128)




num_boost_round = 1000
params = {"objective": "regression",
          "min_data_in_leaf":1000,
          "boosting_type": "gbdt",
          "learning_rate": 0.65,
          "num_leaves": 128,
          "feature_fraction": 0.5, 
          "bagging_freq": 10,
          "bagging_fraction": 0.9,
          "tree_learner":"voting",
          "verbosity": 0,
          "metric": "l2_root",
          "nthread": 4
          }
cv_pred = np.zeros(len(test_id))
kf = kfold.split(X)
for i, (train_fold, test_fold) in enumerate(kf):
    train_t0 = time.time()
    X_train, X_validate, label_train, label_validate =             X[train_fold, :], X[test_fold, :], train_label[train_fold], train_label[test_fold]
    dtrain = lgbm.Dataset(X_train, label_train)
    dvalid = lgbm.Dataset(X_validate, label_validate, reference=dtrain)
    bst = lgbm.train(params, dtrain, num_boost_round, valid_sets=dvalid, verbose_eval=100,early_stopping_rounds=100)
    cv_pred += bst.predict(X_test, num_iteration=bst.best_iteration)
    print ('training & predict time',time.time()-train_t0)
    gc.collect()

cv_pred /= NFOLDS
cv_pred = np.expm1(cv_pred)
submission = test[["test_id"]]
submission["price"] = cv_pred
submission.to_csv("./myNNsubmission.csv", index=False)
print ('overall time',time.time()-t00)

