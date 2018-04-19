
# coding: utf-8

# **Word Vectors and Features**
# 
# I have been wanting to try out a model that combines word vectors with feature engineering. I also wanted my model to use KFold CV with a validation set within each fold. Finally, I wanted to ensemble the predictions to produce a combined model that out performs its individual parts.
# 
# **Over time, this kernel has grown to include XGBoost, LightGBM and CatBoost models. CatBoost has been dropped in favor of NB-SVM. The final output of each mode is ensembled using a technique from another kernel in this competition.**
# 
# This isn't the prettiest kernel with the highest leaderboard score but I hope that it's useful for those less proficient in python than I am. For those farther along, please suggest improvements!
# 
# I will add more annotations, explanations, modify parameters and make changes over the course of the competition. Stay tuned...

# In[ ]:


import numpy as np
import pandas as pd
import random
from sklearn import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from scipy.sparse import hstack, vstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import punkt
stop_words = stopwords.words('english')

import string
import re
import gc

data_path = '../input/'

# For a clean example that achieves
# the same thing (and more), see t35khan's kernel:
# https://www.kaggle.com/t35khan/tfidf-driven-xgboost


# **Load the Files**
# 
# Two of the columns in **train** and **test** seem to have values that cause pandas some confusion. I have set them to "object" type on import.
# 
# We carve out the target variable, called "project_is_approved," from **train** right away. Drop the target column from **train**.
# 
# Read in the sample submission file. We'll use the IDs from that file later.
# 
# Finally, fill in NA values for text columns with "unk" for "unknown." The string "unk" isn't a word but it will add information to your word vectors : )

# In[ ]:


train = pd.read_csv(data_path + 'donorschoose-application-screening/train.csv', dtype={"project_essay_3": object, "project_essay_4": object})#, nrows=10000)
target = train['project_is_approved']
train = train.drop('project_is_approved', axis=1)

test = pd.read_csv(data_path + 'donorschoose-application-screening/test.csv', dtype={"project_essay_3": object, "project_essay_4": object})#, nrows=10000)

sub = pd.read_csv(data_path + 'donorschoose-application-screening/sample_submission.csv')#, nrows=10000)

resources = pd.read_csv(data_path + 'donorschoose-application-screening/resources.csv')#, nrows=10000)

train.fillna(('unk'), inplace=True)
test.fillna(('unk'), inplace=True)


# In[ ]:


# Split off first two categories into their own cols
for i in range(2):
    # cat
    train['cat'+str(i)] = train['project_subject_categories'].str.split(',', expand=True)[i]
    test['cat'+str(i)] = train['project_subject_categories'].str.split(',', expand=True)[i]
    # sub cat
    train['scat'+str(i)] = train['project_subject_subcategories'].str.split(',', expand=True)[i]
    test['scat'+str(i)] = train['project_subject_subcategories'].str.split(',', expand=True)[i]

#train = train.drop('project_subject_categories', axis=1)
#test = test.drop('project_subject_categories', axis=1)

#train = train.drop('project_subject_subcategories', axis=1)
#test = test.drop('project_subject_subcategories', axis=1)

train.head()


# **Feature Engineering**
# 
# The first step in feature engineering is to turn labels, or categorical values, into integers. Generally, XGBoost does better with label encoding than with one hot encoding.
# 
# @opanichev provided a great piece of code so let's use it!

# In[ ]:


# Thanks to opanichev for saving me a few minutes
# https://www.kaggle.com/opanichev/lightgbm-and-tf-idf-starter/code

# Label encoding

df_all = pd.concat([train, test], axis=0)

cols = [
    'teacher_id', 
    'teacher_prefix', 
    'school_state', 
    'project_grade_category', 
    'project_subject_categories', 
    'project_subject_subcategories',
    'cat0',
    'scat0',
    'cat1',
    'scat1'
]

for c in tqdm(cols):
    le = LabelEncoder()
    le.fit(df_all[c].astype(str))
    train[c] = le.transform(train[c].astype(str))
    test[c] = le.transform(test[c].astype(str))
    
del df_all; gc.collect()


# When it comes to feature engineering, get creative! What metrics can we create that may provide more signal to the model than noise?

# In[ ]:


# Feature engineering

# Date and time
train['project_submitted_datetime'] = pd.to_datetime(train['project_submitted_datetime'])
test['project_submitted_datetime'] = pd.to_datetime(test['project_submitted_datetime'])

# Date as int may contain some ordinal value
train['datetime_int'] = train['project_submitted_datetime'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
test['datetime_int'] = test['project_submitted_datetime'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)

# Date parts

train['datetime_day'] = train['project_submitted_datetime'].dt.day
train['datetime_dow'] = train['project_submitted_datetime'].dt.dayofweek
train['datetime_year'] = train['project_submitted_datetime'].dt.year
train['datetime_month'] = train['project_submitted_datetime'].dt.month
train['datetime_hour'] = train['project_submitted_datetime'].dt.hour
train = train.drop('project_submitted_datetime', axis=1)

test['datetime_day'] = test['project_submitted_datetime'].dt.day
test['datetime_dow'] = test['project_submitted_datetime'].dt.dayofweek
test['datetime_year'] = test['project_submitted_datetime'].dt.year
test['datetime_month'] = test['project_submitted_datetime'].dt.month
test['datetime_hour'] = test['project_submitted_datetime'].dt.hour
test = test.drop('project_submitted_datetime', axis=1)

# Essay length
train['e1_length'] = train['project_essay_1'].apply(len)
test['e1_length'] = train['project_essay_1'].apply(len)

train['e2_length'] = train['project_essay_2'].apply(len)
test['e2_length'] = train['project_essay_2'].apply(len)

# Title length
train['project_title_len'] = train['project_title'].apply(lambda x: len(str(x)))
test['project_title_len'] = test['project_title'].apply(lambda x: len(str(x)))

# Has more than 2 essays?
train['has_gt2_essays'] = train['project_essay_3'].apply(lambda x: 0 if x == 'unk' else 1)
test['has_gt2_essays'] = test['project_essay_3'].apply(lambda x: 0 if x == 'unk' else 1)


# Let's create some features from the numerical columns in the **resources.csv** file. You could try merging the descriptive text into the content that gets vectorized but we'll leave that behind for now.

# In[ ]:


# Combine resources file
# Thanks, the1owl! 
# https://www.kaggle.com/the1owl/the-choice-is-yours

resources['resources_total'] = resources['quantity'] * resources['price']

dfr = resources.groupby(['id'], as_index=False)[['resources_total']].sum()
train = pd.merge(train, dfr, how='left', on='id').fillna(-1)
test = pd.merge(test, dfr, how='left', on='id').fillna(-1)

dfr = resources.groupby(['id'], as_index=False)[['resources_total']].mean()
dfr = dfr.rename(columns={'resources_total':'resources_total_mean'})
train = pd.merge(train, dfr, how='left', on='id').fillna(-1)
test = pd.merge(test, dfr, how='left', on='id').fillna(-1)

dfr = resources.groupby(['id'], as_index=False)[['quantity']].count()
dfr = dfr.rename(columns={'quantity':'resources_quantity_count'})
train = pd.merge(train, dfr, how='left', on='id').fillna(-1)
test = pd.merge(test, dfr, how='left', on='id').fillna(-1)

dfr = resources.groupby(['id'], as_index=False)[['quantity']].sum()
dfr = dfr.rename(columns={'quantity':'resources_quantity_sum'})
train = pd.merge(train, dfr, how='left', on='id').fillna(-1)
test = pd.merge(test, dfr, how='left', on='id').fillna(-1)

# We're done with IDs for now
train = train.drop('id', axis=1)
test = test.drop('id', axis=1)


# Concatenate text columns. This may not be optimal but it is efficient.

# In[ ]:


# Thanks to opanichev for saving me a few minutes
# https://www.kaggle.com/opanichev/lightgbm-and-tf-idf-starter/code

train['project_essay'] = train.apply(lambda row: ' '.join([
    str(row['project_title']),
    str(row['project_essay_1']), 
    str(row['project_essay_2']), 
    str(row['project_essay_3']),
    str(row['project_essay_4']),
    str(row['project_resource_summary'])]), axis=1)
test['project_essay'] = test.apply(lambda row: ' '.join([
    str(row['project_title']),
    str(row['project_essay_1']), 
    str(row['project_essay_2']), 
    str(row['project_essay_3']),
    str(row['project_essay_4']),
    str(row['project_resource_summary'])]), axis=1)

train = train.drop([
    'project_title',
    'project_essay_1', 
    'project_essay_2', 
    'project_essay_3', 
    'project_essay_4',
    'project_resource_summary'], axis=1)
test = test.drop([
    'project_title',
    'project_essay_1', 
    'project_essay_2', 
    'project_essay_3', 
    'project_essay_4',
    'project_resource_summary'], axis=1)


# How are we doing with our training data transformations and feature engineering?

# In[ ]:


train.head()


# As you can see, our data is now mostly numeric. Let's transform the concatenated text into numbers, too.
# 
# Next, we'll clean the text up a bit and [lemmatize](https://en.wikipedia.org/wiki/Lemmatisation) it. 

# In[ ]:


w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def prep_text(text):
    text = text.strip().lower()
    text = re.sub('\W+',' ', text)
    text = re.sub(r'(\")', ' ', text)
    text = re.sub(r'(\r)', ' ', text)
    text = re.sub(r'(\n)', ' ', text)
    text = re.sub(r'(\r\n)', ' ', text)
    text = re.sub(r'(\\)', ' ', text)
    text = re.sub(r'\t', ' ', text)
    text = re.sub(r'\:', ' ', text)
    text = re.sub(r'\"\"\"\"', ' ', text)
    text = re.sub(r'_', ' ', text)
    text = re.sub(r'\+', ' ', text)
    text = re.sub(r'\=', ' ', text)
    text = re.sub(' i m ',' i\'m ', text)
    text = re.sub('n t ','n\'t ', text)
    text = re.sub(' re ',' are ', text)
    text = [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]
    return text

train['project_essay'] = train['project_essay'].apply(lambda x: prep_text(x))
test['project_essay'] = test['project_essay'].apply(lambda x: prep_text(x))


# Here's what our new text objects look like:

# In[ ]:


# Note that stop words are handled by the TFIDF vectorzer, below
train['project_essay'][0:20]


# I won't try to explain [TFIDF](http://https://en.wikipedia.org/wiki/Tf%E2%80%93idf) or text vectorization in general. Follow the link in the comment below, and the links from the link, to learn more.

# In[ ]:


# Learn more about NLP from Abishek: 
# https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle
tfv = TfidfVectorizer(norm='l2', min_df=0,  max_features=8000, 
            strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
            ngram_range=(1,2), use_idf=True, smooth_idf=False, sublinear_tf=True,
            stop_words = 'english')


# Should we fit on **train** and **test** individually, to avoid leakyness? Maybe, but I'm not going to do it that way.
# 
# Note that **hstack** is the method by which we combine our engineered features with the TFIDF vectorized text.

# In[ ]:


train_text = train['project_essay'].apply(lambda x: ' '.join(x))
test_text = test['project_essay'].apply(lambda x: ' '.join(x))

# Fitting tfidf on train + test might be leaky
tfv.fit(list(train_text.values) + list(test_text.values))
train_tfv = tfv.transform(train_text)
test_tfv = tfv.transform(test_text)

del train_text, test_text; gc.collect()


# In[ ]:


# Combine text vectors and features
feat_train = train.drop('project_essay', axis=1)
feat_test = test.drop('project_essay', axis=1)

feat_train = csr_matrix(feat_train.values)
feat_test = csr_matrix(feat_test.values)

X_train_stack = hstack([feat_train, train_tfv[0:feat_train.shape[0]]])
X_test_stack = hstack([feat_test, test_tfv[0:feat_test.shape[0]]])

print('Train shape: ', X_train_stack.shape, '\n\nTest Shape: ', X_test_stack.shape)

del train, test, train_tfv, test_tfv; gc.collect()


# In[ ]:


seed = 28 # Get your own seed


# In[ ]:


K = 5 # How many folds do you want? 
kf = KFold(n_splits = K, random_state = seed, shuffle = True)


# **The XGBoost Model**
# 
# We'll set up arrays to capture our CV scores and the predictions made agains the **test** set. Those will be blended to give us more robust predictions and to prevent overfitting.

# In[ ]:


cv_scores = []
xgb_preds = []

for train_index, test_index in kf.split(X_train_stack):
    
    # Split out a validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_stack, target, test_size=0.20, random_state=random.seed(seed))
    
    # params are tuned with kaggle kernels in mind
    xgb_params = {'eta': 0.15, 
                  'max_depth': 7, 
                  'subsample': 0.80, 
                  'colsample_bytree': 0.80, 
                  'objective': 'binary:logistic', 
                  'eval_metric': 'auc', 
                  'seed': seed
                 }
    
    d_train = xgb.DMatrix(X_train, y_train)
    d_valid = xgb.DMatrix(X_valid, y_valid)
    d_test = xgb.DMatrix(X_test_stack)
    
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    model = xgb.train(xgb_params, d_train, 2000, watchlist, verbose_eval=50, early_stopping_rounds=30)
    cv_scores.append(float(model.attributes()['best_score']))
    xgb_pred = model.predict(d_test)
    xgb_preds.append(list(xgb_pred))
    
    del X_train, X_valid, y_train, y_valid, d_train, d_valid, d_test; gc.collect()


# I like to get a look at the CV scores for each fold. Keep track of the average CV score to determine how parameter changes affect your model.

# In[ ]:


print(cv_scores)
print(np.mean(cv_scores))


# Blend predictions...

# In[ ]:


x_preds=[]
for i in range(len(xgb_preds[0])):
    sum=0
    for j in range(K):
        sum+=xgb_preds[j][i]
    x_preds.append(sum / K)


# ... and then take a peek to see how they compare to your last effort or other models. 

# In[ ]:


# Peek at predictions
x_preds[0:10]


# In[ ]:


# XGB preds
x_preds = pd.DataFrame(x_preds)
x_preds.columns = ['project_is_approved']

submid = sub['id']
xsub = pd.concat([submid, x_preds], axis=1)
xsub.to_csv('xgb_submission.csv', index=False)


# **While we're at it...**
# LightGBM is doing well with this data and we have some matrices ready to go... may as well run a second model and blend the results with a weighted average to see if we can get a little **boost**.

# In[ ]:


# LGBM seems to do well with this data
# Check out @opanichev's kernel
# https://www.kaggle.com/opanichev/lightgbm-and-tf-idf-starter
cnt = 0
p_buf = []
n_splits = 5
n_repeats = 1
kf = RepeatedKFold(
    n_splits=n_splits, 
    n_repeats=n_repeats, 
    random_state=seed)
auc_buf = []   

for train_index, valid_index in kf.split(X_train_stack):
    
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_stack, target, test_size=0.20, random_state=random.seed(seed))
    
    print('Fold {}/{}'.format(cnt + 1, n_splits))
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 7,
        'num_leaves': 32,
        'learning_rate': 0.02,
        'feature_fraction': 0.80,
        'bagging_fraction': 0.80,
        'bagging_freq': 5,
        'verbose': 0,
        'lambda_l2': 1,
    }  

    model = lgb.train(
        params,
        lgb.Dataset(X_train, y_train),
        num_boost_round=10000,
        valid_sets=[lgb.Dataset(X_valid, y_valid)],
        early_stopping_rounds=50,
        verbose_eval=100
        )

    p = model.predict(X_valid, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_valid, p)

    print('{} AUC: {}'.format(cnt, auc))

    p = model.predict(X_test_stack, num_iteration=model.best_iteration)
    if len(p_buf) == 0:
        p_buf = np.array(p)
    else:
        p_buf += np.array(p)
    auc_buf.append(auc)

    cnt += 1
    #if cnt > 0: # Comment this to run several folds
    #    break
    
    del model
    gc.collect

auc_mean = np.mean(auc_buf)
auc_std = np.std(auc_buf)
print('AUC = {:.6f} +/- {:.6f}'.format(auc_mean, auc_std))

lgb_preds = p_buf/cnt

del X_valid, y_valid, X_train, y_train; gc.collect()


# In[ ]:


# Peek at lgbm preds
l_preds = pd.DataFrame(lgb_preds)
l_preds.columns = ['project_is_approved']
l_preds.head()

submid = sub['id']
lsub = pd.concat([submid, l_preds], axis=1)
lsub.to_csv('lgbm_submission.csv', index=False)


# **CatBoost performance is pretty bad.** We'll add **NB-SVM** into the mix instead. It's performance is also weak on its own but maybe it will help the ensemble.

# In[ ]:


# Start with fresh data
del X_train_stack, X_test_stack; gc.collect()


# In[ ]:


train = pd.read_csv(data_path + 'donorschoose-application-screening/train.csv', dtype={"project_essay_3": object, "project_essay_4": object})#, nrows=10000)
labels = pd.DataFrame(train['project_is_approved'])
labels.columns = ['project_is_approved']
train = train.drop('project_is_approved', axis=1)

test = pd.read_csv(data_path + 'donorschoose-application-screening/test.csv', dtype={"project_essay_3": object, "project_essay_4": object})#, nrows=10000)

resources = pd.read_csv(data_path + 'donorschoose-application-screening/resources.csv')

sub = pd.read_csv(data_path + 'donorschoose-application-screening/sample_submission.csv')

train.fillna(('unk'), inplace=True) 
test.fillna(('unk'), inplace=True)


# In[ ]:


# Thanks to opanichev for saving me a few minutes
# https://www.kaggle.com/opanichev/lightgbm-and-tf-idf-starter/code

train['project_essay'] = train.apply(lambda row: ' '.join([
    str(row['project_title']),
    str(row['project_essay_1']), 
    str(row['project_essay_2']), 
    str(row['project_essay_3']),
    str(row['project_essay_4']),
    str(row['project_resource_summary'])]), axis=1)
test['project_essay'] = test.apply(lambda row: ' '.join([
    str(row['project_title']),
    str(row['project_essay_1']), 
    str(row['project_essay_2']), 
    str(row['project_essay_3']),
    str(row['project_essay_4']),
    str(row['project_resource_summary'])]), axis=1)

train = train.drop([
    'project_title',
    'project_essay_1', 
    'project_essay_2', 
    'project_essay_3', 
    'project_essay_4',
    'project_resource_summary'], axis=1)
test = test.drop([
    'project_title',
    'project_essay_1', 
    'project_essay_2', 
    'project_essay_3', 
    'project_essay_4',
    'project_resource_summary'], axis=1)

gc.collect()


# In[ ]:


COMMENT = 'project_essay'

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

n = train.shape[0]

vec = TfidfVectorizer(ngram_range=(1,3), tokenizer=tokenize,
               min_df=3, max_df=0.95, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )

#vec = CountVectorizer(ngram_range=(1,3), tokenizer=tokenize,
#               min_df=3, max_df=0.95, strip_accents='unicode')

train_vec = vec.fit_transform(train[COMMENT])
test_vec = vec.transform(test[COMMENT])

train = train.drop('project_essay', axis=1)
test = test.drop('project_essay', axis=1)


# In[ ]:


label_cols = ['project_is_approved']

x = train_vec
test_x = test_vec

def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=4, dual=True)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r

nb_preds = np.zeros((test.shape[0], len(label_cols)))

for i, j in enumerate(label_cols):
    print('fitting: ', j)
    m,r = get_mdl(labels[j])
    nb_preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]


# In[ ]:


# Peek at predictions
nb_preds[0:10]


# In[ ]:


# nb preds to pandas dataframe
nb_preds = pd.DataFrame(nb_preds)
nb_preds.columns = ['project_is_approved']

submid = sub['id']
nbsub = pd.concat([submid, nb_preds], axis=1)
nbsub.to_csv('nb_submission.csv', index=False)


# **Ensemble Predictions**
# Below, we're trying out @matthewa313's logit ensembling kernel. It seems to perform well (better than simple averaging) but the CatBoost model is a drag on overall performance.

# In[ ]:


# From @matthewa313's Ensembling with Logistic Regression (LB 81.947)
# https://www.kaggle.com/matthewa313/ensembling-with-logistic-regression-lb-81-947/code
# CODE OPTIMIZED: https://www.kaggle.com/matthewa313/ensembling-with-lr-v2-lb-82-474

from scipy.special import expit, logit
 
almost_zero = 1e-10
almost_one  = 1-almost_zero

# nb-svm
nbsub.columns = ['id', 'project_is_approved1']
df1 = nbsub # 0.72237

# lgbm
lsub.columns = ['id', 'project_is_approved2']
df2 = lsub # 0.77583

# xgb
xsub.columns = ['id', 'project_is_approved3']
df3 = xsub # 0.77212

df = pd.merge(df1, df2, on='id')
df = pd.merge(df, df3, on='id')

print(df.head())

scores = [0, 0.72237, 0.77583, 0.77212] # public leaderboard scores

wts = [0, 0, 0, 0]

power = 68 # How is this calculated?

wts[1] = scores[1] ** power
wts[2] = scores[2] ** power
wts[3] = scores[3] ** power

print(wts[:])

# Any answers really close to zero or really close to one will be binary
number1 = df['project_is_approved1'].clip(almost_zero,almost_one).apply(logit) * wts[1]
number2 = df['project_is_approved2'].clip(almost_zero,almost_one).apply(logit) * wts[2]
number3 = df['project_is_approved3'].clip(almost_zero,almost_one).apply(logit) * wts[3]

totweight = wts[0] + wts[1] + wts[2] + wts[3]

df['project_is_approved'] = ( number1 + number2 + number3 ) / ( totweight )

df['project_is_approved']  = df['project_is_approved'].apply(expit) 

df[['id', 'project_is_approved']].to_csv("ensembling-logit_submission.csv", index=False)

