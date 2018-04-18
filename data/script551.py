
# coding: utf-8

# **The Choice is Yours**
# 
# This Kernel demonstrates a few feature engineering options, lets get started by importing the python libraries that support comma delimited file data loads for analysis

# In[ ]:


import numpy as np
import pandas as pd
import glob

datafiles = sorted(glob.glob('../input/donorschoose-application-screening/**.csv'))
datafiles = {file.split('/')[-1].split('.')[0]: pd.read_csv(file, encoding='latin-1', low_memory=True) for file in datafiles}
print([k for k in datafiles])


# Lets review the file contents

# In[ ]:


from IPython.display import display
for key in datafiles:
    print(key, len(datafiles[key]))
    display(datafiles[key].head(2))


# Lets utilize a wordcloud to visualize the important keywords in approved project resources

# In[ ]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud
from wordcloud import STOPWORDS
wc = WordCloud(background_color='white', max_words=2000, width=1600, height=1000, stopwords=STOPWORDS)
wc_string = pd.merge(datafiles['train'], datafiles['resources'], how='left', on='id')[['description','project_is_approved']]
wc_string = wc_string[wc_string['project_is_approved']==1]['description'].astype(str).values
wc_string = ' '.join(wc_string)
wc.generate(wc_string)
plt.imshow(wc)


# Here we create a few aggregate resource quantity and price sheet features and combine them with the train and test projects
# * It is good practice to address NA values in left joins, an advance feature engineering option here can be to use sklearn.preprocessing.Imputer

# In[ ]:


print(datafiles['train'].shape, datafiles['test'].shape)
datafiles['resources']['resources_total'] = datafiles['resources']['quantity'] * datafiles['resources']['price']
dfr = datafiles['resources'].groupby(['id'], as_index=False)[['resources_total']].sum()
datafiles['train'] = pd.merge(datafiles['train'], dfr, how='left', on='id').fillna(-1)
datafiles['test'] = pd.merge(datafiles['test'], dfr, how='left', on='id').fillna(-1)

dfr = datafiles['resources'].groupby(['id'], as_index=False)[['resources_total']].mean()
dfr = dfr.rename(columns={'resources_total':'resources_total_mean'})
datafiles['train'] = pd.merge(datafiles['train'], dfr, how='left', on='id').fillna(-1)
datafiles['test'] = pd.merge(datafiles['test'], dfr, how='left', on='id').fillna(-1)

dfr = datafiles['resources'].groupby(['id'], as_index=False)[['quantity']].count()
dfr = dfr.rename(columns={'quantity':'resources_quantity_count'})
datafiles['train'] = pd.merge(datafiles['train'], dfr, how='left', on='id').fillna(-1)
datafiles['test'] = pd.merge(datafiles['test'], dfr, how='left', on='id').fillna(-1)

dfr = datafiles['resources'].groupby(['id'], as_index=False)[['quantity']].sum()
dfr = dfr.rename(columns={'quantity':'resources_quantity_sum'})
datafiles['train'] = pd.merge(datafiles['train'], dfr, how='left', on='id').fillna(-1)
datafiles['test'] = pd.merge(datafiles['test'], dfr, how='left', on='id').fillna(-1)
print(datafiles['train'].shape, datafiles['test'].shape)


# Here we encode categories
# * Advanced encoding options include hashing and weighting

# In[ ]:


from sklearn import *

for c in ['teacher_id','teacher_prefix','school_state', 'project_grade_category']:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(datafiles['train'][c].unique())+list(datafiles['test'][c].unique()))
    datafiles['train'][c] = lbl.fit_transform(datafiles['train'][c].astype(str))
    datafiles['test'][c] = lbl.fit_transform(datafiles['test'][c].astype(str))
    print(c)

    for c in ['project_subject_categories', 'project_subject_subcategories']:
        for i in range(4):
            lbl = preprocessing.LabelEncoder()
            labels = list(datafiles['train'][c].unique())+list(datafiles['test'][c].unique())
            labels = [str(l).split(',')[i] if len(str(l).split(','))>i else '' for l in labels]
            lbl.fit(labels)
            datafiles['train'][c + str(i+1)] = lbl.fit_transform(datafiles['train'][c].map(lambda x: str(x).split(',')[i] if len(str(x).split(','))>i else '').astype(str))
            datafiles['test'][c + str(i+1)] = lbl.fit_transform(datafiles['test'][c].map(lambda x: str(x).split(',')[i] if len(str(x).split(','))>i else '').astype(str))


# In[ ]:


datafiles['test'].head()


# Here we add some date features
# * Based on the data description note on essay data, a feature here can also include a value for when the paragraph responses where changed on February 18, 2010

# In[ ]:


import datetime

for c in ['project_submitted_datetime']:
    for t in ['train','test']:
        datafiles[t][c] = pd.to_datetime(datafiles[t][c])
        datafiles[t][c+'question_balance'] = (datafiles[t][c] < datetime.date(2010, 2, 18)).astype(np.int)
        datafiles[t][c+'quarter'] = datafiles[t][c].dt.year
        datafiles[t][c+'quarter'] = datafiles[t][c].dt.quarter
        datafiles[t][c+'month'] = datafiles[t][c].dt.month
        datafiles[t][c+'day'] = datafiles[t][c].dt.day
        datafiles[t][c+'dow'] = datafiles[t][c].dt.dayofweek
        datafiles[t][c+'wd'] = datafiles[t][c].dt.weekday
        datafiles[t][c+'hr'] = datafiles[t][c].dt.hour
        datafiles[t][c+'m'] = datafiles[t][c].dt.minute
    print(c)


# Here we add description features including:
# * Word length and count features
# * Term Frequency Inverse Document Frequency features (cutoff at 200 features for performance)
# 
# For advanced options you can explore Natural Language Toolkits which include stemming and tokenizing options

# In[ ]:


max_features_ = 200
print(datafiles['train'].shape, datafiles['test'].shape)
for c in ['project_resource_summary', 'project_title', 'project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4']:
    tfidf = feature_extraction.text.TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_df=0.9, min_df=3, max_features=max_features_)
    tfidf.fit(datafiles['train'][c].astype(str))
    for t in ['train','test']:
        datafiles[t][c+'_len'] = datafiles[t][c].map(lambda x: len(str(x)))
        datafiles[t][c+'_wc'] = datafiles[t][c].map(lambda x: len(str(x).split(' ')))
        features = pd.DataFrame(tfidf.transform(datafiles[t][c].astype(str)).toarray())
        features.columns = [c + str(i) for i in range(max_features_)]
        datafiles[t] = pd.concat((datafiles[t], pd.DataFrame(features)), axis=1, ignore_index=False).reset_index(drop=True)
    print(c)
print(datafiles['train'].shape, datafiles['test'].shape)


# Below are the feature columns that will be used and excluded for training and testing

# In[ ]:


col = ['id', 'project_is_approved', 'project_resource_summary', 'project_title', 'project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4', 'project_submitted_datetime', 'project_subject_categories', 'project_subject_subcategories']
col = [c for c in datafiles['train'].columns if c not in col]


# Here we use the XGBoost supervised learning model with an AUC metric and Binary Logistic objective
# * The training file is split as a best practice for better validation metric outputs
# * The learning rate (eta) is set to a high 0.1 for performance with an early stopping option 20 rounds

# Next we blend the results  of a LightGBM supervised learning model with an AUC metric and Binary Logistic objective
# * The training file is split as a best practice for better validation metric outputs
# * The learning rate is set to a high 0.1 for performance with an early stopping option 20 rounds

# In[ ]:


import lightgbm as lgb

x1, x2, y1, y2 = model_selection.train_test_split(datafiles['train'][col],datafiles['train']['project_is_approved'], test_size=0.20, random_state=19)

params = {'learning_rate': 0.1, 'max_depth': 7, 'boosting': 'gbdt', 'objective': 'binary', 'metric': 'auc', 'is_training_metric': True, 'seed': 19}
model2 = lgb.train(params, lgb.Dataset(x1, label=y1), 450, lgb.Dataset(x2, label=y2), verbose_eval=10, early_stopping_rounds=20)
datafiles['test']['project_is_approved'] = model2.predict(datafiles['test'][col], num_iteration=model2.best_iteration)
datafiles['test']['project_is_approved'] = datafiles['test']['project_is_approved'].clip(0+1e12, 1-1e12)


# Here we blend the results of a great [LightGBM and Tf-idf Starter](https://www.kaggle.com/opanichev/lightgbm-and-tf-idf-starter) by Oleg Panichev using the Add Data Source option of Kaggle Kernels for even better results
# 

# In[ ]:


df = pd.read_csv('../input/lightgbm-and-tf-idf-starter/submission.csv')
df = pd.merge(datafiles['test'][['id','project_is_approved']], df, on='id')
df['project_is_approved'] = (df['project_is_approved_x'] + df['project_is_approved_y']*3) / 4
df[['id','project_is_approved']].to_csv('blend_submission.csv', index=False)


# Enjoy the public leaderboard score and the journey of feature engineering!
