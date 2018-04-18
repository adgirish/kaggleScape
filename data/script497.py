
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# ## Read Data ##

# In[ ]:


df_train = pd.read_csv('../input/train.csv', encoding='utf-8')
df_train['id'] = df_train['id'].apply(str)


# In[ ]:


df_test = pd.read_csv('../input/test.csv', encoding='utf-8')
df_test['test_id'] = df_test['test_id'].apply(str)


# In[ ]:


df_all = pd.concat((df_train, df_test))
df_all['question1'].fillna('', inplace=True)
df_all['question2'].fillna('', inplace=True)


# ## Create Vocab ##

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
import itertools


# In[ ]:


counts_vectorizer = CountVectorizer(max_features=10000-1).fit(
    itertools.chain(df_all['question1'], df_all['question2']))
other_index = len(counts_vectorizer.vocabulary_)


# ##Prep Data##

# In[ ]:


import re
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


# In[ ]:


words_tokenizer = re.compile(counts_vectorizer.token_pattern)


# In[ ]:


def create_padded_seqs(texts, max_len=10):
    seqs = texts.apply(lambda s: 
        [counts_vectorizer.vocabulary_[w] if w in counts_vectorizer.vocabulary_ else other_index
         for w in words_tokenizer.findall(s.lower())])
    return pad_sequences(seqs, maxlen=max_len)


# In[ ]:


df_all = df_all.sample(1000) # Just for debugging


# In[ ]:


X1_train, X1_val, X2_train, X2_val, y_train, y_val =     train_test_split(create_padded_seqs(df_all[df_all['id'].notnull()]['question1']), 
                     create_padded_seqs(df_all[df_all['id'].notnull()]['question2']),
                     df_all[df_all['id'].notnull()]['is_duplicate'].values,
                     stratify=df_all[df_all['id'].notnull()]['is_duplicate'].values,
                     test_size=0.3, random_state=1989)


# ##Training##

# In[ ]:


import keras.layers as lyr
from keras.models import Model


# In[ ]:


input1_tensor = lyr.Input(X1_train.shape[1:])
input2_tensor = lyr.Input(X2_train.shape[1:])

words_embedding_layer = lyr.Embedding(X1_train.max() + 1, 100)
seq_embedding_layer = lyr.LSTM(256, activation='tanh')

seq_embedding = lambda tensor: seq_embedding_layer(words_embedding_layer(tensor))

merge_layer = lyr.multiply([seq_embedding(input1_tensor), seq_embedding(input2_tensor)])

dense1_layer = lyr.Dense(16, activation='sigmoid')(merge_layer)
ouput_layer = lyr.Dense(1, activation='sigmoid')(dense1_layer)

model = Model([input1_tensor, input2_tensor], ouput_layer)

model.compile(loss='binary_crossentropy', optimizer='adam')
model.summary()


# In[ ]:


model.fit([X1_train, X2_train], y_train, 
          validation_data=([X1_val, X2_val], y_val), 
          batch_size=128, epochs=6, verbose=2)


# ##Extract Features From Model##

# In[ ]:


features_model = Model([input1_tensor, input2_tensor], merge_layer)
features_model.compile(loss='mse', optimizer='adam')


# In[ ]:


F_train = features_model.predict([X1_train, X2_train], batch_size=128)
F_val = features_model.predict([X1_val, X2_val], batch_size=128)


# ##Train XGBoost##

# In[ ]:


import xgboost as xgb


# In[ ]:


dTrain = xgb.DMatrix(F_train, label=y_train)
dVal = xgb.DMatrix(F_val, label=y_val)


# In[ ]:


xgb_params = {
    'objective': 'binary:logistic',
    'booster': 'gbtree',
    'eval_metric': 'logloss',
    'eta': 0.1, 
    'max_depth': 9,
    'subsample': 0.9,
    'colsample_bytree': 1 / F_train.shape[1]**0.5,
    'min_child_weight': 5,
    'silent': 1
}
bst = xgb.train(xgb_params, dTrain, 1000,  [(dTrain,'train'), (dVal,'val')], 
                verbose_eval=10, early_stopping_rounds=10)


# ##Predict Test##

# In[ ]:


X1_test = create_padded_seqs(df_all[df_all['test_id'].notnull()]['question1'])
X2_test = create_padded_seqs(df_all[df_all['test_id'].notnull()]['question2'])


# In[ ]:


F_test = features_model.predict([X1_test, X2_test], batch_size=128)


# In[ ]:


dTest = xgb.DMatrix(F_test)


# In[ ]:


df_sub = pd.DataFrame({
        'test_id': df_all[df_all['test_id'].notnull()]['test_id'].values,
        'is_duplicate': bst.predict(dTest, ntree_limit=bst.best_ntree_limit)
    }).set_index('test_id')


# In[ ]:


df_sub.head()


# In[ ]:


df_sub['is_duplicate'].hist(bins=100)

