
# coding: utf-8

# **Introduction**
# 
# I previously built a XGboost to label data without context. Thought the results are good, but it's not satisfying.
# 
# And later I built a shallow RNN(due to the limitations of hardwares), and I found the RNN to be so disappoint.
# 
# Therefore, I built a new XGboost model to use the context to label data.
# 
# And most importantly, unlike the previous notebook, **this one is trained with all data, including PLAIN VERBATIM LETTERS etc.**
# 
# The major difference in the XGboost without context and with context is their dataset preparation.
# 
# XGboost with context(we will use XGBwithC for short) need the previous words(regardless of whether it comes from the same sentence or not). So we define a padding and 2 special symbols to let XGBoost understand, what's is the boundary of words and what are spaces that can be safely ignored.
# 
# **Results**
# 
# Both trained on 960,000 samples(words, not sentences) from en_train.csv and both have exactly the same parameters for training.
# 
# After 60 epochs, here are the total different results:
# 
# *XGBoost without context:
# 
# valid-merror:**0.005521**	train-merror:**0.004168***
# 
# *XGBoost with context:
# 
# valid-merror:**0.003729**	train-merror:**0.000811***
# 
# **Analysis**
# 
# I noticed astonishingly, XGBoost with context performs so well that it outperforms XGboost without context by a great margin. And what's more, **XGboost with context nearly perfectly fits to the training dataset it has seen.** I believe, if built larger and fed more data, it might be able to challenge deep RNN.
# 
# **Notebook explaintion**
# 
# Due to the time limit on notebook, I choose 320,000 samples instead of 960,000 to save time.
# 
# You can view outputs in this notebook.
# 
# **Final Words**
# 
# I have really limited time and I lack powerful computers(I only have my faithful alienware 15 r2 by my side)
# 
# Therefore I won't be participating in this competition any more.
# 
# And **I hope this script can help you in your research**.

# In[ ]:


import pandas as pd
import numpy as np
import os
import pickle
import gc
import xgboost as xgb
import numpy as np
import re
import pandas as pd
from sklearn.model_selection import train_test_split

max_num_features = 10
pad_size = 1
boundary_letter = -1
space_letter = 0
max_data_size = 320000

out_path = r'.'
df = pd.read_csv(r'../input/en_train.csv')

x_data = []
y_data =  pd.factorize(df['class'])
labels = y_data[1]
y_data = y_data[0]
gc.collect()
for x in df['before'].values:
    x_row = np.ones(max_num_features, dtype=int) * space_letter
    for xi, i in zip(list(str(x)), np.arange(max_num_features)):
        x_row[i] = ord(xi)
    x_data.append(x_row)

def context_window_transform(data, pad_size):
    pre = np.zeros(max_num_features)
    pre = [pre for x in np.arange(pad_size)]
    data = pre + data + pre
    neo_data = []
    for i in np.arange(len(data) - pad_size * 2):
        row = []
        for x in data[i : i + pad_size * 2 + 1]:
            row.append([boundary_letter])
            row.append(x)
        row.append([boundary_letter])
        neo_data.append([int(x) for y in row for x in y])
    return neo_data

x_data = x_data[:max_data_size]
y_data = y_data[:max_data_size]
x_data = np.array(context_window_transform(x_data, pad_size))
gc.collect()
x_data = np.array(x_data)
y_data = np.array(y_data)

print('Total number of samples:', len(x_data))
print('Use: ', max_data_size)
#x_data = np.array(x_data)
#y_data = np.array(y_data)

print('x_data sample:')
print(x_data[0])
print('y_data sample:')
print(y_data[0])
print('labels:')
print(labels)


# In[ ]:


x_train = x_data
y_train = y_data
gc.collect()

x_train, x_valid, y_train, y_valid= train_test_split(x_train, y_train,
                                                      test_size=0.1, random_state=2017)
gc.collect()
num_class = len(labels)
dtrain = xgb.DMatrix(x_train, label=y_train)
dvalid = xgb.DMatrix(x_valid, label=y_valid)
watchlist = [(dvalid, 'valid'), (dtrain, 'train')]

param = {'objective':'multi:softmax',
         'eta':'0.3', 'max_depth':10,
         'silent':1, 'nthread':-1,
         'num_class':num_class,
         'eval_metric':'merror'}
model = xgb.train(param, dtrain, 50, watchlist, early_stopping_rounds=20,
                  verbose_eval=10)
gc.collect()

pred = model.predict(dvalid)
pred = [labels[int(x)] for x in pred]
y_valid = [labels[x] for x in y_valid]
x_valid = [ [ chr(x) for x in y[2 + max_num_features: 2 + max_num_features * 2]] for y in x_valid]
x_valid = [''.join(x) for x in x_valid]
x_valid = [re.sub('a+$', '', x) for x in x_valid]

gc.collect()

df_pred = pd.DataFrame(columns=['data', 'predict', 'target'])
df_pred['data'] = x_valid
df_pred['predict'] = pred
df_pred['target'] = y_valid
df_pred.to_csv(os.path.join(out_path, 'pred.csv'))

df_erros = df_pred.loc[df_pred['predict'] != df_pred['target']]
df_erros.to_csv(os.path.join(out_path, 'errors.csv'), index=False)

model.save_model(os.path.join(out_path, 'xgb_model'))

