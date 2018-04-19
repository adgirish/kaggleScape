
# coding: utf-8

# # <center> Beginner's guide: NN with multichannel input  in Keras</center>
# 
# _Author: Kirill Vlasov_
# 
# --------
# 
# # Introduction
# In this article we will not discuss types of Neural Network. We will try to build network with multichannel input, because this case is so difficult for novice.  
#   
# 
# __Plan:__
# - Explanation of model’s usefulness
# - How to develop a neural network with multichannel input in Keras.
# - Practice: using this approach in <a href="https://www.kaggle.com/c/donorschoose-application-screening">DonorsChoose Competition</a>
# 
# Let's start!
# 
# # Explanation of model’s usefulness
# Imagine, we have a dataset of images and we need to solve the problem of classification. Probably, we will develop a convolutional neural network. What are you going to do, in order to supplement meta data (texts, some categorical features and etc.) in model?  
# Obviously, we need different types of NN for different types of data, e.g. RNN, CNN and etc. But NN with multichannel input allows to create ONE NN, which could merge all different types of needed NNs. It could divide different flows of calculation, and then merge them together inside one joint NN.
#   
# # How to develop a neural network with multichannel input in Keras.
# 
# - Firts of all, we define the type of each data and choose apropriate type of NN for each type of data. 
# - Then, we develop each NN.
# - By class _concatenate_ of module _layers.merge_ in _Keras_ we merge all outputs of these different NNs
# - Enjoy! :) 
#  
# __That's all!__
#   
# # Practice: using this approach in DonorsChoose Competition </a>
# 
#   
# ## 0. Importing Libraries

# In[ ]:


import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from tqdm import tqdm_notebook
import re
import nltk
from nltk.stem import SnowballStemmer

from keras.preprocessing.text import Tokenizer

from keras.layers import Dense, Activation, Dropout, Flatten, Input
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.models import Model, Sequential 
from keras.layers.recurrent import LSTM

import tensorflow as tf
from keras import backend as K

from keras.layers.merge import concatenate
from keras.utils import plot_model

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# ## 1. Loading and preprocessing data

# In[ ]:


train = pd.read_csv('../input/train.csv', low_memory=False, index_col='id')
test = pd.read_csv('../input/test.csv', low_memory=False, index_col='id')

res = pd.read_csv('../input/resources.csv', low_memory=False, index_col='id')


# ### 1.1. Concatination of train and test

# In[ ]:


train['is_train'] = 1
test['is_train'] = 0


# In[ ]:


df = pd.concat([train, test], axis=0)


# ### 1.2. Generate features from 'res.csv'

# In[ ]:


sum_res = pd.pivot_table(res, index=res.index, aggfunc='sum', values=['price', 'quantity'])
mean_res = pd.pivot_table(res, index=res.index, aggfunc='mean', values=['price', 'quantity'])
median_res = pd.pivot_table(res, index=res.index, aggfunc='median', values=['price', 'quantity'])

df = pd.merge(df, sum_res,left_index=True, right_index=True)
df = pd.merge(df, mean_res,left_index=True, right_index=True, suffixes=('_sum', ''))
df = pd.merge(df, median_res,left_index=True, right_index=True, suffixes=('_mean', '_median'))


# ### 1.3. Type of features

# In[ ]:


df.columns


# In[ ]:


cat_feature = ['school_state', 'teacher_prefix', 
               'project_subject_categories', 'project_subject_subcategories', 'project_grade_category']

target = 'project_is_approved'

text_feature = ['project_title', 'project_resource_summary', 'project_essay_1', 'project_essay_2', 'project_essay_3',
       'project_essay_4' ]

real_feature = ['teacher_number_of_previously_posted_projects', 'price_sum', 'quantity_sum', 'price_mean', 'quantity_mean',
       'price_median', 'quantity_median' ]



# ### 1.4. Preprocessing of features 
# __Categorical__  
# We may just facrorize features of this type

# In[ ]:


for i in cat_feature:
    df[i] = pd.factorize(df[i])[0]

trn_cat = df[cat_feature].values[:182080]
tst_cat = df[cat_feature].values[182080:]


# __Real__  
# Don't forget about _Scalling_

# In[ ]:


SS = StandardScaler()
df_scale = SS.fit_transform(df[real_feature])

trn_real = df_scale[:182080]
tst_real = df_scale[182080:]


# __Text__  
# Processing of text data easily

# In[ ]:


df_text = df[text_feature].fillna(' ')
df_text['full_text'] = ''
for f in text_feature:
    df_text['full_text'] = df_text['full_text'] + df_text[f]


# In[ ]:


stemmer = SnowballStemmer('english')

def clean(text):
    return re.sub('[!@#$:]', '', ' '.join(re.findall('\w{3,}', str(text).lower())))

def stem(text):
    return ' '.join([stemmer.stem(w) for w in text.split()])


# In[ ]:


df_text['full_text'] = df_text['full_text'].apply(lambda x: clean(x))


# In[ ]:


#df_text['full_text'] = df_text['full_text'].apply(lambda x: stem(x)) - don't think about it :)


# In[ ]:


max_words = 500 #more words for more accuracy
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df_text['full_text'])

trn_text = tokenizer.texts_to_matrix(df_text['full_text'][:182080], mode='binary')
tst_text = tokenizer.texts_to_matrix(df_text['full_text'][182080:], mode='binary')


# __Target__

# In[ ]:


y = df[target].values[:182080]


# ## 2. Modeling! 
# ### 2.1. Parameters

# In[ ]:


len_cat = trn_cat.shape[1]
len_real = trn_real.shape[1]
len_text = trn_text.shape[1]


size_embedding = 5000


# ### 2.2. Architecture

# In[ ]:


# categorical channel 
inputs1 = Input(shape=(len_cat,))
dense_cat_1 = Dense(256, activation='relu')(inputs1)
dense_cat_2 = Dense(128, activation='relu')(dense_cat_1)
dense_cat_3 = Dense(64, activation='relu')(dense_cat_2)
dense_cat_4 = Dense(32, activation='relu')(dense_cat_3)
flat1 = Dense(32, activation='relu')(dense_cat_4)



# real channel
inputs2 = Input(shape=(len_real,))
dense_real_1 = Dense(256, activation='relu')(inputs2)
dense_real_2 = Dense(128, activation='relu')(dense_real_1)
dense_real_3 = Dense(64, activation='relu')(dense_real_2)
dense_real_4 = Dense(32, activation='relu')(dense_real_3)
flat2 = Dense(32, activation='relu')(dense_real_4)


# text chanel
inputs3 = Input(shape=(len_text,))
embedding3 = Embedding(size_embedding, 36)(inputs3)
conv3 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding3)
drop3 = Dropout(0.1)(conv3)
pool3 = MaxPooling1D(pool_size=2)(drop3)
flat3 = Flatten()(pool3)

# merge
merged = concatenate([flat1, flat2, flat3])

# interpretation
dense1 = Dense(200, activation='relu')(merged)
dense2 = Dense(20, activation='relu')(dense1)
outputs = Dense(1, activation='sigmoid')(dense2)
model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)

model.summary()


# ### 2.4. Metric  
# Thx Stackoverflow for realization

# In[ ]:


# AUC for a binary classifier
def auc(y_true, y_pred):   
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)    
    return FP/N
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)    
    return TP/P


# ### 2.3. Compilation

# In[ ]:


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy', auc])


# ### 2.4. Fitting

# In[ ]:


batch_size = 1000
model.fit([trn_cat, trn_real, trn_text], y, batch_size=batch_size, epochs=3, validation_split=0.2)


# ### 2.5. Submitting

# In[ ]:


submit = model.predict([tst_cat, tst_real, tst_text], batch_size=batch_size,verbose=1)


# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


submission['project_is_approved'] = submit
submission.to_csv('mi_nn.csv', index=False)


# ## 3.Comparison with non-multichannel type of NN

# In[ ]:


trn_all = np.hstack((trn_cat, trn_real, trn_text))
trn_all.shape


# In[ ]:


model2 = Sequential()
model2.add(Dense(256, input_shape=(trn_all.shape[1],), activation='relu'))
model2.add(Dense(128, activation='relu'))
model2.add(Dense(1, activation='sigmoid'))


# In[ ]:


model2.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy', auc])


# In[ ]:


batch_size = 2000
model2.fit(trn_all, y, batch_size=batch_size, epochs=3, validation_split=0.2)


# # Conclusion
# Certainly, computing power of Kaggle's kernel doesn't allow to build more sophisticated models, but in practice we may experiment with NN with multichannel input to achieve better results. Finally, NN with multichannel input are more flexible and let you work with different types of data. 
# 
# 
# 
# # Links
# - <a href = "https://keras.io" > Keras Documentation </a>
# - <a href = "https://machinelearningmastery.com/develop-n-gram-multichannel-convolutional-neural-network-sentiment-analysis/" >How to Develop an N-gram Multichannel Convolutional Neural Network for Sentiment Analysis </a>
# - <a href = https://towardsdatascience.com/neural-network-architectures-156e5bad51ba> Neural Network Architectures </a>
# 
