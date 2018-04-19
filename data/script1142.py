
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# In[ ]:


pd.set_option('display.float_format', lambda x: '%.3f' % x) # supress scientfic notation


# In[ ]:


df_train = pd.read_csv('../input/train.csv')


# In[ ]:


df_train.iloc[:,2:].sum(0)


# In[ ]:


categories = df_train.columns[2:].values
categories


# In[ ]:


def get_feature_importances(model, analyzer, ngram, lowercase, min_df=10, sampsize=40000):
    tfv = TfidfVectorizer(min_df=min_df,
                          strip_accents='unicode',
                          analyzer=analyzer,
                          ngram_range=(ngram, ngram),
                          lowercase=lowercase)
    df_sample = df_train.sample(sampsize, random_state=123)
    X = tfv.fit_transform(df_sample.comment_text)
    scaler = StandardScaler(with_mean=False)
    scaler.fit(X)
    terms = tfv.get_feature_names()
    #print('#terms:', len(terms))
    var_imp = pd.DataFrame(index=terms)
    for category in categories:
        y = df_sample[category].values
        model.fit(X, y)
        var_imp[category] =  np.sqrt(scaler.var_) * model.coef_[0]
    var_imp = var_imp.sort_values('toxic', ascending=False)
    return var_imp


# In[ ]:


model = LogisticRegression()


# In[ ]:


var_imp = get_feature_importances(model, analyzer='word', ngram=1, lowercase=True)
var_imp.head(10)


# In[ ]:


var_imp.tail(10)


# In[ ]:


var_imp = get_feature_importances(model, analyzer='word', ngram=2, lowercase=True)
var_imp.head(10)


# In[ ]:


var_imp.tail(10)


# In[ ]:


var_imp = get_feature_importances(model, analyzer='word', ngram=3, lowercase=True)
var_imp.head(10)


# In[ ]:


var_imp.tail(10)


# In[ ]:


var_imp.sort_values('severe_toxic').tail(5)


# In[ ]:


var_imp.sort_values('obscene').tail(5)


# In[ ]:


var_imp.sort_values('threat').tail(5)


# In[ ]:


var_imp.sort_values('insult').tail(5)


# In[ ]:


var_imp.sort_values('identity_hate').tail(5)


# In[ ]:


var_imp = get_feature_importances(model, analyzer='word', ngram=4, lowercase=True)
var_imp.head(10)


# In[ ]:


var_imp.tail(10)


# In[ ]:


var_imp = get_feature_importances(model, analyzer='char', ngram=3, lowercase=False)
var_imp.head(10)


# In[ ]:


var_imp.tail(10)

