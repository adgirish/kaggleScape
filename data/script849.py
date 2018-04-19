
# coding: utf-8

# # ELI5: which features are important for price prediction
# 
# Or, explain like I'm 5, how does a linear ridge predict prices?
# 
# [ElL5](http://eli5.readthedocs.io/) is a library that can help us with that, let's see it in action. It has support for many models, including XGBoost and LightGBM, but we'll be using it to analyze the Ridge model from scikit-learn.  Overall modelling strategy is inspired by this beatiful kernel [Ridge Script](https://www.kaggle.com/apapiu/ridge-script) by Alexandru Papiu.

# In[1]:


import eli5
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import mean_squared_log_error


# Load the data:

# In[2]:


train = pd.read_table('../input/train.tsv')


# Apply mostly normal preprocessing, converting ``shipping`` and ``item_condition_id`` to strings to handle them with a count vectorizer too.

# In[3]:


y_train = np.log1p(train['price'])
train['category_name'] = train['category_name'].fillna('Other').astype(str)
train['brand_name'] = train['brand_name'].fillna('missing').astype(str)
train['shipping'] = train['shipping'].astype(str)
train['item_condition_id'] = train['item_condition_id'].astype(str)
train['item_description'] = train['item_description'].fillna('None')


# In[4]:


train.head()


# Do feature extraction in a rather silly way, to put it mildly, with the goal of having a single scikit-learn vectorizer handle all preprocessing. eli5 is nicer to use with default sklearn objects (although you can pass arbitrary feature names), so we go this way. And siliness with ``preprocessor`` is due to scikit-learn not yet playing nicely with pandas dataframe objects, but it's bearable.

# In[10]:


get_ipython().run_cell_magic('time', '', "\n# we need a custom pre-processor to extract correct field,\n# but want to also use default scikit-learn preprocessing (e.g. lowercasing)\ndefault_preprocessor = CountVectorizer().build_preprocessor()\ndef build_preprocessor(field):\n    field_idx = list(train.columns).index(field)\n    return lambda x: default_preprocessor(x[field_idx])\n    \nvectorizer = FeatureUnion([\n    ('name', CountVectorizer(\n        ngram_range=(1, 2),\n        max_features=50000,\n        preprocessor=build_preprocessor('name'))),\n    ('category_name', CountVectorizer(\n        token_pattern='.+',\n        preprocessor=build_preprocessor('category_name'))),\n    ('brand_name', CountVectorizer(\n        token_pattern='.+',\n        preprocessor=build_preprocessor('brand_name'))),\n    ('shipping', CountVectorizer(\n        token_pattern='\\d+',\n        preprocessor=build_preprocessor('shipping'))),\n    ('item_condition_id', CountVectorizer(\n        token_pattern='\\d+',\n        preprocessor=build_preprocessor('item_condition_id'))),\n    ('item_description', TfidfVectorizer(\n        ngram_range=(1, 3),\n        max_features=100000,\n        preprocessor=build_preprocessor('item_description'))),\n])\nX_train = vectorizer.fit_transform(train.values)\nX_train")


# Now let's train the Ridge model and check it's performance on one fold. Nothing interesting going on here yet.

# In[11]:


get_ipython().run_cell_magic('time', '', "\ndef get_rmsle(y_true, y_pred):\n    return np.sqrt(mean_squared_log_error(np.expm1(y_true), np.expm1(y_pred)))\n\ncv = KFold(n_splits=10, shuffle=True, random_state=42)\nfor train_ids, valid_ids in cv.split(X_train):\n    model = Ridge(\n        solver='auto',\n        fit_intercept=True,\n        alpha=0.5,\n        max_iter=100,\n        normalize=False,\n        tol=0.05)\n    model.fit(X_train[train_ids], y_train[train_ids])\n    y_pred_valid = model.predict(X_train[valid_ids])\n    rmsle = get_rmsle(y_pred_valid, y_train[valid_ids])\n    print(f'valid rmsle: {rmsle:.5f}')\n    break")


# Now that the model is fitted, we can check it's most important features with the power of ELI5. Here we pass the ``vectorizer`` - if we didn't have it, we would have to pass ``feature_names`` instead.

# In[12]:


eli5.show_weights(model, vec=vectorizer)


# The part before the double underscore is the vectorizer name, and the feature name goes after that. Let's show more features and get rid of the bias term:

# In[ ]:


eli5.show_weights(model, vec=vectorizer, top=100, feature_filter=lambda x: x != '<BIAS>')


# Another handy feature is analyzing individual predictions. Let's check some predictions from the validation set. You see a summary of various vectorizer's contribution at the top, and then below you can see features highlighed in text.

# In[13]:


eli5.show_prediction(model, doc=train.values[100], vec=vectorizer)


# In[ ]:


eli5.show_prediction(model, doc=train.values[1], vec=vectorizer)


# In[ ]:


eli5.show_prediction(model, doc=train.values[2], vec=vectorizer)


# What can we do with this?
# First, you can examine features to see if they are what you expect - maybe you are missing some important information due to bad tokenization or have a lot of noise features due to insufficient regularization.
# You can also check most erroneous predictions and try to understand why does the model fail on them.
