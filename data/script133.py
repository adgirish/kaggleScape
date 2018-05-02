
# coding: utf-8

# This notebook shows how to build a linear model on features from apps, app labels, phone brands and device models. It uses LogisticRegression classifier from sklearn. 
# 
# It also shows an efficient way of constructing bag-of-apps and bag-of-labels features without concatenating a bunch of strings.

# In[ ]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss


# ## Load data

# In[ ]:


datadir = '../input'
gatrain = pd.read_csv(os.path.join(datadir,'gender_age_train.csv'),
                      index_col='device_id')
gatest = pd.read_csv(os.path.join(datadir,'gender_age_test.csv'),
                     index_col = 'device_id')
phone = pd.read_csv(os.path.join(datadir,'phone_brand_device_model.csv'))
# Get rid of duplicate device ids in phone
phone = phone.drop_duplicates('device_id',keep='first').set_index('device_id')
events = pd.read_csv(os.path.join(datadir,'events.csv'),
                     parse_dates=['timestamp'], index_col='event_id')
appevents = pd.read_csv(os.path.join(datadir,'app_events.csv'), 
                        usecols=['event_id','app_id','is_active'],
                        dtype={'is_active':bool})
applabels = pd.read_csv(os.path.join(datadir,'app_labels.csv'))


# ## Feature engineering
# 
# The features I'm going to use include:
# 
# * phone brand
# * device model
# * installed apps
# * app labels
# 
# I'm going to one-hot encode everything and sparse matrices will help deal with a very large number of features.
# 
# ### Phone brand
# 
# As preparation I create two columns that show which train or test set row a particular device_id belongs to.

# In[ ]:


gatrain['trainrow'] = np.arange(gatrain.shape[0])
gatest['testrow'] = np.arange(gatest.shape[0])


# A sparse matrix of features can be constructed in various ways. I use this constructor:
# 
#     csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
#     where ``data``, ``row_ind`` and ``col_ind`` satisfy the
#     relationship ``a[row_ind[k], col_ind[k]] = data[k]``
#     
# It lets me specify which values to put into which places in a sparse matrix. For phone brand data the `data` array will be all ones, `row_ind` will be the row number of a device and `col_ind` will be the number of brand.

# In[ ]:


brandencoder = LabelEncoder().fit(phone.phone_brand)
phone['brand'] = brandencoder.transform(phone['phone_brand'])
gatrain['brand'] = phone['brand']
gatest['brand'] = phone['brand']
Xtr_brand = csr_matrix((np.ones(gatrain.shape[0]), 
                       (gatrain.trainrow, gatrain.brand)))
Xte_brand = csr_matrix((np.ones(gatest.shape[0]), 
                       (gatest.testrow, gatest.brand)))
print('Brand features: train shape {}, test shape {}'.format(Xtr_brand.shape, Xte_brand.shape))


# ### Device model

# In[ ]:


m = phone.phone_brand.str.cat(phone.device_model)
modelencoder = LabelEncoder().fit(m)
phone['model'] = modelencoder.transform(m)
gatrain['model'] = phone['model']
gatest['model'] = phone['model']
Xtr_model = csr_matrix((np.ones(gatrain.shape[0]), 
                       (gatrain.trainrow, gatrain.model)))
Xte_model = csr_matrix((np.ones(gatest.shape[0]), 
                       (gatest.testrow, gatest.model)))
print('Model features: train shape {}, test shape {}'.format(Xtr_model.shape, Xte_model.shape))


# ### Installed apps features
# 
# For each device I want to mark which apps it has installed. So I'll have as many feature columns as there are distinct apps.
# 
# Apps are linked to devices through events. So I do the following:
# 
# - merge `device_id` column from `events` table to `app_events`
# - group the resulting dataframe by `device_id` and `app` and aggregate
# - merge in `trainrow` and `testrow` columns to know at which row to put each device in the features matrix

# In[ ]:


appencoder = LabelEncoder().fit(appevents.app_id)
appevents['app'] = appencoder.transform(appevents.app_id)
napps = len(appencoder.classes_)
deviceapps = (appevents.merge(events[['device_id']], how='left',left_on='event_id',right_index=True)
                       .groupby(['device_id','app'])['app'].agg(['size'])
                       .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                       .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                       .reset_index())
deviceapps.head()


# Now I can build a feature matrix where the `data` is all ones, `row_ind` comes from `trainrow` or `testrow` and `col_ind` is the label-encoded `app_id`.

# In[ ]:


d = deviceapps.dropna(subset=['trainrow'])
Xtr_app = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.app)), 
                      shape=(gatrain.shape[0],napps))
d = deviceapps.dropna(subset=['testrow'])
Xte_app = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.app)), 
                      shape=(gatest.shape[0],napps))
print('Apps data: train shape {}, test shape {}'.format(Xtr_app.shape, Xte_app.shape))


# ### App labels features
# 
# These are constructed in a way similar to apps features by merging `app_labels` with the `deviceapps` dataframe we created above.

# In[ ]:


applabels = applabels.loc[applabels.app_id.isin(appevents.app_id.unique())]
applabels['app'] = appencoder.transform(applabels.app_id)
labelencoder = LabelEncoder().fit(applabels.label_id)
applabels['label'] = labelencoder.transform(applabels.label_id)
nlabels = len(labelencoder.classes_)


# In[ ]:


devicelabels = (deviceapps[['device_id','app']]
                .merge(applabels[['app','label']])
                .groupby(['device_id','label'])['app'].agg(['size'])
                .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)
                .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)
                .reset_index())
devicelabels.head()


# In[ ]:


d = devicelabels.dropna(subset=['trainrow'])
Xtr_label = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.label)), 
                      shape=(gatrain.shape[0],nlabels))
d = devicelabels.dropna(subset=['testrow'])
Xte_label = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.label)), 
                      shape=(gatest.shape[0],nlabels))
print('Labels data: train shape {}, test shape {}'.format(Xtr_label.shape, Xte_label.shape))


# ### Concatenate all features

# In[ ]:


Xtrain = hstack((Xtr_brand, Xtr_model, Xtr_app, Xtr_label), format='csr')
Xtest =  hstack((Xte_brand, Xte_model, Xte_app, Xte_label), format='csr')
print('All features: train shape {}, test shape {}'.format(Xtrain.shape, Xtest.shape))


# ## Cross-validation

# In[ ]:


targetencoder = LabelEncoder().fit(gatrain.group)
y = targetencoder.transform(gatrain.group)
nclasses = len(targetencoder.classes_)


# In[ ]:


def score(clf, random_state = 0):
    kf = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=random_state)
    pred = np.zeros((y.shape[0],nclasses))
    for itrain, itest in kf:
        Xtr, Xte = Xtrain[itrain, :], Xtrain[itest, :]
        ytr, yte = y[itrain], y[itest]
        clf.fit(Xtr, ytr)
        pred[itest,:] = clf.predict_proba(Xte)
        # Downsize to one fold only for kernels
        return log_loss(yte, pred[itest, :])
        print("{:.5f}".format(log_loss(yte, pred[itest,:])), end=' ')
    print('')
    return log_loss(y, pred)


# In order to make a good logistic regression model we need to choose a value for regularization constant C. Smaller values of C mean stronger regularization and its default value is 1.0. We probably have a lot of mostly useless columns (rare brands, models or apps), so we'd better look at stronger regularization than default.

# In[ ]:


Cs = np.logspace(-3,0,4)
res = []
for C in Cs:
    res.append(score(LogisticRegression(C = C)))
plt.semilogx(Cs, res,'-o');


# Judging by the plot the best value for C is somewhere between 0.01 and 0.1.

# In[ ]:


score(LogisticRegression(C=0.02))


# By default [LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression) classifier solves a multiclass classification problem in a one versus rest fashion. But it's also possible to fit a multinomial model that optimizes the multiclass logloss - exactly the metric we're evaluated on. Let's see if doing that improves our results:

# In[ ]:


score(LogisticRegression(C=0.02, multi_class='multinomial',solver='lbfgs'))


# Yes, it does!

# ## Predict on test data

# In[ ]:


clf = LogisticRegression(C=0.02, multi_class='multinomial',solver='lbfgs')
clf.fit(Xtrain, y)
pred = pd.DataFrame(clf.predict_proba(Xtest), index = gatest.index, columns=targetencoder.classes_)
pred.head()


# In[ ]:


pred.to_csv('logreg_subm.csv',index=True)


# ## What to try next
# 
# - use some aggregates for apps and labels features. Maybe using them instead of simple indicators shown here will improve the score. For example:
#     - calculate the proportion of events where an app appears on each device
#     - calculate the mean of `is_active` field for each app on each device
#     - create some TFIDF-like weighting for apps or labels
# - add features based on event locations and times
# - add feature interactions
# - fit a nonlinear model (neural networks seem to work well here)
# - blend in the results of the [previous script](https://www.kaggle.com/dvasyukova/talkingdata-mobile-user-demographics/brand-and-model-based-benchmarks) for devices that have no events data
# 
# Share your ideas in the comments, fork and improve =).
