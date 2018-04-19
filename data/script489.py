
# coding: utf-8

# #EDA of important features#

# An XGBoost fitted on approximately 700 columns achieves already a LB-Score of ~0.25 without any feature engineering. What can we learn from the 20 most essential numeric features suggested by that model?
# 
# The analysis in this notebook reveals that the positive and negative samples differ significantly  both in their missing-value counts as well as in their missing-value correlation structure. On the other hand, the behavior on non-missing samples is surprisingly similar for positive and negative samples.

# #Data preparation#

# In the first step, we import standard libraries and fix the most essential features as suggested by an XGB oracle.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


feature_names = ['L3_S38_F3960', 'L3_S33_F3865', 'L3_S38_F3956', 'L3_S33_F3857',
       'L3_S29_F3321', 'L1_S24_F1846', 'L3_S32_F3850', 'L3_S29_F3354',
       'L3_S29_F3324', 'L3_S35_F3889', 'L0_S1_F28', 'L1_S24_F1844',
       'L3_S29_F3376', 'L0_S0_F22', 'L3_S33_F3859', 'L3_S38_F3952', 
       'L3_S30_F3754', 'L2_S26_F3113', 'L3_S30_F3759', 'L0_S5_F114']


# We determine the indices of the most important features. After that the training data is loaded

# In[ ]:


numeric_cols = pd.read_csv("../input/train_numeric.csv", nrows = 1).columns.values
imp_idxs = [np.argwhere(feature_name == numeric_cols)[0][0] for feature_name in feature_names]
train = pd.read_csv("../input/train_numeric.csv", 
                index_col = 0, header = 0, usecols = [0, len(numeric_cols) - 1] + imp_idxs)
train = train[feature_names + ['Response']]


# The data is split into positive and negative samples.

# In[ ]:


X_neg, X_pos = train[train['Response'] == 0].iloc[:, :-1], train[train['Response']==1].iloc[:, :-1]


# #Univariate characteristics#

# In order to understand better the predictive power of single features, we compare the univariate distributions of the most important features. First, we divide the train data into batches column-wise to prepare the data for plotting.

# In[ ]:


BATCH_SIZE = 5
train_batch =[pd.melt(train[train.columns[batch: batch + BATCH_SIZE].append(np.array(['Response']))], 
                      id_vars = 'Response', value_vars = feature_names[batch: batch + BATCH_SIZE])
              for batch in list(range(0, train.shape[1] - 1, BATCH_SIZE))]


# After this split, we can now draw violin plots. Due to memory reasons, we have to split the presentation into several cells. For many of the distributions there is no clear difference between the positive and negative samples.

# In[ ]:


FIGSIZE = (12,16)
_, axs = plt.subplots(len(train_batch), figsize = FIGSIZE)
plt.suptitle('Univariate distributions')
for data, ax in zip(train_batch, axs):
    sns.violinplot(x = 'variable',  y = 'value', hue = 'Response', data = data, ax = ax, split =True)


# The data set is characterized by a large proportion of missing values. When comparing the missing values between positive and negative samples, we see striking differences.

# In[ ]:


non_missing = pd.DataFrame(pd.concat([(X_neg.count()/X_neg.shape[0]).to_frame('negative samples'),
                                      (X_pos.count()/X_pos.shape[0]).to_frame('positive samples'),  
                                      ], 
                       axis = 1))
non_missing_sort = non_missing.sort_values(['negative samples'])
non_missing_sort.plot.barh(title = 'Proportion of non-missing values', figsize = FIGSIZE)
plt.gca().invert_yaxis()


# #Correlation structure#

# In the previous section we have seen differences between negative and positive samples for univariate characteristics. We go down the rabbit hole a little further and analyze covariances for the negative and positive samples separately. 

# In[ ]:


FIGSIZE = (13,4)
_, (ax1, ax2) = plt.subplots(1,2, figsize = FIGSIZE)
MIN_PERIODS = 100

triang_mask = np.zeros((X_pos.shape[1], X_pos.shape[1]))
triang_mask[np.triu_indices_from(triang_mask)] = True

ax1.set_title('Negative Class')
sns.heatmap(X_neg.corr(min_periods = MIN_PERIODS), mask = triang_mask, square=True,  ax = ax1)

ax2.set_title('Positive Class')
sns.heatmap(X_pos.corr(min_periods = MIN_PERIODS), mask = triang_mask, square=True,  ax = ax2)


# The difference between the two matrices is sparse except for three specific feature combinations.

# In[ ]:


sns.heatmap(X_pos.corr(min_periods = MIN_PERIODS) -X_neg.corr(min_periods = MIN_PERIODS), 
             mask = triang_mask, square=True)


# Finally, as in the univariate case, we analyze correlations between missing values in  different features.

# In[ ]:


nan_pos, nan_neg = np.isnan(X_pos), np.isnan(X_neg)

triang_mask = np.zeros((X_pos.shape[1], X_pos.shape[1]))
triang_mask[np.triu_indices_from(triang_mask)] = True

FIGSIZE = (13,4)
_, (ax1, ax2) = plt.subplots(1,2, figsize = FIGSIZE)
MIN_PERIODS = 100

ax1.set_title('Negative Class')
sns.heatmap(nan_neg.corr(),   square=True, mask = triang_mask, ax = ax1)

ax2.set_title('Positive Class')
sns.heatmap(nan_pos.corr(), square=True, mask = triang_mask,  ax = ax2)


# For the difference of the missing-value correlation matrices, a striking pattern emerges. A further and more systematic analysis of such missing-value patterns has the potential to beget powerful features.

# In[ ]:


sns.heatmap(nan_neg.corr() - nan_pos.corr(), mask = triang_mask, square=True)

