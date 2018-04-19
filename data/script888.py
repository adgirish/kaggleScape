
# coding: utf-8

# I used this approach in my solution which is the 11th on the Leaderboard. Here I'm infering 4 components in the target's value distribution and I'm showing how did I identify to which of these components does the particular object belong.
# Of course I should mention this forum thread as the main source of the idea: https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/discussion/35382.
# My solution itself is here: https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/discussion/36242

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import cross_val_score,cross_val_predict
train = pd.read_csv('../input/train.csv')
X_train = train.drop(['y'],axis=1)
y_train = train['y']
X_test = pd.read_csv('../input/test.csv')

#
#   Here we drop columns with zero std
#

zero_std = X_train.std()[X_train.std()==0].index
X_train = X_train.drop(zero_std,axis=1)
X_test = X_test.drop(zero_std,axis=1)


# The four components I've mentioned are clearly observable on the distplot

# In[ ]:


sns.distplot(y_train[y_train<170],bins=100,kde=False)


# So we are trying using our features to figure out where (to what component) does the particular object belongs. 
# Here we build a kind of "cluster encoder". It takes categorical feature, computes group means and clusters them to four groups.

# In[ ]:


class cluster_target_encoder:
    def make_encoding(self,df):
        self.encoding = df.groupby('X')['y'].mean()
    def fit(self,X,y):
        df = pd.DataFrame(columns=['X','y'],index=X.index)
        df['X'] = X
        df['y'] = y
        self.make_encoding(df)
        clust = KMeans(4,random_state=0)
        labels = clust.fit_predict(self.encoding[df['X'].values].values.reshape(-1,1))
        df['labels'] = labels
        self.clust_encoding = df.groupby('X')['labels'].median()
    def transform(self,X):
        res = X.map(self.clust_encoding).astype(float)
        return res
    def fit_transform(self,X,y):
        self.fit(X,y)
        return self.transform(X)


# Now as mentioned on forum we use X0 to split the components

# In[ ]:


enc1 = cluster_target_encoder()
labels_train = enc1.fit_transform(X_train['X0'],train['y'])
labels_test = enc1.transform(X_test['X0'])
get_ipython().run_line_magic('pylab', 'inline')
plt.figure(figsize(10,5))
plt.hist(y_train.values[labels_train==0],bins=70,label='cluster 0')
plt.hist(y_train.values[labels_train==1],bins=100,label='cluster 1')
plt.hist(y_train.values[labels_train==2],bins=70,label='cluster 2')
plt.hist(y_train.values[labels_train==3],bins=70,label='cluster 3')
plt.legend()
plt.title('Train targets distribution for all clusters')
plt.xlim((60,170))
plt.show()


# Brilliant, isn't it? But we have a problem. We have some values of X0 in test which we don't have in train, so we have some NaNs in labels_test

# In[ ]:


labels_test[np.isnan(labels_test)].shape


# In fact we can just do nothing. 6 objects is to few to worry about. But instead we can predict these labels using other features. Actually this global four "car clusters" are obvious for most of the algorithms and it's really not a problem to predict them. The problem is to predict what's happening inside the cluster (you can read the Discussions thread for more).
# Let's ensure that we can predict the labels well.

# In[ ]:


cross_val_score(
    X = X_train.select_dtypes(include=[np.number]),
    y = labels_train,
    estimator = xgb.XGBClassifier(),
    cv = 5,
    scoring = 'accuracy')


# As you see the accuracy is super. The last thing we have to do is to predict NaNs in labels_test and we have almost perfect split of these four parts of the mixture

# In[ ]:


est = xgb.XGBClassifier()
est.fit(X_train.select_dtypes(include=[np.number]),labels_train)
labels_test[np.isnan(labels_test)] = est.predict(
    X_test.select_dtypes(include=[np.number]))[np.isnan(labels_test)]
np.isnan(labels_test).any()


# And the last note here. This feature is not the silver bullet. In fact we did not add much information, machine learning algorithms are capable to understand this structure without our help. Take a look how well does xgboost separate these clusters when predicting the y.

# In[ ]:


y_pred = cross_val_predict(
    X = X_train.select_dtypes(include=[np.number]),
    y = y_train,
    estimator = xgb.XGBRegressor(),
    cv = 5)
plt.figure(figsize(10,5))
plt.hist(y_pred[labels_train==0],bins=70,label='cluster 0')
plt.hist(y_pred[labels_train==1],bins=100,label='cluster 1')
plt.hist(y_pred[labels_train==2],bins=70,label='cluster 2')
plt.hist(y_pred[labels_train==3],bins=70,label='cluster 3')
plt.legend()
plt.title('Cross_val_predict distribution for all clusters')
plt.show()


# But this new feature was really very useful for me, you can check my solution to figure out how.
