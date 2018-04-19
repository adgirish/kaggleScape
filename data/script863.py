
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train= pd.read_csv('../input/creditcard.csv')
train.head()


# In[ ]:


import seaborn as sns
sns.heatmap(train.corr())


# In[ ]:


sns.countplot(train['Class'])


# In[ ]:


from sklearn.preprocessing import StandardScaler
train['Amount_n']= StandardScaler().fit_transform(train['Amount'].reshape(-1,1))


# In[ ]:


train.head()


# In[ ]:


train['Time_H']= train['Time']/3600


# In[ ]:


sns.distplot(train['Time_H'])


# In[ ]:


sns.countplot(train['Class'])


# In[ ]:


sns.jointplot(train['Time_H'], train['Class'])


# In[ ]:


train= train.drop(['Time','Time_H','Amount'], axis=1)
train.head()


# In[ ]:


X= train.ix[:, train.columns != 'Class']
y= train.ix[:, train.columns == 'Class']   


# In[ ]:


fraud_count = len(train[train.Class == 1])
fraud_indices = train[train.Class == 1].index
normal_indices = train[train.Class == 0].index

r_normal_indices = np.random.choice(normal_indices, fraud_count, replace = False) # random 

undersample_indices = np.concatenate([fraud_indices,r_normal_indices])
undersample_train = train.iloc[undersample_indices,:]

X_undersample = undersample_train.ix[:, undersample_train.columns != 'Class']
y_undersample = undersample_train.ix[:, undersample_train.columns == 'Class']


# In[ ]:


from sklearn.model_selection import train_test_split
X_tr, X_test, y_tr, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)
X_tr_u, X_test_u, y_tr_u, y_test_u = train_test_split(X_undersample,y_undersample,test_size = 0.3,random_state = 0)
                                                                                                   


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix,auc,roc_auc_score,recall_score,classification_report,precision_recall_curve, roc_curve
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


logreg = LogisticRegression(C = .01, penalty = 'l1')
logreg.fit(X_tr_u,y_tr_u.values.ravel())
y_pred_u= logreg.predict(X_test_u)
y_pred_u_proba=logreg.predict_proba(X_test_u)
print('cm:', confusion_matrix(y_test_u,y_pred_u))
print('cr:', classification_report(y_test_u,y_pred_u))
print('recall_score:', recall_score(y_test_u,y_pred_u))
print('roc_auc_score:',roc_auc_score(y_test_u,y_pred_u))


# In[ ]:


y_predprob_u = logreg.predict_proba(X_test_u)[:, 1]  # default threshold 0.5

plt.hist(y_predprob_u, bins=8)
plt.xlabel('predicted probability of fraud')
plt.ylabel('frequency')
plt.title('Histogram of predicted probabilities') 


# In[ ]:


logreg.fit(X_tr_u,y_tr_u.values.ravel())
y_pred_u_proba = logreg.predict_proba(X_test_u)
thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in thresholds:
    y_test_pred_prob = y_pred_u_proba[:,1] > i
    precision, recall, thresholds= precision_recall_curve(y_test_u,y_test_pred_prob)
    plt.plot(recall, precision,label='Threshold: %s'%i)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.title('Precision-Recall curve')    
    plt.legend(loc='center left')


# In[ ]:


#In order to increase recall(sensitivity), we need to decrease the threshold of the classifier
fpr, tpr, thresholds = roc_curve(y_test_u,y_predprob_u)
roc_auc = auc(fpr,tpr)
plt.plot(fpr,tpr)
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC curve for fraud classifier')
plt.grid(True)


# In[ ]:


roc_auc_score(y_test_u, y_predprob_u)


# In[ ]:


#Decrease the threshold(0.3)for predicting frauds in order to increase the sensitivity of the classifier
from sklearn.preprocessing import binarize
y_pred_class_u_highrecall = binarize([y_predprob_u], 0.3)[0]
print('cm:', confusion_matrix(y_test_u,y_pred_class_u_highrecall))
print('cr:', classification_report(y_test_u,y_pred_class_u_highrecall))
print('recall_score:', recall_score(y_test_u,y_pred_class_u_highrecall))
print('roc_auc_score:',roc_auc_score(y_test_u,y_pred_class_u_highrecall))


# In[ ]:


logreg.fit(X_tr_u,y_tr_u.values.ravel())
y_pred= logreg.predict_proba(X_test)[:,1]    #predicted probabilities for class 1                                          
y_pred_class_highrecall = binarize([y_pred], 0.3)[0]
print('cm:', confusion_matrix(y_test,y_pred_class_highrecall))
print('cr:', classification_report(y_test,y_pred_class_highrecall))
print('recall_score:', recall_score(y_test,y_pred_class_highrecall))
print('roc_auc_score:',roc_auc_score(y_test,y_pred_class_highrecall))


# In[ ]:


logreg.fit(X_tr_u,y_tr_u.values.ravel())
y_pr= logreg.predict_proba(X)[:,1]    #predicted probabilities for class 1                                          
y_pr_class_highrecall = binarize([y_pr], 0.3)[0]
print('cm:', confusion_matrix(y,y_pr_class_highrecall))
print('cr:', classification_report(y,y_pr_class_highrecall))
print('recall_score:', recall_score(y,y_pr_class_highrecall))
print('roc_auc_score:',roc_auc_score(y,y_pr_class_highrecall))

