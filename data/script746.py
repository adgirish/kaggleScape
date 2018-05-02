
# coding: utf-8

# # Tips for Using Scikit-Learn for Evaluation

# I ran into a few things that tripped me up when working with the evaluation metric in Scikit-Learn.
# 
# There are a few examples out there of how to calculate Gini, but I found that basing it off of the `roc_auc_score` within Scikit-Learn seemed cleaner than building on by hand or using some of the other ones that I've seen out there. 
# 
# One of the first things you'd like to see from an algorithm and a dataset is evidence that it's learning. The function `learning_curve` is useful for that and you can pass it a custom scoring function. 
# 
# However, I ran into some unexpected results at first.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.linear_model import LogisticRegression


# In[ ]:


train_data = pd.read_csv('../input/train.csv', na_values='-1')
train_data.fillna(value=train_data.median(), inplace=True)


# **NB**: This kernel doesn't showcase any feature engineering, just some simple interpolation to ensure that a predictive model will run.

# In[ ]:


X_train, y_train = train_data.iloc[:, 2:], train_data.iloc[:, 1]

del train_data


# Given the `roc_auc_score` function built into Scikit-Learn, we can easily compute the Gini coefficient.

# In[ ]:


def gini_normalized(y_actual, y_pred):
    """Simple normalized Gini based on Scikit-Learn's roc_auc_score"""
    gini = lambda a, p: 2 * roc_auc_score(a, p) - 1
    return gini(y_actual, y_pred) / gini(y_actual, y_actual)


# In[ ]:


lr = LogisticRegression()

train_sizes, train_scores, test_scores = learning_curve(
    estimator=lr,
    X=X_train,
    y=y_train,
    train_sizes=np.linspace(0.05, 1, 6),
    cv=5,
    scoring=make_scorer(gini_normalized)
)


# In[ ]:


train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(train_sizes, train_mean, 
         color='blue', marker='o', 
         markersize=5, 
         label='training gini')
plt.fill_between(train_sizes, 
                 train_mean + train_std,
                 train_mean - train_std, 
                 alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, 
         color='green', linestyle='--', 
         marker='s', markersize=5, 
         label='validation gini')
plt.fill_between(train_sizes, 
                 test_mean + test_std,
                 test_mean - test_std, 
                 alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Normalized Gini')
plt.legend(loc='lower right')
plt.ylim([-0.25, 0.25])
plt.show()


# The graph above looks really odd, as both the training and validation normalized Gini values are near zero for all of the dataset sizes. Without digging into it, I would have thought that there was no learning going on at all.

# It turned out that the culprit was how Scikit-Learn scored the hold-out set. By default, it predicts using the `predict` method on the model rather than the `predict_proba` method. The output from `predict` on a classification problem is the class labels while the output from `predict_proba` is the probabilities for the class labels. For computing the Gini value on the results, the output of `predict_proba` is more appropriate. 

# To ensure this happens, we modify the `gini_normalized` function to allow that.

# In[ ]:


def gini_normalized(y_actual, y_pred):
    """Simple normalized Gini based on Scikit-Learn's roc_auc_score"""
    
    # If the predictions y_pred are binary class probabilities
    if y_pred.ndim == 2:
        if y_pred.shape[1] == 2:
            y_pred = y_pred[:, 1]
    gini = lambda a, p: 2 * roc_auc_score(a, p) - 1
    return gini(y_actual, y_pred) / gini(y_actual, y_actual)


# In[ ]:


lr = LogisticRegression()

train_sizes, train_scores, test_scores = learning_curve(
    estimator=lr,
    X=X_train,
    y=y_train,
    train_sizes=np.linspace(0.05, 1, 6),
    cv=5,
    scoring=make_scorer(gini_normalized, needs_proba=True)
)


# In[ ]:


train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(train_sizes, train_mean, 
         color='blue', marker='o', 
         markersize=5, 
         label='training gini')
plt.fill_between(train_sizes, 
                 train_mean + train_std,
                 train_mean - train_std, 
                 alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, 
         color='green', linestyle='--', 
         marker='s', markersize=5, 
         label='validation gini')
plt.fill_between(train_sizes, 
                 test_mean + test_std,
                 test_mean - test_std, 
                 alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Normalized Gini')
plt.legend(loc='lower right')
plt.ylim([0.2, 0.3])
plt.show()


# Once we've adjusted the `gini_normalized` and plotted the results of model fitting at different samples sizes of the training dataset, we see that indeed learning is happening correctly. 
# 
# We also see that our evaluation metric seems correct.

# It might seem simple to veteran competitors, but I think it serves as another lesson learned to ensure that you understand the evaluation metric and that you can reproduce some of the baseline results.
