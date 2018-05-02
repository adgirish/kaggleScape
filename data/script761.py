
# coding: utf-8

# # Montecarlo Model Selection 
# 
# In this notebook we apply a montecarlo based feature selection method to identify a reduced number of features that can be used as a good predictor for this credit card fraud dataset. Using a reduced number of features is of crucial importance when overfitting needs to be prevented. This models has been tested in some other datasets with similar results.  
# 
# We expand the set of explanatory features by computing the products of all features against the rest. When the product of two features has better sorting capabilities than both features individually we include the product in our set of candidate features, a minimum threshold is applied. 
# 
# Additionally the original set of features and the new set resulting from multiplying pairs of feature are transformed by means a logistic distribution. We have observed that this transformation increases predictive capabilities when compared to the ubiquitous normal transformation. 
# 
# In order to measure if a model has a good predictive power we define a modified Jaccard distance. As follows:
#                                                   
# $$Modified\,Jaccard\,Distance = 1 − \dfrac{\sum\limits_{i}{\min{ (target_{i},\,model\, probability_{i})}}}{\sum\limits_{i}{\max{(target_{i},\,model\, probability_{i})}}}$$                               
# The lower the distance the best the model predicts the target.
# 
# In each Montecarlo iteration a reduced set of features, say 5 to 8, is randomly selected, then we compute the Logistic Regression model that best predicts fraud with this features and, finally we compute the modified Jaccard distance from the prediction to the target. The process is repeated for a large number of iterations. Resulting models are sorted by distance. 
# 
# We have tested modified Jaccard distance metric against most common metrics such as ROC, AUC, recall… and found that models with the low values of this modified Jaccard distance have a better balanced results in the rest of metrics. 
# 
# Final model selection is done by choosing the model with me minimum modified Jaccard distance, or any other among those with minimum distance that best fits the test subsample. 

# ### Libraries
# 
# We will use [scikit-learn](http://scikit-learn.org/stable/) for machine learning in general. All additional functions needed can be found as a dataset named [MonteCarloModelSelection_Functions.py](https://www.kaggle.com/forzzeeteam/monte-carlo-model-selection/data). 

# In[ ]:


import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split

import os
print(os.listdir("../input"))

import sys 

sys.path.append ('../input/montecarlomodelselection-functions/')
from MonteCarloModelSelection_Functions import *      


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('autosave', '0')


# ### Loading Dataset 

# In[ ]:


# Loading dataset creditcard
filename = '../input/creditcardfraud/creditcard.csv'   

with open(filename, 'r') as f:
    reader=csv.reader(f, delimiter=',') 
    labels=next(reader)

    raw_data=[]
    for row in reader:
        raw_data.append(row)

data = np.array(raw_data)
data = data.astype(np.float)


# The Amount column is normalized. 

# In[ ]:


# Setting target and data
target = data[:,-1]
dataAmount   = data[:,29]
data   = data[:,1:29]

# Normalising Amount column 
dataAmountNormalize = np.array((dataAmount-np.mean(dataAmount))/np.std(dataAmount))
data = np.c_[ data,dataAmountNormalize]


# In[ ]:


# Output Path
path = './output/'


# ### Transformation 
# All features are tranformed using Univariate Logistic Regression. Normal transformation can be applied too, however we observed better results for the logit transformation. 

# In[ ]:


# Calculating transformed dataset by means of logit or normal method
transformation = 'logit' 
transformed_dataset = Transformation(data, target, transformation)


# Calculate some metrics, initially we will pay special attention to the sorting capabilities of the different features by using different metrics.
# 
# 

# In[ ]:


# Calculating all metric
metric ='all'
global_pi = Calculate_Metrics(transformed_dataset, target, metric, path, transformation)


# A new dataset resulting from combinations of products of features can be found. Following we look for products of features that improve the sorting capabilities of the features. First we select products that result in a “modified Jaccard distance” lower than that of the features independently and at the same time the metric is lower than 0.6.

# In[ ]:


# Calculating new datasets with combinations of products of features using distance metric
threshold = 0.6
transformation = 'logit'
metric = 'all'
metric_prod = 'distance'
new_dataset, new_dataset_df = Products_Analysis(data, transformed_dataset, target, global_pi, metric, metric_prod, transformation, path, threshold)


# Since “distance” did not produce predictive products of features the try “roc”.

# In[ ]:


# Calculating new datasets with combinations of products of features using roc metric
threshold = 0.6
transformation = 'logit'
metric = 'all'
metric_prod = 'roc'
new_dataset, new_dataset_df = Products_Analysis(data, transformed_dataset, target, global_pi, metric, metric_prod, transformation, path, threshold)


# In[ ]:


new_dataset_df.tail(20)


# 18 new combinations have been created: 0 ratio and 16 ratio, 1 ratio and 6 ratio, etc. 

# ### Resampling and Setting up the Training and Testing Sets
# The original dataset has 492 fraud and 284.315 no fraud observations. We split the dataset into train and test as in the following table. 
# 
# |         | DATASET           | TRAIN  | TEST  | 
# | ------------- |:-------------:| -----:|-----:|
# | No Fraud      | 284315 | 199019 |85296 | 
# | Fraud      | 492      |    345| 147 |
# | Total | 284807      |   199364 | 85443 |
# - Dataset

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(new_dataset, target, test_size = 0.3, random_state = 0)


# We split the dataset in order to work with a balanced dataset. Equal number of Fraud/No Fraud observations.
#  
# 
# |         | DATASET           | TRAIN UNDERSAMPLED  | TEST UNDERSAMPLED | 
# | ------------- |:-------------:| -----:|-----:|
# | No Fraud      | 284315 | 688 | 296 | 
# | Fraud      | 492      |   343| 149 |
# | Total | 284807      |   345 | 147 |
# 
# - Resampled dataset

# In[ ]:


# Resampling dataset  
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
np.random.seed(10)
number_records_fraud = target.sum().astype(int)
normal_indices = (target==0).nonzero()[0]
fraud_indices = (target==1).nonzero()[0]
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace = False)
random_normal_indices = np.array(random_normal_indices)
under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])
under_sample_data = new_dataset[under_sample_indices,:]
X_undersample = under_sample_data
y_undersample = target[under_sample_indices]
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample, y_undersample, test_size = 0.3, random_state = 0)


# ### Montecarlo selection of features 
# 
# We performed **Montecarlo simulation** with 10.000 iterations to randomly select 5 features. In each iteration a new model is calibrated and its permormance metrics are calculated.

# In[ ]:


metric = 'Distance'
number_iterations = 10000
number_ini_ratio = 5
number_final_ratio = 5
results= Multivariate_Best_Model(number_iterations, X_train_undersample, y_train_undersample, X_test_undersample, y_test_undersample, metric, path, number_ini_ratio, number_final_ratio)       


# Models resulting from the montecarlo process are stored in a DataFrame with the following columns:
# - *Models*: set of ratios randomly selected.
# - *Metrics*: 'Roc', 'Accuracy', 'Precision', 'Recall', 'F1', 'Auc' metrics calculated for each set of models.
# - *P_def*: model probability.
# - *Prediction*: model prediction.
# - *Score*: score as the argument of the logit funtion.
# - *Betas*: multipliers of each variable in the logit function. 
# - *Distance*: Modified Jaccard Distance. 
# 
# Then models are sorted by the Modified Jaccard Distance
# 

# In[ ]:


results.head(5)


# ### PLOTS AND RESULTS
# 
# - Resampled dataset

# In[ ]:


models_list = [i-1 for i in results['Models'][0]]
bt = results['Betas'][0]
ind_best = models_list 
X_test_b = X_test_undersample[:,ind_best]
X_test_b_1 = np.array([1]*X_test_b.shape[0])
X_test_b_ = np.c_[X_test_b_1, X_test_b]
xtest_bt = np.ravel(np.dot(X_test_b_,np.transpose(bt)))

[tn_u, fp_u, fn_u, tp_u] = Graph(y_test_undersample, xtest_bt)


# - Dataset

# In[ ]:


models_list = [i-1 for i in results['Models'][0]]
bt = results['Betas'][0]
ind_best = models_list 
X_test_b = X_test[:,ind_best]
X_test_b_1 = np.array([1]*X_test_b.shape[0])
X_test_b_ = np.c_[X_test_b_1, X_test_b]
xtest_bt = np.ravel(np.dot(X_test_b_,np.transpose(bt)))

[tn, fp, fn, tp]  = Graph(y_test, xtest_bt)


# ### Conclusion 
# The parsimony principle tells us to choose the simplest explanation that fits the evidence. In this work we used a Montecarlo method to find a model that can explain the target variable, proving that by selecting the appropriate features a model as simple as a Logistic Regression with 5 variables produces predictions that are as good as those coming from more complex models.
