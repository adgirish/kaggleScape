
# coding: utf-8

# Hi all,
# 
# This is my first Kernel on Kaggle.
# 
# I will try to build a rough model using Gaussian Distribution to detect Anamolous transactions.
# 
# **Reason behind using Gaussian Distribution:-**  <br>
# If I can summarize what Andrew Ng has mentioned in his lecture on Anomaly detection is 
# Supervised Classification technique is not the perfect candidate for highly imbalanced data. In this case it is 
#  0.172% (near to 0)
# 
# If We think from the persepctive of building the model to find out the anomalous data which is not seen very frequently 
# We should go for Anomaly detection technique using Gaussian Distribution.  
# 

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
import pandas as pd
import numpy as np
import random as rnd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from numpy import genfromtxt
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score , average_precision_score
from sklearn.metrics import precision_score, precision_recall_curve
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)


# I will be defining the below two functions which are required to calculate Gaussian Distribution of the normalized variables provided in the dataset (V1, V2 ....V28, Amount ).  <br>
# note- These functions will be invoked for building the model
# 
# 1) Find out mu and Sigma for the dataframe variables passed to this function. <br>
#       ----
# 2) Calculate Probability Distribution for the each row (I will explain why we need Probality for each row as we proceed) <br>
#        ----
#        
# Formula:- 
# if each example x has N dimensiona(features) then below formula is used to calculate the P value <br>
# **P(x) = p(x1,u1,sigma1^2)p(x2,u2,sigma2^2)p(x3,u3,sigma3^2).....p(xn,un,sigma'N'^2)**
#       ---

# In[ ]:


def estimateGaussian(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.cov(dataset.T)
    return mu, sigma

def multivariateGaussian(dataset,mu,sigma):
    p = multivariate_normal(mean=mu, cov=sigma)
    return p.pdf(dataset)


# Below is the most crucial function used to detect how well we are doing with our subset (Cross validation subset) .
# I have decided values for Epsilon for detecting the fradulent transactions from the Subsets.  <br><br>
# **(Tip :- Ideally you should provide range of epsilon values, due to time constraint on running this kernel i have provided few values here for demonstration purpose)**
# 
#  **For now remember Epsilon value is the threshold value below which we will mark transaction as Anomalous.**
#            ----
# 
# Rewriting above sentense again 
# P(x) for X if less than the epsilon value then mark that transaction as anomalous transaction. 
# 
# We need to maintain healthy balance between the Recall and Precision . We may get Recall value above 0.80 and close to 0.90 here but at the expense of reducing our precision which is not advisable.
# 

# In[ ]:


def selectThresholdByCV(probs,gt):
    best_epsilon = 0
    best_f1 = 0
    f = 0
    farray = []
    Recallarray = []
    Precisionarray = []
    epsilons = (0.0000e+00, 1.0527717316e-70, 1.0527717316e-50, 1.0527717316e-24)
    #epsilons = np.asarray(epsilons)
    for epsilon in epsilons:
        predictions = (p_cv < epsilon)
        f = f1_score(train_cv_y, predictions, average = "binary")
        Recall = recall_score(train_cv_y, predictions, average = "binary")
        Precision = precision_score(train_cv_y, predictions, average = "binary")
        farray.append(f)
        Recallarray.append(Recall)
        Precisionarray.append(Precision)
        print ('For below Epsilon')
        print(epsilon)
        print ('F1 score , Recall and Precision are as below')
        print ('Best F1 Score %f' %f)
        print ('Best Recall Score %f' %Recall)
        print ('Best Precision Score %f' %Precision)
        print ('-'*40)
        if f > best_f1:
            best_f1 = f
            best_recall = Recall
            best_precision = Precision
            best_epsilon = epsilon    
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.5, 0.7, 0.3])
    #plt.subplot(3,1,1)
    plt.plot(farray ,"ro")
    plt.plot(farray)
    ax.set_xticks(range(5))
    ax.set_xticklabels(epsilons,rotation = 60 ,fontsize = 'medium' )
    ax.set_ylim((0,0.8))
    ax.set_title('F1 score vs Epsilon value')
    ax.annotate('Best F1 Score', xy=(best_epsilon,best_f1), xytext=(best_epsilon,best_f1))
    plt.xlabel("Epsilon value") 
    plt.ylabel("F1 Score") 
    plt.show()
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.5, 0.9, 0.3])
    #plt.subplot(3,1,2)
    plt.plot(Recallarray ,"ro")
    plt.plot(Recallarray)
    ax.set_xticks(range(5))
    ax.set_xticklabels(epsilons,rotation = 60 ,fontsize = 'medium' )
    ax.set_ylim((0,1.0))
    ax.set_title('Recall vs Epsilon value')
    ax.annotate('Best Recall Score', xy=(best_epsilon,best_recall), xytext=(best_epsilon,best_recall))
    plt.xlabel("Epsilon value") 
    plt.ylabel("Recall Score") 
    plt.show()
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.5, 0.9, 0.3])
    #plt.subplot(3,1,3)
    plt.plot(Precisionarray ,"ro")
    plt.plot(Precisionarray)
    ax.set_xticks(range(5))
    ax.set_xticklabels(epsilons,rotation = 60 ,fontsize = 'medium' )
    ax.set_ylim((0,0.8))
    ax.set_title('Precision vs Epsilon value')
    ax.annotate('Best Precision Score', xy=(best_epsilon,best_precision), xytext=(best_epsilon,best_precision))
    plt.xlabel("Epsilon value") 
    plt.ylabel("Precision Score") 
    plt.show()
    return best_f1, best_epsilon


# Lets Read the dataset 
#         ---

# In[ ]:


train_df = pd.read_csv("../input//creditcard.csv")


# In[ ]:


print(train_df.columns.values)


# **Copied below piece of code for visualization from the kernel shared by the expert to identify which features are not much of help in the algorithm. **

# In[ ]:


v_features = train_df.iloc[:,1:29].columns


# In[ ]:


plt.figure(figsize=(12,8*4))
gs = gridspec.GridSpec(7, 4)
for i, cn in enumerate(train_df[v_features]):
    ax = plt.subplot(gs[i])
    sns.distplot(train_df[cn][train_df.Class == 1], bins=50)
    sns.distplot(train_df[cn][train_df.Class == 0], bins=50)
    ax.set_xlabel('')
    ax.set_title('feature: ' + str(cn))
plt.show()


# We see normal distribution of anomalous transation is matching with normal distribution of Normal transaction for below Features 
# 'V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8'
# Better we remove these features 

# In[ ]:


train_df.drop(['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8'], axis =1, inplace = True)


# I have removed Amount and Time feature since they wont add much value in calculating gaussian distribution.

# In[ ]:


train_df.drop(labels = ["Amount","Time"], axis = 1, inplace = True)


# Split the dataset into 2 part one with Class 1 and other with class 0

# In[ ]:


train_strip_v1 = train_df[train_df["Class"] == 1]
train_strip_v0 = train_df[train_df["Class"] == 0]


# In the Anomalized technique  we distribute this large dataset into 3 parts .
# 
# 1) Normal Transactons: classified as 0 , no anomalized transaction should be present here since it is not a supervised method<br>  How to get this dataset :- 60% of normal transactions should be added here. <br> 
# Find out Epsilon by using  min(Probability) command 
# 
# 2) dataset for Cross validation : from the remaining normal transaction take 50 % (i.e. 20 % as a whole since we have already took the data in the first step)  and add 50% of the Anomalized data with this .
# 
# 3) dataset for testing the algorithm :- this step is similar to what we did for Cross validattion. <br>
#  Test dataset = leftover normal transaction + leftover Anomalized data 
# 

# In[ ]:


Normal_len = len (train_strip_v0)
Anomolous_len = len (train_strip_v1)

start_mid = Anomolous_len // 2
start_midway = start_mid + 1

train_cv_v1  = train_strip_v1 [: start_mid]
train_test_v1 = train_strip_v1 [start_midway:Anomolous_len]

start_mid = (Normal_len * 60) // 100
start_midway = start_mid + 1

cv_mid = (Normal_len * 80) // 100
cv_midway = cv_mid + 1

train_fraud = train_strip_v0 [:start_mid]
train_cv    = train_strip_v0 [start_midway:cv_mid]
train_test  = train_strip_v0 [cv_midway:Normal_len]

train_cv = pd.concat([train_cv,train_cv_v1],axis=0)
train_test = pd.concat([train_test,train_test_v1],axis=0)


print(train_fraud.columns.values)
print(train_cv.columns.values)
print(train_test.columns.values)

train_cv_y = train_cv["Class"]
train_test_y = train_test["Class"]

train_cv.drop(labels = ["Class"], axis = 1, inplace = True)
train_fraud.drop(labels = ["Class"], axis = 1, inplace = True)
train_test.drop(labels = ["Class"], axis = 1, inplace = True)


# Choosing Epsilon Values <br>
#     ---
# I calculated P value for all the rows present in Normal Transaction and found the minimum P value 
# by using below command
#  **min(p)** 
#       ---
# similalrly I found the minimum P Value for rest of the datasets and found this value to be very close to 0 and then i found the max(p) value which is again somewhat far from 0. <br><br>
# Instead of looping between the epsilon values (between min and max of P) , i chose set of epsilon values for demonstration purpose to see how well i can perform to find the fraudulent transactions.

# In[ ]:


mu, sigma = estimateGaussian(train_fraud)
p = multivariateGaussian(train_fraud,mu,sigma)
p_cv = multivariateGaussian(train_cv,mu,sigma)
p_test = multivariateGaussian(train_test,mu,sigma)


# Performance wrt to Epsilon values
#     ----
# Check out how well we are performing with the given set of epsilon values from the function called here.

# In[ ]:


fscore, ep= selectThresholdByCV(p_cv,train_cv_y)


# Epsilon value = 1.0527717316e-70 is selected as threshold to identify Anomalous transactions 
# 
# now time to Predict and calculate  F1 , Recall and Precision score for our Test Dataset

# In[ ]:


predictions = (p_test < ep)
Recall = recall_score(train_test_y, predictions, average = "binary")    
Precision = precision_score(train_test_y, predictions, average = "binary")
F1score = f1_score(train_test_y, predictions, average = "binary")    
print ('F1 score , Recall and Precision for Test dataset')
print ('Best F1 Score %f' %F1score)
print ('Best Recall Score %f' %Recall)
print ('Best Precision Score %f' %Precision)


# Lets Visualize our predictions in below scatter plot 
#          -------

# In[ ]:


fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(train_test['V14'],train_test['V11'],marker="o", color="lightBlue")
ax.set_title('Anomalies(in red) vs Predicted Anomalies(in Green)')
for i, txt in enumerate(train_test['V14'].index):
       if train_test_y.loc[txt] == 1 :
            ax.annotate('*', (train_test['V14'].loc[txt],train_test['V11'].loc[txt]),fontsize=13,color='Red')
       if predictions[i] == True :
            ax.annotate('o', (train_test['V14'].loc[txt],train_test['V11'].loc[txt]),fontsize=15,color='Green')


# From the above result we can see that we are able to maintain the balance between Recall and Precision. 
# 
# Precision of around 60% with Recall of 74% is not bad at all when we have such highly unbalanced data. 
# These numbers are not fixed and can vary . 
#  
#  These numbers were different for Cross validation dataset and we shortlisted our Epsilon value by comparing the results of F1 Score.
# 
# I will show you the result we achieved on Cross validation dataset again.

# In[ ]:


predictions = (p_cv < ep)
Recall = recall_score(train_cv_y, predictions, average = "binary")    
Precision = precision_score(train_cv_y, predictions, average = "binary")
F1score = f1_score(train_cv_y, predictions, average = "binary")    
print ('F1 score , Recall and Precision for Cross Validation dataset')
print ('Best F1 Score %f' %F1score)
print ('Best Recall Score %f' %Recall)
print ('Best Precision Score %f' %Precision)


# 
#  Summary of above Algorithm: 
#  
#  1) Find Epsilon value by considering only Normal Transaction.
#  
#  2) Use this Epsilon value on CV dataset (Normal transaction + Anomalous transaction)
#  
#  3) Come up with set of Epsilon values to see how your algorithm performs and note down the Best F1 score along with
#       Recall and Precision percentage 
#       
#  4) Choose the Epsilon value with highest F1 score 
#  
#  5) Use this Epsilon value to predict the Anomalous transaction on Test Dataset   
#  
# Please comment and let me know to help improve this kernel.
