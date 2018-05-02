
# coding: utf-8

# <....Work in progress...>
# 
# Thank you for opening this script!
# 
# I have made all efforts to document each and every step involved in the prediction process so that this notebook acts as a good starting point for new Kagglers and new machine learning enthusiasts.
# 
# Please **upvote** this kernel so that it reaches the top of the chart and is easily locatable by new users. Your comments on how we can improve this kernel is welcome. Thanks.
# ***
# ## Layout of the document
# The prediction process is divided into two notebooks.
# 
# Part 1 : Covers data statistics, data visualization, and feature selection : https://www.kaggle.com/sharmasanthosh/forest-cover-type-prediction/exploratory-study-on-feature-selection
# 
# This notebook : Covers prediction using various algorithms 
# ***
# ## Data statistics
# * Shape
# * Datatypes
# * Description
# * Skew
# * Class distribution
# 
# ## Data Interaction
# * Correlation
# * Scatter plot
# 
# ## Data Visualization
# * Box and density plots
# * Grouping of one hot encoded attributes
# 
# ## Data Cleaning
# * Remove unnecessary columns
# 
# ## Data Preparation
# * Original
# * Delete rows or impute values in case of missing
# * StandardScaler
# * MinMaxScaler
# * Normalizer
# 
# ## Feature selection
# * ExtraTreesClassifier
# * GradientBoostingClassifier
# * RandomForestClassifier
# * XGBClassifier
# * RFE
# * SelectPercentile
# * PCA
# * PCA + SelectPercentile
# * Feature Engineering
# 
# ## Evaluation, prediction, and analysis
# * LDA (Linear algo)
# * LR (Linear algo)
# * KNN (Non-linear algo)
# * CART (Non-linear algo)
# * Naive Bayes (Non-linear algo)
# * SVC (Non-linear algo)
# * Bagged Decision Trees (Bagging)
# * Random Forest (Bagging)
# * Extra Trees (Bagging)
# * AdaBoost (Boosting)
# * Stochastic Gradient Boosting (Boosting)
# * Voting Classifier (Voting)
# * MLP (Deep Learning)
# * XGBoost
# 
# ***

# ## Load raw data:
# 
# Information about all the attributes can be found here:
# 
# https://www.kaggle.com/c/forest-cover-type-prediction/data
# 
# Learning: 
# We need to predict the 'Cover_Type' based on the other attributes. Hence, this is a classification problem where the target could belong to any of the seven classes.

# In[ ]:


# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings('ignore')

# Read raw data from the file

import pandas #provides data structures to quickly analyze data
#Since this code runs on Kaggle server, train data can be accessed directly in the 'input' folder
dataset = pandas.read_csv("../input/train.csv") 

#Drop the first column 'Id' since it just has serial numbers. Not useful in the prediction process.
dataset = dataset.iloc[:,1:]


# ## Data Cleaning
# * Remove unnecessary columns

# In[ ]:


#Removal list initialize
rem = []

#Add constant columns as they don't help in prediction process
for c in dataset.columns:
    if dataset[c].std() == 0: #standard deviation is zero
        rem.append(c)

#drop the columns        
dataset.drop(rem,axis=1,inplace=True)

print(rem)

#Following columns are dropped


# ## Data Preparation
# * Original
# * Delete rows or impute values in case of missing
# * StandardScaler
# * MinMaxScaler
# * Normalizer

# In[ ]:


#get the number of rows and columns
r, c = dataset.shape

#get the list of columns
cols = dataset.columns
#create an array which has indexes of columns
i_cols = []
for i in range(0,c-1):
    i_cols.append(i)
#array of importance rank of all features  
ranks = []

#Extract only the values
array = dataset.values

#Y is the target column, X has the rest
X_orig = array[:,0:(c-1)]
Y = array[:,(c-1)]

#Validation chunk size
val_size = 0.1

#Use a common seed in all experiments so that same chunk is used for validation
seed = 0

#Split the data into chunks
from sklearn import cross_validation
X_train, X_val, Y_train, Y_val = cross_validation.train_test_split(X_orig, Y, test_size=val_size, random_state=seed)

#Import libraries for data transformations
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer

#All features
X_all = []
#Additionally we will make a list of subsets
X_all_add =[]

#columns to be dropped
rem_cols = []
#indexes of columns to be dropped
i_rem = []

#Add this version of X to the list 
X_all.append(['Orig','All', X_train,X_val,1.0,cols[:c-1],rem_cols,ranks,i_cols,i_rem])

#point where categorical data begins
size=10

import numpy

#Standardized
#Apply transform only for non-categorical data
X_temp = StandardScaler().fit_transform(X_train[:,0:size])
X_val_temp = StandardScaler().fit_transform(X_val[:,0:size])
#Concatenate non-categorical data and categorical
X_con = numpy.concatenate((X_temp,X_train[:,size:]),axis=1)
X_val_con = numpy.concatenate((X_val_temp,X_val[:,size:]),axis=1)
#Add this version of X to the list 
X_all.append(['StdSca','All', X_con,X_val_con,1.0,cols,rem_cols,ranks,i_cols,i_rem])

#MinMax
#Apply transform only for non-categorical data
X_temp = MinMaxScaler().fit_transform(X_train[:,0:size])
X_val_temp = MinMaxScaler().fit_transform(X_val[:,0:size])
#Concatenate non-categorical data and categorical
X_con = numpy.concatenate((X_temp,X_train[:,size:]),axis=1)
X_val_con = numpy.concatenate((X_val_temp,X_val[:,size:]),axis=1)
#Add this version of X to the list 
X_all.append(['MinMax', 'All', X_con,X_val_con,1.0,cols,rem_cols,ranks,i_cols,i_rem])

#Normalize
#Apply transform only for non-categorical data
X_temp = Normalizer().fit_transform(X_train[:,0:size])
X_val_temp = Normalizer().fit_transform(X_val[:,0:size])
#Concatenate non-categorical data and categorical
X_con = numpy.concatenate((X_temp,X_train[:,size:]),axis=1)
X_val_con = numpy.concatenate((X_val_temp,X_val[:,size:]),axis=1)
#Add this version of X to the list 
X_all.append(['Norm', 'All', X_con,X_val_con,1.0,cols,rem_cols,ranks,i_cols,i_rem])

#Impute
#Imputer is not used as no data is missing

#List of transformations
trans_list = []

for trans,name,X,X_val,v,cols_list,rem_list,rank_list,i_cols_list,i_rem_list in X_all:
    trans_list.append(trans)


# ## Feature Selection
# Using the rankings produced in :
# https://www.kaggle.com/sharmasanthosh/forest-cover-type-prediction/exploratory-study-on-feature-selection

# In[ ]:


#Select top 75%,50%,25%
ratio_list = [0.75,0.50,0.25]

#Median of rankings for each column
unsorted_rank = [0,8,11,4,5,2,5,7.5,9.5,3,8,28.5,14.5,2,35,19.5,12,14,37,25.5,50,44,9,28,20.5,19.5,40,38,20,38,43,35,44,22,24,33,49,42,46,47,27.5,19,31.5,23,28,42,30.5,46,40,12,13,18]

#List of feature selection models
feat = []

#Add Median to the list 
n = 'Median'
for val in ratio_list:
    feat.append([n,val])   

for trans,s, X, X_val, d, cols, rem_cols, ra, i_cols, i_rem in X_all:
    #Create subsets of feature list based on ranking and ratio_list
    for name, v in feat:
        #Combine importance and index of the column in the array joined
        joined = []
        for i, pred in enumerate(unsorted_rank):
            joined.append([i,cols[i],pred])
        #Sort in descending order    
        joined_sorted = sorted(joined, key=lambda x: x[2])
        #Starting point of the columns to be dropped
        rem_start = int((v*(c-1)))
        #List of names of columns selected
        cols_list = []
        #Indexes of columns selected
        i_cols_list = []
        #Ranking of all the columns
        rank_list =[]
        #List of columns not selected
        rem_list = []
        #Indexes of columns not selected
        i_rem_list = []
        #Split the array. Store selected columns in cols_list and removed in rem_list
        for j, (i, col, x) in enumerate(list(joined_sorted)):
            #Store the rank
            rank_list.append([i,j])
            #Store selected columns in cols_list and indexes in i_cols_list
            if(j < rem_start):
                cols_list.append(col)
                i_cols_list.append(i)
            #Store not selected columns in rem_list and indexes in i_rem_list    
            else:
                rem_list.append(col)
                i_rem_list.append(i)    
        #Sort the rank_list and store only the ranks. Drop the index 
        #Append model name, array, columns selected and columns to be removed to the additional list        
        X_all_add.append([trans,name,X,X_val,v,cols_list,rem_list,[x[1] for x in sorted(rank_list,key=lambda x:x[0])],i_cols_list,i_rem_list])


# In[ ]:


#Import plotting library    
import matplotlib.pyplot as plt    

#Dictionary to store the accuracies for all combinations 
acc = {}

#List of combinations
comb = []

#Append name of transformation to trans_list
for trans in trans_list:
    acc[trans]=[]


# ## Evaluation, prediction, and analysis
# * LDA (Linear algo)

# In[ ]:


#Evaluation of various combinations of LinearDiscriminatAnalysis using all the views

#Import the library
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#Set the base model
model = LinearDiscriminantAnalysis()
algo = "LDA"

##Set figure size
#plt.rc("figure", figsize=(25, 10))

#Accuracy of the model using all features
for trans,name,X,X_val,v,cols_list,rem_list,rank_list,i_cols_list,i_rem_list in X_all:
    model.fit(X[:,i_cols_list],Y_train)
    result = model.score(X_val[:,i_cols_list], Y_val)
    acc[trans].append(result)
    #print(trans+"+"+name+"+%d" % (v*(c-1)))
    #print(result)
comb.append("%s+%s of %s" % (algo,"All",1.0))
        
#Accuracy of the model using a subset of features    
for trans,name,X,X_val,v,cols_list,rem_list,rank_list,i_cols_list,i_rem_list in X_all_add:
    model.fit(X[:,i_cols_list],Y_train)
    result = model.score(X_val[:,i_cols_list], Y_val)
    acc[trans].append(result)
    #print(trans+"+"+name+"+%d" % (v*(c-1)))
    #print(result)
for v in ratio_list:
    comb.append("%s+%s of %s" % (algo,"Subset",v))
    
##Plot the accuracies of all combinations
#fig, ax = plt.subplots()
##Plot each transformation
#for trans in trans_list:
#        plt.plot(acc[trans])
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb)))
#ax.set_xticklabels(comb,rotation='vertical')
##Display the plot
#plt.legend(trans_list,loc='best')    
##Plot the accuracy for all combinations
#plt.show()    

#Best estimated performance is 65%. Occurs when all features are used and without any transformation!
#Performance of MinMax and Normalizer is very poor


# ## Evaluation, prediction, and analysis
# * LR (Linear algo)

# In[ ]:


#Evaluation of various combinations of LogisticRegression using all the views

#Import the library
from sklearn.linear_model import LogisticRegression

C_list = [100]

for C in C_list:
    #Set the base model
    model = LogisticRegression(n_jobs=-1,random_state=seed,C=C)
   
    algo = "LR"

    ##Set figure size
    #plt.rc("figure", figsize=(25, 10))

    #Accuracy of the model using all features
    for trans,name,X,X_val,v,cols_list,rem_list,rank_list,i_cols_list,i_rem_list in X_all:
        model.fit(X[:,i_cols_list],Y_train)
        result = model.score(X_val[:,i_cols_list], Y_val)
        acc[trans].append(result)
        #print(trans+"+"+name+"+%d" % (v*(c-1)))
        #print(result)
    comb.append("%s with C=%s+%s of %s" % (algo,C,"All",1.0))

    #Accuracy of the model using a subset of features    
    for trans,name,X,X_val,v,cols_list,rem_list,rank_list,i_cols_list,i_rem_list in X_all_add:
        model.fit(X[:,i_cols_list],Y_train)
        result = model.score(X_val[:,i_cols_list], Y_val)
        acc[trans].append(result)
        #print(trans+"+"+name+"+%d" % (v*(c-1)))
        #print(result)
    for v in ratio_list:
        comb.append("%s with C=%s+%s of %s" % (algo,C,"Subset",v))
    
##Plot the accuracies of all combinations
#fig, ax = plt.subplots()
##Plot each transformation
#for trans in trans_list:
#        plt.plot(acc[trans])
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb)))
#ax.set_xticklabels(comb,rotation='vertical')
##Display the plot
#plt.legend(trans_list,loc='best')    
##Plot the accuracy for all combinations
#plt.show()    
      
#Best estimated performance is close to 67% with LR when C=100 and all attributes are considered and with standardized data
#Performance improves will increasing value of C
#Performance of Normalizer and MinMax Scaler is poor in general


# ## Evaluation, prediction, and analysis
# * KNN (Non-linear algo)

# In[ ]:


#Evaluation of various combinations of KNN Classifier using all the views

#Import the library
from sklearn.neighbors import KNeighborsClassifier

n_list = [1]

for n_neighbors in n_list:
    #Set the base model
    model = KNeighborsClassifier(n_jobs=-1,n_neighbors=n_neighbors)
   
    algo = "KNN"

    ##Set figure size
    #plt.rc("figure", figsize=(25, 10))

    #Accuracy of the model using all features
    for trans,name,X,X_val,v,cols_list,rem_list,rank_list,i_cols_list,i_rem_list in X_all:
        model.fit(X[:,i_cols_list],Y_train)
        result = model.score(X_val[:,i_cols_list], Y_val)
        acc[trans].append(result)
        #print(trans+"+"+name+"+%d" % (v*(c-1)))
        #print(result)
    comb.append("%s with n=%s+%s of %s" % (algo,n_neighbors,"All",1.0))

    #Accuracy of the model using a subset of features    
    for trans,name,X,X_val,v,cols_list,rem_list,rank_list,i_cols_list,i_rem_list in X_all_add:
        model.fit(X[:,i_cols_list],Y_train)
        result = model.score(X_val[:,i_cols_list], Y_val)
        acc[trans].append(result)
        #print(trans+"+"+name+"+%d" % (v*(c-1)))
        #print(result)
    for v in ratio_list:
        comb.append("%s with n=%s+%s of %s" % (algo,n_neighbors,"Subset",v))
    
##Plot the accuracies of all combinations
#fig, ax = plt.subplots()
##Plot each transformation
#for trans in trans_list:
#        plt.plot(acc[trans])
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb)))
#ax.set_xticklabels(comb,rotation='vertical')
##Display the plot
#plt.legend(trans_list,loc='best')    
##Plot the accuracy for all combinations
#plt.show()    
 
#Best estimated performance is close to 86% when n_neighbors=1 and normalizer is used


# ## Evaluation, prediction, and analysis
# * Naive Bayes

# In[ ]:


#Evaluation of various combinations of Naive Bayes using all the views

#Import the library
from sklearn.naive_bayes import GaussianNB

#Set the base model
model = GaussianNB()
algo = "NB"

##Set figure size
#plt.rc("figure", figsize=(25, 10))

#Accuracy of the model using all features
for trans,name,X,X_val,v,cols_list,rem_list,rank_list,i_cols_list,i_rem_list in X_all:
    model.fit(X[:,i_cols_list],Y_train)
    result = model.score(X_val[:,i_cols_list], Y_val)
    acc[trans].append(result)
    #print(trans+"+"+name+"+%d" % (v*(c-1)))
    #print(result)
comb.append("%s+%s of %s" % (algo,"All",1.0))
        
#Accuracy of the model using a subset of features    
for trans,name,X,X_val,v,cols_list,rem_list,rank_list,i_cols_list,i_rem_list in X_all_add:
    model.fit(X[:,i_cols_list],Y_train)
    result = model.score(X_val[:,i_cols_list], Y_val)
    acc[trans].append(result)
    #print(trans+"+"+name+"+%d" % (v*(c-1)))
    #print(result)
for v in ratio_list:
    comb.append("%s+%s of %s" % (algo,"Subset",v))
    
##Plot the accuracies of all combinations
#fig, ax = plt.subplots()
##Plot each transformation
#for trans in trans_list:
#        plt.plot(acc[trans])
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb)))
#ax.set_xticklabels(comb,rotation='vertical')
##Display the plot
#plt.legend(trans_list,loc='best')    
##Plot the accuracy for all combinations
#plt.show()    

#Best estimated performance is close to 64%. Original with 50% subset outperfoms all transformations of NB


# ## Evaluation, prediction, and analysis
# * CART (Non-linear algo)

# In[ ]:


#Evaluation of various combinations of CART using all the views

#Import the library
from sklearn.tree import DecisionTreeClassifier

d_list = [13]

for max_depth in d_list:
    #Set the base model
    model = DecisionTreeClassifier(random_state=seed,max_depth=max_depth)
   
    algo = "CART"

    #Set figure size
    plt.rc("figure", figsize=(15, 10))

    #Accuracy of the model using all features
    for trans,name,X,X_val,v,cols_list,rem_list,rank_list,i_cols_list,i_rem_list in X_all:
        model.fit(X[:,i_cols_list],Y_train)
        result = model.score(X_val[:,i_cols_list], Y_val)
        acc[trans].append(result)
        #print(trans+"+"+name+"+%d" % (v*(c-1)))
        #print(result)
    comb.append("%s with d=%s+%s of %s" % (algo,max_depth,"All",1.0))

    #Accuracy of the model using a subset of features    
    for trans,name,X,X_val,v,cols_list,rem_list,rank_list,i_cols_list,i_rem_list in X_all_add:
        model.fit(X[:,i_cols_list],Y_train)
        result = model.score(X_val[:,i_cols_list], Y_val)
        acc[trans].append(result)
        #print(trans+"+"+name+"+%d" % (v*(c-1)))
        #print(result)
    for v in ratio_list:
        comb.append("%s with d=%s+%s of %s" % (algo,max_depth,"Subset",v))
    
##Plot the accuracies of all combinations
#fig, ax = plt.subplots()
##Plot each transformation
#for trans in trans_list:
#        plt.plot(acc[trans])
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb)))
#ax.set_xticklabels(comb,rotation='vertical')
##Display the plot
#plt.legend(trans_list,loc='best')    
##Plot the accuracy for all combinations
#plt.show()    
    
#Best estimated performance is close to 79% when max_depth=13 and for Original


# ## Evaluation, prediction, and analysis
# * SVM (Non-linear algo)

# In[ ]:


#Evaluation of various combinations of SVM using all the views

#Import the library
from sklearn.svm import SVC

c_list = [10]

for C in c_list:
    #Set the base model
    model = SVC(random_state=seed,C=C)

    algo = "SVM"

    #Set figure size
    #plt.rc("figure", figsize=(15, 10))

    #Accuracy of the model using all features
    for trans,name,X,X_val,v,cols_list,rem_list,rank_list,i_cols_list,i_rem_list in X_all:
        model.fit(X[:,i_cols_list],Y_train)
        result = model.score(X_val[:,i_cols_list], Y_val)
        acc[trans].append(result)
        #print(trans+"+"+name+"+%d" % (v*(c-1)))
        #print(result)
    comb.append("%s with C=%s+%s of %s" % (algo,C,"All",1.0))

    ##Accuracy of the model using a subset of features    
    #for trans,name,X,X_val,v,cols_list,rem_list,rank_list,i_cols_list,i_rem_list in X_all_add:
    #    model.fit(X[:,i_cols_list],Y_train)
    #    result = model.score(X_val[:,i_cols_list], Y_val)
    #    acc[trans].append(result)
    #    print(trans+"+"+name+"+%d" % (v*(c-1)))
    #    print(result)
    #for v in ratio_list:
    #    comb.append("%s with C=%s+%s of %s" % (algo,C,"Subset",v))
    
##Plot the accuracies of all combinations
#fig, ax = plt.subplots()
##Plot each transformation
#for trans in trans_list:
#        plt.plot(acc[trans])
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb)))
#ax.set_xticklabels(comb,rotation='vertical')
##Display the plot
#plt.legend(trans_list,loc='best')    
##Plot the accuracy for all combinations
#plt.show()    

#Training time is very high compared to other algos
#Performance is very poor for original. Shows the importance of data transformation
#Best estimated performance is close to 77% when C=10 and for StandardScaler with 0.25 subset


# ## Evaluation, prediction, and analysis
# * Bagged Decision Trees (Bagging)

# In[ ]:


#Evaluation of various combinations of Bagged Decision Trees using all the views

#Import the library
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

#Base estimator
base_estimator = DecisionTreeClassifier(random_state=seed,max_depth=13)

n_list = [100]

for n_estimators in n_list:
    #Set the base model
    model = BaggingClassifier(n_jobs=-1,base_estimator=base_estimator, n_estimators=n_estimators, random_state=seed)
   
    algo = "Bag"

    #Set figure size
    plt.rc("figure", figsize=(20, 10))

    #Accuracy of the model using all features
    for trans,name,X,X_val,v,cols_list,rem_list,rank_list,i_cols_list,i_rem_list in X_all:
        model.fit(X[:,i_cols_list],Y_train)
        result = model.score(X_val[:,i_cols_list], Y_val)
        acc[trans].append(result)
        #print(trans+"+"+name+"+%d" % (v*(c-1)))
        #print(result)
    comb.append("%s with n=%s+%s of %s" % (algo,n_estimators,"All",1.0))

    #Accuracy of the model using a subset of features    
    for trans,name,X,X_val,v,cols_list,rem_list,rank_list,i_cols_list,i_rem_list in X_all_add:
        model.fit(X[:,i_cols_list],Y_train)
        result = model.score(X_val[:,i_cols_list], Y_val)
        acc[trans].append(result)
        #print(trans+"+"+name+"+%d" % (v*(c-1)))
        #print(result)
    for v in ratio_list:
        comb.append("%s with n=%s+%s of %s" % (algo,n_estimators,"Subset",v))
    
##Plot the accuracies of all combinations
#fig, ax = plt.subplots()
##Plot each transformation
#for trans in trans_list:
#        plt.plot(acc[trans])
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb)))
#ax.set_xticklabels(comb,rotation='vertical')
##Display the plot
#plt.legend(trans_list,loc='best')    
##Plot the accuracy for all combinations
#plt.show()    

#Best estimated performance is close to 82% when n_estimators is 100 for Original


# ## Evaluation, prediction, and analysis
# * Random Forest (Bagging)

# In[ ]:


#Evaluation of various combinations of Random Forest using all the views

#Import the library
from sklearn.ensemble import RandomForestClassifier

n_list = [100]

for n_estimators in n_list:
    #Set the base model
    model = RandomForestClassifier(n_jobs=-1,n_estimators=n_estimators, random_state=seed)
   
    algo = "RF"

    #Set figure size
    plt.rc("figure", figsize=(20, 10))

    #Accuracy of the model using all features
    for trans,name,X,X_val,v,cols_list,rem_list,rank_list,i_cols_list,i_rem_list in X_all:
        model.fit(X[:,i_cols_list],Y_train)
        result = model.score(X_val[:,i_cols_list], Y_val)
        acc[trans].append(result)
        #print(trans+"+"+name+"+%d" % (v*(c-1)))
        #print(result)
    comb.append("%s with n=%s+%s of %s" % (algo,n_estimators,"All",1.0))

    #Accuracy of the model using a subset of features    
    for trans,name,X,X_val,v,cols_list,rem_list,rank_list,i_cols_list,i_rem_list in X_all_add:
        model.fit(X[:,i_cols_list],Y_train)
        result = model.score(X_val[:,i_cols_list], Y_val)
        acc[trans].append(result)
        #print(trans+"+"+name+"+%d" % (v*(c-1)))
        #print(result)
    for v in ratio_list:
        comb.append("%s with n=%s+%s of %s" % (algo,n_estimators,"Subset",v))
    
##Plot the accuracies of all combinations
#fig, ax = plt.subplots()
##Plot each transformation
#for trans in trans_list:
#        plt.plot(acc[trans])
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb)))
#ax.set_xticklabels(comb,rotation='vertical')
##Display the plot
#plt.legend(trans_list,loc='best')    
##Plot the accuracy for all combinations
#plt.show()    

#Best estimated performance is close to 85% when n_estimators is 100


# ## Evaluation, prediction, and analysis
# * Extra Trees (Bagging)

# In[ ]:


#Evaluation of various combinations of Extra Trees using all the views

#Import the library
from sklearn.ensemble import ExtraTreesClassifier

n_list = [100]

for n_estimators in n_list:
    #Set the base model
    model = ExtraTreesClassifier(n_jobs=-1,n_estimators=n_estimators, random_state=seed)
   
    algo = "ET"

    #Set figure size
    plt.rc("figure", figsize=(20, 10))

    #Accuracy of the model using all features
    for trans,name,X,X_val,v,cols_list,rem_list,rank_list,i_cols_list,i_rem_list in X_all:
        model.fit(X[:,i_cols_list],Y_train)
        result = model.score(X_val[:,i_cols_list], Y_val)
        acc[trans].append(result)
        #print(trans+"+"+name+"+%d" % (v*(c-1)))
        #print(result)
    comb.append("%s with n=%s+%s of %s" % (algo,n_estimators,"All",1.0))

    #Accuracy of the model using a subset of features    
    for trans,name,X,X_val,v,cols_list,rem_list,rank_list,i_cols_list,i_rem_list in X_all_add:
        model.fit(X[:,i_cols_list],Y_train)
        result = model.score(X_val[:,i_cols_list], Y_val)
        acc[trans].append(result)
        #print(trans+"+"+name+"+%d" % (v*(c-1)))
        #print(result)
    for v in ratio_list:
        comb.append("%s with n=%s+%s of %s" % (algo,n_estimators,"Subset",v))
    
##Plot the accuracies of all combinations
#fig, ax = plt.subplots()
##Plot each transformation
#for trans in trans_list:
#        plt.plot(acc[trans])
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb)))
#ax.set_xticklabels(comb,rotation='vertical')
##Display the plot
#plt.legend(trans_list,loc='best')    
##Plot the accuracy for all combinations
#plt.show()    

#Best estimated performance is close to 88% when n_estimators is 100 , StdScaler with 0.75


# ## Evaluation, prediction, and analysis
# * AdaBoost (Boosting)

# In[ ]:


#Evaluation of various combinations of AdaBoost ensemble using all the views

#Import the library
from sklearn.ensemble import AdaBoostClassifier

n_list = [100]

for n_estimators in n_list:
    #Set the base model
    model = AdaBoostClassifier(n_estimators=n_estimators, random_state=seed)
   
    algo = "Ada"

    #Set figure size
    plt.rc("figure", figsize=(20, 10))

    #Accuracy of the model using all features
    for trans,name,X,X_val,v,cols_list,rem_list,rank_list,i_cols_list,i_rem_list in X_all:
        model.fit(X[:,i_cols_list],Y_train)
        result = model.score(X_val[:,i_cols_list], Y_val)
        acc[trans].append(result)
        #print(trans+"+"+name+"+%d" % (v*(c-1)))
        #print(result)
    comb.append("%s with n=%s+%s of %s" % (algo,n_estimators,"All",1.0))

    #Accuracy of the model using a subset of features    
    for trans,name,X,X_val,v,cols_list,rem_list,rank_list,i_cols_list,i_rem_list in X_all_add:
        model.fit(X[:,i_cols_list],Y_train)
        result = model.score(X_val[:,i_cols_list], Y_val)
        acc[trans].append(result)
        #print(trans+"+"+name+"+%d" % (v*(c-1)))
        #print(result)
    for v in ratio_list:
        comb.append("%s with n=%s+%s of %s" % (algo,n_estimators,"Subset",v))
    
##Plot the accuracies of all combinations
#fig, ax = plt.subplots()
##Plot each transformation
#for trans in trans_list:
#        plt.plot(acc[trans])
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb)))
#ax.set_xticklabels(comb,rotation='vertical')
##Display the plot
#plt.legend(trans_list,loc='best')    
##Plot the accuracy for all combinations
#plt.show()    

#Best estimated performance is close to 38% when n_estimators is 100


# ## Evaluation, prediction, and analysis
# * Gradient Boosting (Boosting)

# In[ ]:


#Evaluation of various combinations of Stochastic Gradient Boosting using all the views

#Import the library
from sklearn.ensemble import GradientBoostingClassifier

d_list = [9]

for max_depth in d_list:
    #Set the base model
    model = GradientBoostingClassifier(max_depth=max_depth, random_state=seed)
   
    algo = "SGB"

    #Set figure size
    plt.rc("figure", figsize=(20, 10))

    #Accuracy of the model using all features
    for trans,name,X,X_val,v,cols_list,rem_list,rank_list,i_cols_list,i_rem_list in X_all:
        model.fit(X[:,i_cols_list],Y_train)
        result = model.score(X_val[:,i_cols_list], Y_val)
        acc[trans].append(result)
        #print(trans+"+"+name+"+%d" % (v*(c-1)))
        #print(result)
    comb.append("%s with d=%s+%s of %s" % (algo,max_depth,"All",1.0))

    ##Accuracy of the model using a subset of features    
    #for trans,name,X,X_val,v,cols_list,rem_list,rank_list,i_cols_list,i_rem_list in X_all_add:
    #    model.fit(X[:,i_cols_list],Y_train)
    #    result = model.score(X_val[:,i_cols_list], Y_val)
    #    acc[trans].append(result)
    #    #print(trans+"+"+name+"+%d" % (v*(c-1)))
    #    #print(result)
    #for v in ratio_list:
    #    comb.append("%s with d=%s+%s of %s" % (algo,max_depth,"Subset",v))
    
##Plot the accuracies of all combinations
#fig, ax = plt.subplots()
##Plot each transformation
#for trans in trans_list:
#        plt.plot(acc[trans])
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb)))
#ax.set_xticklabels(comb,rotation='vertical')
##Display the plot
#plt.legend(trans_list,loc='best')    
##Plot the accuracy for all combinations
#plt.show()    

#training time is too high
#Best estimated performance is close to 86% when depth is 7


# ## Evaluation, prediction, and analysis
# * Voting Classifier (Voting)

# In[ ]:


#Evaluation of various combinations of Voting Classifier using all the views

#Import the library
from sklearn.ensemble import VotingClassifier

list_estimators =[]

estimators = []
model1 = ExtraTreesClassifier(n_jobs=-1,n_estimators=100, random_state=seed)
estimators.append(('et', model1))
model2 = RandomForestClassifier(n_jobs=-1,n_estimators=100, random_state=seed)
estimators.append(('rf', model2))
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
base_estimator = DecisionTreeClassifier(random_state=seed,max_depth=13)
model3 = BaggingClassifier(n_jobs=-1,base_estimator=base_estimator, n_estimators=100, random_state=seed)
estimators.append(('bag', model3))

list_estimators.append(['Voting',estimators])

for name, estimators in list_estimators:
    #Set the base model
    model = VotingClassifier(estimators=estimators, n_jobs=-1)
   
    algo = name

    #Set figure size
    plt.rc("figure", figsize=(20, 10))

    #Accuracy of the model using all features
    for trans,name,X,X_val,v,cols_list,rem_list,rank_list,i_cols_list,i_rem_list in X_all:
        model.fit(X[:,i_cols_list],Y_train)
        result = model.score(X_val[:,i_cols_list], Y_val)
        acc[trans].append(result)
        #print(trans+"+"+name+"+%d" % (v*(c-1)))
        #print(result)
    comb.append("%s+%s of %s" % (algo,"All",1.0))

    #Accuracy of the model using a subset of features    
    for trans,name,X,X_val,v,cols_list,rem_list,rank_list,i_cols_list,i_rem_list in X_all_add:
        model.fit(X[:,i_cols_list],Y_train)
        result = model.score(X_val[:,i_cols_list], Y_val)
        acc[trans].append(result)
        #print(trans+"+"+name+"+%d" % (v*(c-1)))
        #print(result)
    for v in ratio_list:
        comb.append("%s+%s of %s" % (algo,"Subset",v))
    
##Plot the accuracies of all combinations
#fig, ax = plt.subplots()
##Plot each transformation
#for trans in trans_list:
#        plt.plot(acc[trans])
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb)))
#ax.set_xticklabels(comb,rotation='vertical')
##Display the plot
#plt.legend(trans_list,loc='best')    
##Plot the accuracy for all combinations
#plt.show()    

#Best estimated performance is close to 86%


# ## Evaluation, prediction, and analysis
# * XGBoost

# In[ ]:


#Evaluation of various combinations of XG Boost using all the views

#Import the library
from xgboost import XGBClassifier

n_list = [300]

for n_estimators in n_list:
    #Set the base model
    model = XGBClassifier(n_estimators=n_estimators, seed=seed,subsample=0.25)
   
    algo = "XGB"

    #Set figure size
    plt.rc("figure", figsize=(20, 10))

    #Accuracy of the model using all features
    for trans,name,X,X_val,v,cols_list,rem_list,rank_list,i_cols_list,i_rem_list in X_all:
        model.fit(X[:,i_cols_list],Y_train)
        result = model.score(X_val[:,i_cols_list], Y_val)
        acc[trans].append(result)
        #print(trans+"+"+name+"+%d" % (v*(c-1)))
        #print(result)
    comb.append("%s with n=%s+%s of %s" % (algo,n_estimators,"All",1.0))

    #Accuracy of the model using a subset of features    
    for trans,name,X,X_val,v,cols_list,rem_list,rank_list,i_cols_list,i_rem_list in X_all_add:
        model.fit(X[:,i_cols_list],Y_train)
        result = model.score(X_val[:,i_cols_list], Y_val)
        acc[trans].append(result)
        #print(trans+"+"+name+"+%d" % (v*(c-1)))
        #print(result)
    for v in ratio_list:
        comb.append("%s with n=%s+%s of %s" % (algo,n_estimators,"Subset",v))
    
##Plot the accuracies of all combinations
#fig, ax = plt.subplots()
##Plot each transformation
#for trans in trans_list:
#        plt.plot(acc[trans])
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb)))
#ax.set_xticklabels(comb,rotation='vertical')
##Display the plot
#plt.legend(trans_list,loc='best')    
##Plot the accuracy for all combinations
#plt.show()    

#Best estimated performance is close to 80% when n_estimators is 300, sub_sample=0.25 , subset=0.75


# ## Evaluation, prediction, and analysis
# * Multi-layer perceptrons (Deep learning)

# In[ ]:


#Evaluation of baseline model of MLP using all the views

#Import libraries for deep learning
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense

#Import libraries for encoding
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

#no. of output classes
y = 7

#random state
numpy.random.seed(seed)

# one hot encode class values
encoder = LabelEncoder()
Y_train_en = encoder.fit_transform(Y_train)
Y_train_hot = np_utils.to_categorical(Y_train_en,y) 
Y_val_en = encoder.fit_transform(Y_val)
Y_val_hot = np_utils.to_categorical(Y_val_en,y) 


# define baseline model
def baseline(v):
     # create model
     model = Sequential()
     model.add(Dense(v*(c-1), input_dim=v*(c-1), init='normal', activation='relu'))
     model.add(Dense(y, init='normal', activation='sigmoid'))
     # Compile model
     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
     return model

# define smaller model
def smaller(v):
 # create model
 model = Sequential()
 model.add(Dense(v*(c-1)/2, input_dim=v*(c-1), init='normal', activation='relu'))
 model.add(Dense(y, init='normal', activation='sigmoid'))
 # Compile model
 model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 return model

# define deeper model
def deeper(v):
 # create model
 model = Sequential()
 model.add(Dense(v*(c-1), input_dim=v*(c-1), init='normal', activation='relu'))
 model.add(Dense(v*(c-1)/2, init='normal', activation='relu'))
 model.add(Dense(y, init='normal', activation='sigmoid'))
 # Compile model
 model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 return model

# Optimize using dropout and decay
from keras.optimizers import SGD
from keras.layers import Dropout
from keras.constraints import maxnorm

def dropout(v):
    #create model
    model = Sequential()
    model.add(Dense(v*(c-1), input_dim=v*(c-1), init='normal', activation='relu',W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(v*(c-1)/2, init='normal', activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(y, init='normal', activation='sigmoid'))
    # Compile model
    sgd = SGD(lr=0.1,momentum=0.9,decay=0.0,nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

# define decay model
def decay(v):
    # create model
    model = Sequential()
    model.add(Dense(v*(c-1), input_dim=v*(c-1), init='normal', activation='relu'))
    model.add(Dense(y, init='normal', activation='sigmoid'))
    # Compile model
    sgd = SGD(lr=0.1,momentum=0.8,decay=0.01,nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model
    
est_list = [('MLP',baseline),('smaller',smaller),('deeper',deeper),('dropout',dropout),('decay',decay)]

for name, est in est_list:
 
    algo = name

    #Set figure size
    plt.rc("figure", figsize=(20, 10))

    #Accuracy of the model using all features
    for trans,name,X,X_val,v,cols_list,rem_list,rank_list,i_cols_list,i_rem_list in X_all:
        model = KerasClassifier(build_fn=est, v=v, nb_epoch=10, verbose=0)
        model.fit(X[:,i_cols_list],Y_train_hot)
        result = model.score(X_val[:,i_cols_list], Y_val_hot)
        acc[trans].append(result)
    #    print(trans+"+"+name+"+%d" % (v*(c-1)))
    #    print(result)
    comb.append("%s+%s of %s" % (algo,"All",1.0))

    ##Accuracy of the model using a subset of features    
    #for trans,name,X,X_val,v,cols_list,rem_list,rank_list,i_cols_list,i_rem_list in X_all_add:
    #    model = KerasClassifier(build_fn=est, v=v, nb_epoch=10, verbose=0)
    #    model.fit(X[:,i_cols_list],Y_train_hot)
    #    result = model.score(X_val[:,i_cols_list], Y_val_hot)
    #    acc[trans].append(result)
    #    print(trans+"+"+name+"+%d" % (v*(c-1)))
    #    print(result)
    #for v in ratio_list:
    #    comb.append("%s+%s of %s" % (algo,"Subset",v))

#Plot the accuracies of all combinations
fig, ax = plt.subplots()
#Plot each transformation
for trans in trans_list:
        plt.plot(acc[trans])
#Set the tick names to names of combinations
ax.set_xticks(range(len(comb)))
ax.set_xticklabels(comb,rotation='vertical')
#Display the plot
plt.legend(trans_list,loc='best')    
#Plot the accuracy for all combinations
plt.show()    

# Best estimated performance is 71% 
# Performance is poor is general. Data transformations make a huge difference.


# ##Make Predictions

# In[ ]:


# Make predictions using Extra Tress Classifier + 0.5 subset as it gave the best estimated performance

n_estimators = 100

#Obtain the list of indexes for the required model
indexes = []
for trans,name,X,X_val,v,cols_list,rem_list,rank_list,i_cols_list,i_rem_list in X_all_add:
    if v == 0.5:
        if trans == 'Orig':
            indexes = i_cols_list
            break

#Best model definition
best_model = ExtraTreesClassifier(n_jobs=-1,n_estimators=n_estimators)
best_model.fit(X_orig[:,indexes],Y)

#Read test dataset
dataset_test = pandas.read_csv("../input/test.csv")
#Drop unnecessary columns
ID = dataset_test['Id']
dataset_test.drop('Id',axis=1,inplace=True)
dataset_test.drop(rem,axis=1,inplace=True)
X_test = dataset_test.values

#Make predictions using the best model
predictions = best_model.predict(X_test[:,indexes])
# Write submissions to output file in the correct format
with open("submission.csv", "w") as subfile:
    subfile.write("Id,Cover_Type\n")
    for i, pred in enumerate(list(predictions)):
        subfile.write("%s,%s\n"%(ID[i],pred))

