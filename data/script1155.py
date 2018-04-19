
# coding: utf-8

# # Using the Wisconsin breast cancer diagnostic data set for predictive analysis
# ## Buddhini Waidyawansa (12-03-2016)
# Attribute Information:
# 
#  - 1) ID number 
#  - 2) Diagnosis (M = malignant, B = benign) 
#  
# -3-32.Ten real-valued features are computed for each cell nucleus:
# 
#  - a) radius (mean of distances from center to points on the perimeter) 
#  - b) texture (standard deviation of gray-scale values) 
#  - c) perimeter 
#  - d) area 
#  - e) smoothness (local variation in radius lengths) 
#  - f) compactness (perimeter^2 / area - 1.0) 
#  - g). concavity (severity of concave portions of the contour) 
#  - h). concave points (number of concave portions of the contour) 
#  - i). symmetry 
#  - j). fractal dimension ("coastline approximation" - 1)
# 
# The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.
# 
# 
# For this analysis, as a guide to predictive analysis I followed the instructions and discussion on "A Complete Tutorial on Tree Based Modeling from Scratch (in R & Python)" at Analytics Vidhya.

# #Load Libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# keeps the plots in one place. calls image as static pngs
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt # side-stepping mpl backend
import matplotlib.gridspec as gridspec # subplots
import mpld3 as mpl

#Import models from scikit learn module:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics


# # Load the data

# In[ ]:


df = pd.read_csv("../input/data.csv",header = 0)
df.head()


# # Clean and prepare data

# In[ ]:


df.drop('id',axis=1,inplace=True)
df.drop('Unnamed: 32',axis=1,inplace=True)
# size of the dataframe
len(df)


# In[ ]:


df.diagnosis.unique()


# In[ ]:


df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})
df.head()


# # Explore data

# In[ ]:


df.describe()


# In[ ]:


df.describe()
plt.hist(df['diagnosis'])
plt.title('Diagnosis (M=1 , B=0)')
plt.show()


# ### nucleus features vs diagnosis

# In[ ]:


features_mean=list(df.columns[1:11])
# split dataframe into two based on diagnosis
dfM=df[df['diagnosis'] ==1]
dfB=df[df['diagnosis'] ==0]


# In[ ]:


#Stack the data
plt.rcParams.update({'font.size': 8})
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8,10))
axes = axes.ravel()
for idx,ax in enumerate(axes):
    ax.figure
    binwidth= (max(df[features_mean[idx]]) - min(df[features_mean[idx]]))/50
    ax.hist([dfM[features_mean[idx]],dfB[features_mean[idx]]], bins=np.arange(min(df[features_mean[idx]]), max(df[features_mean[idx]]) + binwidth, binwidth) , alpha=0.5,stacked=True, normed = True, label=['M','B'],color=['r','g'])
    ax.legend(loc='upper right')
    ax.set_title(features_mean[idx])
plt.tight_layout()
plt.show()


# ###Observations
# 
# 1. mean values of cell radius, perimeter, area, compactness, concavity and concave points can be used in classification of the cancer. Larger values of these parameters tends to show a correlation with malignant tumors. 
# 2. mean values of texture, smoothness, symmetry or fractual dimension does not show a particular preference of one diagnosis over the other. In any of the histograms there are no noticeable large outliers that warrants further cleanup.

# ## Creating a test set and a training set
# Since this data set is not ordered, I am going to do a simple 70:30 split to create a training data set and a test data set.

# In[ ]:


traindf, testdf = train_test_split(df, test_size = 0.3)


# ## Model Classification
# 
# Here we are going to build a classification model and evaluate its performance using the training set.
# 
# 

# In[ ]:


#Generic function for making a classification model and accessing the performance. 
# From AnalyticsVidhya tutorial
def classification_model(model, data, predictors, outcome):
  #Fit the model:
  model.fit(data[predictors],data[outcome])
  
  #Make predictions on training set:
  predictions = model.predict(data[predictors])
  
  #Print accuracy
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print("Accuracy : %s" % "{0:.3%}".format(accuracy))

  #Perform k-fold cross-validation with 5 folds
  kf = KFold(data.shape[0], n_folds=5)
  error = []
  for train, test in kf:
    # Filter training data
    train_predictors = (data[predictors].iloc[train,:])
    
    # The target we're using to train the algorithm.
    train_target = data[outcome].iloc[train]
    
    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)
    
    #Record error from each cross-validation run
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
    
    print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))
    
  #Fit the model again so that it can be refered outside the function:
  model.fit(data[predictors],data[outcome]) 


# ### Logistic Regression model
# 
# Logistic regression is widely used for classification of discrete data. In this case we will use it for binary (1,0) classification.
# 
# Based on the observations in the histogram plots, we can reasonably hypothesize that the cancer diagnosis depends on the mean cell radius, mean perimeter, mean area, mean compactness, mean concavity and mean concave points. We can then  perform a logistic regression analysis using those features as follows:

# In[ ]:


predictor_var = ['radius_mean','perimeter_mean','area_mean','compactness_mean','concave points_mean']
outcome_var='diagnosis'
model=LogisticRegression()
classification_model(model,traindf,predictor_var,outcome_var)


# The prediction accuracy is reasonable. 
# What happens if we use just one predictor? 
# Use the mean_radius:

# In[ ]:


predictor_var = ['radius_mean']
model=LogisticRegression()
classification_model(model,traindf,predictor_var,outcome_var)


# This gives a similar prediction accuracy and a cross-validation score.
# 

# The accuracy of the predictions are good but not great. The cross-validation scores are reasonable. 
# Can we do better with another model?

# ### Decision Tree Model

# In[ ]:


predictor_var = ['radius_mean','perimeter_mean','area_mean','compactness_mean','concave points_mean']
model = DecisionTreeClassifier()
classification_model(model,traindf,predictor_var,outcome_var)


# Here we are over-fitting the model probably due to the large number of predictors.
# Let use a single predictor, the obvious one is the radius of the cell.

# In[ ]:


predictor_var = ['radius_mean']
model = DecisionTreeClassifier()
classification_model(model,traindf,predictor_var,outcome_var)


# The accuracy of the prediction is much much better here.  But does it depend on the predictor?

# Using a single predictor gives a 97% prediction accuracy for this model but the cross-validation score is not that great. 

# ### Randome Forest

# In[ ]:


# Use all the features of the nucleus
predictor_var = features_mean
model = RandomForestClassifier(n_estimators=100,min_samples_split=25, max_depth=7, max_features=2)
classification_model(model, traindf,predictor_var,outcome_var)


# Using all the features improves the prediction accuracy and the cross-validation score is great.

#  An advantage with Random Forest is that it returns a feature importance matrix which can be used to select features. So lets select the top 5 features and use them as predictors.

# In[ ]:


#Create a series with feature importances:
featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print(featimp)


# In[ ]:


# Using top 5 features
predictor_var = ['concave points_mean','area_mean','radius_mean','perimeter_mean','concavity_mean',]
model = RandomForestClassifier(n_estimators=100, min_samples_split=25, max_depth=7, max_features=2)
classification_model(model,traindf,predictor_var,outcome_var)


# Using the top 5 features only changes the prediction accuracy a bit but I think we get a better result if we use all the predictors.
# 
# What happens if we use a single predictor as before? Just check.

# In[ ]:


predictor_var =  ['radius_mean']
model = RandomForestClassifier(n_estimators=100)
classification_model(model, traindf,predictor_var,outcome_var)


# This gives a better prediction accuracy too but the cross-validation is not great.
# 

# ## Using on the test data set

# In[ ]:


# Use all the features of the nucleus
predictor_var = features_mean
model = RandomForestClassifier(n_estimators=100,min_samples_split=25, max_depth=7, max_features=2)
classification_model(model, testdf,predictor_var,outcome_var)


# The prediction accuracy for the test data set using the above Random Forest model is 95%!

# ## Conclusion
# 
# The best model to be used for diagnosing breast cancer as found in this analysis is the Random Forest model with the top 5 predictors, 'concave points_mean','area_mean','radius_mean','perimeter_mean','concavity_mean'. It gives a prediction accuracy of ~95% and a cross-validation score ~ 93% for the test data set.
# 
# 
# I will see if I can improve this more by tweaking the model further and trying out other models in a later version of this analysis.
