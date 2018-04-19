
# coding: utf-8

# # Introduction
# This notebook demos Data Visualisation and various Machine Learning Classification algorithms on Pima Indians dataset.

# In[ ]:


from IPython.display import YouTubeVideo
YouTubeVideo("pN4HqWRybwk")


# # 1) Loading Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import warnings
warnings.filterwarnings('ignore')


# # 2) Data

# In[ ]:


pima = pd.read_csv("../input/diabetes.csv")


# In[ ]:


pima.head()


# #### Additional details about the attributes
# 
# >Pregnancies: Number of times pregnant
# 
# >Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# 
# >BloodPressure: Diastolic blood pressure (mm Hg)
# 
# >SkinThickness: Triceps skin fold thickness (mm)
# 
# >Insulin: 2-Hour serum insulin (mu U/ml)
# 
# >BMI: Body mass index (weight in kg/(height in m)^2)
# 
# >DiabetesPedigreeFunction: Diabetes pedigree function
# 
# >Age: Age (years)
# 
# >Outcome: Class variable (0 or 1)

# In[ ]:


pima.shape


# In[ ]:


pima.describe()


# In[ ]:


pima.groupby("Outcome").size()


# # 3) Data Visualisation
# Let's try to visualise this data

# In[ ]:


pima.hist(figsize=(10,8))


# In[ ]:


pima.plot(kind= 'box' , subplots=True, layout=(3,3), sharex=False, sharey=False, figsize=(10,8))


# In[ ]:


column_x = pima.columns[0:len(pima.columns) - 1]


# In[ ]:


column_x


# In[ ]:


corr = pima[pima.columns].corr()


# In[ ]:


sns.heatmap(corr, annot = True)


# # 4) Feature Extraction

# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[ ]:


X = pima.iloc[:,0:8]
Y = pima.iloc[:,8]
select_top_4 = SelectKBest(score_func=chi2, k = 4)


# In[ ]:


fit = select_top_4.fit(X,Y)
features = fit.transform(X)


# In[ ]:


features[0:5]


# In[ ]:


pima.head()


# So, the top performing features are Glucose, Insulin, BMI, Age 

# In[ ]:


X_features = pd.DataFrame(data = features, columns = ["Glucose","Insulin","BMI","Age"])


# In[ ]:


X_features.head()


# In[ ]:


Y = pima.iloc[:,8]


# In[ ]:


Y.head()


# #  5) Standardization
# It changes the attribute values to Guassian distribution with mean as 0 and standard deviation as 1. It is useful when the algorithm expects the input features to be in Guassian distribution.

# In[ ]:


from sklearn.preprocessing import StandardScaler
rescaledX = StandardScaler().fit_transform(X_features)


# In[ ]:


X = pd.DataFrame(data = rescaledX, columns= X_features.columns)


# In[ ]:


X.head()


# # 6) Binary Classification

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, random_state = 22, test_size = 0.2)


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# In[ ]:


models = []
models.append(("LR",LogisticRegression()))
models.append(("NB",GaussianNB()))
models.append(("KNN",KNeighborsClassifier()))
models.append(("DT",DecisionTreeClassifier()))
models.append(("SVM",SVC()))


# In[ ]:


results = []
names = []
for name,model in models:
    kfold = KFold(n_splits=10, random_state=22)
    cv_result = cross_val_score(model,X_train,Y_train, cv = kfold,scoring = "accuracy")
    names.append(name)
    results.append(cv_result)
for i in range(len(names)):
    print(names[i],results[i].mean())


# # 7) Visualising Results

# In[ ]:


ax = sns.boxplot(data=results)
ax.set_xticklabels(names)


# # 8) Final Prediction using Test Data
# Logistic Regression and SVM provides maximum results.

# In[ ]:


lr = LogisticRegression()
lr.fit(X_train,Y_train)
predictions = lr.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[ ]:


print(accuracy_score(Y_test,predictions))


# In[ ]:


svm = SVC()
svm.fit(X_train,Y_train)
predictions = svm.predict(X_test)


# In[ ]:


print(accuracy_score(Y_test,predictions))


# In[ ]:


print(classification_report(Y_test,predictions))


# In[ ]:


conf = confusion_matrix(Y_test,predictions)


# In[ ]:


label = ["0","1"]
sns.heatmap(conf, annot=True, xticklabels=label, yticklabels=label)

