
# coding: utf-8

# ## The notebook has been divided into three parts
#  1. EDA
#  2. Feature Engineering
#  3. Machine Learning Models
#  
# ** If you like it please up vote it**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


# import the file
Data = pd.read_csv('../input/HR_comma_sep.csv')


# In[ ]:


Data.info() # information about data


# In[ ]:


columns= Data.columns.tolist() # Make a list of name of columns
columns


# ## EDA

# In[ ]:


import seaborn as sns # for Interactive plots
import matplotlib.pyplot as plt # for plots
get_ipython().run_line_magic('matplotlib', 'inline')
import itertools


# In[ ]:


#let count no of of different categories in catgorical variables
#categorical variables are left, promotion_last_5years,sal,salary,work_accident
#more than two or three categories are number_project,time spend_company
categorical=['number_project','time_spend_company','Work_accident','left', 'promotion_last_5years','sales','salary']
fig=plt.subplots(figsize=(10,15))
length=len(categorical)
for i,j in itertools.zip_longest(categorical,range(length)): # itertools.zip_longest for to execute the longest loop
    plt.subplot(np.ceil(length/2),2,j+1)
    plt.subplots_adjust(hspace=.5)
    sns.countplot(x=i,data = Data)
    plt.xticks(rotation=90)
    plt.title("No. of employee")


#  **Observations from graphs**
# 
# 1. the maximum employee are doing project between 3-5
# 
# 2. There is a lot number of employees who are working in company for last three years but after that there is huge drop for 4 year
# 
# 3. The no of employee left are 20 % of total data
# 
# 4. A very less number of employee get the promotion in last 5 year
# 
# 5. Sales department is having maximum no.of employee followed by technical and support
# 
# 6. The maximum employees are getting salary either medium or low

# **Let's see who is leaving the company**

# In[ ]:


# no of employee who left the company 
print("The Number of employee who left the company :",len(Data[Data['left']==1]))
print("The Number of employee who didn't left the company",len(Data[Data['left']==0]))
print("The proportion of employee who left",len(Data[Data['left']==1])/len(Data))


# 1. 23 % of employee left the company which is nearby 1/4 it is huge number

# In[ ]:


####let's Analysis the Categorical and ordinal Variable


# In[ ]:


# here we will do it only for categorical variable
categorical=['number_project','time_spend_company','Work_accident','promotion_last_5years','sales','salary'] # here I have removed left to see who is leaving cpmpany
fig=plt.subplots(figsize=(12,15))# to define the size of figure
length=len(categorical) # no of categorical and ordinal variable
for i,j in itertools.zip_longest(categorical,range(length)): # itertools.zip_longest for to execute the longest loop
    plt.subplot(np.ceil(length/2),2,j+1) # this is to plot the subplots like as 2,2,1 it means 2x2 matrix and graph at 1 
    plt.subplots_adjust(hspace=.5) # to adjust the distance between subplots
    sns.countplot(x=i,data = Data,hue="left") # To plot the countplot of variable with hue left
    plt.xticks(rotation=90) # to rotate the xticks by 90 such that no xtixks overlap each other
    plt.title("No.of employee who left") # to plot the title of graph


# #### observations
# 
# 1. Some interesting observations are here :
# 
# 2. Those who promotion in last 5 years they didn't leave i.e All those left they didn't get the promotion in last 5 years
# 
# 3. from these graphs we can't make so much interpenetration because it is like that sales department have more number of employees so more number of employees left so for this we need the calculation of proportions.
# 
# 4. But one point interesting to note that those who have spent 5 years in company are leaving more in proportion.it may be because they didn't get the promotion in last 5 years
# 
# 5. And who have spent more than 6 years they are not leaving it may be because affection to the company 
# 
# 6. and more interesting point here those all who have done 7 projects have left the company it seems to like that the person for more than 6 years they are not leaving the company so it means those have done more number of projects in less year they are leaving the company
# 
# 7. So here we are talking about in may be to get the full detail we will do further analysis
# 
# 8. Now I will go ahead with analysis of proportion

# In[ ]:


# Lets Calcualte proportion for above same
fig=plt.subplots(figsize=(12,15))
for i,j in itertools.zip_longest(categorical,range(length)):# itertools.zip_longest for to execute the longest loop
    Proportion_of_data = Data.groupby([i])['left'].agg(lambda x: (x==1).sum()).reset_index()# only counting the number who left 
    Proportion_of_data1=Data.groupby([i])['left'].count().reset_index() # Counting the total number 
    Proportion_of_data2 = pd.merge(Proportion_of_data,Proportion_of_data1,on=i) # mergeing two data frames
    # Now we will calculate the % of employee who left category wise
    Proportion_of_data2["Proportion"]=(Proportion_of_data2['left_x']/Proportion_of_data2['left_y'])*100 
    Proportion_of_data2=Proportion_of_data2.sort_values(by="Proportion",ascending=False).reset_index(drop=True)#sorting by percentage
    plt.subplot(np.ceil(length/2),2,j+1)
    plt.subplots_adjust(hspace=.5)
    sns.barplot(x=i,y='Proportion',data=Proportion_of_data2)
    plt.xticks(rotation=90)
    plt.title("percentage of employee who left")
    plt.ylabel('Percentage')


# #### Observations
# 
# 1. First of all try to understand what are these graph showing. These graph are showing the percentage of the employee who left category wise means like that in sales department there were 4100 employee and approx 1000 of them left so the percentage of this will be nearby 25 %
# 
# 2. Now looking at the number of projects the employees who have done 7 project they left the company and then it comes who have done 2 projects 60 % of them have left the company maybe there are fired by company. 
# 
# 3. Then coming to time spend in company the top is 5 years followed by 4 and 6 it means the younger(2 or 3) and older ones(more than 6) are leaving less company then the adult ones(4,5,6)
# 
# 4. Now who got promotion in last 5 year it seems to be conflict with previous observation in that we said who got promotion in last 5 year haven't left the but here it is 5 % of them it is because of scale because who left is a very less number so it didn't come in graph 
# 
# 5. Now coming to department in previous graphs we saw in sales department more employee left the group but if we talk in percentage then HR department is at the top followed by Accounts and technical. Management peoples are in less percentage it is because may be they are at higher positions
# 
# 6. In case of salary the employee who are getting a low salary are leaving more than the medium and  high. Those who are getting high salary only 5 % of left the company
# 
# 7. I enjoy exploring the data so I will go for another analysis

# In[ ]:


# Let see who is getting Promotions proportion wise.
fig=plt.subplots(figsize=(12,15))
categorical=['number_project','time_spend_company','sales','salary']
length=len(categorical)
for i,j in itertools.zip_longest(categorical,range(length)):# itertools.zip_longest for to execute the longest loop
    Proportion_of_data = Data.groupby([i])['promotion_last_5years'].agg(lambda x: (x==1).sum()).reset_index()# only counting the number who left 
    Proportion_of_data1=Data.groupby([i])['promotion_last_5years'].count().reset_index() # Counting the total number 
    Proportion_of_data2 = pd.merge(Proportion_of_data,Proportion_of_data1,on=i) # mergeing two data frames
    # Now we will calculate the % of employee who  category wise
    Proportion_of_data2["Proportion"]=(Proportion_of_data2['promotion_last_5years_x']/Proportion_of_data2['promotion_last_5years_y'])*100 
    Proportion_of_data2=Proportion_of_data2.sort_values(by="Proportion",ascending=False).reset_index(drop=True)#sorting by percentage
    plt.subplot(np.ceil(length/2),2,j+1)
    plt.subplots_adjust(hspace=.3)
    sns.barplot(x=i,y='Proportion',data=Proportion_of_data2)
    plt.xticks(rotation=90)
    plt.title("Pecentage of employee getting promotion")
    plt.ylabel('Percentage')


# #### Observations
# 1. what the hell company is this. The employee who have done 7 projects are not getting promotions so why as we saw in previous graph a large number of employee left the company because they are not getting promotions while those have done 3 or 4 projects are getting Promotion. I think it is very unfair.
# 
# 2. The employee getting promotion who are for more time in company. The employee who have spent 7 years at the top followed by 10 years and 8 years. The employee who have spent 5 or 6 years in company have got promotion in very less proportion as we saw in previous graphs the employee who have spent 5 years left the most so it may be due to they are not getting promotion
# 
# 3. Management employee are getting promotion in high proportions followed by marketing. This was expected
# 
# 4. Employee getting high salaries are promoted more

# In[ ]:


# I want to have a look at the which department is accident prone

Proportion_of_data = Data.groupby(["sales"])["Work_accident"].agg(lambda x: (x==1).sum()).reset_index()# only counting the number who left 
Proportion_of_data1=Data.groupby(["sales"])["Work_accident"].count().reset_index() # Counting the total number 
Proportion_of_data2 = pd.merge(Proportion_of_data,Proportion_of_data1,on='sales') # mergeing two data frames
# Now we will calculate the % of employee who  category wise
Proportion_of_data2["Proportion"]=(Proportion_of_data2['Work_accident_x']/Proportion_of_data2['Work_accident_y'])*100 
Proportion_of_data2=Proportion_of_data2.sort_values(by="Proportion",ascending=False).reset_index(drop=True)#sorting by percentage
sns.barplot(x='sales',y='Proportion',data=Proportion_of_data2)
plt.xticks(rotation=90)
plt.title('Department wise accident')
plt.ylabel('Percentage')



# 1. The management department seems to be accident prone and Random.It is surprising me. why management is the accident prone ? I don't know !

# ###### Let's Start with continues variable

# In[ ]:


continues_variable=['satisfaction_level','last_evaluation','average_montly_hours']
categorical_variable=['promotion_last_5years','sales','salary','left','time_spend_company','number_project']
Data['Impact']=(Data['number_project']/Data["average_montly_hours"])*100


# In[ ]:


def pointplot(x, y, **kwargs): # making a function to plot point plot
    sns.pointplot(x=x, y=y)      
    x=plt.xticks(rotation=90)


# In[ ]:


# Start with Satisfaction Level
categorical_variable=['promotion_last_5years','sales','salary','left','time_spend_company','number_project']
f = pd.melt(Data, id_vars=["satisfaction_level"], value_vars=categorical_variable)
g=sns.FacetGrid(f,col='variable',col_wrap=2,sharex=False,sharey=False,size=5)

g.map(pointplot,"value","satisfaction_level")


# ##### Observations
# 
# 1. The point plot represent the mean of quantity the point show the mean of the satisfaction_level for that category and the vertical line show the variation of the data for that category. For details [Pointplot](http://seaborn.pydata.org/tutorial/categorical.html)
# 
# 2. by observing it who got promotions are more satisfied then the who didn't. It is a obvious fact but those who got promotion are very differ in there opinions as we can see a long vertical line indicating this while for who didn't get promotion are matching in their opinions
# 
# 3. same is for who left were less satisfied then the who didn't. If I think as a business Analyst my first question will be why were they not satisfied ? There are many answers we will try to get correct one in next analysis.
# 
# 4. looking at salary one obvious fact is there who is getting higher salary are more satisfied than others.
# 
# 5. by looking at department wise it seems the accounting and hr employee are very less satisfied compare to others.by looking at previous graph we can see that the department which get less promotion are less satisfied.
# 
# 6. now looking at the time spend at company there is sharp drop for the people who had spent 4 years in company.we can say it is due to promotion but people who have spent 5 years got less promotion too but they are very much satisfied compare to 4 years.
# 
# 7. now for number of projects the employee who have completed 7 projects are very much less satisfied that is only due to promotion because employee who have completed less number of projects are getting more promotions.
# 

# In[ ]:


# Now with last evaluation
f = pd.melt(Data, id_vars=['last_evaluation'], value_vars=categorical_variable)
g=sns.FacetGrid(f,col='variable',col_wrap=2,sharex=False,sharey=False,size=5)

g.map(pointplot,"value",'last_evaluation')


# ##### Observations
# 
# 1. This data is for Last evaluation from last evaluation it means the evaluation done by company
# 
# 2. Now look at the promotion graph the employee with less last evaluation are getting the promotion as it for who didn't get a promotion have a very high mean.
# 
# 3. By looking at department wise it seems management employee have a good  last evaluation compare to others.
# 
# 4. The high paid employee have a very bad last evolution and they are getting promotion too (as we can see from previous analysis)
# 
# 5. By looking at the employee who left have a good mean last evaluation then who haven't.But the range of employee is spread it may be because who have very less last evaluation may be fired by the company. and those who have very good last evaluation left because they didn't get promotion as we can see from promotion graph
# 
# 6. Coming to time spend the employee who have spend 5 years have a very good last evaluation followed by 4 years and 6 years.
# 
# 7. Looking at the number of projects the employee who have completed 7 projects have a very good last evaluation.
# 
# 8. By looking above two observation why the left have employee who have good last evaluation it is due to employee who have done 7 project have very good last evaluation and they didn't get promotion also so why they left the company.

# In[ ]:


# Now let's explore the variable average_monthly_hours
# With a new type of plot violin plot
Categorical_variable= ['sales','salary','time_spend_company','number_project']
fig=plt.subplots(figsize=(12,15))
length=len(Categorical_variable)
for i,j in itertools.zip_longest(Categorical_variable,range(length)): # itertools.zip_longest for to execute the longest loop
    plt.subplot(np.ceil(length/2),2,j+1) # here j repersent the number of graphs like as subplot 221
    plt.subplots_adjust(hspace=.5) # to get the space between subplots
    sns.violinplot(x=i,y="average_montly_hours", data = Data,hue="left",split=True,scale="count")#here count scales the width of violin
    plt.xticks(rotation=90)


# ###### Observation
# 
# 1. From above graphs it is clear two type of people left the company one who are working hard and other who are working very less. From this we can conclude that who are working less are may be fired by company and other one leave because they didn't get progress.
# 
# 2. For the different departments the pattern is same means all the departments are same in their opinions.
# 
# 3. in The time spent graph we can see that who have spent only 2 years a less number of employee left the company but for 3 years the employee who has left were working less than 160 hours from it we can conclude that they were fired by company but for 4 ,5, 6 the employee who left were working nearly more than 250 hours so it suggest that after this much of hard work they didn't get progress so why they left.
# 
# 4. for number of projects who have done only 2 projects and working very less are fired by company ( we can see in left) and the employee who have completed 7 projects they were wotking really hard but all of them left the company it os only because they didn't get promotion same is for who have done 6 projects

# ## 2. Feature Engineering

# 1. Now we have to predict who will left the company before going ahead lets do a part of feature engineering means selecting important features for the prediction.
# 
# 2. This can be done in many ways :-
# 
# 3. First of all remove all correlated variables 
# 4. We can use Randomforest to find important features
# 

# In[ ]:


# Now we have to predict who will left the company beofore going ahead lets do a part of feature engineering selecting import

# Let's plot the correlation Matrix
Data.drop('Impact',axis=1,inplace=True)
corr= Data.corr()


# In[ ]:


plt.figure(figsize=(12,10))
sns.heatmap(corr,annot=True,cbar=True,cmap="coolwarm")
plt.xticks(rotation=90)


# 1. Here no variables are so much correlated so that we can say that all variables are uncorrelated
# 2. so no need to remove any features lets get important features by using Randomforestclassifier

# In[ ]:


from sklearn.preprocessing import LabelEncoder # For change categorical variable into int
from sklearn.metrics import accuracy_score 
le=LabelEncoder()
Data['salary']=le.fit_transform(Data['salary'])
Data['sales']=le.fit_transform(Data['sales'])


# In[ ]:


# we can select importance features by using Randomforest Classifier
from sklearn.ensemble import RandomForestClassifier 
model= RandomForestClassifier(n_estimators=100)
feature_var = Data.ix[:,Data.columns != "left"]
pred_var = Data.ix[:,Data.columns=='left']
model.fit(feature_var,pred_var.values.ravel())


# In[ ]:


featimp = pd.Series(model.feature_importances_,index=feature_var.columns).sort_values(ascending=False)
print(featimp)


# ### 3.1 Machine Learning Models
# 
# 1. In this part we are going to use different machine learning models with all features variable and also top 5 important features given by Random forest Classifier
# 
# 2. In this part we will use simple models 

# In[ ]:


# Importing Machine learning models library used for classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.naive_bayes import GaussianNB as GB
from sklearn.svm import SVC


# In[ ]:


def Classification_model(model,Data,x,y): # here x is the variable which are used for prediction
    # y is the prediction variable
    train,test = train_test_split(Data,test_size= 0.33)
    train_x = Data.ix[train.index,x] # Data for training only with features
    train_y = Data.ix[train.index,y] # Data for training only with predcition variable
    test_x = Data.ix[test.index,x] # same as for training 
    test_y = Data.ix[test.index,y]
    model.fit(train_x,train_y.values.ravel())
    pred=model.predict(test_x)
    accuracy=accuracy_score(test_y,pred)
    return accuracy


# In[ ]:


All_features=['satisfaction_level',
'number_project',
'time_spend_company',
'average_montly_hours',
'last_evaluation',
'sales',
'salary',
'Work_accident',       
'promotion_last_5years']
print(All_features)
Important_features = ['satisfaction_level',
'number_project',
'time_spend_company',
'average_montly_hours',
'last_evaluation']
print(Important_features)
Pred_var = ["left"]
print(Pred_var)


# In[ ]:


# Lets us make a list of models
models=["RandomForestClassifier","Gaussian Naive Bays","KNN","Logistic_Regression","Support_Vector"]
Classification_models = [RandomForestClassifier(n_estimators=100),GB(),knn(n_neighbors=7),LogisticRegression(),SVC()]
Model_Accuracy = []
for model in Classification_models:
    Accuracy=Classification_model(model,Data,All_features,Pred_var)
    Model_Accuracy.append(Accuracy)


# In[ ]:


Accuracy_with_all_features = pd.DataFrame(
    { "Classification Model" :models,
     "Accuracy with all features":Model_Accuracy
     
    })


# In[ ]:


Accuracy_with_all_features.sort_values(by="Accuracy with all features",ascending=False).reset_index(drop=True)


# #### Observation 
# 
# 1. The Random Forest is at the top followed by Support_vector and KNN
# 
# 2. These all are giving accuracy more than 90% for validation data i.e. test data that is not bad
# 
# 3. In Next we will try same with but only with important features suggested by RandomForest 

# In[ ]:


# Lets try with Important features
Model_Accuracy = []
for model in Classification_models:
    Accuracy=Classification_model(model,Data,Important_features,Pred_var) # Just instead of all features give only important features
    Model_Accuracy.append(Accuracy)


# In[ ]:


Accuracy_with_important_features = pd.DataFrame(
    { "Classification Model" :models,
     "Accuracy with Important features":Model_Accuracy
     
    })
Accuracy_with_important_features.sort_values(by="Accuracy with Important features",ascending=False).reset_index(drop=True)


# 1. By using the important features there is a slight increase in accuracy for all classification models
# 
# 2. The gaussian navie bays and KNN show a increase of 2 % in accuracy

# ### 3.2 Machine Learning Models With Cross Validation
# 
# 1. In this we will do the cross validation with the models to get there mean accuracy
# 
# 2. From 3.1 we came to know that by using important features there is increase in the accuracy for all models so in this we will go only with important features

# In[ ]:


from sklearn.model_selection import cross_val_score # This is used for to caculate the score of cross validation by using Kfold
def Classification_model_CV(model,Data,x,y): # here x is the variable which are used for prediction
    # y is the prediction variable
    data_x = Data.ix[:,x] # Here no need of training and test data because in cross validation it splits data into 
    
    # train and test itself # data_x repersent features
    data_y = Data.ix[:,y] # data for predication
    data_y=data_y.values.ravel()
    scores= cross_val_score(model,data_x,data_y,scoring="accuracy",cv=10)
    print(scores) # print the scores
    print('')
    accuracy=scores.mean()
    return accuracy


# In[ ]:


models=["RandomForestClassifier","Gaussian Naive Bays","KNN","Logistic_Regression","Support_Vector"]
Classification_models = [RandomForestClassifier(n_estimators=100),GB(),knn(n_neighbors=7),LogisticRegression(),SVC()]
Model_Accuracy = []
for model,z in zip(Classification_models,models):
    print(z) # Print the name of model
    print('')
    Accuracy=Classification_model_CV(model,Data,Important_features,Pred_var)
    
    Model_Accuracy.append(Accuracy)


# In[ ]:


Accuracy_with_CV = pd.DataFrame(
    { "Classification Model" :models,
     "Accuracy with CV":Model_Accuracy
     
    })
Accuracy_with_CV.sort_values(by="Accuracy with CV",ascending=False).reset_index(drop=True)


# ### 3.3 Machine Learning Models With Parameter tuning
# 
# 1. In this we will use  Grid SearchCV to find the best parameter for a model 
# 
# 2. in this we will use important features too

# In[ ]:


from sklearn.model_selection import GridSearchCV 
def Classification_model_GridSearchCV(model,Data,x,y,params):
    
    # here params repersent Parameters
    data_x = Data.ix[:,x]  
    data_y = Data.ix[:,y] 
    data_y=data_y.values.ravel()
    clf = GridSearchCV(model,params,scoring="accuracy",cv=5)
    clf.fit(data_x,data_y)
    print("best score is :")
    print(clf.best_score_)
    print('')
    print("best estimator is :")
    print(clf.best_estimator_)

    return (clf.best_score_)


# In[ ]:


models=["RandomForestClassifier","Gaussian Naive Bays","KNN","Logistic_Regression","Support_Vector"]
Model_Accuracy=[]
model = RandomForestClassifier()
param_grid = {'n_estimators':(70,80,90,100),'criterion':('gini','entropy'),'max_depth':[25,30]}
Accuracy=Classification_model_GridSearchCV(model,Data,Important_features,Pred_var,param_grid)
Model_Accuracy.append(Accuracy)


# In[ ]:


model = GB()
param_grid={}
Accuracy=Classification_model_GridSearchCV(model,Data,Important_features,Pred_var,param_grid)
Model_Accuracy.append(Accuracy)


# In[ ]:


model=knn()
param_grid={'n_neighbors':[5,15],'weights':('uniform','distance'),'p':[1,5]}
Accuracy=Classification_model_GridSearchCV(model,Data,Important_features,Pred_var,param_grid)
Model_Accuracy.append(Accuracy)


# In[ ]:


model=LogisticRegression()
param_grid={'C': [0.01,0.1,1,10],'penalty':('l1','l2')}
Accuracy=Classification_model_GridSearchCV(model,Data,Important_features,Pred_var,param_grid)
Model_Accuracy.append(Accuracy)


# In[ ]:


model=SVC()
param_grid={'C': [1,10,20,100],'gamma':[0.1,1,10]} 
Accuracy=Classification_model_GridSearchCV(model,Data,Important_features,Pred_var,param_grid)
Model_Accuracy.append(Accuracy)


# In[ ]:


Accuracy_with_GridSearchCV = pd.DataFrame(
    { "Classification Model" :models,
     "Accuracy with GridSearchCV":Model_Accuracy
     
    })
Accuracy_with_GridSearchCV.sort_values(by="Accuracy with GridSearchCV",ascending=False).reset_index(drop=True)


# In[ ]:


Comparison=pd.merge(pd.merge(pd.merge(Accuracy_with_all_features,Accuracy_with_important_features,on='Classification Model'),Accuracy_with_CV,on='Classification Model'),Accuracy_with_GridSearchCV,on='Classification Model')


# In[ ]:


Comparison1=Comparison.ix[:,["Classification Model","Accuracy with all features","Accuracy with Important features","Accuracy with CV","Accuracy with GridSearchCV"]]


# In[ ]:


Comparison1


# **Conclusion**
# 
#  1. Here we can compare the accuracy obtained by different Classification Models  with different strategy
#  2. For A quick revision
#  3. Accuracy with all features means the all features of data were used for prediction of will employee left or not?  this accuracy is obtained on the test data which was not used in training.
#  4. Accuracy with important features means the same as above but here only 5 most important features were used. The importance of features we got by using Random Forest Classifier.
#  5. Accuracy with CV means the mean of accuracies which were obtained on iteration of one CV. here 10 iterations were used
#  6. Accuracy with GridSearchCV means the best score obtained after tuning the model.  Here for CV only 5 folds were used
# 
#  7. Thank you
# 
#  

# **Some Links**
# 
#  1. [Plotting with categorical Variable](http://seaborn.pydata.org/tutorial/categorical.html)
#  2. [Random Forest Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
#  3. [Support Vector Machine](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
#  4. [KNN](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
#  5. [Logistic Regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
#  6. [Gaussian Naive Bays](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
# 
#  7. [Cross validation](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)
# 
#  8. [Grid SearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
# 
#  9. In comments please share resources from where I can learn other algorithm like as XGboost and other Ensemble methods
# 
# 

# **Upcoming**
# 
#  1. I will try to learn about Xgboost 
