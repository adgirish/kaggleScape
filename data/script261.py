
# coding: utf-8

# I am predicting which employees are the most likely to change jobs, explore why, and come up with recommendations what employers can do to keep staff.

# In[ ]:


# Import the modules

import pandas as pd
import numpy as np
from scipy import stats
import sklearn as sk
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import svm
import xgboost as xgb


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV

sns.set(style='white', context='notebook', palette='deep') 

from sklearn import metrics
from sklearn.metrics import accuracy_score



# In[ ]:


#Load data
data = pd.read_csv('../input/HR_comma_sep.csv')


# # Data Inspection

# The first step is to check for missing values in the data set. That's import so that we can put the summary statistics in context. Missing data can lead to a false impression of the data distribution and we need to be aware of that.

# In[ ]:


print(data.isnull().sum())


# We got lucky and there are no missing data, which is usually not the case.

# Next, let's check with what kind of variables we are dealing with, i.e. are the variables categorical, numerical etc.? 

# In[ ]:


data.dtypes


# So, **satisfaction_level** and **last_evaluation** are continuous numerical data but we need to confirm that. They could be misclassified and should really be integers. 
# 
# **sales** and **salary** are strings and probably categorical.
# 
# The other variables are integers but it is not quite clear whether they are numerical or really categorical data.
# 
# Let's look at the first few lines of the dataset to get a fell for what the data looks like.

# In[ ]:


data.head()


# And now let's look at a brief summary of the data. The describe function only works on numerical data unfortunately. We will get to a summary of categorical variables shortly though.

# In[ ]:


data.describe()


# Let's look at this visually.

# In[ ]:


#histograms
warnings.filterwarnings('ignore')
plt.figure(figsize=[12,10])

plt.subplot(331)
plt.xlabel('satisfaction_level', fontsize=12)
plt.ylabel('distribution', fontsize=12)
sns.distplot(data['satisfaction_level'], kde=False)

plt.subplot(332)
plt.xlabel('last_evaluation', fontsize=12)
#plt.ylabel('distribution', fontsize=12)
sns.distplot(data['last_evaluation'], kde=False)

plt.subplot(333)
plt.xlabel('number_project', fontsize=12)
#plt.ylabel('distribution', fontsize=12)
sns.distplot(data['number_project'], kde=False)

plt.subplot(334)
plt.xlabel('average_montly_hours', fontsize=12)
plt.ylabel('distribution', fontsize=12)
sns.distplot(data['average_montly_hours'], kde=False)

plt.subplot(335)
plt.xlabel('time_spend_company', fontsize=12)
#plt.ylabel('distribution', fontsize=12)
sns.distplot(data['time_spend_company'], kde=False)

plt.subplot(336)
plt.xlabel('Work_accident', fontsize=12)
#plt.ylabel('distribution', fontsize=12)
sns.distplot(data['Work_accident'], kde=False)

plt.subplot(337)
plt.xlabel('left', fontsize=12)
plt.ylabel('distribution', fontsize=12)
sns.distplot(data['left'], kde=False)

plt.subplot(338)
plt.xlabel('promotion_last_5years', fontsize=12)
#plt.ylabel('distribution', fontsize=12)
sns.distplot(data['promotion_last_5years'], kde=False)


# Now, we have a good idea of the distributions and what kind of data they are. 
# **satisfaction_level**, **last_evaluation**, **number_project**, **average_montly_hours**, and **time_spend_company** are numerical variables.
# **Work_accident**, **promotion_last_5_years**, and **left** are categorical data where 1 means *yes* and 0 means *no*.
# We can also see some interesting bi-modal distribtuion and a very high number satisfaction levels of 0 but more of that in the next section.

# Let's move on to the categorical variables we have not looked at yet.

# In[ ]:


data['sales'].value_counts()


# In[ ]:


count = data.groupby(data['sales']).count()
count = pd.DataFrame(count.to_records())
count = count.sort_values(by= 'left', ascending = False)
count = count['sales']

sns.countplot(y='sales', data=data, order=count)


# In[ ]:


data['salary'].value_counts()


# In[ ]:


count = data.groupby(data['salary']).count()
count = pd.DataFrame(count.to_records())
count = count.sort_values(by= 'left', ascending = False)
count = count['salary']

sns.countplot(y='salary', data=data, order=count)


# # Data Analysis

# Let's first start off with calculating the baseline prediction, which is the proportion of leavers out of all employees. We will use this in the next section to compare against the predictive models. For now, it will be useful to check whether there are any types of jobs that are more risk of employees leaving etc. 

# In[ ]:


#Calculate the baseline prediction accuracy
#colour code left
left = data[data['left']==1]
stayed = data[data['left']==0]
left_col = 'blue'
stayed_col = 'red'

print('Left: %i (%.1f percent), Stayed: %i (%.1f percent), Total: %i'     %(len(left), 1.*len(left)/len(data)*100.0,      len(stayed), 1.*len(stayed)/len(data)*100.0, len(data)))


# Let's compare leavers and stayers and plot them against each other.

# In[ ]:


#plot the variables against their survival rates
warnings.filterwarnings('ignore')
plt.figure(figsize=[12,10])

plt.subplot(331)
plt.xlabel('satisfaction_level', fontsize=12)
plt.ylabel('distribution', fontsize=12)
sns.kdeplot(left['satisfaction_level'].dropna().values, color=left_col)
sns.kdeplot(stayed['satisfaction_level'].dropna().values, color=stayed_col)
plt.plot([0.5, 0.5], [3, 0], linewidth=2, color='black')

plt.subplot(332)
plt.xlabel('last_evaluation', fontsize=12)
plt.ylabel('distribution', fontsize=12)
sns.kdeplot(left['last_evaluation'].dropna().values, color=left_col)
sns.kdeplot(stayed['last_evaluation'].dropna().values, color=stayed_col)
plt.plot([0.57, 0.57], [3, 0], linewidth=2, color='black')
plt.plot([0.82, 0.82], [3, 0], linewidth=2, color='black')

plt.subplot(333)
sns.barplot('number_project', 'left', data=data)
plt.plot([-1, 10], [0.238, 0.238], linewidth=2, color='black')

plt.subplot(334)
plt.xlabel('average_montly_hours', fontsize=12)
plt.ylabel('distribution', fontsize=12)
sns.kdeplot(left['average_montly_hours'].dropna().values, color=left_col)
sns.kdeplot(stayed['average_montly_hours'].dropna().values, color=stayed_col)
plt.plot([160, 160], [0.012, 0], linewidth=2, color='black')
plt.plot([240, 240], [0.012, 0], linewidth=2, color='black')

plt.subplot(335)
sns.barplot('time_spend_company', 'left', data=data)
plt.plot([-1, 10], [0.238, 0.238], linewidth=2, color='black')

plt.subplot(336)
sns.barplot('Work_accident', 'left', data=data)
plt.plot([-1, 10], [0.238, 0.238], linewidth=2, color='black')

plt.subplot(337)
sns.barplot('promotion_last_5years', 'left', data=data)
plt.plot([-1, 10], [0.238, 0.238], linewidth=2, color='black')

plt.subplot(338)
sns.barplot('sales', 'left', data=data)
plt.plot([-1, 10], [0.238, 0.238], linewidth=2, color='black')
plt.xticks(rotation=90)

plt.subplot(339)
sns.barplot('salary', 'left', data=data)
plt.plot([-1, 10], [0.238, 0.238], linewidth=2, color='black')

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)


# To summarise:
# 
# **satisfaction_level**: If the satisfaction level is smaller than 0.5 employees almost without exception more likely to leave than to stay. There is an uptick in leavers at levl of around 0.8. These must the top performers who most likely have been poached by a competitor.
# 
# **last_evaluation**: Employees who have been evaluated recently or a long time ago are more likely to leave. Perhaps the recent evaluation made the employee realise that it's not the right job for them or they received poor feedback. An evaluation that happened long time ago could indicate that these employees have received any feedback or could make their voice heard in a long time and have become 'estranged' from the company.
# 
# **number_project**: Employees with few projections or very many are more likely to leave.
# 
# **average_montly_hours**: This is a similar story to number_projects. These two variables are probably correlated with each other.
# 
# **time_spend_company**: Once an employee has worked longer than 3 years, they are more likely to leave.
# 
# **Work_accident**: Interestingly, there is a higher proportion of employees to leave that had an accident than those who did not. This might be due to a very smaller number of accidents and thus a statistical fluke. 
# 
# **promotion_last_5_years**: It's not surprising to see that employees who have been promoted are more likely to stay.
# 
# **sales**: Management and R&D are more likely to stay than other job categories. On the other hand HR and Accounting are more likely to leave compared to the average.
# 
# **salary**: Not surprisingly employees with higher pay are more likely to stay.

# Let's check the accidents quickly. 

# In[ ]:


accident = data[data['Work_accident']==1]
no_accident = data[data['Work_accident']==0]

print('Accident: %i (%.1f percent), No accident: %i (%.1f percent), Total: %i'     %(len(accident), 1.*len(accident)/len(data)*100.0,      len(no_accident), 1.*len(no_accident)/len(data)*100.0, len(data)))


# Turns out there are actually quite a few accidents. It is an interesting observation but let's see if it really matters.

# Next, let's find out if any of the variables are correlated with each other.

# In[ ]:


data.head()


# In[ ]:


#I need to drop the categorical varaibles as calculating their correlations doesn't make any sense.
datacor = data.drop(['Work_accident', 'promotion_last_5years', 'sales', 'salary', 'left'], axis=1)

plt.figure(figsize=(14,12))
sns.heatmap(datacor.corr(), vmax=0.6, square=True, annot=True)


# After we have looked at the variables individually, let's look at them in the context of other variables. First I want to find out whether salaries have anything to do with job categories having different probabilities of leaving. 
# 
# What really stands out is that there are proportionally a lot more highly paid positions in management. That could help explain the lower probability in leaving. The other job categories all look fairly similar.

# In[ ]:


#How do the salaries amongst different job categories look like?
sns.factorplot("salary", col="sales", col_wrap=5,
                   data=data,
                    kind="count", size=2.5, aspect=.8, sharey = False)


# In[ ]:


#Where do accidents occur?
sns.factorplot("Work_accident", col="sales", col_wrap=5,
                   data=data,
                    kind="count", size=2.5, aspect=.8, sharey = False)


# This equally distributed across all job categories

# In[ ]:


#Who got the promotions?
sns.factorplot("promotion_last_5years", col="sales", col_wrap=5,
                   data=data,
                    kind="count", size=2.5, aspect=.8, sharey = False)


# Management (and Marketing + RandD) seem to have more promotions 

# In[ ]:


#Who is the most satisfied
median = data.groupby(['sales']).median()
median = pd.DataFrame(median.to_records())
median = median.sort_values(by='satisfaction_level', ascending = False)
median = median['sales']

sns.boxplot('sales', 'satisfaction_level',order=median, data=data)


# Satisfaction levels medians are almost the same across all job categories with the exception of accounting and HR.
# 

# In[ ]:


#Who works the longest hours?


median = data.groupby(['sales']).median()
median = pd.DataFrame(median.to_records())
median = median.sort_values(by='average_montly_hours', ascending = False)
median = median['sales']

sns.boxplot('sales', 'average_montly_hours',order=median, data=data)


# In[ ]:


#Is satisfaction level correlated with working hours?

sns.jointplot(x='satisfaction_level', y='average_montly_hours', data=data, alpha=0.1)


# In[ ]:


#Satisfaction vs salary

median = data.groupby(['salary']).median()
median = pd.DataFrame(median.to_records())
median = median.sort_values(by='satisfaction_level', ascending = False)
median = median['salary']

sns.violinplot('salary', 'satisfaction_level',order=median, data=data)


# Satisfaction levels seem fairly equally distributed across the three salary categories but the mean of *low* is slightly lower than for the other two categories.

# # Data Preprocessing

# In this section I will one-hot encode **salary** and **sales** and create a training and testing set. 
# 
# One-hot encoding means that we create a new column for each type of **sales** - HR, accounting etc. will all have their own column with each cell containing either 1 or 0. Each column is now effectively a dummy variable. If an employee works in HR the corresponding cell will be filled with 1 and otherwise with 0.
# 
# One-hot encoding is important for categorical variables that contain values other than 1 and 0. If this isn't done, Python will convert Accounting, HR, Marketing et.c to the numbers 0, 1, 2... which will lead to misleading results.
# 
# Splitting the data into training and testing sets allows us to check how well the model performs on unseen data, the testing set.

# In[ ]:


#Select the variables to be one-hot encoded
one_hot_features = ['salary', 'sales']

# For each categorical column, find the unique number of categories. This tells us how many columns we are adding to the dataset.
longest_str = max(one_hot_features, key=len)
total_num_unique_categorical = 0
for feature in one_hot_features:
    num_unique = len(data[feature].unique())
    print('{col:<{fill_col}} : {num:d} unique categorical values.'.format(col=feature, 
                                                                          fill_col=len(longest_str),
                                                                          num=num_unique))
    total_num_unique_categorical += num_unique
print('{total:d} columns will be added during one-hot encoding.'.format(total=total_num_unique_categorical))


# In[ ]:


# Convert categorical variables into dummy/indicator variables (i.e. one-hot encoding).
one_hot_encoded = pd.get_dummies(data[one_hot_features])
one_hot_encoded.info(verbose=True, memory_usage=True, null_counts=True)


# In[ ]:


#Let's check everything looks like as we were expecting
one_hot_encoded.head()


# In[ ]:


#Delete the columns salary and sales...
data = data.drop(['salary', 'sales'], 1)

#...and add the new one-hot encoded variables
data = pd.concat([data, one_hot_encoded], axis=1)
data.head()


# In[ ]:


#Split data into training and testing set with 80% of the data going into training
training, testing = train_test_split(data, test_size=0.2, random_state=0)
print("Total sample size = %i; training sample size = %i, testing sample size = %i"     %(data.shape[0],training.shape[0],testing.shape[0]))


# In[ ]:


#This creates a list with all column names, which will be used to subset the tables 
cols = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident',
'promotion_last_5years', 'salary_high', 'salary_low', 'salary_medium', 'sales_IT', 'sales_RandD', 'sales_accounting',
'sales_hr', 'sales_management', 'sales_marketing', 'sales_product_mng', 'sales_sales', 'sales_support', 'sales_technical']
tcols = np.append(['left'],cols)

#X are the variables/features that help predict y, which tells us whether an employee left or stayed. This is done for both 
#training and testing
df = training.loc[:,tcols]
X = df.loc[:,cols]
y = np.ravel(df.loc[:,['left']])

df_test = testing.loc[:,tcols]
X_test = df_test.loc[:,cols]
y_test = np.ravel(df_test.loc[:,['left']])


# # Data Models

# In this section I will first calculate the baseline, which is the accuracy of predicting the most frequent class, which is 76.2% for *stayed*.
# 
# I then run several models, which have to beat an accuracy of 76.2 in order to be an improvement to prediction. After I ran all models I collate the accuracy score and present my findings.

# In[ ]:


#Baseline
print('Left: %i (%.1f percent), Stayed: %i (%.1f percent), Total: %i'     %(len(left), 1.*len(left)/len(data)*100.0,      len(stayed), 1.*len(stayed)/len(data)*100.0, len(data)))
base_score = len(stayed)/len(data)
print('This is the score to beat:', base_score)


# In[ ]:


#Logistic Regression
clf_log = LogisticRegression()
clf_log = clf_log.fit(X,y)
score_log = cross_val_score(clf_log, X, y, cv=5).mean()
print(score_log)


# In[ ]:


# Perceptron

clf_pctr = Perceptron(
    class_weight='balanced'
    )
clf_pctr = clf_pctr.fit(X,y)
score_pctr = cross_val_score(clf_pctr, X, y, cv=5).mean()
print(score_pctr)


# In[ ]:


k_range = range(1,26)
scores=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X,y)
    y_pred=knn.predict(X_test)
    scores.append(accuracy_score(y_test,y_pred))

plt.plot(k_range,scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Testing accuracy')


# In[ ]:


# KNN - based on the graph n_neighbors should be 1 or 2 but 10 seems to give a better result

clf_knn = KNeighborsClassifier(
    n_neighbors=10,
    weights='distance'
    )
clf_knn = clf_knn.fit(X,y)
score_knn = cross_val_score(clf_knn, X, y, cv=5).mean()
print(score_knn)


# In[ ]:


#SVM

clf_svm = svm.SVC(
    class_weight='balanced'
    )
clf_svm.fit(X, y)
score_svm = cross_val_score(clf_svm, X, y, cv=5).mean()
print(score_svm)


# In[ ]:


# Bagging

bagging = BaggingClassifier(
    KNeighborsClassifier(
        n_neighbors=10,
        weights='distance'
        ),
    oob_score=True,
    max_samples=0.5,
    max_features=1.0
    )
clf_bag = bagging.fit(X,y)
score_bag = clf_bag.oob_score_
print(score_bag)


# In[ ]:


# Decision Tree

clf_tree = tree.DecisionTreeClassifier(
    #max_depth=3,
    class_weight="balanced",
    min_weight_fraction_leaf=0.01
    )
clf_tree = clf_tree.fit(X,y)
score_tree = cross_val_score(clf_tree, X, y, cv=5).mean()
print(score_tree)


# In[ ]:


# Random Forest

clf_rf = RandomForestClassifier(
    n_estimators=1000, 
    max_depth=None, 
    min_samples_split=10 
    #class_weight="balanced", 
    #min_weight_fraction_leaf=0.02 
    )
clf_rf = clf_rf.fit(X,y)
score_rf = cross_val_score(clf_rf, X, y, cv=5).mean()
print(score_rf)


# In[ ]:


# Extremely Randomised Trees

clf_ext = ExtraTreesClassifier(
    max_features='auto',
    bootstrap=True,
    oob_score=True,
    n_estimators=1000,
    max_depth=None,
    min_samples_split=10
    #class_weight="balanced",
    #min_weight_fraction_leaf=0.02
    )
clf_ext = clf_ext.fit(X,y)
score_ext = cross_val_score(clf_ext, X, y, cv=5).mean()
print(score_ext)


# In[ ]:


# Gradient Boosting

clf_gb = GradientBoostingClassifier(
            #loss='exponential',
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=3,
            subsample=0.5,
            random_state=0).fit(X, y)
clf_gb.fit(X,y)
score_gb = cross_val_score(clf_gb, X, y, cv=5).mean()
print(score_gb)


# In[ ]:


# Ada Boost

clf_ada = AdaBoostClassifier(n_estimators=400, learning_rate=0.1)
clf_ada.fit(X,y)
score_ada = cross_val_score(clf_ada, X, y, cv=5).mean()
print(score_ada)


# In[ ]:


#eXtreme Gradient Boosting

clf_xgb = xgb.XGBClassifier(
    max_depth=2,
    n_estimators=500,
    subsample=0.5,
    learning_rate=0.1
    )
clf_xgb.fit(X,y)
score_xgb = cross_val_score(clf_xgb, X, y, cv=5).mean()
print(score_xgb)


# The results are in and tabulated below and shown graphically. We can boost prediction accuracy from 76.2% to 98% using Gradient Boosting or Random Forest, which is a significant increase. This is the cross-validation score using the training set only. In the next section I will test accuracy using the trained models on the unseen testing set.

# In[ ]:


models = pd.DataFrame({
        'Model' : ['Baseline', 'Logistic Regression', 'Perceptron', 'KNN', 'SVM', 'Bagging', 'Decision Tree',
                   'Random Forest', 'Extra Tree', 'Gradient Boosting', 'ADA Boosting', 'XGBoost'],
        'Score' : [base_score, score_log, score_pctr, score_knn, score_svm, score_bag, score_tree, score_rf, score_ext, score_gb, 
                    score_ada, score_xgb]  
})

models = models.sort_values(by='Score', axis=0, ascending=False, inplace=False, kind='quicksort', na_position='last')

models


# In[ ]:


sns.barplot(y=models.Model, x=models.Score)


# # Model Validation on test set

# We get similar scores using the test set

# In[ ]:


score_log_test = clf_log.score(X_test, y_test)
score_pctr_test = clf_pctr.score(X_test, y_test)
score_knn_test = clf_knn.score(X_test, y_test)
score_svm_test = clf_svm.score(X_test, y_test)
score_bag_test = clf_bag.score(X_test, y_test)
score_tree_test = clf_tree.score(X_test, y_test)
score_rf_test = clf_rf.score(X_test, y_test)
score_ext_test = clf_ext.score(X_test, y_test)
score_gb_test = clf_gb.score(X_test, y_test)
score_ada_test = clf_ada.score(X_test, y_test)
score_xgb_test = clf_xgb.score(X_test, y_test)


# In[ ]:


models_test = pd.DataFrame({
        'Model' : ['Baseline', 'Logistic Regression', 'Perceptron', 'KNN', 'SVM', 'Bagging', 'Decision Tree',
                   'Random Forest', 'Extra Tree', 'Gradient Boosting', 'ADA Boosting', 'XGBoost'],
        'Score' : [base_score, score_log_test, score_pctr_test, score_knn_test, score_svm_test, score_bag_test, score_tree_test,
                   score_rf_test, score_ext_test, score_gb_test, 
                    score_ada_test, score_xgb_test]  
})

models_test = models_test.sort_values(by='Score', axis=0, ascending=False, inplace=False, kind='quicksort', na_position='last')

models_test


# In[ ]:


sns.barplot(y=models_test.Model, x=models_test.Score)


# # Feature Importance using Random Forest as Example

# The models are doing a great job at predicting who is going to leave or stay but they are a bit of a black box. So, what factors are influencing the decision of an employee to leave. This is useful information for the company so that they can tackle their turn-over and focus on the right factors.
# 
# As an example, I will investigate what factors were the most important in predicting the outcome in Random Forest.  

# In[ ]:


importance = pd.DataFrame(list(zip(X.columns, np.transpose(clf_rf.feature_importances_)))             ).sort_values(1, ascending=False)
importance


# In[ ]:


importances = clf_rf.feature_importances_

std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],  
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]),X.columns[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()

