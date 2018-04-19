
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns


# In[ ]:


df = pd.read_csv('../input/HR_comma_sep.csv')


# In[ ]:


df.info()


# In[ ]:


df.head()


# In[ ]:


df['sales'].unique()


# In[ ]:


df['promotion_last_5years'].unique()


# In[ ]:


df['salary'].unique()


# In[ ]:


df.mean()


# ### tell me the daily working hour

# In[ ]:


df.mean()['average_montly_hours']/30


# ## tell me the # of people who has left

# In[ ]:


print('# of people left = {}'.format(df[df['left']==1].size))
print('# of people stayed = {}'.format(df[df['left']==0].size))
print('protion of people who left in 5 years = {}%'.format(int(df[df['left']==1].size/df.size*100)))


# ### 1, Corelation Matrix overall

# In[ ]:


corrmat = df.corr()
f, ax = plt.subplots(figsize=(4, 4))
# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()


# ### As expected:
# 
# * 1, The score of evaluation and satisfaction_level are highly correlated,  and the less left
# * 2, The more number_project in hands, the more average_montly_hours, and this result in a higher score of evaluation but makes employees less satisfied. 
# And they spend more time in company, btw.
# * 3, Being promoted(aka. level up) makes poeple happier, doing more job and being less likely to run away

# ### 2, Corelation Matrix by salaries

# In[ ]:


corrmat_low = df[df['salary'] == 'low'].corr()
corrmat_medium = df[df['salary'] == 'medium'].corr()
corrmat_high = df[df['salary'] == 'high'].corr()

sns.heatmap(corrmat_low, vmax=.8, square=True,annot=True,fmt='.2f')


# In[ ]:


sns.heatmap(corrmat_medium, vmax=.8, square=True,annot=True,fmt='.2f')


# In[ ]:


sns.heatmap(corrmat_high, vmax=.8, square=True,annot=True,fmt='.2f')


# ### Even though I print out the correlation digits, it's still hard to sell how salary affect people's mentality

# In[ ]:


df_low = df[df['salary'] == 'low']
df_medium = df[df['salary'] == 'medium']
df_high = df[df['salary'] == 'high']

print('# of low salary employees= ',df_low.shape[0])
print('# of medium salary employees= ',df_medium.shape[0])
print('# of high salary employees= ',df_high.shape[0])


# In[ ]:


fmt = '{:<22}{:<25}{}'

print(fmt.format('', 'mean', 'std'))
for i, (mean, std) in enumerate(zip(df_low.mean(), df_low.std())):
    print(fmt.format(df_low.columns[i], mean, std))
print('\n')
for i, (mean, std) in enumerate(zip(df_medium.mean(), df_medium.std())):
    print(fmt.format(df_low.columns[i], mean, std))
print('\n')
for i, (mean, std) in enumerate(zip(df_high.mean(), df_high.std())):
    print(fmt.format(df_low.columns[i], mean, std))


# ### Now it's apparent that:
# 
# * high salary employees spend more time in company but less monthly working hours than the others.
# * high salary employees have been promoted more and have felt more satisfied.
# * high salary employees tend to choose stay rather than left.
# * high salary employees make a little bit more work accidents than the others.

# ### Sales by Salary all feature plot

# In[ ]:


sns.factorplot("sales", col="salary", col_wrap=4, data=df, kind="count", size=10, aspect=.8)


# ### Satisfaction_level by Sales

# In[ ]:


df.groupby('sales').mean()['satisfaction_level'].plot(kind='bar',color='r')


# ### Accountants are the most unhappy employees:(
# ### And I've plenty of accounting friends. Gotta let them know:)

# ### 1, Predict 'left == 1' by the other features

# In[ ]:


from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn import svm


# In[ ]:


df_copy = pd.get_dummies(df)
df_copy.head()


# In[ ]:


df1 = df_copy
y = df1['left'].values
df1 = df1.drop(['left'],axis=1)
X = df1.values


# In[ ]:


Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.50)


# In[ ]:


log_reg = LogisticRegression()
log_reg.fit(Xtrain, ytrain)
y_val_l = log_reg.predict_proba(Xtest)
print("Validation accuracy: ", sum(pd.DataFrame(y_val_l).idxmax(axis=1).values
                                   == ytest)/len(ytest))


# In[ ]:


sdg = SGDClassifier()
sdg.fit(Xtrain, ytrain)
y_val_l = sdg.predict(Xtest)
print("Validation accuracy: ", sum(y_val_l
                                   == ytest)/len(ytest))


# In[ ]:


radm = RandomForestClassifier()
radm.fit(Xtrain, ytrain)
y_val_l = radm.predict_proba(Xtest)
print("Validation accuracy: ", sum(pd.DataFrame(y_val_l).idxmax(axis=1).values
                                   == ytest)/len(ytest))


# In[ ]:


clf = radm


# ### RadomForest scores so high! It actually make sense because (we make up mind to quit a job by a serial decision making. (aka following a decision tree(?) in our mind))

# In[ ]:


indices = np.argsort(radm.feature_importances_)[::-1]

# Print the feature ranking
print('Feature ranking:')

for f in range(df1.shape[1]):
    print('%d. feature %d %s (%f)' % (f+1 , indices[f], df1.columns[indices[f]],
                                      radm.feature_importances_[indices[f]]))


# ### The above shows what are the primary factors for employees to quit the job.
# 
# * 1, satisfaction_level
# * 2, time_spend_company
# * 3, number_project
# * 4, last_evaluation
# * 5, work_accident
# ### All make sense~

# ## 2, Predict Salary by the other features

# In[ ]:


df_copy = df
y = LabelEncoder().fit(df['salary']).transform(df['salary'])
df2 = df_copy.drop(['salary'],axis=1)
X = pd.get_dummies(df2).values


# In[ ]:


Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)


# In[ ]:


radm = RandomForestClassifier()
radm.fit(Xtrain, ytrain)
y_val_l = radm.predict_proba(Xtest)
print("Validation accuracy: ", sum(pd.DataFrame(y_val_l).idxmax(axis=1).values
                                   == ytest)/len(ytest))


# In[ ]:


log_reg = LogisticRegression()
log_reg.fit(Xtrain, ytrain)
y_val_l = log_reg.predict_proba(Xtest)
print("Validation accuracy: ", sum(pd.DataFrame(y_val_l).idxmax(axis=1).values
                                   == ytest)/len(ytest))


# ### Not so great. It's hard to determine salary just by the data provided.

# ## 3, Predict Sales by the other features

# In[ ]:


df_copy = df
y = LabelEncoder().fit(df['sales']).transform(df['sales'])
df2 = df_copy.drop(['sales'],axis=1)
X = pd.get_dummies(df2).values


# In[ ]:


Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)


# In[ ]:


radm = RandomForestClassifier()
radm.fit(Xtrain, ytrain)
y_val_l = radm.predict_proba(Xtest)
print("Validation accuracy: ", sum(pd.DataFrame(y_val_l).idxmax(axis=1).values
                                   == ytest)/len(ytest))


# In[ ]:


log_reg = LogisticRegression()
log_reg.fit(Xtrain, ytrain)
y_val_l = log_reg.predict_proba(Xtest)
print("Validation accuracy: ", sum(pd.DataFrame(y_val_l).idxmax(axis=1).values
                                   == ytest)/len(ytest))


# ### Even poorer. It makes sense because our data doesn't provide any information on what do employees do.

# ## 4, Predict who will leave soon

# In[ ]:


stay = df[df['left'] == 0]
stay_copy = pd.get_dummies(stay)


# In[ ]:


df1 = stay_copy
y = df1['left'].values
df1 = df1.drop(['left'],axis=1)
X = df1.values


# In[ ]:


pred = clf.predict_proba(X)


# ### tell me the # of employees will definitely leave

# In[ ]:


sum(pred[:,1]==1)


# In[ ]:


stay['will leave the job'] = pred[:,1]


# ### show who will likely to leave with probability greater than or equal to 50%

# In[ ]:


stay[stay['will leave the job']>=0.5]

