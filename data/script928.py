
# coding: utf-8

# # *Women and kids first! (c) Titanic*
# 

# ![Titanic](https://www.usnews.com/dims4/USNEWS/4f3cd50/2147483647/thumbnail/970x647/quality/85/?url=http%3A%2F%2Fmedia.beam.usnews.com%2F0e%2Fe187dd2f8f1fe5be9058fa8eef419e%2F7018FE_DA_080929titanic.jpg)

# # Visualization of titanic dataset
# This notebook presents a profound exploratory analysis of the dataset in order to provide understanding of the dependencies and interesting facts. Simple Logistic regression was used to perform classification.
# 
# Four ML techniques are used to do prediction: RandomForest, LogisticRegression, KNeighbours and the Ensemble.
# 
# Logistic Regression performed the best with a score of 0.80383.
# 
# 
# 
# *****************
# **I will happy to hear some remarks or suggestions and feel free to upvote if you like it :)**
# 
# **Have fun with the data!**
# *****************

# In[ ]:


import pandas as pd
import numpy as np
import collections, re
import copy

from pandas.tools.plotting import scatter_matrix
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('bmh')

get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from sklearn.grid_search import GridSearchCV

pd.set_option('display.max_columns', 500)
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.info()


# # 1. Exploratory analysis
# ## Basic Information about the table

# In[ ]:


train.head(2)


# In[ ]:


train.describe()


# Average Age is 29 years and ticket price is 32.
# As there are 681 unique tickets and there is no way to extract less detailed information we exclude this variable. There are 891 unique names but we could take a look on the title of each person to understand if the survival rate of people from high society was higher

# In[ ]:


train.describe(include=['O'])


# In[ ]:


## exctract cabin letter
def extract_cabin(x):
    return x!=x and 'other' or x[0]
train['Cabin_l'] = train['Cabin'].apply(extract_cabin)


# ## 1.1 Superficial overview of each variable

# Just a quick look on variables we are dealing with.

# In[ ]:


plain_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Cabin_l']
fig, ax = plt.subplots(nrows = 2, ncols = 3 ,figsize=(20,10))
start = 0
for j in range(2):
    for i in range(3):
        if start == len(plain_features):
            break
        sns.barplot(x=plain_features[start], y='Survived', data=train, ax=ax[j,i])
        start += 1


# #### A citate from a movie: 'Children and women first'. 
# * Sex: Survival chances of women are higher.
# * Pclass: Having a first class ticket is beneficial for the survival.
# * SibSp and Parch: middle size families had higher survival rate than the people who travelled alone or big families. The reasoning might be that alone people would want to sacrifice themselves to help others. Regarding the big families I would explain that it is hard to manage the whole family and therefore people would search for the family members insetad of getting on the boat.
# * Embarked C has a higher survival rate. It would be interesting to see if, for instance, the majority of Pclass 1 went on board in embarked C.

# ## 1.2 Survival by Sex and Age

# In[ ]:


sv_lab = 'survived'
nsv_lab = 'not survived'
fig, ax = plt.subplots(figsize=(5,3))
ax = sns.distplot(train[train['Survived']==1].Age.dropna(), bins=20, label = sv_lab, ax = ax)
ax = sns.distplot(train[train['Survived']==0].Age.dropna(), bins=20, label = nsv_lab, ax = ax)
ax.legend()
_ = ax.set_ylabel('KDE')

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
females = train[train['Sex']=='female']
males = train[train['Sex']=='male']

ax = sns.distplot(females[females['Survived']==1].Age.dropna(), bins=30, label = sv_lab, ax = axes[0], kde =False)
ax = sns.distplot(females[females['Survived']==0].Age.dropna(), bins=30, label = nsv_lab, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(males[males['Survived']==1].Age.dropna(), bins=30, label = sv_lab, ax = axes[1], kde = False)
ax = sns.distplot(males[males['Survived']==0].Age.dropna(), bins=30, label = nsv_lab, ax = axes[1], kde = False)
ax.legend()
_ = ax.set_title('Male')


# * Survival rate of boys is higher than of the adult men. However, the same fact does not hold for the girls. and between 13 and 30 is lower. Take it into consideration while engineering the variable: we could specify a categorical variable as young and adult.
# * For women the survival chances are higher between 14 and 40 age. For men of the same age the survival chances are flipped.

# ## 1.3 Survival by Class,  Embarked and Fare.

# ## 1.3.1 Survival by Class and Embarked

# In[ ]:


_ = sns.factorplot('Pclass', 'Survived', hue='Sex', col = 'Embarked', data=train)
_ = sns.factorplot('Pclass', 'Survived', col = 'Embarked', data=train)


# * As noticed already before, the class 1 passangers had a higher survival rate.
# * All women who died were from the 3rd class. 
# * Embarked in Q as a 3rd class gave you slighly better survival chances than embarked in S for the same class.
# * In fact, there is a very high variation in survival rate in embarked Q among 1st and 2nd class. The third class had the same survival rate as the 3rd class embarked C. We will exclude this variable embarked Q. From crosstab we see that there were only 5 passengers in embarked Q with the 1st and 2nd class. That explains large variation in survival rate and a perfect separation of men and women in Q.

# In[ ]:


tab = pd.crosstab(train['Embarked'],train['Pclass'])
print(tab)
tab_prop = tab.div(tab.sum(1).astype(float), axis=0)
tab_prop.plot(kind="bar", stacked=True)


# ## 1.3.2 Fare and class distribution

# In[ ]:


ax = sns.boxplot(x="Pclass", y="Fare", hue="Survived", data=train)
ax.set_yscale('log')


# * It appears that the higher the fare was in the first class the higher survival chances a person from the 1st had.

# ## 1.3.3 Class and age distribution

# In[ ]:


_ = sns.violinplot(x='Pclass', y='Age', hue = 'Survived', data=train, split=True)


# * Interesting note that Age decreases proportionally with the Pclass, meaning most old passangers are from 1st class. We will construct a new feature Age*Class to intefere the this findig. 
# * The younger people from 1st had higher survival chanches than older from the same class.
# * Majority (from the 3rd class) and most children from the 2nd class survived.

# ## 1.4 Survival rate regarding the family members

# In[ ]:


# To get the full family size of a person, added siblings and parch.
#fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(15, 5))
train['family_size'] = train['SibSp'] + train['Parch'] + 1 
test['family_size'] = test['SibSp'] + test['Parch'] + 1 
axes = sns.factorplot('family_size','Survived', 
                      hue = 'Sex', 
                      data=train, aspect = 4)


# Assumption: the less people was in your family the faster you were to get to the boat. The more people they are the more managment is required. However, if you had no family members you might wanted to help others and therefore sacrifice.
# 
# * The females traveling with up to 2 more family members had a higher chance to survive. However, a high variation of survival rate appears once family size exceeds 4 as mothers/daughters would search longer for the members and therefore the chanes for survival decrease.
# * Alone men might want to sacrifice and help other people to survive. 

# ## 1.5 Survival rate by the title
# * Barplots show that roalties had normally 1st or 2nd class tickets. However, people with the title Master had mostly 3rd class. In fact, a title 'Master' was given to unmarried boys. You can see that the age of of people with this title is less than 13.
# * Women and roalties had higher survival rate. (There are only two titlted women in the train class and both have survived, I would put them into Mrs class)
# * The civils and reverends a lower one due to the fact that they had/wanted to help people.

# In[ ]:


train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
print(collections.Counter(train['Title']).most_common())
test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
print()
print(collections.Counter(test['Title']).most_common())


# In[ ]:


tab = pd.crosstab(train['Title'],train['Pclass'])
print(tab)
tab_prop = tab.div(tab.sum(1).astype(float), axis=0)
tab_prop.plot(kind="bar", stacked=True)


# Investigate who were masters. The age is less than 12.

# In[ ]:


max(train[train['Title']== 'Master'].Age)


# In[ ]:


_ = sns.factorplot('Title','Survived', data=train, aspect = 3)


# We will group the roalties and assign masters to Mr and due to the fact that there were not so many roaly women, we will assign then to Mrs.

# In[ ]:


#train['Title'].replace(['Master','Major', 'Capt', 'Col', 'Countess','Dona','Lady', 'Don', 'Sir', 'Jonkheer', 'Dr'], 'titled', inplace = True)
train['Title'].replace(['Master','Major', 'Capt', 'Col','Don', 'Sir', 'Jonkheer', 'Dr'], 'titled', inplace = True)
#train['Title'].replace(['Countess','Dona','Lady'], 'titled_women', inplace = True)
#train['Title'].replace(['Master','Major', 'Capt', 'Col','Don', 'Sir', 'Jonkheer', 'Dr'], 'titled_man', inplace = True)
train['Title'].replace(['Countess','Dona','Lady'], 'Mrs', inplace = True)
#train['Title'].replace(['Master'], 'Mr', inplace = 'True')
train['Title'].replace(['Mme'], 'Mrs', inplace = True)
train['Title'].replace(['Mlle','Ms'], 'Miss', inplace = True)


# In[ ]:


g = sns.factorplot('Title','Survived', data=train, aspect = 3)


# ## 1.6 Survival rate by cabin
# Cabin is supposed to be less distingushing, also taking into consideration that most of the values are missing.

# In[ ]:


def extract_cabin(x):
    return x!=x and 'other' or x[0]
train['Cabin_l'] = train['Cabin'].apply(extract_cabin)
print(train.groupby('Cabin_l').size())
sns.factorplot('Cabin_l','Survived', 
               order = ['other', 'A','B', 'C', 'D', 'E', 'F', 'T' ], 
               aspect = 3, 
               data=train)


# ## 1.7 Correlation of the variables
# * Pclass is slightly correlated with Fare as logically, 3rd class ticket would cost less than the 1st class.
# * Pclass is also slightly correlated with Survived
# * SibSp and Parch are weakly correlated as basically they show how big the family size is.
# 

# In[ ]:


plt.figure(figsize=(8, 8))
corrmap = sns.heatmap(train.drop('PassengerId',axis=1).corr(),square=True, annot=True)


# ## 2. FEATURE SELECTION AND ENGINEERING
# ## 2.1 Impute values
# First, we check how many nas there is in general. If there is only small amount then we can just exclude those individuals. Considering that there are 891 training samples, 708 do not have missing values. 183 samples have na values. It is better to impute. There are different techniques one can impute the values.

# In[ ]:


train.shape[0] - train.dropna().shape[0]


# Check wich columns to impute in which set. It shows the number of na-values in each column.

# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# Embarked: fill embarked with a major class

# In[ ]:


max_emb = np.argmax(train['Embarked'].value_counts())
train['Embarked'].fillna(max_emb, inplace=True)


# Pclass: because there is only one missing value in Fare we will fill it with a median of the corresponding Pclass

# In[ ]:


indz = test['Fare'].index[test['Fare'].apply(np.isnan)].tolist
print(indz)
pclass = test['Pclass'][152]
fare_test = test[test['Pclass']==pclass].Fare.dropna()
fare_train = train[train['Pclass']==pclass].Fare
fare_med = (fare_test + fare_train).median()
print(fare_med)
test.loc[152,'Fare'] = fare_med


# There are several imputing techniques, we will use the random number from the range mean +- std 

# In[ ]:


ages = np.concatenate((test['Age'].dropna(), train['Age'].dropna()), axis=0)
std_ages = ages.std()
mean_ages = ages.mean()
train_nas = np.isnan(train["Age"])
test_nas = np.isnan(test["Age"])
np.random.seed(122)
impute_age_train  = np.random.randint(mean_ages - std_ages, mean_ages + std_ages, size = train_nas.sum())
impute_age_test  = np.random.randint(mean_ages - std_ages, mean_ages + std_ages, size = test_nas.sum())
train["Age"][train_nas] = impute_age_train
test["Age"][test_nas] = impute_age_test
ages_imputed = np.concatenate((test["Age"],train["Age"]), axis = 0)


# In[ ]:


train['Age*Class'] = train['Age']*train['Pclass']
test['Age*Class'] = test['Age']*test['Pclass']


# Check if we disrupted the distribution somehow.

# In[ ]:


_ = sns.kdeplot(ages_imputed, label = 'After imputation')
_ = sns.kdeplot(ages, label = 'Before imputation')


# ## 2.2 ENGENEER VALUES

# Integrate into test the title feature

# In[ ]:


test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

test['Title'].replace(['Master','Major', 'Capt', 'Col','Don', 'Sir', 'Jonkheer', 'Dr'], 'titled', inplace = True)
test['Title'].replace(['Countess','Dona','Lady'], 'Mrs', inplace = True)
#test['Title'].replace(['Master'], 'Mr', inplace = True)
test['Title'].replace(['Mme'], 'Mrs', inplace = True)
test['Title'].replace(['Mlle','Ms'], 'Miss', inplace = True)


# Seperate young and adult people

# In[ ]:


train['age_cat'] = None
train.loc[(train['Age'] <= 13), 'age_cat'] = 'young'
train.loc[ (train['Age'] > 13), 'age_cat'] = 'adult'

test['age_cat'] = None
test.loc[(test['Age'] <= 13), 'age_cat'] = 'young'
test.loc[(test['Age'] > 13), 'age_cat'] = 'adult'


# Drop broaden variables. As we have seen from describe there are too many unique values for Ticket and missing values for Cabin

# In[ ]:


train_label = train['Survived']
test_pasId = test['PassengerId']
drop_cols = ['Name','Ticket', 'Cabin', 'SibSp', 'Parch', 'PassengerId']
train.drop(drop_cols + ['Cabin_l'], 1, inplace = True)
test.drop(drop_cols, 1, inplace = True)


# Convert Pclass into categorical variable

# In[ ]:


train['Pclass'] = train['Pclass'].apply(str)
test['Pclass'] = test['Pclass'].apply(str)


# Create dummy variables for categorical data.

# In[ ]:


train.drop(['Survived'], 1, inplace = True)
train_objs_num = len(train)
dataset = pd.concat(objs=[train, test], axis=0)
dataset = pd.get_dummies(dataset)
train = copy.copy(dataset[:train_objs_num])
test = copy.copy(dataset[train_objs_num:])


# In[ ]:


droppings = ['Embarked_Q', 'Age']
#droppings += ['Sex_male', 'Sex_female']

test.drop(droppings, 1, inplace = True)
train.drop(droppings,1, inplace = True)


# In[ ]:


train.head(5)


# ## CLASSIFICATION
# 

# In[ ]:


def prediction(model, train, label, test, test_pasId):
    model.fit(train, label)
    pred = model.predict(test)
    accuracy = cross_val_score(model, train, label, cv = 5)

    sub = pd.DataFrame({
            "PassengerId": test_pasId,
            "Survived": pred
        })    
    return [model, accuracy, sub]


# ## 1. Random Forest
# There are many categorical features, so I have chosen random forest to do the classification.

# In[ ]:


rf = RandomForestClassifier(n_estimators=80, min_samples_leaf = 2, min_samples_split=2, random_state=110)
acc_random_forest = prediction(rf, train, train_label, test, test_pasId)
importances = pd.DataFrame({'feature':train.columns,'importance':np.round(rf.feature_importances_,3)})
importances = importances.sort_values('importance', ascending=False).set_index('feature')
#acc_random_forest[2].to_csv('~/Desktop/random_forest.txt', index=False)
print (importances)
importances.plot.bar()
print(acc_random_forest[1])

test_predictions = acc_random_forest[0].predict(test)
test_predictions = test_predictions.astype(int)
submission = pd.DataFrame({
        "PassengerId": test_pasId,
        "Survived": test_predictions
    })

submission.to_csv("titanic_submission_randomforest.csv", index=False)


# ## 2. Logistic Regression
# 

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(train['Fare'].values.reshape(-1, 1))
train['Fare'] = scaler.transform(train['Fare'].values.reshape(-1, 1)) 
test['Fare'] = scaler.transform(test['Fare'].values.reshape(-1, 1))  

scaler = StandardScaler().fit(train['Age*Class'].values.reshape(-1, 1))
train['Age*Class'] = scaler.transform(train['Age*Class'].values.reshape(-1, 1)) 
test['Age*Class'] = scaler.transform(test['Age*Class'].values.reshape(-1, 1))  



lr  = LogisticRegression(random_state=110)
acc = prediction(lr, train, train_label, test, test_pasId)
print(acc[1])

test_predictions = acc[0].predict(test)
test_predictions = test_predictions.astype(int)
submission = pd.DataFrame({
        "PassengerId": test_pasId,
        "Survived": test_predictions
    })
submission.to_csv("titanic_submission_logregres.csv", index=False)

#train.columns.tolist()
print(list(zip(acc[0].coef_[0], train.columns.tolist())))


# ## 3. KNeighbours

# In[ ]:


kn = KNeighborsClassifier()
acc = prediction(kn, train, train_label, test, test_pasId)
print(acc[1])
test_predictions = acc[0].predict(test)
test_predictions = test_predictions.astype(int)
submission = pd.DataFrame({
        "PassengerId": test_pasId,
        "Survived": test_predictions
    })
submission.to_csv("titanic_submission_kn.csv", index=False)


# ## 4. Ensemble
# 

# In[ ]:


from sklearn.ensemble import VotingClassifier
eclf1 = VotingClassifier(estimators=[
        ('lr', lr), ('rf', rf)], voting='soft')
eclf1 = eclf1.fit(train, train_label)
test_predictions = eclf1.predict(test)
test_predictions = test_predictions.astype(int)
submission = pd.DataFrame({
        "PassengerId": test_pasId,
        "Survived": test_predictions
    })

submission.to_csv("titanic_submission.csv", index=False)


# ## 5. Grid Search
# Find optimal parameters for the logistic regression.

# In[ ]:


def grid_search(clf, X, Y, parameters, cv):
    grid_model = GridSearchCV(estimator=clf, param_grid=parameters, cv=cv)
    grid_model.fit(X, Y)
    #grid_model.cv_results_
    print("Best Score:", grid_model.best_score_," / Best parameters:", grid_model.best_params_)
    return grid_model.best_params_


# In[ ]:


param_range = np.logspace(-6, 5, 12)
parameters = dict(C= param_range, penalty = ['l1', 'l2'])
grid_search(lr, train, train_label, parameters, 5)


# In[ ]:


lr  = LogisticRegression(random_state=110, penalty= 'l1', C= 100)
acc = prediction(lr, train, train_label, test, test_pasId)
print(acc[1])

test_predictions = acc[0].predict(test)
test_predictions = test_predictions.astype(int)
submission = pd.DataFrame({
        "PassengerId": test_pasId,
        "Survived": test_predictions
    })
submission.to_csv("titanic_submission_logregres_tuned_scaled.csv", index=False)


# **********
# I will be happy to hear remarks or comments. If you liked the Kernel, please upvote :)
#     
# Have fun with the data!
# *****
