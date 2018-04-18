
# coding: utf-8

# **This kernel shows:**
# 
#  - Data exploration with some simple yet useful graphs.
#  - Data cleaning.
#  - Feature creation.
#  - Feature selection.
#  - Model score benchmark.
#  - Train and test.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
from sklearn.metrics import cohen_kappa_score
from sklearn.ensemble import BaggingClassifier

import warnings
warnings.filterwarnings('ignore')


# **I will work with three datasets:**
# 
#  - train: contains the information from train.csv. This one will be used to get statistics and graphs.
#  - test: contains the information from test.csv. This one will be used to get the predicted labels.
#  - data: contains both train and test. This is the one where all data manipulation will be done.

# In[ ]:


# Read the data:
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
data = pd.concat([train,test],ignore_index=True)
labels=train["Survived"]


# In[ ]:


# Functions used in the kernel

# Create a graph that groups, counts and check survivors per group
def survival_rate(column,t):
    df=pd.DataFrame()
    df['total']=train.groupby(column).size()
    df['survived'] = train.groupby(column).sum()['Survived']
    df['percentage'] = round(df['survived']/df['total']*100,2)
    print(df)

    df['survived'].plot(kind=t)
    df['total'].plot(kind=t,alpha=0.5,title="Survivors per "+str(column))
    plt.show()

# If age is less than 1, we return 1. Else, we return the original age.
def normalize_age_below_one(age):
    if age < 1:
        return 1
    else:
        return age

# Group ages in buckets
def group_age(value):
    if value <= 10:
        return "0-10"
    elif value <= 20:
        return "10-20"
    elif value <= 30:
        return "20-30"
    elif value <= 40:
        return "30-40"
    elif value <= 50:
        return "40-50"
    elif value <= 60:
        return "50-60"
    elif value <= 70:
        return "60-70"
    elif value <= 80:
        return "70-80"
    elif value <= 90:
        return "80-90"
    else:
        return "No data"

# Change sex type to integers
def sex(value):
    if value == "male":
        return 0
    else:
        return 1

# Change embarked type to integers
def embarked(value):
    if value == "C":
        return 0
    elif value =="Q":
        return 1
    else:
        return 2

# Clean title and convert to numeric.
data["TitleClean"] = data["Name"].str.extract('(\w*\.)', expand=True)
def title_to_int(value):
    if value == "Capt.":
        return 0
    elif value == "Col.":
        return 1
    elif value == "Countess.":
        return 2
    elif value == "Don.":
        return 3
    elif value == "Dr.":
        return 4
    elif value == "Jonkheer.":
        return 5
    elif value == "Lady.":
        return 6
    elif value == "Major.":
        return 7
    elif value == "Master.":
        return 8
    elif value == "Miss.":
        return 9
    elif value == "Mlle.": #Same as miss
        return 9
    elif value == "Mme.":
        return 11
    elif value == "Mr.":
        return 12
    elif value == "Mrs.":
        return 13
    elif value == "Ms.":
        return 14
    elif value == "Rev.":
        return 15
    elif value == "Sir.":
        return 16
    elif value == "Dona.": # Same as Mrs
        return 13
    else:
        return np.nan
    
# Test a bunch of models. If NL is false, Neural Networks are not tested (they are pretty slow)
def lets_try(NL):
    results={}
    def test_model(clf):
        
        cv = KFold(n_splits=10)
        fbeta_scorer = make_scorer(fbeta_score, beta=1)
        cohen_scorer = make_scorer(cohen_kappa_score)
        accu = cross_val_score(clf, features, labels, cv=cv)
        fbeta = cross_val_score(clf, features, labels, cv=cv,scoring=fbeta_scorer)
        cohen = cross_val_score(clf, features, labels, cv=cv,scoring=cohen_scorer)
        scores=[accu.mean(),fbeta.mean(),cohen.mean()]
        return scores

    # Decision Tree
    clf = tree.DecisionTreeClassifier()
    results["Decision Tree"]=test_model(clf)
    # Logistic Regression
    clf = LogisticRegression()
    results["Logistic Regression"]=test_model(clf)
    # SVM Linear
    clf = svm.LinearSVC()
    results["Linear SVM"]=test_model(clf)
    # SVM RBF
    clf = svm.SVC()
    results["RBF SVM"]=test_model(clf)
    # Gaussian Bayes
    clf = GaussianNB()
    results["Gaussian Naive Bayes"]=test_model(clf)
    # Random Forest
    clf=RandomForestClassifier()
    results["Random Forest"]=test_model(clf)
    # AdaBoost with Decision Trees
    clf=AdaBoostClassifier()
    results["AdaBoost"]=test_model(clf)
    # SGDC
    clf=SGDClassifier()
    results["SGDC"]=test_model(clf)
    # Bagging
    clf=BaggingClassifier()
    results["Bagging"]=test_model(clf)
    # Neural Networks
    if NL:
        clf=MLPClassifier()
        results["Neural Network"]=test_model(clf)
    
    results = pd.DataFrame.from_dict(results,orient='index')
    results.columns=["Accuracy","F-Score", "Cohen Kappa"] 
    results=results.sort(columns=["Accuracy","F-Score", "Cohen Kappa"],ascending=False)
    results.plot(kind="bar",title="Model Scores")
    axes = plt.gca()
    axes.set_ylim([0,1])
    return plt


# # Data Exploration #
# 
# In this section I will do some data exploration to try to find some relations between the survival changes and some of the most important features present in the data.

# ###Data Set's Characteristics###
# 
# The data to analyse includes information from passengers of Titanic, that collided with an iceberg on 15 April 1912. 
# 
# First we get some basic information from the datasets. Things I am interesting in are **rows**, **columns** and **cells without data**. This will help us to have some interesting information like:
# 
#  1. The size of our dataset.
#  2. The different features we have.
#  3. How much data is missing from each feature.
# 
# For future reference, these are column's descriptions:
# 
#     survival        Survival
#                     (0 = No; 1 = Yes)
#     pclass          Passenger Class
#                     (1 = 1st; 2 = 2nd; 3 = 3rd)
#     name            Name
#     sex             Sex
#     age             Age
#     sibsp           Number of Siblings/Spouses Aboard
#     parch           Number of Parents/Children Aboard
#     ticket          Ticket Number
#     fare            Passenger Fare
#     cabin           Cabin
#     embarked        Port of Embarkation
#                     (C = Cherbourg; Q = Queenstown; S = Southampton)

# This is the information we have in the training data set.

# In[ ]:


# Count the number of rows
print("*** Number of rows: " + str(train.shape[0]))
total = train.shape[0]
print("\n")

# List all the columns
print("*** Columns: " + str(train.columns.values), end="\n")

# Count the number of NaNs each column has.
print("\n*** NaNs per column:")
print(pd.isnull(train).sum())


# ###Passenger's Gender###
# 
# The following section checks the **gender distribution** and the **survival percentage**. Data output:
# 
#  1. Passengers per gender and the percentage over total number of passengers
#  2. Passengers that survived per gender and the percentage over the total number of passengers per gender

# In[ ]:


# Change gender's text to integers
data["Sex"] = data["Sex"].apply(sex)

# Draw survival per sex
survival_rate("Sex","barh")


# ###Passengers' Class###
# 
# The following section checks the **class distribution** and the **survival percentage**.  Data output:
# 
#  1. Passengers per class and the percentage over total number of passengers.
#  2. Passengers that survived per class and the percentage over the total number of passengers per class.

# In[ ]:


# Draw survival per Class
survival_rate("Pclass","barh")


# ###Passengers' Age###
# 
# The following section checks the **age distribution**. First some data manipulation will be done:
# 
#  1. The data set contain ages less than 1 for those people with only months of life. Those will be changed to 1.
#  2. Create new column with buckets of ages, in 10 years ranges.
# 
# Then, the usual graph representation showing the ages in groups and the survival rate.

# In[ ]:


print("*** Number of people with age less than 1 (months):")
print(train[train["Age"] < 1.0].shape[0])

# Those with age <1, changed to 1
data['Age'] = data['Age'].apply(normalize_age_below_one)

# Create new feature with data in buckets
data["AgeGroup"] = data["Age"].apply(group_age)
train["AgeGroup"] = train["Age"].apply(group_age)

# Draw survival per age group
survival_rate("AgeGroup","bar")


# ###Passengers' Fare###
# 
# The following section checks the **fare distribution** and the **average fare per class**. The analysis shows that fare is mostly related with "Class", so no need to check the survival rate since survival per class has been already analysed.

# In[ ]:


# Get Fare statistics
print("*** Fare statistics:")
print(train["Fare"].describe())

# Seems that some people paid nothing:
print("\n*** People with fare 0:")
nothing = train[train["Fare"] == 0]
print(nothing[["Name","Sex","Age","Pclass","Survived"]])

# Graph average Fare per Class
train.groupby("Pclass").mean()['Fare'].plot(kind="bar",title="Average Fare per Class")
plt.show()


# ## Passengers' Port of Embarkation ##
# 
# The following section checks the **port distribution** and the **survival percentage**.

# In[ ]:


# Change embarkation data type to integers
data["Embarked"] = data["Embarked"].apply(embarked)

# Graph survived per port of embarkation
survival_rate("Embarked","bar")


# ## Family members ##
# 
# The following section checks the **family distribution**. First some data manipulation will be done:
# 
#  - New column, **FamilyMembers** will be added. It counts the number of family members of that particular passenger.
# 
# I will also check large families (more than 5 members) and see if there are families where no member survived at all.
# 
# Then, the usual graph representation showing the family members and the survival rate.

# In[ ]:


data["FamilyMembers"]=data["SibSp"]+data["Parch"]
train["FamilyMembers"]=train["SibSp"]+data["Parch"]

print("*** Family statistics, members:")
print("Min: " + str(train["FamilyMembers"].min()))
print("Average: " + str(round(train["FamilyMembers"].mean(),2)))
print("Max: " + str(train["FamilyMembers"].max()), end="\n\n")

print("*** Average family members per Class:")
print(train.groupby("Pclass").mean()['FamilyMembers'], end="\n\n")

# Families with more than 5 members
large_families=train[train["FamilyMembers"]>= 5]
large_families_by_ticket=large_families.groupby("Ticket").sum()['Survived']
print("*** Large families by ticket. Did all family die?:")
print(large_families_by_ticket==0, end="\n\n")

# Largest family where all members died
largest_family_ticket=train["Ticket"][train["FamilyMembers"]==10].iloc[0]
name=train["Name"][train["Ticket"]==largest_family_ticket].iloc[0]
print("*** Largest family, all members died: "+ name.split(",")[0], end="\n\n")
# More info: http://www.bbc.com/news/uk-england-cambridgeshire-17596264

survival_rate("FamilyMembers","bar")


# ## Passenger's Tickets ##
# 
# As we have seen in previous section, tickets hide some information that could be valuable. Every member of the same family has the same ticket number. That means that we can use that information to in some way group people by family.
# 
# There is no single ticket identification, but the most common one are numbers. Therefore, a regex will help is to get those and store in a new column.

# In[ ]:


train["Ticket"].head()


# In[ ]:


data["TicketClean"] = data["Ticket"].str.extract('(\d{2,})', expand=True)
data["TicketClean"].head()


# There are 8 NaN tickets because they didn't have a number in them. We are going to manually assign those.

# In[ ]:


print("Rows with NaN: " + str(pd.isnull(data["TicketClean"]).nonzero()[0]))
print("Ticket number: ")
print(str(data["Ticket"].ix[179]))
print(str(data["Ticket"].ix[271]))
print(str(data["Ticket"].ix[302]))
print(str(data["Ticket"].ix[597]))
print(str(data["Ticket"].ix[772]))
print(str(data["Ticket"].ix[841]))
print(str(data["Ticket"].ix[1077]))
print(str(data["Ticket"].ix[1193]))


# To cleanup the data I am going to:
# 
#  - Convert "TicketClean" to number.
#  - Assign median value to the first group.
#  - Assign median+std() to the second group.
#  - Assign median-std() to the third group.
#  - Then manually update the values.

# In[ ]:


data["TicketClean"] = data["Ticket"].str.extract('(\d{3,})', expand=True)
data["TicketClean"] = data["TicketClean"].apply(pd.to_numeric)
med1=data["TicketClean"].median()
med2=data["TicketClean"].median()+data["TicketClean"].std()
med3=data["TicketClean"].median()-data["TicketClean"].std()
data.set_value(179, 'TicketClean', int(med1))
data.set_value(271, 'TicketClean', int(med1))
data.set_value(302, 'TicketClean', int(med1))
data.set_value(597, 'TicketClean', int(med1))
data.set_value(772, 'TicketClean', int(med2))
data.set_value(841, 'TicketClean', int(med2))
data.set_value(1077, 'TicketClean', int(med2))
data.set_value(1193, 'TicketClean', int(med2))
data["TicketClean"].head()


# ## Passenger's Name ##
# 
# Names also hide some valuable information. Like family name, and also tittle abbreviations like Mr., Miss. and so on. I am going to extract those titles and examine them to see if there is something useful we can do with it.

# In[ ]:


data["TitleClean"] = data["Name"].str.extract('(\w*\.)', expand=True)
data.groupby(data["TitleClean"]).size()


# In[ ]:


data["TitleClean"] = data["TitleClean"].apply(title_to_int)


# ## Machine Learning ##

# ### Data Balance ###
# 
# It is important to check if our data is balanced. That means that in this binary classification problem we should have the same number of rows for each possible outcome. If the data is imbalanced, our accuracy score could be wrong and the model could have problems to generalise. 
# 
# The graph shows that our data is imbalanced, so the things we can do are:
# 
#  - Use a different score. Apart from Accuracy and F-Score, I will also check Cohen's Kappa.
# 
# **Cohen’s kappa**: Classification accuracy normalized by the imbalance of the classes in the data.
# http://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/
# 
#  - **TODO**. Resample the data adding copies of instances from the under-represented class .

# In[ ]:


df=pd.DataFrame()
df['total']=train.groupby("Survived").size()
df=df['total']/train.shape[0]
df.plot(kind="bar",title="Label's Balance")
axes = plt.gca()
axes.set_ylim([0,1])
plt.show()


# ### Data Preparation ###
# 
# First some data cleaning:
# 
#  - Drop useless columns.
#  - Fill NaN ages and Fare with average from "Title" and "Pclass" groups.
#  - Separate features and labels

# In[ ]:


data.head()


# In[ ]:


remove=['Name','Cabin','Ticket', 'AgeGroup']
for column in remove:
    data = data.drop(column, 1)

# Add missing ages. If there is a NaN, change it with the average for that title group.
list_nan=pd.isnull(data["Age"]).nonzero()
# Get a pd with the mean age for each title
means = data.groupby("TitleClean").mean()['Age']
# for each row with NaN, we write the average
for i in list_nan[0]:
    temp_title = data["TitleClean"].ix[i]
    data.set_value(i, 'Age', int(means[temp_title]))

# Add missing fare. If there is a NaN, change it with the average for that Pclass.
list_nan=pd.isnull(data["Fare"]).nonzero()
# Get a pd with the mean age for each title
means = data.groupby("Pclass").mean()['Fare']
# for each row with NaN, we write the average
for i in list_nan[0]:
    temp_class = data["Pclass"].ix[i]
    data.set_value(i, 'Fare', int(means[temp_class]))


# In[ ]:


# Prepare features
train=data[data['Survived'].isin([0, 1])]
#labels=train["Survived"]
train=train.drop("Survived", 1)
train=train.drop('PassengerId', 1)
features=train

# Prepare testing data
test=data[~data['Survived'].isin([0, 1])]
test=test.drop("Survived", 1)


# In[ ]:


lets_try(NL=False).show()


# Seems that Random Forest and AdaBoost perform better. Both allow us to extract information about what are the most important features to take a decision. In the next graph we see which are the most important features for RandomForest.

# In[ ]:


def draw_best_features():
    clf=RandomForestClassifier()
    clf.fit(features,labels)
    importances = clf.feature_importances_
    names=features.columns.values

    pd.Series(importances*100, index=names).plot(kind="bar")
    plt.show()
    
draw_best_features()


# In[ ]:


# Now let's test only with relevant features
#best_features=["Pclass","Sex","Age","Fare","FamilyMembers", "TicketClean", "TitleClean"]
best_features=["Pclass","Sex","Age","Fare", "TicketClean", "TitleClean"]
features=features[best_features]
features.head()


# Most of the models require the data to be standardised, so I am going to use a scaler and then check the scores again. There should be a huge difference.

# In[ ]:


scaler = MinMaxScaler()
features_backup=features
features = scaler.fit_transform(features)
pd.DataFrame(features).head()


# In[ ]:


lets_try(NL=False).show()


# In[ ]:


features=features_backup
cv = KFold(n_splits=5)

parameters = {'n_estimators': [10,20,30,40,50],
               'min_samples_split' :[2,3,4,5],
               'min_samples_leaf' : [1,2,3]
             }

clf = RandomForestClassifier()
grid_obj = GridSearchCV(clf, parameters, cv=cv)
grid_fit = grid_obj.fit(features, labels)
best_clf = grid_fit.best_estimator_ 

best_clf.fit(features,labels)


# In[ ]:


PassengerId=test["PassengerId"]


# In[ ]:


#remove=['PassengerId','SibSp', 'Parch', 'Embarked']
#for column in remove:
#    test = test.drop(column, 1)


# In[ ]:


test=test[best_features]
test.head()


# In[ ]:


predictions=best_clf.predict(test)

sub = pd.DataFrame({
        "PassengerId": PassengerId,
        "Survived": predictions
    })
sub.to_csv("titanic_submission.csv", index=False)

