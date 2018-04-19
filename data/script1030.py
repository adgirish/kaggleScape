
# coding: utf-8

# # Hello World! to the Machine Learning with Titanic dataset
# 
# This notebook covers the basics of **Data Exploration, Analysis and Prediction** in Python. This is hands-on Machine learning. This is my first Kaggle submission. I made this notebook while learning. 
# ### **Suggestions are always welcome & feel free to upvote! :) **
# 

# ### Contents :
# 1. Data Loading
# 2. Data Exploration
# 3. Feature Engineering
# 4. Applying Machine Learning
# 5. Selecting the best-fitted model
# 6. Submitting the file

# ## DATA LOADING
# *     Loding modules
# *    Loding Data
# *  Understanding the Data

# ## 1. Loading modules

# In[1]:


# pandas, numpy
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import Series,DataFrame

# matplotlib, seaborn
import matplotlib.pyplot as plt
import seaborn as sns
# to make the plots visible in notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning modules are imported in later part

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import warnings
warnings.filterwarnings("ignore")


# ## 2. Loading Data

# In[2]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train_test_data = [train, test]


# ## 3. Understanding Data

# In[3]:


train.head()


# In[4]:


train.describe()


# In[5]:


print ('TRAINING DATA\n')
train.info()
print ("----------------------------------------\n")
print ('TESTING DATA\n')
test.info()


# In[6]:


print ('#MISSING VALUES IN TRAINING DATA')
train.isnull().sum()


# In[7]:


print ('#MISSING VALUES IN TESTING DATA')
test.isnull().sum()


# Here we can evaluate that **training data** has missing values for *Age, Cabin, Embarked*  and **testing data **has missing values for *Age, Fare, Cabin, Embarked*.

# In[8]:


train.describe(include=['O'])


# ## DATA EXPLORATION
# Here we will see the relationship of various features with the *Survival*, as that is what we have to predict ultimately. Data is analysed and observations are made.
# * Pclass vs. Survival
# * Sex vs. Survival
# * Age vs. Survival
# * Embarked vs. Survival
# * Fare vs. Survival
# * Parch vs. Survival
# * SibSp vs. Survival

# In[9]:


survived = train[train['Survived'] == 1]
not_survived = train[train['Survived'] == 0]


# In[10]:


sns.countplot(x='Survived',data=train)


# In[11]:


print ("Survived: %i (%.1f%%)" %(len(survived), float(len(survived))/len(train)*100.0))
print ("Not Survived: %i (%.1f%%)" %(len(not_survived), float(len(not_survived))/len(train)*100.0))
print ("Total: %i" % (len(train)))


# **Observation : **Nearly two-third of the Passengers on Titanic did not survive. 

# ### Pclass vs. Survival

# In[12]:


train.Pclass.value_counts()


# In[13]:


train.groupby('Pclass').Survived.value_counts()


# In[14]:


sns.factorplot(x="Pclass", hue="Sex", col="Survived",data=train, kind="count",size=5, aspect=1, palette="BuPu");


# In[15]:


train.groupby('Pclass').Survived.mean()


# In[16]:


sns.factorplot(x="Pclass", y="Survived", data=train,size=5, kind="bar", palette="BuPu", aspect=1.3)


# **Observation** : People in First-Class have maximum chances of survival than second and third. People of higher ranking/class were rescued first.

# ### Sex vs. Survival

# In[17]:


train.Sex.value_counts()


# In[18]:


train.groupby('Sex').Survived.value_counts()


# In[19]:


train.groupby('Sex').Survived.mean()


# In[20]:


sns.factorplot(x='Sex',y='Survived',data=train, size=5, palette='RdBu_r', ci=None, kind='bar', aspect=1.3)


# In[21]:


plt.style.use('bmh')
sns.factorplot(x='Pclass', y='Survived', hue='Sex', col='Embarked', data=train)


# **Observation** : Female Passengers have much greater chance of survival. As it goes *"Ladies First"*.

# ### Age vs. Survival

# In[22]:


# Filling NaN values
for dataset in train_test_data:
    avg = dataset['Age'].mean()
    std = dataset['Age'].std()
    null_count = dataset['Age'].isnull().sum()
    random = np.random.randint(avg-std, avg+std, size=null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = random
    dataset['Age'] = dataset['Age'].astype(int)


# In[23]:


age_survived = train['Age'][train['Survived'] == 1]
age_not_survived = train['Age'][train['Survived'] == 0]

# Plot
plt.style.use('bmh')
sns.kdeplot(age_survived, shade=True, label = 'Survived')
sns.kdeplot(age_not_survived, shade=True, label = 'Not Survived')


# In[24]:


plt.style.use('bmh')
sns.set_color_codes("deep")
fig , (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(17,5))
sns.violinplot(x="Pclass", y="Age", hue="Survived", data=train, palette={0: "b", 1: "r"},split=True, ax=ax1)
sns.violinplot(x="Embarked", y="Age", hue="Survived", data=train,palette={0: "b", 1: "r"}, split=True, ax=ax2)
sns.violinplot(x="Sex", y="Age", hue="Survived", data=train,palette={0: "b", 1: "r"}, split=True, ax=ax3)


# **Observation** : 
# 
# From Pclass violinplot:
# * 1st Pclass has very few children as compared to other two classes.
# * 1st Plcass has more old people as compared to other two classes.
# * Almost all children (between age 0 to 10) of 2nd Pclass survived.
# * Most children of 3rd Pclass survived.
# * Younger people of 1st Pclass survived as compared to its older people.
# 
# From Sex violinplot:
# * Most male children (between age 0 to 14) survived.
# * Females with age between 18 to 40 have better survival chance.
# * Old women have better survival chance than old men.

# ### Embarked vs. Survival

# In[25]:


train.Embarked.value_counts()


# In[26]:


# As there are only 2 missing values, we will fill those by most occuring "S"
train['Embarked'] = train['Embarked'].fillna('S')
train.Embarked.value_counts()


# In[27]:


train.groupby('Embarked').Survived.value_counts()


# In[28]:


train.groupby('Embarked').Survived.mean()


# In[29]:


sns.factorplot(x='Embarked', y='Survived', data=train, size=4, aspect=2.5)


# **Observation** : Those who embarked from C, survived the most. They might be sitting near exit/window/passway/door.

# ### Fare vs. Survival 

# In[30]:


# As there is one missing value in test data, fill it with the median.
test['Fare'] = test['Fare'].fillna(test['Fare'].median())

# Convert the Fare to integer values
train['Fare'] = train['Fare'].astype(int)
test['Fare'] = test['Fare'].astype(int)

# Compute the Fare for Survived and Not Survived
fare_not_survived = train["Fare"][train["Survived"] == 0]
fare_survived = train["Fare"][train["Survived"] == 1]


# In[31]:


sns.factorplot(x="Survived", y="Fare", data=train,size=5, kind="bar", ci=None, aspect=1.3)


# In[32]:


train["Fare"][train["Survived"] == 1].plot(kind='hist', alpha=0.6, figsize=(15,3),bins=100, xlim=(0,60))
train["Fare"][train["Survived"] == 0].plot(kind='hist', alpha=0.4, figsize=(15,3),bins=100, xlim=(0,60), title='Fare of Survived(Red) and Not Survived(Blue)')


# 
# **Observation**: There are more number of passengers with cheaper fare but their *Survival* rate is low. 

# ### Parch vs. Survival

# In[33]:


train.Parch.value_counts()


# In[34]:


train.groupby('Parch').Survived.value_counts()


# In[35]:


train.groupby('Parch').Survived.mean()


# In[36]:


sns.barplot(x='Parch',y='Survived', data=train, ci=None, palette="Blues_d")


# ### SibSp vs. Survival

# In[37]:


train.SibSp.value_counts()


# In[38]:


train.groupby('SibSp').Survived.value_counts()


# In[39]:


train.groupby('SibSp').Survived.mean()


# In[40]:


sns.barplot(x='SibSp', y='Survived', data=train, ci=None, palette="Blues_d")


# In[41]:


pd.crosstab(train['SibSp'], train['Parch'])


# In[42]:


# Correlation of features
# Negative numbers : inverse proportionality
# Positive numbers : direct proportionality
plt.figure(figsize=(15,6))
sns.heatmap(train.drop('PassengerId',axis=1).corr(), square=True, annot=True, center=0)


# In[43]:


train.isnull().sum()


# In[44]:


test.isnull().sum()


# All the missing values are filled. The *Cabin* & *Ticket* columns have to be dropped as it does not draw any relation with the Survival of passenger.

# ## Feature Engineering
# Basic Rule: 
# * Map all the string values to numerical. 
# * Drop unnecessary features to avoid overfitting & underfitting.

# In[45]:


train.dtypes.index


# ### 1. PassengerId
# 
# Index is not required in training dataset but does in testing data(submission purpose).

# In[46]:


del train['PassengerId']
train.head()


# ### 2.  Pclass
# This feature is considered. This gives the major idea about the Survival of Passenger. 
# * First class passengers have greater chance of survival.
# * Third class passengers have the least. 

# In[47]:


pd.crosstab(train['Pclass'], train['Survived'])


# ### 3. Name
# Is there any correlation between the passenger's *Title* and chance of  his/her *Survival?*

# In[48]:


train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.')
test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.')

# Delete the 'Name' columns from datasets
del train['Name']
del test['Name']


# In[49]:


train['Title'].value_counts()


# In[50]:


pd.crosstab(train['Title'], train['Pclass'])


# In[51]:


pd.crosstab(train['Title'], train['Survived'])


# In[52]:


for data in train_test_data:
    data['Title'] = data['Title'].replace(['Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Others')
    data['Title'] = data['Title'].replace(['Mlle', 'Ms'],'Miss')
    data['Title'] = data['Title'].replace(['Mme', 'Lady'],'Mrs')   
    
train.groupby('Title').Survived.mean()


# In[53]:


for data in train_test_data:
    data['Title'] = data['Title'].map({ 'Master':0, 'Miss':1, 'Mr':2, 'Mrs':3, "Others":4 }).astype(int)


# In[54]:


train.head()


# But here we see that *Title* is irrelevant to the Survival. Because:
# - It's distribution is not even to distinguish the survival chance of passengers. 
# - It's vague distribution (wrt Pclass) shows that it is highly deviated. Example, Titles such as *Col, Capt, Don, Dr, Jonkheer* belongs to 1st class but none of them had hight rate of survival, this contradicts with the fact that 1st class passengers had greater chance of survival.
# - The given data is not enough to train the model and thus made show variance for test data.

# In[55]:


for data in train_test_data:
    del data['Title']


# ### 4. Sex
# This is another very important feature.
# * Females have greater chance of survival

# In[56]:


def person(per):
    age,sex = per
    return 'child' if age < 16 else sex

train['Person'] = train[['Age', 'Sex']].apply(person, axis=1)
test['Person'] = test[['Age', 'Sex']].apply(person, axis=1)

# As 'Sex' column is not required.
del train['Sex']
del test['Sex']


# In[57]:


train.head()


# In[58]:


train['Person'].value_counts()


# In[59]:


train.groupby('Person').Survived.mean()


# In[60]:


g = sns.PairGrid(train, y_vars="Survived",x_vars="Person",size=3.5, aspect=1.7)
g.map(sns.pointplot, color=sns.xkcd_rgb["plum"])


# In[61]:


for data in train_test_data:
    data['Person'] = data['Person'].map({ 'female':0, 'male':1, 'child':3 }).astype(int)
train.head()


# ### 5. Embarked

# In[62]:


test.Embarked.value_counts()


# In[63]:


for data in train_test_data:
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)


# In[64]:


train.head()


# ### 6. Age

# In[65]:


# Divide 'Age' into groups
a = pd.cut(train['Age'], 5)
print (train.groupby(a).Survived.mean())


# In[66]:


# Assign number to Age limits
for data in train_test_data:
    data.loc[ data['Age'] <= 16, 'Age'] = 0
    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
    data.loc[ data['Age'] > 64, 'Age'] = 4  


# In[67]:


train.head()


# ### 7. Fare

# In[68]:


f = pd.qcut(train['Fare'], 4)
print (train.groupby(f).Survived.mean())


# In[69]:


for data in train_test_data:
    data.loc[ data['Fare'] <= 7.91, 'Fare'] = 0
    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare']   = 2
    data.loc[ data['Fare'] > 31, 'Fare'] = 3
    data['Fare'] = data['Fare'].astype(int)


# In[70]:


train.head()


# ### 8. Parch & SibSp
# 
# Consider Parents Children(Parch) & Sibling Spouse (SibSp) as Family. Adding this will give *Family*.

# In[71]:


for data in train_test_data:
    data['Family'] = data['Parch'] + data['SibSp']
    data['Family'].loc[data['Family'] > 0] = 1
    data['Family'].loc[data['Family'] == 0] = 0

for data in train_test_data:
    del data['Parch']
    del data['SibSp']


# In[72]:


train.head()


# ### 9. Cabin & Ticket
# Drop them!

# In[73]:


for data in train_test_data:
    del data['Cabin']
    del data['Ticket']


# In[74]:


train.head()


# ## MACHINE LEARNING

# In[75]:


# Importing modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# ### Splitting Training-Testing Data

# In[76]:


X = train.drop('Survived', axis=1)
y = train.Survived

# Split into train-test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Shape of you
print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)


# ### Classification Algorithms
# 
# As this is Classification problem(eg: 0/1; yes/no; high/low; r/g/b) we will use following algorithms:
# * Logistic Regression
# * Guassian Naive Bayes 
# * Decision Trees
# * Random Forest
# * SVC
# * LinearSVC
# * AdaBoost
# * K-nearest neighbours (KNN)
# * Perceptron
# * Stochastic Gradient Descent (SGD)
# * Bagging
# 

# ### 1. Logistic Regression

# In[77]:


logReg = LogisticRegression()
logReg.fit(X_train, y_train)
y_pred = logReg.predict(X_test)
print ('Score: %.2f%%' % (round(logReg.score(X_test, y_test)*100, 4)))
print ('Accuracy: %.2f' % (accuracy_score(y_test, y_pred)))


# ### 2. Guassian Naive Bayes 

# In[78]:


naive_clf = GaussianNB()
naive_clf.fit(X_train, y_train)
y_pred = naive_clf.predict(X_test)
print ('Score: %.2f%%' % (round(naive_clf.score(X_test, y_test)*100, 4)))
print ('Accuracy: %.2f' % (accuracy_score(y_test, y_pred)))


# ### 3. Decision Trees

# In[79]:


dtree_clf = DecisionTreeClassifier()
dtree_clf.fit(X_train, y_train)
y_pred = dtree_clf.predict(X_test)
print ('Score: %.2f%%' % (round(dtree_clf.score(X_test, y_test)*100, 4)))
print ('Accuracy: %.2f' % (accuracy_score(y_test, y_pred)))


# ### 4. Random Forest

# In[80]:


rtree_clf = RandomForestClassifier(n_estimators=100)
rtree_clf.fit(X_train, y_train)
y_pred = rtree_clf.predict(X_test)
print ('Score: %.2f%%' % (round(rtree_clf.score(X_test, y_test)*100, 4)))
print ('Accuracy: %.2f' % (accuracy_score(y_test, y_pred)))


# ### 5. SVM

# In[81]:


svc_clf = SVC()
svc_clf.fit(X_train, y_train)
y_pred = svc_clf.predict(X_test)
print ('Score: %.2f%%' % (round(svc_clf.score(X_test, y_test)*100, 4)))
print ('Accuracy: %.2f' % (accuracy_score(y_test, y_pred)))


# ### 6. Linear SVC

# In[82]:


linear_clf = LinearSVC()
linear_clf.fit(X_train, y_train)
y_pred = linear_clf.predict(X_test)
print ('Score: %.2f%%' % (round(linear_clf.score(X_test, y_test)*100, 4)))
print ('Accuracy: %.2f' % (accuracy_score(y_test, y_pred)))


# ### 7. KNN

# In[83]:


k_range = list(range(1, 30))
k_values = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    k_values.append(acc)
    print (k,acc)

plt.plot(k_range, k_values)
plt.xlabel('K Values')
plt.ylabel('Accuracy')


# In[84]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print ('Score: %.2f%%' % (round(knn.score(X_test, y_test)*100, 4)))
print ('Accuracy: %.2f' % (accuracy_score(y_test, y_pred)))


# ### 8. AdaBoost

# In[85]:


e_range = list(range(1, 25))
estimator_values = []
for est in e_range:
    ada = AdaBoostClassifier(n_estimators=est)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    estimator_values.append(acc)
    print (est,acc)

plt.plot(e_range, estimator_values)
plt.xlabel('estimator values')
plt.ylabel('Accuracy')


# In[86]:


ada = AdaBoostClassifier(n_estimators=7)
ada.fit(X_train, y_train)
y_pred = ada.predict(X_test)
print ('Score: %.2f%%' % (round(ada.score(X_test, y_test)*100, 4)))
print ('Accuracy: %.2f' % (accuracy_score(y_test, y_pred)))


# ### 9. Perceptron

# In[87]:


iteration_values = []
for i in range(1,30):
    clf = Perceptron(max_iter=i, tol=None)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    iteration_values.append(acc)
    print (i,acc)

# Plot
plt.plot(range(1,30), iteration_values)
plt.xlabel('max_iter')
plt.ylabel('Accuracy')


# In[88]:


per_clf = Perceptron(max_iter=4, tol=None)
per_clf.fit(X_train, y_train)
y_pred = per_clf.predict(X_test)
print ('Score: %.2f%%' % (round(per_clf.score(X_test, y_test)*100, 4)))
print ('Accuracy: %.2f' % (accuracy_score(y_test, y_pred)))


# ### 11. Stochastic Gradient Decent (SGD)

# In[89]:


sgd_clf = SGDClassifier(max_iter=8, tol=None)
sgd_clf.fit(X_train, y_train)
y_pred = sgd_clf.predict(X_test)
print ('Score: %.2f%%' % (round(sgd_clf.score(X_test, y_test)*100, 4)))
print ('Accuracy: %.2f' % (accuracy_score(y_test, y_pred)))


# ### 12. Bagging

# In[90]:


e_range = list(range(1, 30))
estimator_values = []
for est in e_range:
    ada = BaggingClassifier(n_estimators=est)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    estimator_values.append(acc)
    print (est,acc)

plt.plot(e_range, estimator_values)
plt.xlabel('estimator values')
plt.ylabel('Accuracy')


# In[91]:


bag = BaggingClassifier()
bag.fit(X_train, y_train)
y_pred = bag.predict(X_test)
print ('Score: %.2f%%' % (round(bag.score(X_test, y_test)*100, 4)))
print ('Accuracy: %.2f' % (accuracy_score(y_test, y_pred)))


# ## Submission!

# In[92]:


test.head()


# In[93]:


submission = pd.DataFrame({
    "PassengerId" : test['PassengerId'],
    "Survived" : rtree_clf.predict(test.drop('PassengerId', axis=1))
})


# In[94]:


# submission.to_csv('titanic.csv', index=False)
submission.head()


# ### Thank you! **Suggestions are always welcome & feel free to upvote! :) **
# 
