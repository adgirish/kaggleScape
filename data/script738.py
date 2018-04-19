
# coding: utf-8

# ## Forewords ##

# Main task is to predict if a particular person would have survived in the titanic crash or not. We have to develop a predictive algorithm and train it with the train data and check its accuracy with test data (Someone please take a movie titled "How to train your algorithm" like "How to train your dragon"!!)
# 
# When I started working in Python, a question was constantly ringing in my head – “This task can be easily done in Excel with few clicks. Why should I blast my head in Python ?” I know most of aspiring data scientists would ask themselves this same question. So I have also included appropriate Microsoft Excel equivalents for tasks wherever applicable in this problem.
# 
# My opinion is that you are totally free to use any tools of your choice if you know what you are doing and know where to go.
# 

# So the steps are,
# 
# **Data Exploration :**<br/>
#         Import necessary packages<br/>
#         Import data and explore it<br/>
#         Features are categorical or numerical ?<br/>
# 
# **Feature Engineering :**<br/>
#         Finding the impact of features on the survival rate<br/>
#  	Heat map for correlation between features <br/>
# 
# **Data Wrangling :**<br/>
# 	Dropping unwanted features<br/>
# 	Creating new features<br/>
# 	Mapping categorical features to numbers<br/>
#         Treating features containing null values<br/>
#         Heat map for correlation between features <br/>
# 
# **Predicting passenger’s fate :**<br/>
# 	Short descriptions for algorithms<br/>
# 	Trying them all<br/>
# 	Model Evaluation<br/>
# 	Final verdict

# ## Data Exploration ##

# **Import necessary packages**<br/>
# Python comes with various packages that can be used instantly loaded and put in action. Packages are created by other fellow developers and shared publicly so that other developers don"t want to write the same code again and again.
# 
# Our first step is to import the necessary packages into our analysis. It can be done as follows :

# In[ ]:


# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# **Import data and explore it**<br/>
# Once we have all the tools for analysis, we are set to import dataset. It can be done as follows :

# In[ ]:


# train data has been assigned to a variable called train
# test data has been assigned to a variable called test
# created a variable called combine and assigning a list object to it containing both train and test
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
combine = [train, test]


# **Lets explore the dataset that we imported freshly**<br/>
# Consider this step like unboxing your newly bought phone. Once we get a new phone, we will check every features of the phone. Similarly we will check the features of dataset if they have any missing value, whether they are numerical or categorical etc

# In[ ]:


# First lets see what are the features available in our dataset
print(train.columns.values)
# we have 12 features


# In[ ]:


# lets check how many of them are numerical and how many of them are catagorical
train.head()
# we have 7 numerical features and 3 text features and 1 mixed feature


# Categorical features : Survived, Sex, and Embarked<br/>
# Ordinal features : Pclass<br/>
# Continous features : Age, Fare<br/>
# Discrete features : SibSp, Parch<br/>
# 

# In[ ]:


# checking if there is any missing values in train dataset
train.info()


# Age, Cabin and Embarked are the features that have missing values in the Train dataset

# In[ ]:


# checking if there is any missing values in train dataset
test.info()


# Age, Fare, Cabin and Embarked are the features that have missing values in the Test dataset

# **Digging further into the Train dataset**

# In[ ]:


train.describe()


# **Following conclusions can be drawn from the above table :**<br/>
# 1. Survived is a categorical feature (0 or 1)<br/>
# 2. Pclass (Passenger class) have values 1,2 or 3<br/>
# 3. Minimum age of the passengers is 4 months and maximum age is 80<br/>
# 4. Fare varies hugely, leaping to a maximum of $ 512.32<br/>
# 

# **Lets hear what categorical variables tell us**

# In[ ]:


train.describe(include=['O'])


# **Following conclusions can be drawn from the above table :** <br/>
# 1. Most people embarked at "S" port and there are 3 embarking ports.<br/>
# 2. We don't have lots of cabin data (anyway that is not very important as the cabin number have nothing to do with if the passenger survived or not) <br/>
# 3. 577 men were there in train dataset<br/>

# ## Feature Engineering ##

# **Finding the impact of features on the survival rate**

# **Pclass**

# In[ ]:



# Lets check if pclass attribute have any impact on the survival rate :
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# From the above table, we can conclude that most of the 1st class passengers were saved and most of the 3rd class passengers are left to die. Can you understand the politics here ?

# **Sex**

# In[ ]:


# Lets check if sex attribute have any impact on the survival rate :
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# Again a valuable insight. 74% of female passengers were saved and only 18.8% of male passengers survived. Any feminists reading this ??

# **SibSp**

# In[ ]:


# Lets check what is the probability of survival of passengers with siblings onboard
train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# Above table tells us that if a passenger is alone onboard with no siblings, he have 34.53% survival rate. The graph roughly decreases if the number of siblings increase.
# It makes sense right ???? That is, if I have a sibling on board, I will try to save them instead of saving myself in the first place.<br/>
# 
# Also we should remember that this feature includes spouses too. So we can conclude the first row in the above table shows higher survival rate because more of the wives or husbands were saved by their counterparts.

# **Parch**

# In[ ]:


# Lets check how many people survived who had one or two parent onboard with them
train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# **Family Size**

# At this point, we can create a new feature called "Family size" and analyse it. This feature is the summation of Parch and SibSp. It gives us a consolidated data so that we can check if survival rate have anything to do with family size of the passengers. 

# In[ ]:


for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())


# We can see a better result from the above table, but lets dig a bit deeper and categorize passengers if they are alone 

# In[ ]:


for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())


# This is really a good insight. If someone is alone on the ship, he have only 30.35% of survival chance. If they are accompanied by any of their family members, they get 50.56% of survival rate.

# **Lets create some visuals**

# I am going to plot histograms for Age attribute to find which age group survived the most

# In[ ]:


g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)


# **From the above chart, we can draw below insights :**<br/>
# 1. Most of the children below age 4 were saved<br/>
# 2. Most of the aged passengers (80 years) were saved<br/>
# 3. Passengers between age group 15 to 35 died a lot<br/>
# 4. Age is an important factor and should be used to train our algorithm (dragon!!)

# In[ ]:


# Plotting Pclass againt survival
g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Pclass', bins=3)


# **From the above chart, we can draw below insights :**<br/>
# 1. Most of the people traveled in 3rd class ticket died<br/>
# 2. Most of the 1st class passengers were saved <br/>
# 3. Pclass should definitely be taken into consideration while training our dragon<br/>

# In[ ]:


# In the below chart, we are visualizing 4 attributes : Fare, Embarked, Survived and sex
grid = sns.FacetGrid(train, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.8, ci=None)
grid.add_legend()


# **From the above chart, we can draw below insights :**<br/>
# 1. People embarked at "C" port survived the most. From the above chart, it is obvious that they paid more for the ticket. It means most of the 1st class passengers are embarked at "C" port. This clearly shows the inter relationship between survival rate and port of embarkation.
# 2. This shows that port of embarkation have some relationship with survival rate, so add embarkation feature in our model.

# **Heat map for correlation between features**

# In[ ]:


colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Feature correlations', y=1.05, size=15)
sns.heatmap(train.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


# From the above heat map we can get, the correlations between all the attributes. Numbers closer to 1 have strong positive correlation between the attributes (For ex, SibSp and FamilySize). Positive correlation means, if one attribute increases its value, another attribute also increase.
# 
# Numbers closer to -1 have strong negative correlation between the attributes (For ex, SibSp and IsAlone). Negative correlation means, if one attribute increases its value, another attribute decreases.

# ## Data Wrangling ##

# Lets pack everything and clean data to feed it into our predictive algorithm

# **Dropping unwanted features**

# Some of the features in the dataset will be clinging there for no good. They can be eliminated, making the dataset simpler. <br/>
# <br/>
# Here, we can eliminate "Ticket" and "Cabin" attributes, as they don't have anything to do with the survival rate. Number printed on the ticket cannot predict if a passenger will die or survive. Similarly number written on their cabin door also cannot predict anything. I would call this as "Common Sense".

# In[ ]:


# We are dropping the attributes from both test and training datasets to maintain consistency
train = train.drop(['Ticket', 'Cabin'], axis=1)
test = test.drop(['Ticket', 'Cabin'], axis=1)
combine = [train, test]


# In[ ]:


# lets check
train.head ()


# OK done !!

# **Creating new features**

# We are going to create a feature called "Salutation" depending on the title the passengers had in their name. For example, Miss, Mr etc.

# In[ ]:


for dataset in combine:
    dataset['Salutation'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Salutation'], train['Sex'])


# We got the Salutations. Lets categorize them and check for any patters hiding in it.

# In[ ]:


for dataset in combine:
    dataset['Salutation'] = dataset['Salutation'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Salutation'] = dataset['Salutation'].replace('Mlle', 'Miss')
    dataset['Salutation'] = dataset['Salutation'].replace('Ms', 'Miss')
    dataset['Salutation'] = dataset['Salutation'].replace('Mme', 'Mrs')
    
train[['Salutation', 'Survived']].groupby(['Salutation'], as_index=False).mean()


# From the above table, if someone (probably females) have titles as "Miss" or "Mrs", they have huge chances of survival. So if you happened to be in a drowning Titanic, don't search for a life jacket, instead pretend like a lady (like Mr.Bean) and you will be saved most probably (no offense.. just for fun)

# **Lets map Salutations to ordinal numbers**

# In[ ]:


Salutation_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Salutation'] = dataset['Salutation'].map(Salutation_mapping)
    dataset['Salutation'] = dataset['Salutation'].fillna(0)

train.head()


# Perfect !!!

# Now we can drop "Name" attribute without any loss. Here "PassengerID" attribute is also clinging in dataset without any value adding. Let it go.

# In[ ]:


train = train.drop(['Name', 'PassengerId'], axis=1)
test = test.drop(['Name'], axis=1)
combine = [train, test]


# In[ ]:


train.head ()


# ## Mapping categorical variables to numbers ##

# Lets map sex first

# In[ ]:


for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train.head()


# Done !!!

# ## Treating features containing null values ##

# We are going to guess missing values in Age feature using other features like Pclass and Gender. Using the below six combinations, we are going to guess the missing ages.

# In[ ]:


# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')
grid = sns.FacetGrid(train, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


# In[ ]:


# Lets create a null array with (2,3) size
guess_ages = np.zeros((2,3))
guess_ages


# Now we are gonna write a code which will iterate over Sex (0 & 1) and Pclass (1,2,3) and fill the null matrix

# In[ ]:


for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess = dataset[(dataset['Sex'] == i) &                                   (dataset['Pclass'] == j+1)]['Age'].dropna()

            age_guess = guess.median()
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train.head()


# We cannot process the age as a whole number. Lets create bands in the age range and check for its correlation with survival rate.

# In[ ]:


train['AgeBand'] = pd.cut(train['Age'], 5)
train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


# Replacing the age bands with ordinal numbers just like we did in Sex and Salutation

# In[ ]:


for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train.head()


# Done !!!

# Dropping the Age Band feature

# In[ ]:


train = train.drop(['AgeBand'], axis=1)
combine = [train, test]
train.head()


# **Completing missed values in Embarked feature**

# In[ ]:


# Check which port have frequent occurance in our dataset
freq_port = train.Embarked.dropna().mode()[0]
freq_port


# Lets fill the two blanks in Embarkation attribute with "S"

# In[ ]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# So People embarked at C port have better survival rate. Now lets convert the categories of Embarkation feature into numerical values.

# In[ ]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train.head()


# Now, there is only one null value in "Fare" attribute and we are going to fill it with median value value.

# In[ ]:


test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)
test.head()


# Now lets create Fare band with 4 segments

# In[ ]:


train['FareBand'] = pd.qcut(train['Fare'], 3)
train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


# Map numbers for the bands

# Creating an artificial feature by multiplying age and class

# In[ ]:


for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)


# In[ ]:


for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train = train.drop(['FareBand'], axis=1)
combine = [train, test]
    
train.head(10)


# In[ ]:


test.head (10)


# ## Heat map for correlation between features ##

# Lets find the correlations between all the attributes along with newly created features using a heat map.

# In[ ]:


colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Feature correlations', y=1.05, size=15)
sns.heatmap(train.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


# ## Predicting Passenger's fate ##

# **Short Descriptions for Algorithms**

# There is a large number of Machine Learning Algorithms available today for various applications. Based on our current criteria, we can narrow down to these algorithms :<br/>
# Logistic Regression<br/>
# KNN or k-Nearest Neighbors<br/>
# Support Vector Machines<br/>
# Naive Bayes classifier<br/>
# Decision Tree<br/>
# Random Forrest<br/>
# Perceptron<br/>
# Artificial neural network<br/>
# RVM or Relevance Vector Machine<br/>
# 
# Check this link out to learn more about these algorithms

# **Lets try every above algorithm and check which one gives us highest accuracy**

# **Logistic Regression**

# In[ ]:


# Lets prepare dataset to feed into the algorithm
# We are dropping "Survived" column from train data and "PassengerID" from test data
X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# In[ ]:


# Lets run Logistic Regression algorithm

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# Logistic regression gives us 82.15 confidence score (measure of accuracy)

# **Support Vector Machines**

# In[ ]:


svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# **KNN or k-Nearest Neighbors**

# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# **Naive Bayes classifier**

# In[ ]:


gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# **Perceptron**

# In[ ]:


perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron


# **Linear SVC**

# In[ ]:


linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc


# **Stochastic Gradient Descent**

# In[ ]:


sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd


# **Decision Tree**

# In[ ]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# **Random Forest**

# In[ ]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# ## Evaluating Models ##

# Lets tabulate all the confidence scores and check which algorithm is the best fit for this problem

# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# In[ ]:


# This is the final submission
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred
    })
submission


# In[ ]:


# Taking a look at complete predictions
print(submission.to_string())

