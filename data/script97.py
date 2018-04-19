
# coding: utf-8

# ## Introduction

# This is my first kernel on Kaggle. The following credits are due:
# 
# - https://www.kaggle.com/sinakhorami/titanic-best-working-classifier
# - https://www.kaggle.com/frederikh/12-500-feet-under-the-sea
# - Aurelien Geron 'Hands-On Machine Learning with Scikit-Learn & Tensorflow'
# 
# EDIT: 
# 1. Added KFold validation to get more representative assessments of each model performance
# 2. Added hard voting classifier for ensemble learning
# 3. Added suggestions for further improvements
# 

# ## Data exploration

# First, import all the relevant modules.

# In[ ]:


import pandas as pd
import numpy as np
import re as re
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
pd.options.mode.chained_assignment = None
from sklearn.model_selection import KFold


# Next, load training and test data.

# In[ ]:


data_train = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')


# Have a look at the training data.

# In[ ]:


data_train.head(5)


# We can see that 'Survived' column contains the output labels so we can extract that already:

# In[ ]:


y_train = data_train.iloc[:, 1]


# Have a look at the test data.

# In[ ]:


data_test.head(5)


# See which columns have missing values in training data:

# In[ ]:


data_train.isnull().sum()


# See which columns have missing values in test data:

# In[ ]:


data_test.isnull().sum()


# Before we start filling the missing values for training and test data, let's create an array with all the data (training and test):

# In[ ]:


all_data = pd.concat([data_train, data_test], axis=0)


# Let's have a look at missing values across the board:

# In[ ]:


all_data.isnull().sum()


# ## Resolve missing data

# ### Missing ports of embarkation

# There are 2 missing values for the port that the passenger embarked on in training data. Let's have a look at these two passengers:

# In[ ]:


all_data[pd.isnull(all_data['Embarked'])]


# Both are first class passengers, that paid £80. Let's have a look if we can find similar passengers:

# In[ ]:


all_data[(all_data['Pclass'] == 1) & (all_data['Fare'] > 70) & (all_data['Fare'] < 90)]


# Looking at this, it seems unlikely that these two passengers embarked at Quenstown, but there doesn't seem to be any obvious separation between Cherbourg and Southampton. So let's just fill in the missing port values with the port the most passengers embarked on (port S):

# In[ ]:


all_data.iloc[:, 2] = all_data.iloc[:, 2].fillna(all_data.mode(0)['Embarked'][0])


# Confirm there are no missing embarkation port values in training data:

# In[ ]:


all_data.isnull().sum()


# ### Missing fare

# There is one missing ticket fare value in test data. Let's have a look at this passenger:

# In[ ]:


all_data[pd.isnull(all_data['Fare'])]


# It's a 3rd class passenger so let's fill the missing value with the mean fare for passengers of the same class:

# In[ ]:


third_class_pass = all_data[(all_data['Pclass'] == 3)]
all_data.iloc[:, 3] = all_data.iloc[:, 3].fillna(third_class_pass.mean(0)['Fare'])


# Confirm there are no missing fare values in test data:

# In[ ]:


all_data.isnull().sum()


# ### Missing age

# There are multiple age values missing in both training and test data. Let's first have a look at the data rows with missing age:

# In[ ]:


all_data[pd.isnull(all_data['Age'])]


# Most of the missing age values seem to come from third class passengers. Let's have a look if there are notible mean age differences between different passenger classes:

# In[ ]:


all_data.groupby(['Pclass'], as_index=False)['Age'].mean()


# Indeed it seems that first class passengers were the oldest and third class passengers - the youngest. Let's have a look further if there was a difference between male and female average ages:

# In[ ]:


all_data.groupby(['Pclass', 'Sex'], as_index=False)['Age'].mean()


# It seems like, on average, women were younger across all three passenger classes. We might be able to extract further information around womens' age by looking at their title. Let's first extract all the titles.

# In[ ]:


def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

# create new column
all_data['Title'] = [get_title(i) for i in all_data['Name']]

pd.crosstab(all_data['Title'], all_data['Sex'])


# Next let's create a 'Married' column and assign titles that indicate the woman is married the value 1, and 0 otherwise.

# In[ ]:


all_data['Married'] = [1 if i in ['Mrs', 'Countess', 'Mme', 'Dona'] else 0 for i in all_data['Title']]

all_data.groupby(['Pclass', 'Sex', 'Married'], as_index=False)['Age'].mean()


# Indeed there is quite a big difference between the average age of married and unmarried women in all passenger classes. Let's have a look if the median of the different classes is much different (would indicate outliers pushing the mean in either direction):

# In[ ]:


all_data.groupby(['Pclass', 'Sex', 'Married'], as_index=False)['Age'].median()


# The median values are close to the means.  Let's use the medians. 

# In[ ]:


for i in range(len(all_data)):
    # if age is null
    if pd.isnull(all_data.iloc[i, 0]):
        # if passenger male
        if all_data.iloc[i, 8] == 'male':
            # age estimate based on passenger class
            all_data.iloc[i, 0] = {1: 42, 2: 29.5, 3:25}[all_data.iloc[i, 7]]
        else:
            # if woman not married
            if all_data.iloc[i, 13] == 0:
                # age estimate based on passenger class
                all_data.iloc[i, 0] = {1: 30, 2: 20, 3:18}[all_data.iloc[i, 7]]
            else:
                all_data.iloc[i, 0] = {1: 45, 2: 30.5, 3:31}[all_data.iloc[i, 7]]


# Confirm there are no missing age values:

# In[ ]:


all_data.isnull().sum()


# Let's leave the missing Cabin values unfilled and let's separate out the training and test data again:

# In[ ]:


X_train = all_data[:891]
X_test = all_data[891:]


# ## Feature selection

# ### Gender

# Let's have a look if gender had an influence on survival rate:

# In[ ]:


data_train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()


# There seems to be a strong link between being female and surviving. That's a good feature to choose for the model.

# ### Passenger class

# Now let's have a look if passenger class is a good predictor of survival:

# In[ ]:


data_train[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).mean()


# Passenger class seems to be another good predictor of survival rate, where 1st class passengers were most likely to survive and 3rd class passengers - least likely.

# ### Ticket fare

# Intuitively one would expect a similar trend to that of passenger class when looking at ticket prices. Let's first plot the distribution of ticket prices.

# In[ ]:


fig,ax = plt.subplots(1)
plt.plot(data_train['PassengerId'], data_train['Fare'], 'r.')
plt.title('Ticket fares (training data)')
ax.set_ylabel('Ticket fare (£)')
ax.set_xlabel('Passenger Id')
plt.show()


# There seems to be several distinct clusters around ticket prices. Let's drop the rows where ticket price is 0 and then have a look at ticket price ranges per class - if these ranges do not follow the class boundaries neatly then we might be able to extract additional information from ticket price feature that we have not yet cought within passenger class feature.

# In[ ]:


data_train_no_zero_fares = data_train[data_train['Fare'] != 0]
data_train_no_zero_fares[["Pclass", "Fare"]].groupby(['Pclass'], as_index=False).min()


# In[ ]:


data_train_no_zero_fares[["Pclass", "Fare"]].groupby(['Pclass'], as_index=False).max()


# We can see for the 1st class passengers payed between £5.00 and ~£512.33 for their ticket, 2nd class - between £10.50 and £73.50 and 3rd class - between ~£4.01 and £69.55. The ticket prices indeed do not follow the class boundaries neatly, so it is worth exploring it further as a potential feature. Let's zoom in to look into more detail at ticket prices that were less than £100:

# In[ ]:


low_fares = data_train[data_train['Fare'] < 100]


# In[ ]:


fig,ax = plt.subplots(1)
plt.plot(low_fares['PassengerId'], low_fares['Fare'], 'r.')
plt.title('Ticket fares (training data)')
ax.set_ylabel('Ticket fare (£)')
ax.set_xlabel('Passenger Id')
plt.show()


# Looking at this and the previous plot, let's divide the data into four distinct groups: between £0 and £30, between £30 and £100, between £100 and £300 and finally three outliers quite above £300.

# In[ ]:


def createFaresRangeColumn(X):
    conditions = [
        (X['Fare'] > 300),
        (X['Fare'] > 100),
        (X['Fare'] > 30)
    ]
    choices = [0, 1, 2]
    X['FaresRange'] = np.select(conditions, choices, default=3)
    return X

X_train = createFaresRangeColumn(X_train)   
X_test = createFaresRangeColumn(X_test)


# Now let's have a look if fares are a good predictor of survival rates:

# In[ ]:


X_train[["FaresRange", "Survived"]].groupby(['FaresRange'], as_index=False).mean()


# And indeed it is - the greater the passenger fare, the more likely the passenger was to survive. 

# ### Age

# Now let's have a look at the age of the passenger. Let's plot the initial age data (without the extrapolated values):

# In[ ]:


fig,ax = plt.subplots(1)
plt.plot(data_train['PassengerId'], data_train['Age'], 'b.')
plt.title('Age (training data)')
ax.set_ylabel('Age (years)')
ax.set_xlabel('Passenger Id')
plt.show()


# There seems to be three distinct age group clusters: less than 15 years old, 15-35 years old and more than 35 years old. Let's create an age group column:

# In[ ]:


def createAgeRangeColumn(X):
    conditions = [
        (X['Age'] > 35),
        (X['Age'] > 15),
    ]
    choices = [2, 1]
    X['AgeRange'] = np.select(conditions, choices, default=0)
    return X

X_train = createAgeRangeColumn(X_train)   
X_test = createAgeRangeColumn(X_test)


# Now let's check if age group had an effect of survival:

# In[ ]:


X_train[["AgeRange", "Survived"]].groupby(['AgeRange'], as_index=False).mean()


# It seems like being in 'less than 15 years old' range increased the chances of survival, but other age ranges didn't have much of an effect. Let's check the proportions of different passenger classes amongst different age groups to see if the differences can be explained away by non-age related effects:

# In[ ]:


X_train[["AgeRange", "Pclass"]].groupby(['AgeRange'], as_index=False).mean()


# No, it seems that despite the fact that younger passengers were more likely to be lower class, which we saw earlier was indicative of lower chances of survival, they were still more likely to survive than older passengers. Hence, age seems to be a good feature to use in our model training as well. 

# ### Port of embarkation

# Now let's have a look if port of embarkation affected the chances of survival:

# In[ ]:


X_train[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean()


# It seems like passengers that embarked in Cherbourg had higher survival rates but this might be due to passenger class/fare rather than the port itself. Let's have a look at the average class numbers per port:

# In[ ]:


X_train[['Embarked', 'Pclass']].groupby(['Embarked'], as_index=False).mean()


# Indeed it seems like the proportion of upper class passengers boarded in Cherbourg was higher than in the other two ports. Hence, we will not use the port data as a feature in our model. 

# ### Family size

# Finally let's explore the family size as a potential feature. 

# In[ ]:


X_train['FamilySize'] = X_train['SibSp'] + X_train['Parch'] + 1
X_test['FamilySize'] = X_test['SibSp'] + X_test['Parch'] + 1


# In[ ]:


fig,ax = plt.subplots(1)
plt.plot(X_train['PassengerId'], X_train['FamilySize'], 'g.')
plt.title('Family size (training data)')
ax.set_ylabel('Family size (no. people)')
ax.set_xlabel('Passenger Id')
plt.show()


# Great majority of people travelled on their own. Let's have a look if travelling on your own had an effect the survival.

# In[ ]:


X_train['TraveledAlone'] = (X_train['FamilySize'] == 1).astype(int)
X_test['TraveledAlone'] = (X_test['FamilySize'] == 1).astype(int)


# In[ ]:


X_train[['TraveledAlone', 'Survived']].groupby(['TraveledAlone'], as_index=False).mean()


# The result indicates that travelling alone indeed decresed chances of survival. Let's check if this can't be explained away by other factors:

# In[ ]:


X_train[['TraveledAlone', 'Pclass']].groupby(['TraveledAlone'], as_index=False).mean()


# In[ ]:


X_train[['TraveledAlone', 'FaresRange']].groupby(['TraveledAlone'], as_index=False).mean()


# It seems that those travelling alone on average payed more for the fare and were higher class customers. That should make them more likely to survive but it was not the case. Hence, this looks like a good feature to include.

# So we will construct our models using the following features: Sex, Pclass, Fare, Age and TravelledAlone.

# In[ ]:


X_train_features = X_train[['Sex', 'Pclass', 'Fare', 'Age', 'TraveledAlone']]
X_test_features = X_test[['Sex', 'Pclass', 'Fare', 'Age', 'TraveledAlone']]
X_train_features.head(5)


# ## Model construction

# Let's first turn the gender values in numerical labels:

# In[ ]:


labelencoder = LabelEncoder()
X_train_features.loc[:, 'Sex'] = labelencoder.fit_transform(X_train_features.loc[:, 'Sex'])
X_test_features.loc[:, 'Sex'] = labelencoder.transform(X_test_features.loc[:, 'Sex'])
X_train_features.head(5)


# Next, let's scale the Fare and Age values to be in similar range to the rest of features:

# In[ ]:


scaler = StandardScaler()
X_train_features[['Fare', 'Age']] = scaler.fit_transform(X_train_features[['Fare', 'Age']])
X_test_features[['Fare', 'Age']] = scaler.fit_transform(X_test_features[['Fare', 'Age']])
X_train_features.head(5)


# Finally, let's one-hot-encode the passenger class:

# In[ ]:


onehotencoder = OneHotEncoder(categorical_features = [1])
X_train_enc = onehotencoder.fit_transform(X_train_features).toarray()
X_test_enc = onehotencoder.transform(X_test_features).toarray()
print(X_train_enc[0, :])


# Now let's assess a number of different classfiers. Since we have a small training data set, we will use KFold cross validaton to get a more representative assessment of each model performance. 

# First let's start with the basic [Kneighbors Classifier] (http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html):

# In[ ]:


def kfold_assessment(clf):
    k_fold = KFold(5)
    for k, (train, val) in enumerate(k_fold.split(X_train_enc, y_train)):
        clf.fit(X_train_enc[train], y_train[train])
        print("[fold {0}],  score: {1:.5f}".format(k, clf.score(X_train_enc[val], y_train[val])))

kn_clf = KNeighborsClassifier(n_neighbors=3)
kfold_assessment(kn_clf)


# Next, let's have a look how a Logistic Regression classfier (http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) is performing

# In[ ]:


lr_clf = LogisticRegression(solver="lbfgs", C=10)
kfold_assessment(lr_clf)


# Next, let's have a look at probabilistic Gaussian Naive Bayes classifier performance (http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html):

# In[ ]:


gnb_clf = GaussianNB()
kfold_assessment(gnb_clf)


# Next, let's see how Linear Support Vector Machine classifier is performing (http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)

# In[ ]:


svm_clf = LinearSVC(C=1)
kfold_assessment(svm_clf)


# Next, let's start having a look at ensemble learners. First let's start with [Random Forest Classifier] (http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html):

# In[ ]:


rf_clf = RandomForestClassifier(max_depth=6, n_estimators=100, n_jobs=-1)
kfold_assessment(rf_clf)


# Next - the AdaBoost Classifier (http://scikit-learn.org/0.15/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)

# In[ ]:


ab_clf = AdaBoostClassifier()
kfold_assessment(ab_clf)


# And finally - the Gradient Boosting Classifier (http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)

# In[ ]:


gb_clf = GradientBoostingClassifier()
kfold_assessment(gb_clf)


# We can see that all of them have comparable performances with only notably poorer performance of the Naive Bayes Classifier. Since they arive at the assessment using different techniques, they will tend to make different errors as well. If we combine all the models in some way, we usually get a better performance than from any individual model, since the errors of any individual model are in a way 'compensated for' by the other models. This is the whole idea behind ensemble learning. In this particular case, we can use a soft voting classifier, that looks at probabilities of each individual classifier prediction - the more confident the predictor the more weight its 'vote' gets in the ensemble.

# In[ ]:


classifiers = [
    ('KNeighbors', KNeighborsClassifier(n_neighbors=3)), 
    ('LogisticRegression', LogisticRegression(solver="lbfgs", C=10)), 
    ('GaussianNB', GaussianNB()),
    ('SupportVectorMachine', SVC(probability=True)),
    ('Random Forest', RandomForestClassifier(max_depth=4, n_estimators=100, n_jobs=-1)), 
    ('AdaBoost', AdaBoostClassifier()), 
    ('GradientBoosting', GradientBoostingClassifier())
]
vc = VotingClassifier(estimators=classifiers, voting='soft')
vc = vc.fit(X_train_features, y_train)

preds = vc.predict(X_test_features)


# Let's produce the output file:

# In[ ]:


submission = pd.DataFrame({
        "PassengerId": data_test["PassengerId"],
        "Survived": preds
    })
submission.to_csv('titanic.csv', index=False)


# This gives the 0.78468 score on the leaderboard. 

# ## Suggestions for further improvements in leaderboard score:

# 1. Hyperparameter tuning - I haven't done any hyperparameter tuning above. I included the links to the docs of each of the models utilised above, have a look what hyperparameters can be specificied for each of the models and try to find the optimal ones using GridSearchCV. 
# 2. Other models - the above is not an exhaustive list of models that can be used. For example, you can also have a look at a basic neural net model. Keras has a scikit-learn API but I found it doesn't play well with the scikit-learn's Voting Classifier since Keras model's don't return a flat array. Hence, you might need to build your own Voting Classfier implementation if you are exploring beyond scikit-learn's own models. 
