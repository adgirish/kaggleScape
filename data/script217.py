
# coding: utf-8

# # Table of Contents
# 
# ### 1. Introduction

# ### 2. Exploratory Data Analysis (EDA)
# * [Import Libraries and Settings](#import-lib)
# * [Import Dataset](#import-data)
# * [Dataset Checking](#data-check)
# * [Features Visualization](#feature-visual)

# ### 3. Feature Engineering
# 
# * [Fill the Missing data](#fill-missing-data)
# * [Reshape Dataset](#reshape-data)
# * [Dataset Cleanup and Spliting](#cleanup-split)
# * [Encoding Data for Analysis](#cleanup-split)
# * [Check for Correlations](#correlation)
# * [Feature Importances](#importance)
# * [Feature Selection](#feature-selection)
# * [Dummy Variables Encoding](#dummy).

# ### 4. Machine Learning
# * [Fundamental](#fundamental)
# * [Hyper Parameters Tuning by Grid Search](#grid-search) 
# * [Train the Algorithms with Optimized Parameters](#optimized-train) 
# * [Submit the Prediction](#submission)
# 

# ### 5. Closing
# * [Room for Improvements](#improvement-room)
# * [Kernel References](#kernel-refs)
# * [References](#refs)

# # 1. Introduction
# 
# In this notebook, I start with exploratory data analysis (EDA) in order to understand the problem and find the hidden information inside each feature. Second, I reengineer the existing features to modify them and create new features to better explain the dataset to machine learning mode. When the dataset is ready, I teach the machine learning models, in this case, classifiers with the dataset  I prepared. Lastly I use the ensemble technique to try reaching out for the better score on test dataset.
# 
# After study many things from visualization to feature engineering for 2 weeks, I come back and attempt to redesign this notebook into proper steps of data science pipeline, while focusing more on data analysis and feature engineering, which are the most important steps to improve machine learning model's performance according to many articles written by data scientists.
# 
# **Change Logs**
# 
# **2018/02/21**: Fix xgboost of ensemble part. 2000 estimators broke runtime limit, so I reduced it to 800 estimators.

# # 2. Exploratory Data Analysis (EDA)

# ## Import Libraries and Settings <a class="anchor" id="import-lib" ></a>

# In[ ]:


# Basic Libraries
import numpy as np 
import pandas as pd 

# Feature Scaling
from sklearn.preprocessing import RobustScaler

# Visaulization
import matplotlib.pyplot as plt
import seaborn as sns

# Classifier (machine learning algorithm) 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Evaluation
from sklearn.model_selection import cross_val_score, cross_val_predict

# Parameter Tuning
from sklearn.model_selection import GridSearchCV

# Settings
pd.options.mode.chained_assignment = None # Stop warning when use inplace=True of fillna


# ## Import Dataset <a class="anchor" id="import-data"></a>

# In[ ]:


train_set =  pd.read_csv('../input/train.csv')
test_set =  pd.read_csv('../input/test.csv')


# ## Dataset Checking <a class="anchor" id="data-check"></a>
# 
# First, lets take a look on how training dataset and test dataset look like.

# In[ ]:


train_set.head()


# In[ ]:


test_set.head()


# Throughout this notebook, I will pretend that testing set (test_set) is never exists until the model is trained, to simulates the real-life scenario where the data to be predicted comes later.
# 
# Next, I check the details about both training dataset and test dataset. I hide the result of each code block to save up space. Click on "Output" tab on the right-hand side of each code block to see these details.
# 
# Also I use .isnull().sum() to check for the missing data (NaN).

# In[ ]:


len(train_set)


# In[ ]:


train_set.describe()


# In[ ]:


train_set.isnull().sum()


# In[ ]:


len(test_set)


# In[ ]:


test_set.describe()


# In[ ]:


test_set.isnull().sum()


# We found that there are missing data in Age, Cabin and Embarked,  especially many missing data in Cabin column of both train_set and test_set. I did a little search about Titanic's cabin here. I found the fact and even a discussion on Kaggle about the case.
# 
# [Suites and Cabins for Passengers on the Titanic](http://www.dummies.com/education/history/suites-and-cabins-for-passengers-on-the-titanic/)
# 
# [Is cabin an important predictor?](https://www.kaggle.com/c/titanic/discussion/4693)
# 
# In my conclusion, everyone on the Titanic, regardless of their classes and whether they are passengers or crews, SHOULD have their own cabin to sleep. But for some reason, most of the cabin data are missing.

# ## Features Visualization <a class="anchor" id="feature-visual"></a>
#  
# The first thing to do is to visualize the data and find any valuable/hidden information inside each feature. 
# In this section, I tried to visualize existing data first without filling the missing data yet to keep things in steps, hence I won't process text feature like Name yet, so there is no visualization for that feature.
# 
# I use these methods to create graphs based on type of each feature.

# In[ ]:


# Continuous Data Plot
def cont_plot(df, feature_name, target_name, palettemap, hue_order, feature_scale): 
    df['Counts'] = "" # A trick to skip using an axis (either x or y) on splitting violinplot
    fig, [axis0,axis1] = plt.subplots(1,2,figsize=(10,5))
    sns.distplot(df[feature_name], ax=axis0);
    sns.violinplot(x=feature_name, y="Counts", hue=target_name, hue_order=hue_order, data=df,
                   palette=palettemap, split=True, orient='h', ax=axis1)
    axis1.set_xticks(feature_scale)
    plt.show()
    # WARNING: This will leave Counts column in dataset if you continues to use this dataset

# Categorical/Ordinal Data Plot
def cat_plot(df, feature_name, target_name, palettemap): 
    fig, [axis0,axis1] = plt.subplots(1,2,figsize=(10,5))
    df[feature_name].value_counts().plot.pie(autopct='%1.1f%%',ax=axis0)
    sns.countplot(x=feature_name, hue=target_name, data=df,
                  palette=palettemap,ax=axis1)
    plt.show()

    
survival_palette = {0: "black", 1: "orange"} # Color map for visualization


# ### **Pclass** (Ticket class)

# In[ ]:


cat_plot(train_set, 'Pclass','Survived', survival_palette)


# If you don't already know yet, the Pclass = 1 are the person who paid for their ticket in high-class, while Pclass = 2 are the middle-class and Pclass = 3 are the less fortunate class. 
# 
# Try to look at how much different between survived and not survived within each class to see if this entire feature has impact on survivability or not. You can see that it has so much differences between classes here, regardless of numbers of people, Pclass1 is the only class which has more survivors, not much different on Pclass2 and only few people from Pclass3 are survived.
# 
# When being in Pclass MATTER for each person to survive or not, you know that this feature is important.
# 
# ### **Sex**

# In[ ]:


cat_plot(train_set, 'Sex','Survived', survival_palette)


# Female clearly has better survival chance than male.
# 
# ### **Age**
# As I checked above, Age has some missing data. Visualization method can't deal with missing data, so I drop (delete) the rows with missing data temporally. You see that I didn't overwrite train_set and create new dataframe for visualization instead (age_set_nonan).

# In[ ]:


age_set_nonan = train_set[['Age','Survived']].copy().dropna(axis=0)
cont_plot(age_set_nonan, 'Age', 'Survived', survival_palette, [1, 0], range(0,100,10))


# Because of the distribution of Seaborn's violinplot, the graph is shifting to the left side of Age=0 here. Try to press the graph back to the right in your mind when reading this.
# 
# You can see that there were more survivors around age lower than 15, then suddenly more person died on around age 15 to 35. After age 35 numbers of survivors and the losts are equal again. Lastly after around age 60, they rarely got survivors on those ages.
# 
# With this information, I think it is a good idea to classify them into categorical feature based on these differences in rough survivors/losts ratio. The range of age I decided to go for are 
# 1. less than 15 years  
# 2. from 15 to 35 years  
# 3. from 35 to 60 years 
# 4. more than 60 years
# 
# ### **SipSp**

# In[ ]:


cat_plot(train_set, 'SibSp', 'Survived', survival_palette)


# Only people with 1 sibling or spouse has a bit more survivors compares to person who were alone or had more than that.
# 
# ### **Parch**

# In[ ]:


cat_plot(train_set, 'Parch', 'Survived', survival_palette)


# Similar to SibSp, the person with few numbers parents/children on board had the best survivability.
# 
# Because SibSp and Parch has a very close meaning, "Family". If I combine these 2 features together, maybe I could see more differences in each classes of these features. That's the idea I got from [Anisotropic](https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python/) and this will be used on the next part..
# 
# ### **Fare**

# In[ ]:


fare_set = train_set[['Fare','Survived']].copy() # Copy dataframe so method won't leave Counts column in train_set
cont_plot(fare_set, 'Fare', 'Survived', survival_palette, [1, 0], range(0,550,50))


# Althrough the graph has clear difference here, but lets zoom-in to check. 

# In[ ]:


fare_set_mod = train_set[['Fare','Survived']].copy()
fare_set_mod['Counts'] = "" 
fig, axis = plt.subplots(1,1,figsize=(10,5))
sns.violinplot(x='Fare', y="Counts", hue='Survived', hue_order=[1, 0], data=fare_set_mod,
               palette=survival_palette, split=True, orient='h', ax=axis)
axis.set_xticks(range(0,100,10))
axis.set_xlim(-20,100)
plt.show()


# It is harder to read than Age's graph. As before, try to shift graph back to the right. I estimate that the peak of black graph are at Fare = 20.
# 
# Similar to Age, there were more survivors on the first part of graph on Fare less than 10. Numbers of person who died peak at around Fare 10 to 30 and gradually reduce as Fare goes higher.
# 
# I decided to categorize Fare into ranges of
# 1. less than Fare = 10
# 2. from Fare = 10 to 30
# 3. from Fare = 35 to 60
# 4. Fare =  60 or more
# 
# ### **Embarked**
# The same as Age, there are missing data in Embarked.

# In[ ]:


emb_set_nonan = train_set[['Embarked','Survived']].copy().dropna(axis=0)
cat_plot(train_set, 'Embarked','Survived', survival_palette)


# About 2 of 3 of people from port S and Q didn't survive, while not much different in people from port C. 
# 
# When combine with domain knowledge of the situation on Titanic, I couldn't see how embarking from different port impact on survivability on a ship. Unless passengers from port C has some kind of privilege for the seat on rescue boat, this probably be just a coincidence.
# 
# ### **Ticket**
# 
# For Ticket feature I could not find any information or pattern that could be extracted from them, so it will dropped later.

# # 3. Feature Engineering
# 
# As I heard from many data scientists, this is the most important part, which has the most impact on the accuracy of machine learning model and there is a reason for that. 
# 
# Try to imagine that you are a teacher and machine learning models are students. If you just read the textbook to the student, they probably won't learn that much. But if you prepare your lesson before class, categorize the information so the students can see the differences and memorize them better. The result is much better that way.
# 
# The same as in machine learning, instead of feeding raw data to the model, you can make it better by modify the existing features and/or create new features. We can use the information from the previous EDA part to help us decide how to do that.

# ## Fill the Missing data <a class="anchor" id="fill-missing-data"></a>
# 
# About test_set, I said that I will pretend that test_set doesn't exist before I created the models. But in real practice, I will have to do these same processes on test_set anyway, so I'm doing them right now.

# In[ ]:


train_set.describe()


# In[ ]:


test_set.describe()


# I decide to fill missing data in Age and Fare by median value. While filling Embarked with the same value as the most occurance.
# 
# About Cabin, since there are too many missing data, filling them with the most occurance might make it very biased. So I decide to make a new feature called "HasCabin" instead (0 if Cabin data is missing, and 1 if it has Cabin data).

# In[ ]:


combined_set = [train_set, test_set] # combined 2 datasets for more efficient processing

for dataset in combined_set:
    dataset["Age"].fillna(dataset["Age"].median(), inplace=True)
    dataset["Fare"].fillna(dataset["Fare"].median(), inplace=True)

train_set["Embarked"].fillna(train_set["Embarked"].value_counts().index[0], inplace=True)


# Re-check for missing data.

# In[ ]:


train_set.isnull().sum()


# In[ ]:


test_set.isnull().sum()


# Only Cabin has missing data, but I'm going to use HasCabin instead and drop Cabin later.
# 
# ## Reshape Dataset <a class="anchor" id="reshape-data"></a>
# 
# Next, I categorize Age and Fare as I made a decision on EDA part. Combine SipSp and Parch by sum them up.
# 
# For Name feature, I got the idea from other Kagglers and articles to extract Titles from name. But with my own modifications.

# In[ ]:


age_bins = [0,15,35,45,60,200]
age_labels = ['15-','15-35','35-45','40-60','60+']
fare_bins = [0,10,30,60,999999]
fare_labels = ['10-','10-30','30-60','60+']

def get_title(dataset, feature_name):
    return dataset[feature_name].map(lambda name:name.split(',')[1].split('.')[0].strip())

for dataset in combined_set:
    dataset['AgeRange'] = pd.cut(dataset['Age'], bins=age_bins, labels=age_labels, include_lowest=True)
    dataset['FareRange'] = pd.cut(dataset['Fare'], bins=fare_bins, labels=fare_labels, include_lowest=True)
    dataset['FamilySize'] = dataset['SibSp'] + train_set['Parch']
    dataset['HasCabin'] = dataset['Cabin'].notnull().astype(int) # NaN Cabins will become 0, otherwise 1
    dataset['Title'] = get_title(dataset, 'Name')


# Lets check how these new features affect the survival rate
# 
# ### **AgeRange**

# In[ ]:


cat_plot(train_set, 'AgeRange','Survived', survival_palette)


# ### **FareRange**

# In[ ]:


cat_plot(train_set, 'FareRange','Survived', survival_palette)


# ### **FamilySize**

# In[ ]:


cat_plot(train_set, 'FamilySize','Survived', survival_palette)


# ### **HasCabin**

# In[ ]:


cat_plot(train_set, 'HasCabin','Survived', survival_palette)


# ### **Title**

# In[ ]:


fig, axis = plt.subplots(1,1,figsize=(12,5))
sns.countplot(x='Title', hue='Survived', data=train_set,
                  palette=survival_palette,ax=axis)
plt.show()

print(train_set['Title'].value_counts())


# We can see that there are difference in survivability on each new features. 
# 
# For FamilySize we can see more difference than before, but I could make them more categorized but grouping numbers of family with the same survivors ratio together.

# In[ ]:


for dataset in combined_set:
    dataset['Family'] = ''
    dataset.loc[dataset['FamilySize'] == 0, 'Family'] = 'alone'
    dataset.loc[(dataset['FamilySize'] > 0) & (dataset['FamilySize'] <= 3), 'Family'] = 'small'
    dataset.loc[(dataset['FamilySize'] > 3) & (dataset['FamilySize'] <= 6), 'Family'] = 'medium'
    dataset.loc[dataset['FamilySize'] > 6, 'Family'] = 'large'


# In[ ]:


cat_plot(train_set, 'Family','Survived', survival_palette)


# For Title, I think it has too many varies here with very low occurances, so I try to categorize them too.
# I got an idea of how to process Title [here](https://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html)
# but with my own modifications and my own researchs
# 
# https://en.wikipedia.org/wiki/French_honorifics
# 
# https://en.wikipedia.org/wiki/The_Reverend

# In[ ]:


title_dict = {
                "Mr" :        "Mr",
                "Miss" :      "Miss",
                "Mrs" :       "Mrs",
                "Master" :    "Master",
                "Dr":         "Scholar",
                "Rev":        "Religious",
                "Col":        "Officer",
                "Major":      "Officer",
                "Mlle":       "Miss",
                "Don":        "Noble",
                "the Countess":"Noble",
                "Ms":         "Mrs",
                "Mme":        "Mrs",
                "Capt":       "Noble",
                "Lady" :      "Noble",
                "Sir" :       "Noble",
                "Jonkheer":   "Noble"
            }

for dataset in combined_set:
    dataset['TitleGroup'] = dataset.Title.map(title_dict)


# Because I used handwriting dictionary to create TitleGroup feature, there might be some titles which only exists in test_set and will be converted to NaN value. Lets check for those Title and fill it with proper TitleGroup, in real practice we will have to do this when we process the test_set.

# In[ ]:


print(test_set[test_set['TitleGroup'].isnull() == True])


# In[ ]:


test_set.at[414, 'TitleGroup'] = 'Noble' # A record with Dona title


# Lets check TitleGroup, I saw that there are so many Mr. title with Survived = 0, so I set y limit to see shorter bars better.

# In[ ]:


fig, axis = plt.subplots(1,1,figsize=(12,5))
sns.countplot(x='TitleGroup', hue='Survived', data=train_set,
              palette=survival_palette,ax=axis)
axis.set_ylim(0, 200)
plt.show()


# ## Dataset Cleanup and Spliting<a class="anchor" id="cleanup-split"></a>
# 
# I cleanup unused or transformed features and split Survived column into a series, so both training dataset and test dataset are identical.

# In[ ]:


X_train = train_set.drop(['Survived','PassengerId','Name','Age','Fare','Ticket','Cabin','SibSp','Parch','Title','FamilySize'], axis=1)
X_test = test_set.drop(['PassengerId','Name','Age','Fare','Ticket','Cabin','SibSp','Parch','Title','FamilySize'], axis=1)

y_train = train_set['Survived']  # Relocate Survived target feature to y_train


# ## Encoding Data for Analysis <a class="anchor" id="encoding"></a>
# Many methods of Python libraries don't accept text input, so I need to encode categorical features into ordinal numbers. I encode them manually, because I want to assign the label's order. 
# 
# Also I use a copy of training dataset. I will tell you the reason I don't overwrite training dataset with ordinal numbers later.

# In[ ]:


X_train_analysis = X_train.copy()
X_train_analysis['Sex'] = X_train_analysis['Sex'].map({'male': 0, 'female': 1}).astype(int)
X_train_analysis['Embarked'] = X_train_analysis['Embarked'].map({'C': 0, 'Q': 1, 'S': 2}).astype(int)
X_train_analysis['Family'] = X_train_analysis['Family'].map({'alone': 0, 'small': 1, 'medium': 2, 'large': 3}).astype(int)

agerange_dict = dict(zip(age_labels, list(range(len(age_labels)))))
X_train_analysis['AgeRange'] = X_train_analysis['AgeRange'].map(agerange_dict).astype(int)

farerange_dict = dict(zip(fare_labels, list(range(len(fare_labels)))))
X_train_analysis['FareRange'] = X_train_analysis['FareRange'].map(farerange_dict).astype(int)

titlegroup_labels = list(set(title_dict.values()))
titlegroup_dict = dict(zip(titlegroup_labels, list(range(len(titlegroup_labels)))))
X_train_analysis['TitleGroup'] = X_train_analysis['TitleGroup'].map(titlegroup_dict).astype(int)


# ## Check for Correlations <a class="anchor" id="correlation"></a>
# 
# When 2 features or more have correlation, that means they are explaining each others while giving only a few or none of new information. Try to imagine if TitleGroup feature only has 2 classes, 'Mr.' and 'Miss.'. We can be sure that all male data would have Mr. title and all female have Miss. title. Features with correlation would lead to overfitting on machine learning model, which might result in high accuracy on training dataset while decrease accuracy on test dataset.
# 
# But even there are the correlations, we can't carelessly cut down the features. As I saw comments from Anton Lytyakov and GeekYoung. I start researching and study how [LongYin](https://www.kaggle.com/longyin2/titanic-machine-learning-from-disaster-0-842/notebook) did with his Random Forest classifier. I put some features I dropped in the early version back, and found that accuracy is actually improved.
# 
# The lesson I learned is not to drop features by judging too early, atleast until I done the analysis is done.. 
# 

# In[ ]:


colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Correlation between Features', y=1.05, size = 15)
sns.heatmap(X_train_analysis.corr(),
            linewidths=0.1, 
            vmax=1.0, 
            square=True, 
            cmap=colormap, 
            linecolor='white', 
            annot=True)


# The one with the most positive correlation here is 0.59 between FareRange VS HasCabin and the most negative correlation is -0.73 between HasCabin and FareRange VS Pclass. Positive correlation means when you increase one of them, another will also increase. Negative correlation means when you increase one of them anothe will decrease instead.
# 
# ## Feature Importances <a class="anchor" id="importance"></a>
# 
# Another way to analyze how much impact a feature could make on target feature (Survived) is to use feature_importances_ property from machine learning algorithm library.

# In[ ]:


rforest_checker = RandomForestClassifier(random_state = 0)
rforest_checker.fit(X_train_analysis, y_train)
importances_df = pd.DataFrame(rforest_checker.feature_importances_, columns=['Feature_Importance'],
                              index=X_train_analysis.columns)
importances_df.sort_values(by=['Feature_Importance'], ascending=False, inplace=True)
print(importances_df)


# The result here is quite random. But the same thing is TitleGroup is always the most important. 
# 
# This is what I got at the time I wrote this.
# 

# In[ ]:


my_imp_dict = {'Feature Importance' : pd.Series([0.360313, 0.113686, 0.109495, 0.103845, 0.100966, 0.099818, 0.056429, 0.055449],
             index=['TitleGroup', 'Family', 'Pclass', 'Sex','FareRange', 'AgeRange', 'HasCabin', 'Embarked'])}
my_imp_df = pd.DataFrame(my_imp_dict)
print(my_imp_df)


# ## Feature Selection <a class="anchor" id="feature-selection"></a>
# 
# The time has come to select features to use for training model. I decide to use all the current features except HasCabin and Embarked. There are the reasons for that.
# 
# 1. HasCabin and Embarked got low importance score comparing to the rest.
# 2. HasCabin is one of the features with high correlation.
# 3. Embarked make no sense of how it could impact survivability.

# In[ ]:


X_train = X_train.drop(['HasCabin','Embarked'], axis=1)
X_test = X_test.drop(['HasCabin','Embarked'], axis=1)


# So these are the features I use in this run.

# In[ ]:


# TitleGroup, Family, Pclass, Sex, FareRange, AgeRange


# ## Dummy Variables Encoding <a class="anchor" id="dummy"></a>
# 
# Machine learning algorithms cannot process text and categorical variables, unless they have build-in function for it. So we have to convert categorical variables into numerical variables. But encoding them to ordinal values could make algorithm bias on how large the numbers are, this is why I don't encode train_set with ordinal numbers. So I convert them into dummy variables which are binary.

# In[ ]:


X_train = pd.get_dummies(X_train, columns=['TitleGroup','Family','Pclass','Sex','AgeRange','FareRange'])
X_test = pd.get_dummies(X_test, columns=['TitleGroup','Family','Pclass','Sex','AgeRange','FareRange'])


# The multi-collinearity is what we want to avoid when using dummy variables. When 2 or more variables are heavily correlated to each other, for example, 'Sex' feature when converted it turns into Sex_male and Sex_female dummy variables. When Sex_male = 0, you can be sure that Sex_female is always 1 and vice versa. In other words, we have duplicated variables which are always different. This situation is also called "Dummy Variable Trap".
# 
# We can avoid this by exclude any 1 dummy for each feature. In this case, I chose to cut the first dummy variable of each feature.

# In[ ]:


X_train = X_train.drop(['Pclass_1','Sex_female','TitleGroup_Master','AgeRange_15-','FareRange_10-','Family_alone'], axis=1)
X_test = X_test.drop(['Pclass_1','Sex_female','TitleGroup_Master','AgeRange_15-','FareRange_10-','Family_alone'], axis=1)


# These are the remaining categorical features as dummy variables in our dataset.

# In[ ]:


X_train.head()


# # 4. Machine Learning
# 
# ## Fundamental <a class="anchor" id="fundamental"></a>
# 
# The following is the usual way to train the classifier algorithm (model) and submit the prediction. In this example, we use Logistic Regression classifier.

# In[ ]:


# Train the Classifier
## classifier = LogisticRegression()
## classifier.fit(X_train, y_train)

# Predict the Test Data
## y_pred = classifier.predict(X_test)

# Submit Prediction Result
## passengerId = test_set['PassengerId']
## submission = pd.DataFrame({ 'PassengerId' : passengerId, 'Survived' : y_pred })
## submission.to_csv('submission.csv', index=False)


# But because completition's data provider does not give us a correct answer for the prediction. We have 3 options to test accuracy of each classifier to see which one has the best accuracy and the least variance.
# 
# 1. Use training set to train and try to predict the same training set
# 2. Split new training set and test set from the current training set (in this case, X_train and y_train).
# 3. Use k-fold Cross Validation technique.
# 
# The first option tends to deliver a high accuracy, but not because the model is good, but because the model is trying to predict the data, which it already have seen. This option should be avoided at all cost.
# 
# The second  option is quite simple to do by using train_test_split library, but the problem is sometime training set and data set are not split evenly and give us a very biased data. For example, a new training set with Pclass = 1 and 2 but has no Pclass = 3 at all. Which could make us misjudge about which model is the best.
# 
# The third  option, k-fold Cross Validation, split training data into k splits then train and make prediction k times. This technique helps us to reduce the chance of overfitting and biased data.
# 
# The following is the proceduces of how to use k-fold Cross Validation.

# In[ ]:


## classifier = LogisticRegression()

# First we need to train the classifier as usual
## classifier.fit(X_train, y_train)

# estimator = the classifier algorithm to use, cv = number of cross validation split
## acc_logreg = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

# You can check the accuracy score for each split. In this case, 10 accuracy scores
## print(acc_logreg)

# Get mean of accuracy score of all cross validations
## acc_logreg.mean() 

# Standard deviation = differences of the accuracy score in each cross validations. the less = less variance = the better
## acc_logreg.std() 


# Next, we are going to train and validate these classification models.
# 1. Logistic Regression
# 2. Kernel Support Vector Machine (Kernel SVM)
# 4. Decision Tree
# 5. Random Forest
# 
# In this version of notebook, instead of train the models first, I use Grid Search technique to find the best hyper parameters for each model first while check for accuracy on training set. 
# 
# After that, I train all models with their best hyper parameters
# 
# If you want to see what I did with other algorthms, please check the older version of notebook, such as version 56. 

# ## Hyper Parameters Tuning by Grid Search <a class="anchor" id="grid-search"></a>
# The hyper parameters such as, in Random Forest, n_estimators and criterion can be changed to improve prediction accuracy of the model. But this process, when repeat for multiple times, is really time-comsuming, so instead we use a library called Grid Search to help us test different hyper parameters all at once.
# 
# Note that you can also try other hyper parameters and other values than what listed below too. But beware that if you include too many parameters and values, the process time could be very long.
# 
# If you want to know what I did with parameters tuning on other algorithms, please check the version 56.

# ### Logistics Regression
# 
# The hyper parameters which could be adjusted to improve accuracy for Logistics Regression are...
# 
# 1. C
# 2. penalty

# In[ ]:


params_logreg = [{'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1','l2']}]
grid_logreg = GridSearchCV(estimator = LogisticRegression(),
                           param_grid = params_logreg,
                           scoring = 'accuracy',
                           cv = 10)
grid_logreg = grid_logreg.fit(X_train, y_train)
best_acc_logreg = grid_logreg.best_score_
best_params_logreg = grid_logreg.best_params_


# ### Kernel SVM
# 
# According to this [link](https://stackoverflow.com/questions/12616492/scikit-learns-gridsearchcv-with-linear-kernel-svm-takes-too-long) Grid Search on Kernel SVM will take absurdly a lot of process time if you don't normalize the data with continuous feature first.

# In[ ]:


"""
X_train_norm = X_train.copy()
X_train_norm = X_train_norm.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
X_train_norm.head()
"""


# But since there is no continuous feature on the dataset, I don't need to normalize it.

# Parameters to adjust in Kernel SVM are as follows
# 1. C
# 2. kernel
# 3. degree (only for poly kernel)
# 4. gamma (only for rbf, poly and sigmoid kernel
# 
# Each inside {} is a branch of parameters we are trying. For example, the first branch we are trying linear kernel with different C value. The second branch we are trying rbf kernel with different C and gamma. The third branch we are trying with different C, degree and gamma. This way we can avoid using unnecessary parameters, in this case, degree and gamma on linear kernel.

# In[ ]:


params_ksvm = [{'C': [0.1, 1, 10, 100], 'kernel': ['linear']},
               {'C': [0.1, 1, 10, 100], 'kernel': ['rbf'],
                'gamma': [0.1, 0.2, 0.3, 0.4, 0.5]},
               {'C': [0.1, 1, 10, 100], 'kernel': ['poly'],
                'degree': [1, 2, 3],
                'gamma': [0.1, 0.2, 0.3, 0.4, 0.5]}]
grid_ksvm = GridSearchCV(estimator = SVC(random_state = 0),
                         param_grid = params_ksvm,
                         scoring = 'accuracy',
                         cv = 10,
                         n_jobs=-1)
grid_ksvm = grid_ksvm.fit(X_train, y_train)  # Replace X_train with X_train_norm here if you need
best_acc_ksvm = grid_ksvm.best_score_
best_params_ksvm = grid_ksvm.best_params_


# ### Decision Tree
# 
# According to [LongYin](https://www.kaggle.com/longyin2/titanic-machine-learning-from-disaster-0-842/notebook) and also my own testings. min_samples_split, min_samples_leaf and max_features are also important in Random Forest and Decision Tree.
# 
# Note that on max_features, the algorithm will not allow us to input 'None' to it.

# In[ ]:


params_dtree = [{'min_samples_split': [5, 10, 15, 20],
                 'min_samples_leaf': [1, 2, 3],
                 'max_features': ['auto', 'log2']}]
grid_dtree = GridSearchCV(estimator = DecisionTreeClassifier(criterion = 'gini', 
                                                             random_state = 0),
                            param_grid = params_dtree,
                            scoring = 'accuracy',
                            cv = 10,
                            n_jobs=-1)
grid_dtree = grid_dtree.fit(X_train, y_train)
best_acc_dtree = grid_dtree.best_score_
best_params_dtree = grid_dtree.best_params_


# ### Random Forest
# 
# The same as Decision Tree, but also has n_estimators.

# In[ ]:


params_rforest = [{'n_estimators': [200, 300],
                   'max_depth': [5, 7, 10],
                   'min_samples_split': [2, 4]}]
grid_rforest = GridSearchCV(estimator = RandomForestClassifier(criterion = 'gini', 
                                                               random_state = 0, n_jobs=-1),
                            param_grid = params_rforest,
                            scoring = 'accuracy',
                            cv = 10,
                            n_jobs=-1)
grid_rforest = grid_rforest.fit(X_train, y_train)
best_acc_rforest = grid_rforest.best_score_
best_params_rforest = grid_rforest.best_params_


# Because it taking too long to process, I put a full list of parameters for Random Forest here, while reduce above parameters and values but still including the best one.

# In[ ]:


""" params_rforest = [{'n_estimators': [100, 200, 500, 800], 
                   'min_samples_split': [5, 10, 15, 20],
                   'min_samples_leaf': [1, 2, 3],
                   'max_features': ['auto', 'log2']}] """


# ### Grid Search Score

# In[ ]:


grid_score_dict = {'Best Score': [best_acc_logreg,best_acc_ksvm,best_acc_dtree,best_acc_rforest],
                   'Optimized Parameters': [best_params_logreg,best_params_ksvm,best_params_dtree,best_params_rforest],
                  }
pd.DataFrame(grid_score_dict, index=['Logistic Regression','Kernel SVM','Decision Tree','Random Forest'])


# For parameters of classifiers which are too long to show in table.

# In[ ]:


best_params_dtree


# In[ ]:


best_params_rforest


# ## Train the Algorithms with Optimized Parameters <a class="anchor" id="optimized-train"></a>
# 
# After trained the models, I keep both predict accuracy on training dataset via cross validation (y_pred_train) and prediction on test dataset (y_pred_test) for use in the next section.

# In[ ]:


logreg = LogisticRegression(C = 1, penalty = 'l1')
logreg.fit(X_train, y_train)
y_pred_train_logreg = cross_val_predict(logreg, X_train, y_train)
y_pred_test_logreg = logreg.predict(X_test)


# In[ ]:


ksvm = SVC(C = 1, gamma = 0.2, kernel = 'rbf', random_state = 0)
ksvm.fit(X_train, y_train)   # Replace X_train with X_train_norm here if you need
y_pred_train_ksvm = cross_val_predict(ksvm, X_train, y_train)
y_pred_test_ksvm = ksvm.predict(X_test)


# In[ ]:


dtree = DecisionTreeClassifier(criterion = 'gini', max_features='auto', min_samples_leaf=1, min_samples_split=5, random_state = 0)
dtree.fit(X_train, y_train)
y_pred_train_dtree = cross_val_predict(dtree, X_train, y_train)
y_pred_test_dtree = dtree.predict(X_test)


# In[ ]:


rforest = RandomForestClassifier(max_depth = 7, min_samples_split=4, n_estimators = 200, random_state = 0) # Grid Search best parameters
rforest.fit(X_train, y_train)
y_pred_train_rforest = cross_val_predict(rforest, X_train, y_train)
y_pred_test_rforest = rforest.predict(X_test)


# ## Ensemble Models
# I learned this techique from  [Anisotropic](https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python/). Please refers to his notebook for more information. But in my case, I decided to try and use predictions from cross_val_predict instead of Out-of-Folds predictions.

# In[ ]:


second_layer_train = pd.DataFrame( {'Logistic Regression': y_pred_train_logreg.ravel(),
                                    'Kernel SVM': y_pred_train_ksvm.ravel(),
                                    'Decision Tree': y_pred_train_dtree.ravel(),
                                    'Random Forest': y_pred_train_rforest.ravel()
                                    } )
second_layer_train.head()

X_train_second = np.concatenate(( y_pred_train_logreg.reshape(-1, 1), y_pred_train_ksvm.reshape(-1, 1), 
                                  y_pred_train_dtree.reshape(-1, 1), y_pred_train_rforest.reshape(-1, 1)),
                                  axis=1)
X_test_second = np.concatenate(( y_pred_test_logreg.reshape(-1, 1), y_pred_test_ksvm.reshape(-1, 1), 
                                 y_pred_test_dtree.reshape(-1, 1), y_pred_test_rforest.reshape(-1, 1)),
                                 axis=1)

xgb = XGBClassifier(
        n_estimators= 800,
        max_depth= 4,
        min_child_weight= 2,
        gamma=0.9,                        
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread= -1,
        scale_pos_weight=1).fit(X_train_second, y_train)

y_pred = xgb.predict(X_test_second)


# ## Submit the Predictions <a class="anchor" id="submission"></a>

# ### Submit Prediction Result

# In[ ]:


passengerId = np.array(test_set['PassengerId']).astype(int)
submission = pd.DataFrame({ 'PassengerId' : passengerId, 'Survived' : y_pred })
# Check if dataframe has 418 entries and 2 columns or not
print(submission.shape)

submission.to_csv('submission.csv', index=False)


# ## Current Score <a class="anchor" id="current-score"></a>
# 
# Both are using current features (TitleGroup, Family, Pclass, Sex, FareRange, AgeRange)
# 
# Random Forest without Ensemble : 0.78947 
# 
# Emsemble by XGBoost: 0.79425 [Current Best Score] Improved from 0.77511 after reduce estimators from 2000 to 800

# # 5. Closing
# Up until now I have done everything I think it could be done. My score has been improved from the beginning, but I think I spent too much time trying to raise the accuracy to just above 80%. I decide to move on the next dataset to gain different set of skills. Maybe I could comeback and improve this notebook somedays.

# ## Room for Improvements <a class="anchor" id="improvement-room"></a>
# If you have any recommendation to improve my analysis or data processing please feel free to leave a comment. I'm trying to improve myself as a data scienetist and your help is always welcome.
# 
# *List of things to do*
# - Make Notebook shorter and more  compact

# ## Kernel References <a class="anchor" id="kernel-refs"></a>
# 
# - https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python/
# - https://www.kaggle.com/rajatshah/scikit-learn-ml-from-start-to-finish
# - https://www.kaggle.com/berhag/titanic-machine-learning-algorithms
# - https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python/
# - https://www.kaggle.com/omarelgabry/a-journey-through-titanic
# - https://www.kaggle.com/startupsci/titanic-data-science-solutions
# - https://www.kaggle.com/longyin2/titanic-machine-learning-from-disaster-0-842/notebook
# 
# ## References <a class="anchor" id="refs"></a>
# 
# - http://www.algosome.com/articles/dummy-variable-trap-regression.html
# - https://www.analyticsvidhya.com/blog/2015/06/start-journey-kaggle/#case-3
# - https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
# - https://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html
