
# coding: utf-8

# This document is a thorough overview of my process for building a predictive model for Kaggle's Titanic competition. I will provide all my essential steps in this model as well as the reasoning behind each decision I made. This model achieves a score of 82.78%, which is in the top 3% of all submissions at the time of this writing. This is a great introductory modeling exercise due to the simple nature of the data, yet there is still a lot to be gleaned from following a process that ultimately yields a high score.
# 
# You can get my original code on my GitHub: https://github.com/zlatankr/Projects/tree/master/Titanic  
# You get also read my write-up on my blog:  https://zlatankr.github.io/posts/2017/01/30/kaggle-titanic 

# ### The Problem

# We are given information about a subset of the Titanic population and asked to build a predictive model that tells us whether or not a given passenger survived the shipwreck. We are given 10 basic explanatory variables, including passenger gender, age, and price of fare, among others. More details about the competition can be found on the Kaggle site, [here](https://www.kaggle.com/c/titanic). This is a classic binary classification problem, and we will be implementing a random forest classifer.

# ### Exploratory Data Analysis

# The goal of this section is to gain an understanding of our data in order to inform what we do in the feature engineering section.  
# 
# We begin our exploratory data analysis by loading our standard modules.

# In[ ]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# We then load the data, which we have downloaded from the Kaggle website ([here](https://www.kaggle.com/c/titanic/data) is a link to the data if you need it).

# In[ ]:


train = pd.read_csv(os.path.join('../input', 'train.csv'))
test = pd.read_csv(os.path.join('../input', 'test.csv'))


# First, let's take a look at the summary of all the data. Immediately, we note that `Age`, `Cabin`, and `Embarked` have nulls that we'll have to deal with. 

# In[ ]:


train.info()


# It appears that we can drop the `PassengerId` column, since it is merely an index. Note, however, that some people have reportedly improved their score with the `PassengerId` column. However, my cursory attempt to do so did not yield positive results, and moreover I would like to mimic a real-life scenario, where an index of a dataset generally has no correlation with the target variable.

# In[ ]:


train.head()


# ## Survived
# 
# So we can see that 62% of the people in the training set died. This is slightly less than the estimated 67% that died in the actual shipwreck (1500/2224).

# In[ ]:


train['Survived'].value_counts(normalize=True)


# In[ ]:


sns.countplot(train['Survived'])


# ## Pclass
# 
# Class played a critical role in survival, as the survival rate decreased drastically for the lowest class. This variable is both useful and clean, and I will be treating it as a categorical variable. 

# In[ ]:


train['Survived'].groupby(train['Pclass']).mean()


# In[ ]:


sns.countplot(train['Pclass'], hue=train['Survived'])


# ## Name  
# 
# The `Name` column as provided cannot be used in the model. However, we might be able to extract some meaningful information from it.

# In[ ]:


train['Name'].head()


# First, we can obtain useful information about the passenger's title. Looking at the distribution of the titles, it might be useful to group the smaller sized values into an 'other' group, although I ultimately choose not to do this.

# In[ ]:


train['Name_Title'] = train['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
train['Name_Title'].value_counts()


# I have relatively high hopes for this new variable we created, since the survival rate appears to be either significantly above or below the average survival rate, which should help our model.

# In[ ]:


train['Survived'].groupby(train['Name_Title']).mean()


# Additionally, looking at the relationship between the length of a name and survival rate appears to indicate that there is indeed a clear relationship. What might this mean? Are people with longer names more important, and thus more likely to be prioritized in a shipwreck? 

# In[ ]:


train['Name_Len'] = train['Name'].apply(lambda x: len(x))
train['Survived'].groupby(pd.qcut(train['Name_Len'],5)).mean()


# In[ ]:


pd.qcut(train['Name_Len'],5).value_counts()


# ## Sex

# "Women and children first," goes the famous saying. Thus, we should expect females to have a higher survival rate than males, and indeed that is the case. We expect this variable to be very useful in our model.

# In[ ]:


train['Sex'].value_counts(normalize=True)


# In[ ]:


train['Survived'].groupby(train['Sex']).mean()


# ## Age
# 
# There are 177 nulls for `Age`, and they have a 10% lower survival rate than the non-nulls. Before imputing values for the nulls, we will include an `Age_null` flag just to make sure we can account for this characteristic of the data. 

# In[ ]:


train['Survived'].groupby(train['Age'].isnull()).mean()


# Upon first glance, the relationship between age and survival appears to be a murky one at best. However, this doesn't mean that the variable will be a bad predictor; at deeper levels of a given decision tree, a more discriminant relationship might open up.

# In[ ]:


train['Survived'].groupby(pd.qcut(train['Age'],5)).mean()


# In[ ]:


pd.qcut(train['Age'],5).value_counts()


# ## SibSp  
# 
# Upon first glance, I'm not too convinced of the importance of this variable. The distribution and survival rate between the different categories does not give me much hope.

# In[ ]:


train['Survived'].groupby(train['SibSp']).mean()


# In[ ]:


train['SibSp'].value_counts()


# ## Parch
# 
# Same conclusions as `Sibsp`: passengers with zero parents or children had a lower likelihood of survival than otherwise, but that survival rate was only slightly less than the overall population survival rate. 

# In[ ]:


train['Survived'].groupby(train['Parch']).mean()


# In[ ]:


train['Parch'].value_counts()


# When we have two seemingly weak predictors, one thing we can do is combine them to get a stronger predictor. In the case of `SibSp` and `Parch`, we can combine the two variables to get a 'family size' metric, which might (and in fact does) prove to be a better predictor than the two original variables. 

# ## Ticket  
# 
# The `Ticket` column seems to contain unique alphanumeric values, and is thus not very useful on its own. However, we might be able to extract come predictive power from it. 

# In[ ]:


train['Ticket'].head(n=10)


# One piece of potentially useful informatin is the number of characters in the `Ticket` column. This could be a reflection of the 'type' of ticket a given passenger had, which could somehow indicate their chances of survival. One theory (which may in fact be verifiable) is that some characteristic of the ticket could indicate the location of the passenger's room, which might be a crucial factor in their escape route, and consequently their survival.

# In[ ]:


train['Ticket_Len'] = train['Ticket'].apply(lambda x: len(x))


# In[ ]:


train['Ticket_Len'].value_counts()


# Another piece of information is the first letter of each ticket, which, again, might be indicative of a certain attribute of the ticketholders or their rooms.

# In[ ]:


train['Ticket_Lett'] = train['Ticket'].apply(lambda x: str(x)[0])


# In[ ]:


train['Ticket_Lett'].value_counts()


# In[ ]:


train.groupby(['Ticket_Lett'])['Survived'].mean()


# ## Fare
# 
# There is a clear relationship between `Fare` and `Survived`, and I'm guessing that this relationship is similar to that of `Class` and `Survived`.

# In[ ]:


pd.qcut(train['Fare'], 3).value_counts()


# In[ ]:


train['Survived'].groupby(pd.qcut(train['Fare'], 3)).mean()


# Looking at the relationship between `Class` and `Fare`, we do indeed see a clear relationship. 

# In[ ]:


pd.crosstab(pd.qcut(train['Fare'], 5), columns=train['Pclass'])


# ## Cabin
# 
# This column has the most nulls (almost 700), but we can still extract information from it, like the first letter of each cabin, or the cabin number. The usefulness of this column might be similar to that of the `Ticket` variable.

# #### Cabin Letter

# We can see that most of the cabin letters are associated with a high survival rate, so this might very well be a useful variable. Because there aren't that many unique values, we won't do any grouping here, even if some of the values have a small count.

# In[ ]:


train['Cabin_Letter'] = train['Cabin'].apply(lambda x: str(x)[0])


# In[ ]:


train['Cabin_Letter'].value_counts()


# In[ ]:


train['Survived'].groupby(train['Cabin_Letter']).mean()


# #### Cabin Number
# 
# Upon first glance, this appears to be useless. Not only do we have ~700 nulls which will be difficult to impute, but the correlation with `Survived` is almost zero. However, the cabin numbers as a whole do seem to have a high surival rate compared to the population average, so we might want to keep this just in case for now.

# In[ ]:


train['Cabin_num'] = train['Cabin'].apply(lambda x: str(x).split(' ')[-1][1:])
train['Cabin_num'].replace('an', np.NaN, inplace = True)
train['Cabin_num'] = train['Cabin_num'].apply(lambda x: int(x) if not pd.isnull(x) and x != '' else np.NaN)


# In[ ]:


pd.qcut(train['Cabin_num'],3).value_counts()


# In[ ]:


train['Survived'].groupby(pd.qcut(train['Cabin_num'], 3)).mean()


# In[ ]:


train['Survived'].corr(train['Cabin_num'])


# ## Embarked
# 
# Looks like the Cherbourg people had a 20% higher survival rate than the other embarking locations. This is very likely due to the high presence of upper-class passengers from that location.

# In[ ]:


train['Embarked'].value_counts()


# In[ ]:


train['Embarked'].value_counts(normalize=True)


# In[ ]:


train['Survived'].groupby(train['Embarked']).mean()


# In[ ]:


sns.countplot(train['Embarked'], hue=train['Pclass'])


# ### Feature Engineering

# Having done our cursory exploration of the variables, we now have a pretty good idea of how we want to transform our variables in preparation for our final dataset. We will perform our feature engineering through a series of helper functions that each serve a specific purpose. 

# This first function creates two separate columns: a numeric column indicating the length of a passenger's `Name` field, and a categorical column that extracts the passenger's title.

# In[ ]:


def names(train, test):
    for i in [train, test]:
        i['Name_Len'] = i['Name'].apply(lambda x: len(x))
        i['Name_Title'] = i['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
        del i['Name']
    return train, test


# Next, we impute the null values of the `Age` column by filling in the mean value of the passenger's corresponding title and class. This more granular approach to imputation should be more accurate than merely taking the mean age of the population.

# In[ ]:


def age_impute(train, test):
    for i in [train, test]:
        i['Age_Null_Flag'] = i['Age'].apply(lambda x: 1 if pd.isnull(x) else 0)
        data = train.groupby(['Name_Title', 'Pclass'])['Age']
        i['Age'] = data.transform(lambda x: x.fillna(x.mean()))
    return train, test


# We combine the `SibSp` and `Parch` columns into a new variable that indicates family size, and group the family size variable into three categories.

# In[ ]:


def fam_size(train, test):
    for i in [train, test]:
        i['Fam_Size'] = np.where((i['SibSp']+i['Parch']) == 0 , 'Solo',
                           np.where((i['SibSp']+i['Parch']) <= 3,'Nuclear', 'Big'))
        del i['SibSp']
        del i['Parch']
    return train, test


# The `Ticket` column is used to create two new columns: `Ticket_Lett`, which indicates the first letter of each ticket (with the smaller-n values being grouped based on survival rate); and `Ticket_Len`, which indicates the length of the `Ticket` field. 

# In[ ]:


def ticket_grouped(train, test):
    for i in [train, test]:
        i['Ticket_Lett'] = i['Ticket'].apply(lambda x: str(x)[0])
        i['Ticket_Lett'] = i['Ticket_Lett'].apply(lambda x: str(x))
        i['Ticket_Lett'] = np.where((i['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), i['Ticket_Lett'],
                                   np.where((i['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']),
                                            'Low_ticket', 'Other_ticket'))
        i['Ticket_Len'] = i['Ticket'].apply(lambda x: len(x))
        del i['Ticket']
    return train, test


# The following two functions extract the first letter of the `Cabin` column and its number, respectively. 

# In[ ]:


def cabin(train, test):
    for i in [train, test]:
        i['Cabin_Letter'] = i['Cabin'].apply(lambda x: str(x)[0])
        del i['Cabin']
    return train, test


# In[ ]:


def cabin_num(train, test):
    for i in [train, test]:
        i['Cabin_num1'] = i['Cabin'].apply(lambda x: str(x).split(' ')[-1][1:])
        i['Cabin_num1'].replace('an', np.NaN, inplace = True)
        i['Cabin_num1'] = i['Cabin_num1'].apply(lambda x: int(x) if not pd.isnull(x) and x != '' else np.NaN)
        i['Cabin_num'] = pd.qcut(train['Cabin_num1'],3)
    train = pd.concat((train, pd.get_dummies(train['Cabin_num'], prefix = 'Cabin_num')), axis = 1)
    test = pd.concat((test, pd.get_dummies(test['Cabin_num'], prefix = 'Cabin_num')), axis = 1)
    del train['Cabin_num']
    del test['Cabin_num']
    del train['Cabin_num1']
    del test['Cabin_num1']
    return train, test


# We fill the null values in the `Embarked` column with the most commonly occuring value, which is 'S.'

# In[ ]:


def embarked_impute(train, test):
    for i in [train, test]:
        i['Embarked'] = i['Embarked'].fillna('S')
    return train, test


# We also fill in the one missing value of `Fare` in our test set with the mean value of `Fare` from the training set (transformations of test set data must always be fit using training data).

# In[ ]:


test['Fare'].fillna(train['Fare'].mean(), inplace = True)


# Next, because we are using scikit-learn, we must convert our categorical columns into dummy variables. The following function does this, and then it drops the original categorical columns. It also makes sure that each category is present in both the training and test datasets.

# In[ ]:


def dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked', 'Ticket_Lett', 'Cabin_Letter', 'Name_Title', 'Fam_Size']):
    for column in columns:
        train[column] = train[column].apply(lambda x: str(x))
        test[column] = test[column].apply(lambda x: str(x))
        good_cols = [column+'_'+i for i in train[column].unique() if i in test[column].unique()]
        train = pd.concat((train, pd.get_dummies(train[column], prefix = column)[good_cols]), axis = 1)
        test = pd.concat((test, pd.get_dummies(test[column], prefix = column)[good_cols]), axis = 1)
        del train[column]
        del test[column]
    return train, test


# Our last helper function drops any columns that haven't already been dropped. In our case, we only need to drop the `PassengerId` column, which we have decided is not useful for our problem (by the way, I've confirmed this with a separate test). Note that dropping the `PassengerId` column here means that we'll have to load it later when creating our submission file.

# In[ ]:


def drop(train, test, bye = ['PassengerId']):
    for i in [train, test]:
        for z in bye:
            del i[z]
    return train, test


# Having built our helper functions, we can now execute them in order to build our dataset that will be used in the model:a

# In[ ]:


train = pd.read_csv(os.path.join('../input', 'train.csv'))
test = pd.read_csv(os.path.join('../input', 'test.csv'))
train, test = names(train, test)
train, test = age_impute(train, test)
train, test = cabin_num(train, test)
train, test = cabin(train, test)
train, test = embarked_impute(train, test)
train, test = fam_size(train, test)
test['Fare'].fillna(train['Fare'].mean(), inplace = True)
train, test = ticket_grouped(train, test)
train, test = dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked', 'Ticket_Lett',
                                                                     'Cabin_Letter', 'Name_Title', 'Fam_Size'])
train, test = drop(train, test)


# We can see that our final dataset has 45 columns, composed of our target column and 44 predictor variables. Although highly dimensional datasets can result in high variance, I think we should be fine here. 

# In[ ]:


print(len(train.columns))


# ### Hyperparameter Tuning

# We will use grid search to identify the optimal parameters of our random forest model. Because our training dataset is quite small, we can get away with testing a wider range of hyperparameter values. When I ran this on my 8 GB Windows machine, the process took less than ten minutes. I will not run it here for the sake of saving myself time, but I will discuss the results of this grid search.

# from sklearn.model_selection import GridSearchCV  
# from sklearn.ensemble import RandomForestClassifier
# 
# rf = RandomForestClassifier(max_features='auto',
#                                 oob_score=True,
#                                 random_state=1,
#                                 n_jobs=-1)
# 
# param_grid = { "criterion"   : ["gini", "entropy"],
#              "min_samples_leaf" : [1, 5, 10],
#              "min_samples_split" : [2, 4, 10, 12, 16],
#              "n_estimators": [50, 100, 400, 700, 1000]}
# 
# gs = GridSearchCV(estimator=rf,
#                   param_grid=param_grid,
#                   scoring='accuracy',
#                   cv=3,
#                   n_jobs=-1)
# 
# gs = gs.fit(train.iloc[:, 1:], train.iloc[:, 0])
# 
# print(gs.best_score_)   
# print(gs.best_params_)  
# print(gs.cv_results_)

# Looking at the results of the grid search:  
# 
# 0.838383838384  
# {'min_samples_split': 10, 'n_estimators': 700, 'criterion': 'gini', 'min_samples_leaf': 1}  
# 
# ...we can see that our optimal parameter settings are not at the endpoints of our provided values, meaning that we do not have to test more values. What else can we say about our optimal values? The `min_samples_split` parameter is at 10, which should help mitigate overfitting to a certain degree. This is especially good because we have a relatively large number of estimators (700), which could potentially increase our generalization error.

# ### Model Estimation and Evaluation<a name="model"></a>

# We are now ready to fit our model using the optimal hyperparameters. The out-of-bag score can give us an unbiased estimate of the model accuracy, and we can see that the score is 82.94%, which is only a little higher than our final leaderboard score.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(criterion='gini', 
                             n_estimators=700,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
rf.fit(train.iloc[:, 1:], train.iloc[:, 0])
print("%.4f" % rf.oob_score_)


# Let's take a brief look at our variable importance according to our random forest model. We can see that some of the original columns we predicted would be important in fact were, including gender, fare, and age. But we also see title, name length, and ticket length feature prominently, so we can pat ourselves on the back for creating such useful variables.

# In[ ]:


pd.concat((pd.DataFrame(train.iloc[:, 1:].columns, columns = ['variable']), 
           pd.DataFrame(rf.feature_importances_, columns = ['importance'])), 
          axis = 1).sort_values(by='importance', ascending = False)[:20]


# Our last step is to predict the target variable for our test data and generate an output file that will be submitted to Kaggle. 

# In[ ]:


predictions = rf.predict(test)
predictions = pd.DataFrame(predictions, columns=['Survived'])
test = pd.read_csv(os.path.join('../input', 'test.csv'))
predictions = pd.concat((test.iloc[:, 0], predictions), axis = 1)
predictions.to_csv('y_test15.csv', sep=",", index = False)


# ## Conclusion
# 
# This exercise is a good example of how far basic feature engineering can take you. It is worth mentioning that I did try various other models before arriving at this one. Some of the other variations I tried were different groupings for the categorical variables (plenty more combinations remain), linear discriminant analysis on a couple numeric columns, and eliminating more variables, among other things. This is a competition with a generous allotment of submission attempts, and as a result, it's quite possible that even the leaderboard score is an overestimation of the true quality of the model, since the leaderboard can act as more of a validation score instead of a true test score. 
# 
# I welcome any comments and suggestions.
