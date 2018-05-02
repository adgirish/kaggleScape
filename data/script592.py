
# coding: utf-8

# *This tutorial is part of the [Learn Machine Learning](https://www.kaggle.com/learn/machine-learning/) series. In this step, you will learn what data leakage is and how to prevent it.* 
# 
# 
# # What is Data Leakage
# Data leakage is one of the most important issues for a data scientist to understand. If you don't know how to prevent it, leakage will come up frequently, and it will ruin your models in the most subtle and dangerous ways.  Specifically, leakage causes a model to look accurate until you start making decisions with the model, and then the model becomes very inaccurate. This tutorial will show you what leakage is and how to avoid it.
# 
# There are two main types of leakage: **Leaky Predictors** and a **Leaky Validation Strategies.**
# 
# ## Leaky Predictors
# This occurs when your predictors include data that will not be available at the time you make predictions. 
# 
# For example, imagine you want to predict who will get sick with pneumonia. The top few rows of your raw data might look like this:
# 
# | got_pneumonia | age | weight |  male | took_antibiotic_medicine | ... |
# |:-------------:|:---:|:------:|:-----:|:------------------------:|-----|
# |     False     |  65 |   100  | False |           False          | ... |
# |     False     |  72 |   130  |  True |           False          | ... |
# |      True     |  58 |   100  | False |           True           | ... |
# -
# 
# 
# People take antibiotic medicines after getting pneumonia in order to recover. So the raw data shows a strong relationship between those columns. But *took_antibiotic_medicine* is frequently changed **after** the value for *got_pneumonia* is determined. This is target leakage.
# 
# The model would see that anyone who has a value of `False` for `took_antibiotic_medicine` didn't have pneumonia.  Validation data comes from the same source, so the pattern will repeat itself in validation, and the model will have great validation (or cross-validation) scores. But the model will be very inaccurate when subsequently deployed in the real world.
# 
# To prevent this type of data leakage, any variable updated (or created) after the target value is realized should be excluded. Because when we use this model to make new predictions, that data won't be available to the model.
# 
# ![Leaky Data Graphic](https://i.imgur.com/CN4INKb.png)

# ## Leaky Validation Strategy
# 
# A much different type of leak occurs when you aren't careful distinguishing training data from validation data.  For example, this happens if you run preprocessing (like fitting the Imputer for missing values) before calling train_test_split.  Validation is meant to be a measure of how the model does on data it hasn't considered before.  You can corrupt this process in subtle ways if the validation data affects the preprocessing behavoir..  The end result?  Your model will get very good validation scores, giving you great confidence in it, but perform poorly when you deploy it to make decisions.
# 
# 
# ## Preventing Leaky Predictors
# There is no single solution that universally prevents leaky predictors. It requires knowledge about your data, case-specific inspection and common sense.
# 
# However, leaky predictors frequently have high statistical correlations to the target.  So two tactics to keep in mind:
# * To screen for possible leaky predictors, look for columns that are statistically correlated to your target.
# * If you build a model and find it extremely accurate, you likely have a leakage problem.
# 
# ## Preventing Leaky Validation Strategies
# 
# If your validation is based on a simple train-test split, exclude the validation data from any type of *fitting*, including the fitting of preprocessing steps.  This is easier if you use [scikit-learn Pipelines](https://www.kaggle.com/dansbecker/pipelines).  When using cross-validation, it's even more critical that you use pipelines and do your preprocessing inside the pipeline.
# 
# # Example
# We will use a small dataset about credit card applications, and we will build a model predicting which applications were accepted (stored in a variable called *card*).  Here is a look at the data:

# In[ ]:


import pandas as pd

data = pd.read_csv('../input/AER_credit_card_data.csv', 
                   true_values = ['yes'],
                   false_values = ['no'])
print(data.head())


# We can see with `data.shape` that this is a small dataset (1312 rows), so we should use cross-validation to ensure accurate measures of model quality

# In[ ]:


data.shape


# In[ ]:


from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

y = data.card
X = data.drop(['card'], axis=1)

# Since there was no preprocessing, we didn't need a pipeline here. Used anyway as best practice
modeling_pipeline = make_pipeline(RandomForestClassifier())
cv_scores = cross_val_score(modeling_pipeline, X, y, scoring='accuracy')
print("Cross-val accuracy: %f" %cv_scores.mean())


# With experience, you'll find that it's very rare to find models that are accurate 98% of the time.  It happens, but it's rare enough that we should inspect the data more closely to see if it is target leakage.
# 
# Here is a summary of the data, which you can also find under the data tab:
# 
#  - **card:** Dummy variable, 1 if application for credit card accepted, 0 if not
#  - **reports:** Number of major derogatory reports
#  - **age:** Age n years plus twelfths of a year
#  - **income:** Yearly income (divided by 10,000)
#  - **share:** Ratio of monthly credit card expenditure to yearly income
#  - **expenditure:** Average monthly credit card expenditure
#  - **owner:** 1 if owns their home, 0 if rent
#  - **selfempl:** 1 if self employed, 0 if not.
#  - **dependents:** 1 + number of dependents
#  - **months:** Months living at current address
#  - **majorcards:** Number of major credit cards held
#  - **active:** Number of active credit accounts
# 
# A few variables look suspicious.  For example, does **expenditure** mean expenditure on this card or on cards used before appying?
# 
# At this point, basic data comparisons can be very helpful:

# In[ ]:


expenditures_cardholders = data.expenditure[data.card]
expenditures_noncardholders = data.expenditure[~data.card]

print('Fraction of those who received a card with no expenditures: %.2f'       %(( expenditures_cardholders == 0).mean()))
print('Fraction of those who received a card with no expenditures: %.2f'       %((expenditures_noncardholders == 0).mean()))


# Everyone with `card == False` had no expenditures, while only 2% of those with `card == True` had no expenditures.  It's not surprising that our model appeared to have a high accuracy. But this seems a data leak, where expenditures probably means *expenditures on the card they applied for.**. 
# 
# Since **share** is partially determined by **expenditure**, it should be excluded too.  The variables **active**, **majorcards** are a little less clear, but from the description, they sound concerning.  In most situations, it's better to be safe than sorry if you can't track down the people who created the data to find out more.
# 
# We would run a model without leakage as follows:

# In[ ]:


potential_leaks = ['expenditure', 'share', 'active', 'majorcards']
X2 = X.drop(potential_leaks, axis=1)
cv_scores = cross_val_score(modeling_pipeline, X2, y, scoring='accuracy')
print("Cross-val accuracy: %f" %cv_scores.mean())


# This accuracy is quite a bit lower, which on the one hand is disappointing.  However, we can expect it to be right about 80% of the time when used on new applications, whereas the leaky model would likely do much worse then that (even in spite of it's higher apparent score in cross-validation.).
# 
# # Conclusion
# Data leakage can be multi-million dollar mistake in many data science applications. Careful separation of training and validation data is a first step, and pipelines can help implement this separation.  Leaking predictors are a more frequent issue, and leaking predictors are harder to track down. A combination of caution, common sense and data exploration can help identify leaking predictors so you remove them from your model.
# 
# # Exercise
# Review the data in your ongoing project.  Are there any predictors that may cause leakage?  As a hint, most datasets from Kaggle competitions don't have these variables. Once you get past those carefully curated datasets, this becomes a common issue.
# 
# Click **[here](https://www.kaggle.com/learn/machine-learning)** to return the main page for *Learning Machine Learning.*
