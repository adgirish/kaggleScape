
# coding: utf-8

# *This tutorial is part Level 2 in the [Learn Machine Learning](https://www.kaggle.com/learn/machine-learning) curriculum. This tutorial picks up where Level 1 finished, so you will get the most out of it if you've done the exercise from Level 1.*
# 
# In this step, you will learn three approaches to dealing with missing values. You will then learn to compare the effectiveness of these approaches on any given dataset.* 
# 
# # Introduction
# 
# There are many ways data can end up with missing values. For example
# - A 2 bedroom house wouldn't include an answer for _How large is the third bedroom_
# - Someone being surveyed may choose not to share their income
# 
# Python libraries represent missing numbers as **nan** which is short for "not a number".  You can detect which cells have missing values, and then count how many there are in each column with the command:
# ```
# print(data.isnull().sum())
# ```
# 
# Most libraries (including scikit-learn) will give you an error if you try to build a model using data with missing values. So you'll need to choose one of the strategies below.
# 
# ---
# ## Solutions
# 
# 
# ## 1) A Simple Option: Drop Columns with Missing Values
# If your data is in a DataFrame called `original_data`, you can drop columns with missing values. One way to do that is
# ```
# data_without_missing_values = original_data.dropna(axis=1)
# ```
# 
# In many cases, you'll have both a training dataset and a test dataset.  You will want to drop the same columns in both DataFrames. In that case, you would write
# 
# ```
# cols_with_missing = [col for col in original_data.columns 
#                                  if original_data[col].isnull().any()]
# redued_original_data = original_data.drop(cols_with_missing, axis=1)
# reduced_test_data = test_data.drop(cols_with_missing, axis=1)
# ```
# If those columns had useful information (in the places that were not missing), your model loses access to this information when the column is dropped. Also, if your test data has missing values in places where your training data did not, this will result in an error.  
# 
# So, it's somewhat usually not the best solution. However, it can be useful when most values in a column are missing.
# 
# 
# 
# ## 2) A Better Option: Imputation
# Imputation fills in the missing value with some number. The imputed value won't be exactly right in most cases, but it usually gives more accurate models than dropping the column entirely.
# 
# This is done with
# ```
# from sklearn.preprocessing import Imputer
# my_imputer = Imputer()
# data_with_imputed_values = my_imputer.fit_transform(original_data)
# ```
# The default behavior fills in the mean value for imputation.  Statisticians have researched more complex strategies, but those complex strategies typically give no benefit once you plug the results into sophisticated machine learning models.
# 
# One (of many) nice things about Imputation is that it can be included in a scikit-learn Pipeline. Pipelines simplify model building, model validation and model deployment.
# 
# ## 3) An Extension To Imputation
# Imputation is the standard approach, and it usually works well.  However, imputed values may by systematically above or below their actual values (which weren't collected in the dataset). Or rows with missing values may be unique in some other way. In that case, your model would make better predictions by considering which values were originally missing.  Here's how it might look:
# ```
# # make copy to avoid changing original data (when Imputing)
# new_data = original_data.copy()
# 
# # make new columns indicating what will be imputed
# cols_with_missing = (col for col in new_data.columns 
#                                  if new_data[c].isnull().any())
# for col in cols_with_missing:
#     new_data[col + '_was_missing'] = new_data[col].isnull()
# 
# # Imputation
# my_imputer = Imputer()
# new_data = my_imputer.fit_transform(new_data)
# ```
# 
# In some cases this approach will meaningfully improve results. In other cases, it doesn't help at all.
# 
# ---
# # Example (Comparing All Solutions)
# 
# We will see am example predicting housing prices from the Melbourne Housing data.  To master missing value handling, fork this notebook and repeat the same steps with the Iowa Housing data.  Find information about both in the **Data** section of the header menu.
# 
# 
# ### Basic Problem Set-up

# In[ ]:


import pandas as pd

# Load data
melb_data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

melb_target = melb_data.Price
melb_predictors = melb_data.drop(['Price'], axis=1)

# For the sake of keeping the example simple, we'll use only numeric predictors. 
melb_numeric_predictors = melb_predictors.select_dtypes(exclude=['object'])


# ### Create Function to Measure Quality of An Approach
# We divide our data into **training** and **test**. If the reason for this is unfamiliar, review [Welcome to Data Science](https://www.kaggle.com/dansbecker/welcome-to-data-science-1).
# 
# We've loaded a function `score_dataset(X_train, X_test, y_train, y_test)` to compare the quality of diffrent approaches to missing values. This function reports the out-of-sample MAE score from a RandomForest.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(melb_numeric_predictors, 
                                                    melb_target,
                                                    train_size=0.7, 
                                                    test_size=0.3, 
                                                    random_state=0)

def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)


# ### Get Model Score from Dropping Columns with Missing Values

# In[ ]:


cols_with_missing = [col for col in X_train.columns 
                                 if X_train[col].isnull().any()]
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_test  = X_test.drop(cols_with_missing, axis=1)
print("Mean Absolute Error from dropping columns with Missing Values:")
print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test))


# ### Get Model Score from Imputation

# In[ ]:


from sklearn.preprocessing import Imputer

my_imputer = Imputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_test = my_imputer.transform(X_test)
print("Mean Absolute Error from Imputation:")
print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))


# ### Get Score from Imputation with Extra Columns Showing What Was Imputed

# In[ ]:


imputed_X_train_plus = X_train.copy()
imputed_X_test_plus = X_test.copy()

cols_with_missing = (col for col in X_train.columns 
                                 if X_train[col].isnull().any())
for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()

# Imputation
my_imputer = Imputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)

print("Mean Absolute Error from Imputation while Track What Was Imputed:")
print(score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test))


# # Conclusion
# In this case, the extension didn't make a big difference. As mentioned before, this can vary widely from one dataset to the next (largely determined by whether rows with missing values are intrinsically like or unlike those without missing values).

# # Your Turn
# 1) Find some columns with missing values in your dataset.
# 
# 2) Use the Imputer class so you can impute missing values
# 
# 3) Add columns with missing values to your predictors. 
# 
# If you find the right columns, you may see an improvement in model scores. That said, the Iowa data doesn't have a lot of columns with missing values.  So, whether you see an improvement at this point depends on some other details of your model.
# 
# Once you've added the Imputer, keep using those columns for future steps.  In the end, it will improve your model (and in most other datasets, it is a big improvement). 
# 
# # Keep Going
# Once you've added the Imputer and included columns with missing values, you are ready to [add categorical variables](https://www.kaggle.com/dansbecker/using-categorical-data-with-one-hot-encoding), which is non-numeric data representing categories (like the name of the neighborhood a house is in).
# 
# ---
# 
# Part of the **[Learn Machine Learning](https://www.kaggle.com/learn/machine-learning)** track.
