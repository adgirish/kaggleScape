
# coding: utf-8

# *This tutorial is part of the [Machine Learning](https://www.kaggle.com/learn/machine-learning) series. In this step, you will learn what a "categorical" variable is, as well as the most common approach for handling this type of data.* 
# 
# 
# # Introduction
# Categorical data is data that takes only a limited number of values.  
# 
# For example, if you people responded to a survey about which what brand of car they owned, the result would be categorical (because the answers would be things like _Honda_,  _Toyota_, _Ford_, _None_, etc.). Responses fall into
# a fixed set of categories.
# 
# You will get an error if you try to plug these variables into most machine learning models in Python without "encoding" them first.  Here we'll show the most popular method for encoding categorical variables.
# 
# ---
# 
# ## One-Hot Encoding : The Standard Approach for Categorical Data
# One hot encoding is the most widespread approach, and it works very well unless your categorical variable takes on a large number of values (i.e. you generally won't it for variables taking more than 15 different values.  It'd be a poor choice in some cases with fewer values, though that varies.)
# 
# One hot encoding creates new (binary) columns, indicating the presence of each possible value from the original data.  Let's work through an example.
# 
# ![Imgur](https://i.imgur.com/mtimFxh.png)
# 
# The values in the original data are _Red_, _Yellow_ and _Green_.  We create a separate column for each possible value. Wherever the original value was _Red_, we put a 1 in the _Red_ column.  
# 
# ---
# 
# # Example
# 
# Let's see this in code. We'll skip the basic data set-up code, so you can start at the point where you have **train_predictors**, **test_predictors** DataFrames. This data contains housing characteristics. You will use them to predict home prices, which are stored  in a Series called **target**.
# 

# In[ ]:


# Read the data
import pandas as pd
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

# Drop houses where the target is missing
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)

target = train_data.SalePrice

# Since missing values isn't the focus of this tutorial, we use the simplest
# possible approach, which drops these columns. 
# For more detail (and a better approach) to missing values, see
# https://www.kaggle.com/dansbecker/handling-missing-values
cols_with_missing = [col for col in train_data.columns 
                                 if train_data[col].isnull().any()]                                  
candidate_train_predictors = train_data.drop(['Id', 'SalePrice'] + cols_with_missing, axis=1)
candidate_test_predictors = test_data.drop(['Id'] + cols_with_missing, axis=1)

# "cardinality" means the number of unique values in a column.
# We use it as our only way to select categorical columns here. This is convenient, though
# a little arbitrary.
low_cardinality_cols = [cname for cname in candidate_train_predictors.columns if 
                                candidate_train_predictors[cname].nunique() < 10 and
                                candidate_train_predictors[cname].dtype == "object"]
numeric_cols = [cname for cname in candidate_train_predictors.columns if 
                                candidate_train_predictors[cname].dtype in ['int64', 'float64']]
my_cols = low_cardinality_cols + numeric_cols
train_predictors = candidate_train_predictors[my_cols]
test_predictors = candidate_test_predictors[my_cols]


# Pandas assigns a data type (called a dtype) to each column or Series.  Let's see a random sample of dtypes from our prediction data:

# In[ ]:


train_predictors.dtypes.sample(10)


# **Object** indicates a column has text (there are other things it could be theoretically be, but that's unimportant for our purposes). It's most common to one-hot encode these "object" columns, since they can't be plugged directly into most models.  Pandas offers a convenient function called **get_dummies** to get one-hot encodings. Call it like this:

# In[ ]:


one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)


# Alternatively, you could have dropped the categoricals. To see how the approaches compare, we can calculate the mean absolute error of models built with two alternative sets of predictors:
# 1.  One-hot encoded categoricals as well as numeric predictors
# 2. Numerical predictors, where we drop categoricals.
# 
# One-hot encoding usually helps, but it varies on a case-by-case basis.  In this case, there doesn't appear to be any meaningful benefit from using the one-hot encoded variables.

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

def get_mae(X, y):
    # multiple by -1 to make positive MAE score instead of neg value returned as sklearn convention
    return -1 * cross_val_score(RandomForestRegressor(50), 
                                X, y, 
                                scoring = 'neg_mean_absolute_error').mean()

predictors_without_categoricals = train_predictors.select_dtypes(exclude=['object'])

mae_without_categoricals = get_mae(predictors_without_categoricals, target)

mae_one_hot_encoded = get_mae(one_hot_encoded_training_predictors, target)

print('Mean Absolute Error when Dropping Categoricals: ' + str(int(mae_without_categoricals)))
print('Mean Abslute Error with One-Hot Encoding: ' + str(int(mae_one_hot_encoded)))


# ## Applying to Multiple Files
# 
# So far, you've one-hot-encoded your training data.  What about when you have multiple files (e.g. a test dataset, or some other data that you'd like to make predictions for)?  Scikit-learn is sensitive to the ordering of columns, so if the training dataset and test datasets get misaligned, your results will be nonsense.  This could happen if a categorical had a different number of values in the training data vs the test data.
# 
# Ensure the test data is encoded in the same manner as the training data with the align command:

# In[ ]:


one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)
one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)
final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
                                                                    join='left', 
                                                                    axis=1)



# The align command makes sure the columns show up in the same order in both datasets (it uses column names to identify which columns line up in each dataset.)  The argument `join='left'` specifies that we will do the equivalent of SQL's _left join_.  That means, if there are ever columns that show up in one dataset and not the other, we will keep exactly the columns from our training data.  The argument `join='inner'` would do what SQL databases call an _inner join_, keeping only the columns showing up in both datasets.  That's also a sensible choice.
# 
# # Conclusion
# The world is filled with categorical data. You will be a much more effective data scientist if you know how to use this data. Here are resources that will be useful as you start doing more sophisticated work with cateogircal data.
# 
# * **Pipelines:** Deploying models into production ready systems is a topic unto itself. While one-hot encoding is still a great approach, your code will need to built in an especially robust way.  Scikit-learn pipelines are a great tool for this. Scikit-learn offers a [class for one-hot encoding](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) and this can be added to a Pipeline.  Unfortunately, it doesn't handle text or object values, which is a common use case. 
# 
# * **Applications To Text for Deep Learning:** [Keras](https://keras.io/preprocessing/text/#one_hot) and [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/one_hot) have fuctionality for one-hot encoding, which is useful for working with text.
# 
# * **Categoricals with Many Values:** Scikit-learn's [FeatureHasher](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html#sklearn.feature_extraction.FeatureHasher) uses [the hashing trick](https://en.wikipedia.org/wiki/Feature_hashing) to store high-dimensional data.  This will add some complexity to your modeling code.
# 
# # Your Turn
# Use one-hot encoding to allow categoricals in your course project.  Then add some categorical columns to your **X** data. If you choose the right variables, your model will improve quite a bit.  Once you've done that, **[Click Here](https://www.kaggle.com/learn/machine-learning)** to return to Learning Machine Learning where you can continue improving your model.
