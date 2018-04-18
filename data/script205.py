
# coding: utf-8

# *This tutorial is part of the [Learn Machine Learning](https://www.kaggle.com/learn/machine-learning) series. In this step, you will learn how and why to use pipelines to clean up your modeling code.* 
# 
# # What Are Pipelines
# 
# Pipelines are a simple way to keep your data processing and modeling code organized.  Specifically, a pipeline bundles preprocessing and modeling steps so you can use the whole bundle as if it were a single step.
# 
# Many data scientists hack together models without pipelines, but Pipelines have some important benefits. Those include:
# 1. **Cleaner Code:** You won't need to keep track of your training (and validation) data at each step of processing.  Accounting for data at each step of processing can get messy.  With a pipeline, you don't need to manually keep track of each step.
# 2. **Fewer Bugs:** There are fewer opportunities to mis-apply a step or forget a pre-processing step.
# 3. **Easier to Productionize:** It can be surprisingly hard to transition a model from a prototype to something deployable at scale.  We won't go into the many related concerns here, but pipelines can help.
# 4. **More Options For Model Testing:** You will see an example in the next tutorial, which covers cross-validation.
# 
# ---

# # Example
# 
# We won't focus on the data loading. For now, you can imagine you are at a point where you already have train_X, test_X, train_y and test_y. 

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Read Data
data = pd.read_csv('../input/melb_data.csv')
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]
y = data.Price
train_X, test_X, train_y, test_y = train_test_split(X, y)



# 
# You have a modeling process that uses an Imputer to fill in missing values, followed by a RandomForestRegressor to make predictions.  These can be bundled together with the **make_pipeline** function as shown below.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer

my_pipeline = make_pipeline(Imputer(), RandomForestRegressor())


# You can now fit and predict using this pipeline as a fused whole.

# In[ ]:


my_pipeline.fit(train_X, train_y)
predictions = my_pipeline.predict(test_X)


# For comparison, here is the code to do the same thing without pipelines

# In[ ]:


my_imputer = Imputer()
my_model = RandomForestRegressor()

imputed_train_X = my_imputer.fit_transform(train_X)
imputed_test_X = my_imputer.transform(test_X)
my_model.fit(imputed_train_X, train_y)
predictions = my_model.predict(imputed_test_X)


# This particular pipeline was only a small improvement in code elegance. But pipelines become increasingly valuable as your data processing becomes increasingly sophisticated.

# # Understanding Pipelines
# Most scikit-learn objects are either **transformers** or **models.** 
# 
# **Transformers** are for pre-processing before modeling.  The Imputer class (for filling in missing values) is an example of a transformer.  Over time, you will learn many more transformers, and you will frequently use multiple transformers sequentially. 
# 
# **Models** are used to make predictions. You will usually preprocess your data (with transformers) before putting it in a model.  
# 
# You can tell if an object is a transformer or a model by how you apply it.  After fitting a transformer, you apply it with the *transform* command.  After fitting a model, you apply it with the *predict* command. Your pipeline must start with transformer steps and end with a model.  This is what you'd want anyway.
# 
# Eventually you will want to apply more transformers and combine them more flexibly.  We will cover this later in an Advanced Pipelines tutorial.
# 
# # Your Turn
# Take your modeling code and convert it to use pipelines.  For now, you'll need to do one-hot encoding of categorical variables outside of the pipeline (i.e. before putting the data in the pipeline).
