
# coding: utf-8

# *This tutorial is part of the series [Learning Machine Learning](kaggle.com/learn/machine-learning).*
# 
# # Choosing the Prediction Target
# 
# You have the code to load your data, and you know how to index it. You are ready to choose which column you want to predict. This column is called the **prediction target**. There is a convention that the prediction target is referred to as **y**. Here is an example doing that with the example data.

# In[ ]:


# This code loads the data. You have seen it before, so you don't need to focus on it here.

import pandas as pd

melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

# The Melbourne data has somemissing values (some houses for which some variables weren't recorded.)
# We'll learn to handle missing values in a later tutorial.  
# Your Iowa data doesn't have missing values in the predictors you use. 
# So we will take the simplest option for now, and drop those houses from our data. 
#Don't worry about this much for now, though the code is:

# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)


# In[ ]:


y = melbourne_data.Price


# # Choosing Predictors
# Next we select the predictors. Sometimes, you will want to use all of the variables except the target..
# 
# It's possible to model with non-numeric variables, but we'll start with a narrower set of numeric variables.  In the example data, the predictors will be chosen as:

# In[ ]:


melbourne_predictors = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']


# By convention, this data is called **X**.

# In[ ]:


X = melbourne_data[melbourne_predictors]


# ---
# # Building Your Model
# 
# You will use the **scikit-learn** library to create your models.  When coding, this library is written as **sklearn**, as you will see in the sample code. Scikit-learn is easily the most popular library for modeling the types of data typically stored in DataFrames. 
# 
# The steps to building and using a model are:
# * **Define:** What type of model will it be?  A decision tree?  Some other type of model? Some other parameters of the model type are specified too.
# * **Fit:** Capture patterns from provided data. This is the heart of modeling.
# * **Predict:** Just what it sounds like
# * **Evaluate**: Determine how accurate the model's predictions are.
# 
# Here is the example for defining and fitting the model.

# In[ ]:


from sklearn.tree import DecisionTreeRegressor

# Define model
melbourne_model = DecisionTreeRegressor()

# Fit model
melbourne_model.fit(X, y)


# The output describes some parameters about the type of model you've built. Don't worry about it for now.

# In practice, you'll want to make predictions for new houses coming on the market rather than the houses we already have prices for. But we'll make predictions for the first rows of the training data to see how the predict function works.
# 

# In[ ]:


print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))


# ---
# 
# # Your Turn
# Now it's time for you to define and fit a model for your data (in your notebook).
# 1. Select the target variable you want to predict. You can go back to the list of columns from your earlier commands to recall what it's called (*hint: you've already worked with this variable*). Save this to a new variable called `y`.
# 2. Create a **list** of the names of the predictors we will use in the initial model.  Use just the following columns in the list (you can copy and paste the whole list to save some typing, though you'll still need to add quotes):
#     * LotArea
#     * YearBuilt
#     * 1stFlrSF
#     * 2ndFlrSF
#     * FullBath
#     * BedroomAbvGr
#     * TotRmsAbvGrd
# 
# 3. Using the list of variable names you just created, select a new DataFrame of the predictors data. Save this with the variable name `X`.
# 4. Create a `DecisionTreeRegressorModel` and save it to a variable (with a name like my_model or iowa_model). Ensure you've done the relevant import so you can run this command.
# 5. Fit the model you have created using the data in `X` and the target data you saved above.
# 6. Make a few predictions with the model's `predict` command and print out the predictions.  
# 
# 
# ---
# 
# # Continue
# 
# You've built a decision tree model that can predict the prices of houses based on their characteristics.  It's natural to ask how accurate the model's predictions will be, and measuring accuracy is necessary for us to see whether or not other approaches improve our model. 
# 
# Move on to the [next page](https://www.kaggle.com/dansbecker/model-validation) to see how we measure model accuracy.
# 
