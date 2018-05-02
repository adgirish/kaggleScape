
# coding: utf-8

# *This tutorial is part of the [Learn Machine Learning](https://www.kaggle.com/learn/machine-learning) series. In this step, you will learn to use model validation to measure the quality of your model. Measuring model quality is the key to iteratively improving your models.*
# 
# # What is Model Validation
# You've built a model. But how good is it?
# 
# You'll need to answer this question for almost every model you ever build. In most (though not necessarily all) applications, the relevant measure of model quality is predictive accuracy. In other words, will the model's predictions be close to what actually happens.
# 
# Some people try answering this problem by making predictions with their *training data*. They compare those predictions to the actual target values in the *training data*. This approach has a critical shortcoming, which you will see in a moment (and which you'll subsequently see how to solve).
# 
# Even with this simple approach, you'll need to summarize the model quality into a form that someone can understand. If you have predicted and actual home values for 10000 houses, you will inevitably end up with a mix of good and bad predictions. Looking through such a long list would be pointless.
# 
# There are many metrics for summarizing model quality, but we'll start with one called Mean Absolute Error (also called MAE). Let's break down this metric starting with the last word, error.
# 
# The prediction error for each house is: <br>
# error=actual−predicted
#  
# So, if a house cost \$150,000 and you predicted it would cost \$100,000 the error is \$50,000.
# 
# With the MAE metric, we take the absolute value of each error. This converts each error to a positive number. We then take the average of those absolute errors. This is our measure of model quality. In plain English, it can be said as
# 
# On average, our predictions are off by about X
# 
# We first load the Melbourne data and create X and y. That code isn't shown here, since you've already seen it a couple times.

# In[ ]:


# Data Loading Code Hidden Here
import pandas as pd

# Load data
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 
# Filter rows with missing values
filtered_melbourne_data = melbourne_data.dropna(axis=0)
# Choose target and predictors
y = filtered_melbourne_data.Price
melbourne_predictors = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_melbourne_data[melbourne_predictors]


# We then create the Decision tree model with this code:

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(X, y)


# The calculation of mean absolute error in the Melbourne data is

# In[ ]:


from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)


# # The Problem with "In-Sample" Scores
# 
# The measure we just computed can be called an "in-sample" score. We used a single set of houses (called a data sample) for both building the model and for calculating it's MAE score. This is bad.
# 
# Imagine that, in the large real estate market, door color is unrelated to home price. However, in the sample of data you used to build the model, it may be that all homes with green doors were very expensive. The model's job is to find patterns that predict home prices, so it will see this pattern, and it will always predict high prices for homes with green doors.
# 
# Since this pattern was originally derived from the training data, the model will appear accurate in the training data.
# 
# But this pattern likely won't hold when the model sees new data, and the model would be very inaccurate (and cost us lots of money) when we applied it to our real estate business.
# 
# Even a model capturing only happenstance relationships in the data, relationships that will not be repeated when new data, can appear to be very accurate on in-sample accuracy measurements.
# 
# # Example
# 
# Models' practical value come from making predictions on new data, so we should measure performance on data that wasn't used to build the model. The most straightforward way to do this is to exclude some data from the model-building process, and then use those to test the model's accuracy on data it hasn't seen before. This data is called **validation data**.
# 
# The scikit-learn library has a function train_test_split to break up the data into two pieces, so the code to get a validation score looks like this:

# In[ ]:


from sklearn.model_selection import train_test_split

# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))


# # Your Turn
# 1. Use the `train_test_split` command to split up your data.
# 2. Fit the model with the training data
# 3. Make predictions with the validation predictors
# 4. Calculate the mean absolute error between your predictions and the actual target values for the validation data.
# 
# # Continue
# Now that you can measure model performance, you are ready to run some experiments comparing different models.  It's an especially fun part of machine learning.  Once you've done the steps above in your notebook, **[click here](https://www.kaggle.com/dansbecker/underfitting-overfitting-and-model-optimization) ** to continue
