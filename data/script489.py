
# coding: utf-8

# *This tutorial is part of the series [Learn Machine Learning](https://www.kaggle.com/learn/machine-learning). At the end of this step, you will be able to use your first sophisticated machine learning model, the Random Forest.*
# 
# # Introduction
# 
# Decision trees leave you with a difficult decision. A deep tree with lots of leaves will overfit because each prediction is coming from historical data from only the few houses at its leaf. But a shallow tree with few leaves will perform poorly because it fails to capture as many distinctions in the raw data.
# 
# Even today's most sophisticated modeling techniques face this tension between underfitting and overfitting. But, many models have clever ideas that can lead to better performance. We'll look at the **random forest** as an example.
# 
# The random forest uses many trees, and it makes a prediction by averaging the predictions of each component tree. It generally has much better predictive accuracy than a single decision tree and it works well with default parameters. If you keep modeling, you can learn more models with even better performance, but many of those are sensitive to getting the right parameters. 
# 
# # Example
# 
# You've already seen the code to load the data a few times. At the end of data-loading, we have the following variables:
# - train_X
# - val_X
# - train_y
# - val_y

# In[ ]:


import pandas as pd
    
# Load data
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 
# Filter rows with missing values
melbourne_data = melbourne_data.dropna(axis=0)
# Choose target and predictors
y = melbourne_data.Price
melbourne_predictors = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_predictors]

from sklearn.model_selection import train_test_split

# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)


# We build a RandomForest similarly to how we built a decision tree in scikit-learn.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))


# # Conclusion 
# There is likely room for further improvement, but this is a big improvement over the best decision tree error of 250,000. There are parameters which allow you to change the performance of the Random Forest much as we changed the maximum depth of the single decision tree. But one of the best features of Random Forest models is that they generally work reasonably even without this tuning.
# 
# You'll soon learn the XGBoost model, which provides better performance when tuned well with the right parameters (but which requires some skill to get the right model parameters).
# 
# # Your Turn
# Run the RandomForestRegressor on your data. You should see a big improvement over your best Decision Tree models. 
# 
# # Continue 
# You will see more big improvements in your models as soon as you start the  Intermediate track in*Learn Machine Learning* . But you now have a model that's a good starting point to compete in a machine learning competition!
# 
# Follow **[these steps](https://www.kaggle.com/dansbecker/submitting-from-a-kernel/)** to make submissions for your current model. Then you can watch your progress in subsequent steps as you climb up the leaderboard with your continually improving models.
