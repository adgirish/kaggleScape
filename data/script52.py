
# coding: utf-8

# *This tutorial is part of the series [Learn Machine Learning](https://www.kaggle.com/learn/machine-learning). At the end of this step, you will understand the concepts of underfitting and overfitting, and you will be able to apply these ideas to optimize your model accuracy.*
# 
# # Experimenting With Different Models
# 
# Now that you have a trustworthy way to measure model accuracy, you can experiment with alternative models and see which gives the best predictions.  But what alternatives do you have for models?
# 
# You can see in scikit-learn's [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) that the decision tree model has many options (more than you'll want or need for a long time). The most important options determine the tree's depth.  Recall from [page 2](https://www.kaggle.com/dansbecker/first-data-science-scenario-page-2/) that a tree's depth is a measure of how many splits it makes before coming to a prediction.  This is a relatively shallow tree
# 
# ![Depth 2 Tree](http://i.imgur.com/R3ywQsR.png)
# 
# In practice, it's not uncommon for a tree to have 10 splits between the top level (all houses and a leaf).  As the tree gets deeper, the dataset gets sliced up into leaves with fewer houses.  If a tree only had 1 split, it divides the data into 2 groups. If each group is split again, we would get 4 groups of houses.  Splitting each of those again would create 8 groups.  If we keep doubling the number of groups by adding more splits at each level, we'll have \\(2^{10}\\) groups of houses by the time we get to the 10th level. That's 1024 leaves.  
# 
# When we divide the houses amongst many leaves, we also have fewer houses in each leaf.  Leaves with very few houses will make predictions that are quite close to those homes' actual values, but they may make very unreliable predictions for new data (because each prediction is based on only a few houses).
# 
# This is a phenomenon called **overfitting**, where a model matches the training data almost perfectly, but does poorly in validation and other new data.  On the flip side, if we make our tree very shallow, it doesn't divide up the houses into very distinct groups.  
# 
# At an extreme, if a tree divides houses into only 2 or 4, each group still has a wide variety of houses. Resulting predictions may be far off for most houses, even in the training data (and it will be bad in validation too for the same reason). When a model fails to capture important distinctions and patterns in the data, so it performs poorly even in training data, that is called **underfitting**.  
# 
# Since we care about accuracy on new data, which we estimate from our validation data, we want to find the sweet spot between underfitting and overfitting.  Visually, we want the low point of the (red) validation curve in
# 
# ![underfitting_overfitting](http://i.imgur.com/2q85n9s.png)
# 
# # Example
# There are a few alternatives for controlling the tree depth, and many allow for some routes through the tree to have greater depth than other routes.  But the *max_leaf_nodes* argument provides a very sensible way to control overfitting vs underfitting.  The more leaves we allow the model to make, the more we move from the underfitting area in the above graph to the overfitting area.
# 
# We can use a utility function to help compare MAE scores from different values for *max_leaf_nodes*:
# 

# In[ ]:


from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)


# The data is loaded into **train_X**, **val_X**, **train_y** and **val_y** using the code you've already seen (and which you've already written).

# In[ ]:


# Data Loading Code Runs At This Point
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

from sklearn.model_selection import train_test_split

# split data into training and validation data, for both predictors and target
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)


# We can use a for-loop to compare the accuracy of models built with different values for *max_leaf_nodes.*

# In[ ]:


# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


# Of the options listed, 500 is the optimal number of leaves.  Apply the function to your Iowa data to find the best decision tree.
# ---
# 
# # Conclusion
# 
# Here's the takeaway: Models can suffer from either:
# - **Overfitting:** capturing spurious patterns that won't recur in the future, leading to less accurate predictions, or 
# - **Underfitting:** failing to capture relevant patterns, again leading to less accurate predictions. 
# 
# We use **validation** data, which isn't used in model training, to measure a candidate model's accuracy. This lets us try many candidate models and keep the best one. 
# 
# But we're still using Decision Tree models, which are not very sophisticated by modern machine learning standards. 
# 
# ---
# # Your Turn
# In the near future, you'll be efficient writing functions like `get_mae` yourself.  For now, just copy it over to your work area.  Then use a for loop that tries different values of *max_leaf_nodes* and calls the *get_mae* function on each to find the ideal number of leaves for your Iowa data.
# 
# You should see that the ideal number of leaves for Iowa data is less than the ideal number of leaves for the Melbourne data. Remember, that a lower MAE is better.
# 
# ---
# 
# # Continue
# **[Click here](https://www.kaggle.com/dansbecker/random-forests)** to learn your first sophisticated Machine Learning model, the Random Forest. It is a clever extrapolation of the decision tree model that consistently leads to more accurate predictions.
