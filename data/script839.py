
# coding: utf-8

# *This is part of Kaggle's [Learn Machine Learning](https://www.kaggle.com/learn/machine-learning) series.*
# 
# # Selecting and Filtering Data
# Your dataset had  too many variables to wrap your head around, or even to print out nicely.  How can you pare down this overwhelming amount of data to something you can understand?
# 
# To show you the techniques, we'll start by picking a few variables using our intuition. Later tutorials will show you statistical techniques to  automatically prioritize variables.
# 
# Before we can choose variables/columns, it is helpful to see a list of all columns in the dataset. That is done with the **columns** property of the DataFrame (the bottom line of code below).

# In[ ]:


import pandas as pd

melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 
print(melbourne_data.columns)


# There are many ways to select a subset of your data. We'll start with two main approaches:  
# 
# ## Selecting a Single Column
# You can pull out any variable (or column) with **dot-notation**.  This single column is stored in a **Series**, which is broadly like a DataFrame with only a single column of data.  Here's an example:

# In[ ]:


# store the series of prices separately as melbourne_price_data.
melbourne_price_data = melbourne_data.Price
# the head command returns the top few lines of data.
print(melbourne_price_data.head())


# ## Selecting Multiple Columns
# You can select multiple columns from a DataFrame by providing a list of column names inside brackets. Remember, each item in that list should be a string (with quotes).

# In[ ]:


columns_of_interest = ['Landsize', 'BuildingArea']
two_columns_of_data = melbourne_data[columns_of_interest]


# We can verify that we got the columns we need with the **describe** command.

# In[ ]:


two_columns_of_data.describe()


# # Your Turn
# In the notebook with your code:
# 1. Print a list of the columns
# 2. From the list of columns, find a name of the column with the sales prices of the homes. Use the dot notation to extract this to a variable (as you saw above to create `melbourne_price_data`.)
# 3. Use the `head` command to print out the top few lines of the variable you just created.
# 4. Pick any two variables and store them to a new DataFrame (as you saw above to create `two_columns_of_data`.)
# 5. Use the describe command with the DataFrame you just created to see summaries of those variables. <br>
# 
# ---
# # Continue
# Now that you can specify what data you want for a model, you are ready to **[build your first model](https://www.kaggle.com/dansbecker/your-first-scikit-learn-model)** in the next step.
