
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import random

# Random pets column
pet_list = ["cat", "hamster", "alligator", "snake"]
pet = [random.choice(pet_list) for i in range(1,15)]

# Random weight of animal column
weight = [random.choice(range(5,15)) for i in range(1,15)]

# Random length of animals column
length = [random.choice(range(1,10)) for i in range(1,15)]

# random age of the animals column
age = [random.choice(range(1,15)) for i in range(1,15)]

# Put everyhting into a dataframe
df = pd.DataFrame()
df["animal"] = pet
df["age"] = age
df["weight"] = weight
df["length"] = length

# make a groupby object
animal_groups = df.groupby("animal")


# 
# # Groupby 
# ---
# *This tutorial roughly picks up from the <a href="https://www.kaggle.com/crawford/python-merge-tutorial/">Python Merge Tutorial</a> but also works as a stand alone Groupby tutorial. If you come from a background in SQL and are familiar with GROUP BY,  you can scroll through this to see some examples of the syntax. *
# <br><br>
# 
# Groupby is a pretty simple concept. You create a grouping of categories and apply a function to the categories. It's a simple concept but it's an extremely valuable technique that's widely used in data science. The value of groupby really comes from it's ability to aggregate data efficiently, both in performance and the amount code it takes. In real data science projects, you'll be dealing with large amounts of data and trying things over and over, so efficiency really becomes an important consideration. 
# <br><br>
# 
# # Understanding Groupby
# Here's a super simple dataframe to illustrate some examples. We'll be grouping the data by the "animal" column where there are four categories of animals: 
# - alligators
# - cats
# - snakes
# - hamsters

# In[ ]:


df


# So one question we could ask about the animal data might be, "What's the average weight of all the snakes, cats, hamsters, and alligators?" To find the average weight of each category of animal, we'll group the animals by animal type and then apply the mean function. We could apply other functions too. We could apply "sum" to add up all the weights, "min" to find the lowest, "max" to get the highest, or "count" just to get a count of each animal type. 
# <br><br>
# 
# This is a short list of some aggregation functions that I find handy but it's definitely not a complete list of possible operations.
# <br>
# 
# <table>
# <tr>
#     <td><b>Summary statistics</b></td>
#     <td><b>Numpy operations</b></td>
#     <td><b>More complex operations</b></td>
# </tr>
# <tr>
#     <td>mean</td>
#     <td>np.mean</td>
#     <td>.agg()</td>
# </tr>
# <tr>
#     <td>median</td>
#     <td>np.min</td>
#     <td>agg(["mean", "median"])</td>
# </tr>
# <tr>
#     <td>min</td>
#     <td>np.max</td>
#     <td>agg(custom_function())</td>
# </tr>
# <tr>
#     <td>max</td>
#     <td>np.sum</td>
# </tr>
# <tr>
#     <td>sum</td>
#     <td>np.product</td>
# </tr>
# <tr>
#     <td>describe</td>
# </tr>
# <tr>
#     <td>count or size</td>
# </tr>
# </table>
# 
# <br><br>
# 
# These two lines of code group the animals then apply the mean function to the weight column.
# ```python
# # Group by animal category
# animal_groups = df.groupby("animal")
# # Apply mean function to wieght column
# animal_groups['weight'].mean()
# ```
# <br><br>
# 
# Here's what happens when you run that code:
# 
# 
# ### 1. Group the unique values from the animal column 
# <img src="https://imgur.com/DRl1wil.jpg" width=400 alt="group stuff">
# <br><br>
# 
# ### 2. Now there's a bucket for each group
# <img src="https://imgur.com/Q9fHw1O.jpg" width=250 alt="make buckets">
# <br><br>
# 
# ### 3. Toss the other data into the buckets 
# <img src="https://imgur.com/A29SKAY.jpg" width=500 alt="add data">
# <br><br>
# 
# ### 4. Apply a function on the weight column of each bucket
# <img src="https://imgur.com/xZnMuPZ.jpg" width=700 alt="calculate something">

# 
# 
# ### Here is the code again:
# ```python
# # Group by category
# animal_groups = df.groupby("animal")
# 
# # Apply the "mean" function to the weight column
# animal_groups['weight'].mean()
# 
# # Or apply the "max" function to the age column
# animal_groups['age'].max()
# ```
# <br><br>
# 
# 
# 

# # Example
# ---
# 
# This is the same dataset used in the <a href="https://www.kaggle.com/crawford/python-merge-tutorial/">Python Merge Tutorial</a>. In that tutorial we merged restaurant ratings with restaurant parking lot info. There are four possible classes of parking: `None, Public, Valet, and Yes`. We can group the restaurants by the type of parking available then get the average rating for restaurants in each parking category. Basically, we want to know if restaurants with parking lots have higher service_ratings.  
# 
# Here are the steps:
# - Merge two dataframes together (Restaurant ratings merged with restaurant parking lot info)
# - Create groups based on the types of parking available at the restaurants 
# - Calculate the average ratings for each group of parking

# In[ ]:


# Load restaurant ratings and parking lot info
ratings = pd.read_csv("../input/rating_final.csv")
parking = pd.read_csv("../input/chefmozparking.csv")

# Merge the dataframes
df = pd.merge(left=ratings, right=parking, on="placeID", how="left")

# Show the merged data
df.head()


# In[ ]:


# Group by the parking_lot column
parking_group = df.groupby("parking_lot")

# Calculate the mean ratings
parking_group["service_rating"].mean()


# Those ratings seem low... But we don't know what scale the ratings are on. If we calculate the summary statistics (mean, std, min, max, etc) we can see the scope of ratings. A common way to calculate quick summary statistics on groupby objects is to apply the "describe" function. 

# In[ ]:


parking_group['service_rating'].describe()


# With all of the summary statistics in front of us we can see that the lowest rating for all parking categories is 0 and the highest is 2. I would guess that users were asked to rate restaurants with 1-3 stars which equate to 0, 1, or 2 in the data. So do restaurants with valet parking have higher service_ratings?  
# <br><br>

# # Conclusion
# 
# 
# You've now seen how to efficiently group categorical data and apply aggregate functions like "mean", "sum" and "describe". Fork this kernel and try out some different aggregate functions. Mean, median, prod, sum, std, and var are a few examples. If you're feeling confident, try implementing merge and groupby on a datset by yourself.
# <br><br>
