
# coding: utf-8

# *This tutorial is part of the [Learn Machine Learning](https://www.kaggle.com/learn/machine-learning) educational track.*
# 
# # Starting Your Project
# 
# You are about to build a simple model and then continually improve it. It is easiest to keep one browser tab (or window) for the tutorials you are reading, and a separate browser window with the code you are writing. You will continue writing code in the same place even as you progress through the sequence of tutorials.
# 
# ** The starting point for your project is at [THIS LINK](https://www.kaggle.com/dansbecker/my-model/).  Open that link in a new tab. Then hit the "Fork Notebook" button towards the top of the screen.**
# 
# ![Imgur](https://i.imgur.com/GRtMTWw.png)
# 
# **You will see examples predicting home prices using data from Melbourne, Australia. You will then write code to build a model predicting prices in the US state of Iowa. The Iowa data is pre-loaded in your coding notebook.**
# 
# ### Working in Kaggle Notebooks
# You will be coding in a "notebook" environment. These allow you to easily see your code and its output in one place.  A couple tips on the Kaggle notebook environment:
# 
# 1) It is composed of "cells."  You will write code in the cells. Add a new cell by clicking on a cell, and then using the buttons in that look like this. ![Imgur](https://i.imgur.com/Lscji3d.png) The arrows indicate whether the new cell goes above or below your current location. <br><br>
# 2) Execute the code in the current cell with the keyboard shortcut Control-Enter.
# 
# 
# ---
# # Using Pandas to Get Familiar With Your Data
# 
# The first thing you'll want to do is familiarize yourself with the data.  You'll use the Pandas library for this.  Pandas is the primary tool that modern data scientists use for exploring and manipulating data.  Most people abbreviate pandas in their code as `pd`.  We do this with the command

# In[ ]:


import pandas as pd


# The most important part of the Pandas library is the DataFrame.  A DataFrame holds the type of data you might think of as a table. This is similar to a sheet in Excel, or a table in a SQL database. The Pandas DataFrame has powerful methods for most things you'll want to do with this type of data.  Let's start by looking at a basic data overview with our example data from Melbourne and the data you'll be working with from Iowa.
# 
# The example will use data at the file path **`../input/melbourne-housing-snapshot/melb_data.csv`**.  Your data will be available in your notebook at `../input/train.csv` (which is already typed into the sample code for you).
# 
# We load and explore the data with the following:

# In[ ]:


# save filepath to variable for easier access
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path) 
# print a summary of the data in Melbourne data
print(melbourne_data.describe())


# # Interpreting Data Description
# The results show 8 numbers for each column in your original dataset. The first number, the **count**,  shows how many rows have non-missing values.  
# 
# Missing values arise for many reasons. For example, the size of the 2nd bedroom wouldn't be collected when surveying a 1 bedroom house. We'll come back to the topic of missing data.
# 
# The second value is the **mean**, which is the average.  Under that, **std** is the standard deviation, which measures how numerically spread out the values are.
# 
# To interpret the **min**, **25%**, **50%**, **75%** and **max** values, imagine sorting each column from lowest to highest value.  The first (smallest) value is the min.  If you go a quarter way through the list, you'll find a number that is bigger than 25% of the values and smaller than 75% of the values.  That is the **25%** value (pronounced "25th percentile").  The 50th and 75th percentiles are defined analgously, and the **max** is the largest number.
# 
# --- 
# # Your Turn
# **Remember, the notebook you want to "fork" is [here](https://www.kaggle.com/dansbecker/my-model/).**
# 
# Run the equivalent commands (to read the data and print the summary) in the code cell below.  The file path for your data is already shown in your coding notebook. Look at the mean, minimum and maximum values for the first few fields. Are any of the values so crazy that it makes you think you've misinterpreted the data?
# 
# There are a lot of fields in this data.  You don't need to look at it all quite yet.
# 
# When your code is correct, you'll see the size, in square feet, of the smallest lot in your dataset.  This is from the **min** value of **LotArea**, and you can see the **max** size too.  You should notice that it's a big range of lot sizes! 
# 
# You'll also see some columns filled with `...`.  That indicates that we had too many columns of data to print, so the middle ones were omitted from printing.
# 
# We'll take care of both issues in the next step.
# 
# # Continue
# Move on to the next [page](https://www.kaggle.com/dansbecker/Selecting-And-Filtering-In-Pandas/) where you will focus in on the most relevant columns.
