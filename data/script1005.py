
# coding: utf-8

# # About this Tutorial
# * Length: 30 minutes to 1 hour
# * Prerequisites: Basic programming experience (Python strongly preferred)
# * Goals: 
#   * Get exposure to fundamental analysis tools
#   * Fork this kernel 
#   * Make a bare-minimum barplot
#   * Customize barplots
# <br><br>

# # Table of Contents
# - Introduction
# - Dataset
# - Importing
# - Dataframes
# 	- Soccer fields are dataframes
# - Exploring data
# 	- Quick tools
# 		- Head and Tail
# 		- Columns
# 		- Index
# - Wrangling data
# - Plotting data
# 	- Bare minimum barplot
# 	- Subsetting
# 	- Rotating labels
# 	- Changing the color palette
# 	- Main titles and Axis titles
# - Your turn!
# 

# # A Humble Introduction to Data Analysis: <a id="#introduction"></a>
# 
# ---
# This is a humble introduction to data analysis and is meant for people that are new to data science. Originally, I wanted to create a different tutorial called, "The Absolute Best and Most Mind Blowing Tutorial You've Ever Heard Of in Regards to Data Science." It was going to be descriptive, informative, and it was going to blow your socks off with data lazers and be nothing short of award-winning. But that's, like, way too much work. And also we have to tread lightly around new data scientists because they're like squirrels and run away if you move too fast.
# 
# ![Data Squirrel](https://media.giphy.com/media/l0IyitCtDpBafoAbm/giphy.gif)
# <br><br>
# 
# My main goal is to get you involved with the wonderful Kaggle community by getting you into analysis mode right away. Kernel notebooks are a great tool for this because I can type a bunch of jibberish here and you can 'fork' (aka copy) this kernel and build upon my jibberish with your own jibberish! This isn't the kind of tutorial that will make you a pro after reading it. But at the very least, you'll make bargraphs and then who knows? Maybe you print them out and put them on your grandmother's refrigerator or something. Maybe you set up a booth at a data science convention and sell highly-desirable *autographed* versions of your bargraphs. The data world is your oyster (whatever that means)!
# <br><br>
# 
# 

# We're going to be using a smattering of Matplotlib and Seaborn for making bargraphs in this tutorial. Matplotlib is the main plotting library for Python and Seaborn is a wrapper for Matplotlib. That basically means that Seaborn has some "high-level" functions that use matplotlib functions under the hood. Seaborn makes some things easier but we'll still need to use matplotlib functions.
# <br><br>
# 
# Without further ado, allow me to introduce our dataset:

# # Dataset <a id="#dataset"></a>
# ### FAA Wildlife Strikes
# 
# ---
# This is a dataset that the U.S. Federal Aviation Administration puts together and comes from https://wildlife.faa.gov/, the Wildlife Strike database.  Long story short, Wildlife strikes occur when an aircraft and animal collide (usually birds). As you can probably imagine, that isn't good for anyone, especially the birds. When birds collide with aircraft, pilots fill out a report and submit it to the FAA. The dataset contains information such as the kind of damage done and which phase of flight it occurred. 
# <br><br>
# 
#  

# In[ ]:


# Numpy is generally used for making fancier lists called arrays and series. 
import numpy as np 

# Pandas is super important, it's the foundation data analysis library we're using.
import pandas as pd 

# Matplotlib is the python plotting library and folks generally import it as "plt"
import matplotlib.pyplot as plt 

# Seaborn is a wrapper for Matplotlib and makes some things easier, generally imported as "sns"
import seaborn as sns 


# # Importing <a id="#importing"></a>
# How to get at the data in the CSV file?
# 
# ----
# 
# By default, Kernels store our datasets in **../input/** 
# 
# We will store the contents of the CSV in a data structure called a **dataframe**
# 
# - *Pandas.read_csv() puts the contents of a CSV file into a dataframe.*
# - *We're using the argument low_memory=False because this dataset has some funky datatypes and gives us a warning*

# In[ ]:


# Dataset location
database = '../input/database.csv'

# Read in a CSV file and store the contents in a dataframe (df)
df = pd.read_csv(database, low_memory=False)


# # Dataframes <a id="#dataframes"></a>
# ### What the heck is a dataframe?
# 
# ---
# 
# Dataframes are common data structures in data analysis. Here is the definition of a dataframe, straight from the Pandas documentation:
# 
# > Two-dimensional size-mutable, potentially heterogeneous tabular data structure with labeled axes (rows and columns).
# 
# In human terms, that means a dataframe has rows and columns, can change size, and possibly has mixed data types (text and/or numbers for example).  

# ### Soccer Fields are Dataframes<a id="#soccer"></a>
# 
# --- 
# 
# In terms of structure, consider a soccer field (or a football pitch if you like). A soccer field is a soccer player's playing field. 
# Likewise, dataframes can be thought of as a data scientist's playing field (one of them anyway). We can also say the soccer players represent the data in a dataframe. 
# <br><br>
# 
# Soccer Field | Soccer Frame  | Soccer Dataframe
# :-------------------------:|:-------------------------:|:--------------------:
# <img src="http://i.imgur.com/ZEkm8qV.jpg" alt="Soccer Field" width="200"> | <img src="http://i.imgur.com/HO0gADu.jpg"  alt="Data"  width="200">  |  <img src="http://i.imgur.com/cwP0Meo.jpg"  alt='Dataframe'  width="200">
# 
# 
# 
# Discalimer: This is possibly not the best representation of a dataframe. But I challenge anyone to make a better graphic in Microsoft Paint! The important thing to take away from this message is that a soccer field and a dataframe are both 2-dimensional structures. 

# # Exploring the data <a id="#exploring"></a>
# ### Tools for your toolbox
# 
# ---
# 
# Data science works kind of like this: 
# 1. Examine the data
# 2. Create some hypotheses about the data 
# 3. Test the hypotheses
# 4. Report your results
# <br><br>
# 
# This tutorial is all about #1, examining the data.  We are going to examine the hell out of this data. We're going to examine the data then make sweet bar graphs about the data. So how do we examine data? You may be surprised to learn that examining data can be accomplished by looking at,  gazing upon, or poking around the data. When we're poking around our data, it's called an Exploratory Data Analysis (EDA) which is literally a fancy phrase for looking at data. While looking at data, you look for relationships and trends (fancy words for lines) to form hypotheses. Data science is all about lines and I personally start foaming at the mouth when I see lines in data. You may or may not also foam at the mouth, it's unclear if thats normal or not.
# <br><br>
# 
# 
# And moving on...
# 
# 
# <img src="http://i.imgur.com/cwP0Meo.jpg" width=200 style="float: right; padding: 0 0 0 0;"> 
# 
# Part of an exploratory data analysis means we need to learn about the rows and columns (aka dimensions) of the dataframe.  If you had never heard of soccer before, you'd probably want to know how far the field was from end to end. That's a terrible example, but still it's fundamentally helpful to get a sense of how "long" and "wide" your datafame is. 
# 
# This AMAZING DATA VIZ that I created to the right is 3 wide x 3 long, meaning it has 3 rows and 3 columns.
# 
# The rows are labeled: 
# * Strikers
# * Midfield
# * Defense
# <br><br>
# 
# The columns are labeled: 
# * Left field
# * Center
# * Right field
# <br><br>
# 
# 
# If you've gotten this far in life, you probably already know all of that. So now that I've spent entirely too much time drawing and talking about what rows and columns are, let's actually learn about our dataframe's rows and columns... How many are there?
# 
# 
# 
# In order to get a quick check of the number of rows and columns in our actual dataframe, we can use the **shape** method like this:
# 
# 

# In[ ]:


df.shape


# 174104 rows and  66 columns
# 
# > Row then Column is generally how these things are reported. 
# <br><br>
# 
# There are a ton of methods you can use on your dataframe while your exploring. Here sare some that I use:
# 

# 
# ### Quick tools<a id="#quick"></a>
# That I use
# 
# ---
# 1. Shape
#  - I don't actually use shape that much, maybe like once per dataset.
#  
# 2. Head and Tail
#  - I use these the most, they're good for catching mistakes.
#  
# 3. Columns
#  - My short-term memory lasts 30 minutes so I use this every 30 minutes (to remember the column names)
#  
# 4. Index
#  - I use this about 100 times per day for various reasons.
# 

# ### Head and Tail <a id="#headtail"></a>
# Previewing a dataframe
# 
# ---
# 
# The **head** method shows you the first few rows at the top of the dataframe and **tail** shows you lines at the bottom. By default they show you the first/last 5 rows. I mostly use this to get a quick glance and do a quick scan for unwanted values. In this particular dataset, we don't want values like NA, NaN, UNK, or UNKNOWN.
# <br><br>

# In[ ]:


# Head() the dataframe
df.head(4)


# ### Columns <a id="#columns"></a>
#  Accessing column names
# 
# ---
# 
# The **columns** method will show you the names of each column. We can later use the actual column names to filter out data or modify column names. For example, you might want to remove the white spaces in column names.

# In[ ]:


# Accessing the features (column names)
df.columns


# 
# ### Index <a id="#index"></a>
# Accessing the index (row names)
# 
# ---
# 
# The **index** method will show you the names of the rows if they exist. If rows aren't named, then they are indexed by number. Since our dataframe doesn't have named rows, it isn't very informative right now. But sometimes it's helpful so I wanted to include it.
# 
# [https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Index.html][3]
#   [3]: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Index.html

# In[ ]:


# Accessing the index (row names)
df.index


# # Wrangling <a id="#wrangling"></a>
# ### Splitting data into manageable chunks
# 
# ---
# 
# In this section I'm going to split the dataframe into a few manageable arrays for you to use when you're making plots. I remember seeing some nasty NA, NaN, and UNKNOWN entries (NULL values) when we were examining the dataframe earlier so I'll clean that up too. For simplicity and maximum graph pretty-ness, I'm going to remove all the rows that have any kind of Null value. These arrays will will be perfect for using as your x- and y-axes in the plots.

# In[ ]:


# Re-reading the data file to clean up NULL values that make ugly graphs
df = pd.read_csv(database,
                 low_memory=False,             ### Prevents low_memory warning
                 na_values=['UNKNOWN', 'UNK'], ### Adds UNKNOWN and UNK to list of NULLs
                 na_filter=True,               ### Detect NA/NaN as actual NULL values
                 skip_blank_lines=True)        ### Skip boring blank lines


# In[ ]:


# These are the columns we're going to take from the original dataframe
subset_list = ["Incident Year",
               "Incident Month",
               "Incident Day",
               "Operator",
               "State",
               "Airport", 
               "Flight Phase",
               "Species Name"]

# We're saving them into a new dataframe
df = pd.DataFrame(data=df, columns=subset_list)

# ...dropping NA's
df = df.dropna(thresh=8)

# ...and resetting the index 
df = df.reset_index(drop=True)


# 
# 
# # Plotting <a id="#plotting"></a>
# ### Tools for fancy plots, charts, and graphs
# 
# ---
# 
# In our dataframe, each row is a unique birdstrike report and each row also only has one Operator. One operator may show up multiple times if they reported multiple birdstrikes so if we count the number of times an Operator shows up, we will have the number of birdstrikes the operator was involved in.
# <br><br>
# 
# We'll use these three methods to count the number of birdstrikes for each operator, then separate the data, and insert the separate arrays into the x- and y-axes:
# 
# * `value_counts()`: This is a method that counts how many times a thing shows up in the array. For example in the Operator column, it will return the Airline Operator names and how many times each one was counted. This will work for any of our columns, so keep that in mind.
# <br><br>
# 
# * `get_values()`: Returns all the unique values and the number of times they occur.
# <br><br>
# 
# * `index`: This was oe of the "Handy Data Exploration Methods" and is how we access the index of the dataframe.
# 
# *In the code below I'm also using the **head()** method to get just the first 10 operators*

# In[ ]:


df["Operator"].value_counts().head(10)


# That looks like a nice table. I bet it would look cool in a barplot!  To do that we'll use `get_values()` and `index` to separate the Operators from the counts. Operators will go on the x-axis and counts will go on the y-axis.

# In[ ]:


# Get the numnber of occurances of each operator
operator_counts = df.Operator.value_counts()

# Split and Save the Operator names in a variable
operators = operator_counts.index

# Split and Save the counts in another variable
counts = operator_counts.get_values()


# ### Bare minimum barplot<a id="#bareminimum"></a>
# These arrays (operaters and counts) and this code are all you need to make the bare minimum barplot.
# 
# `sns.barplot(x=operators, y=counts)`
# 
# 

# In[ ]:


# Create barplot object
barplot = sns.barplot(x=operators, y=counts)


# ### Subsetting <a id="#sbusetting"></a>
# The plot above shows every single unique airline operator in the dataset and there are too many to fit on the plot.  This is not something you should be proud of and certainly don't hang it up on your refrigerator. When this happens, you need to get creative and plot more specifically. One way to be more specific is to plot the first 5 or 10 by taking a "slice" or subset like this:
# `x=operators[:10], y=counts[:10]`
# <br><br> 
# There are many ways to filter the data, this is just a convenient one because they are sorted.

# In[ ]:


# Create barplot object
barplot = sns.barplot(x=operators[:10], y=counts[:10])


# ### Rotating labels <a id="#rotating"></a>
# The barplot looks like a barplot now! But the labels are all on jumbled up so we don't want to start autographing and selling them yet.
# 
# This is how you rotate the 'xtick labels'
# 
# Play around with it! Try 90, 45, 30, 69, 1, -95, 420, -214. Try them all!
# 
# `plt.xticks(rotation=90)`
# 
# 

# In[ ]:


# Create barplot object
plt.xticks(rotation=90)
barplot = sns.barplot(x=operators[:10], y=counts[:10])


# ### Changing the color palette <a id="#palette"></a>
# 
# The Seaborn colors look pretty cool by default and there are six variations of their default theme: deep, muted, pastel, bright, dark, and colorblind. You can use one of the Matplotlib or Seaborn built-in palettes or *make your own*.
# <br><br>
# 
# Make a list of colors:
# 
# > `my_palette = ["SlateGray", "CornflowerBlue", "PeachPuff", "MediumSeaGreen"]`
# 
# Use sns.color_palette() to create a palette object
# 
# > `current_palette = sns.color_palette(my_palette)`
# 
# and sns.set_palette() to set the palette and tell it how many colors you need (we want 10). 
# 
# > `sns.set_palette(current_palette, 10)`
# <br><br>
# 
# These websites show some of the different color schemes and are way more helpful than I'll ever be so take a look at them.
# 
# http://seaborn.pydata.org/tutorial/color_palettes.html 
# 
# https://chrisalbon.com/python/seaborn_color_palettes.html
# 

# In[ ]:


# Create and Set a color palette with ridiculous color names
my_palette = ["SlateGray", "CornflowerBlue", "Gold", "SpringGreen"]
current_palette = sns.color_palette(my_palette)
sns.set_palette(current_palette, 10)

# Rotate the x-axis labels
plt.xticks(rotation=90)

# Create the barplot object
barplot = sns.barplot(x=operators[:10], y=counts[:10])


# ### Main Titles and Axis Titles <a id="#titles"></a>
# We're almost there! But what are we looking at exactly? We see some airlines and some kind of count of something. If your best friend walked in and saw this bargraph would they know what it meant? 
# <br><br>
# 
# The answer is no, because we don't have titles.
# <br><br>
# 
# Last but not least, your plots MUST have proper titles! It has been shown in countless studies that plotters, charters, and fancy graphs makers make 200% more money than people that don't use plot titles. I made that up, I have no idea if that's true. Regardless, a person in your industry should be able to look at your plot and understand what the plot is saying! Pretty colors, graphics, and lines mean nothing without some description of what they represent. 
# 
# Don't neglect your graphs - give them titles!

# In[ ]:


# Create and Set the color palette
paired_palette = sns.color_palette("colorblind")
sns.set_palette(paired_palette, 10)

# Rotate the x-labels
plt.xticks(rotation=45)

# Add the x-axis title
plt.xlabel("x-axis Title: Airline operators", fontsize=20)

# Add the y-axis title
plt.ylabel("y-axis Title: Number of birdstrikes", fontsize=20)

# Add the plot title
plt.title("Main title: Birdstrikes per Airline Operator", fontsize=20)

# Create the plot
barplot = sns.barplot(x=operators[:10], y=counts[:10])


# # Your turn!
# ---
# ### You have these arrays to play with:
# * Incident Year 
#     * df["Incident Year"]
# * Incident Month
#     * df["Incident Month"]
# * Incident Day
#     * df["Incident Day"]
# * Operator
#     * df["Operator"]
# * State
#     * df["State"]
# * Airport
#     * df["Airport"]
# * Flight Phase
#     * df["Flight Phase"]
# * Species Name
#     * df["Species Name"]
# <br><br>
# 
# ### We used these functions:
# - get_values()
# - value_counts()
# - index
# <br><br>
# 
# ### Excercises:
# 
# 1) Try swapping the x- and y-axes
# - What happens if you use month vs species? 
#     - Hint: It will be a disaster if you don't trim it down somehow!
# - Can you see any kind of seasonal relationship?
# - Which phase of flight has the most number of strikes?
# <br><br>
# 
# 2) Try rotating labels 30 or 45 degrees 
# - What does it look like if you rotate it 180? 
# - 270? 
# - How about the y-labels?
# <br><br>
# 
# 3) Play around with the color palettes 
# - I guarantee you can waste an hour of your day with colors alone.
# <br><br>
# 
# You can and should use whatever tools you want, these are just the ones I talked about. In fact, I probably used like 100 more than I told you about just in making this tutorial. Remember back when I said I wanted to make a different tutorial that would blow your socks off with data lazers? Well, at some point I was like "omg so much stuff" and now here we are.
# <br><br>
# 
# 
