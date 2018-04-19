
# coding: utf-8

# ### All days of the challange:
# 
# * [Day 1: Handling missing values](https://www.kaggle.com/rtatman/data-cleaning-challenge-handling-missing-values)
# * [Day 2: Scaling and normalization](https://www.kaggle.com/rtatman/data-cleaning-challenge-scale-and-normalize-data)
# * [Day 3: Parsing dates](https://www.kaggle.com/rtatman/data-cleaning-challenge-parsing-dates/)
# * [Day 4: Character encodings](https://www.kaggle.com/rtatman/data-cleaning-challenge-character-encodings/)
# * [Day 5: Inconsistent Data Entry](https://www.kaggle.com/rtatman/data-cleaning-challenge-inconsistent-data-entry/)
# ___
# 
# Welcome to day 5 of the 5-Day Data Challenge! (Can you believe it's already been five days??) Today, we're going to learn how to clean up inconsistent text entries. To get started, click the blue "Fork Notebook" button in the upper, right hand corner. This will create a private copy of this notebook that you can edit and play with. Once you're finished with the exercises, you can choose to make your notebook public to share with others. :)
# 
# > **Your turn!** As we work through this notebook, you'll see some notebook cells (a block of either code or text) that has "Your Turn!" written in it. These are exercises for you to do to help cement your understanding of the concepts we're talking about. Once you've written the code to answer a specific question, you can run the code by clicking inside the cell (box with code in it) with the code you want to run and then hit CTRL + ENTER (CMD + ENTER on a Mac). You can also click in a cell and then click on the right "play" arrow to the left of the code. If you want to run all the code in your notebook, you can use the double, "fast forward" arrows at the bottom of the notebook editor.
# 
# Here's what we're going to do today:
# 
# * [Get our environment set up](#Get-our-environment-set-up)
# * [Do some preliminary text pre-processing](#Do-some-preliminary-text-pre-processing)
# * [Use fuzzy matching to correct inconsistent data entry](#Use-fuzzy-matching-to-correct-inconsistent-data-entry)
# 
# 
# Let's get started!

# # Get our environment set up
# ________
# 
# The first thing we'll need to do is load in the libraries we'll be using. Not our datasets, though: we'll get to those later!
# 
# > **Important!** Make sure you run this cell yourself or the rest of your code won't work!

# In[ ]:


# modules we'll use
import pandas as pd
import numpy as np

# helpful modules
import fuzzywuzzy
from fuzzywuzzy import process
import chardet

# set seed for reproducibility
np.random.seed(0)


# When I tried to read in the `PakistanSuicideAttacks Ver 11 (30-November-2017).csv`file the first time, I got a character encoding error, so I'm going to quickly check out what the encoding should be...

# In[ ]:


# look at the first ten thousand bytes to guess the character encoding
with open("../input/PakistanSuicideAttacks Ver 11 (30-November-2017).csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))

# check what the character encoding might be
print(result)


# And then read it in with the correct encoding. (If this look unfamiliar to you, check out [yesterday's challenge](https://www.kaggle.com/rtatman/data-cleaning-challenge-character-encodings/).) 

# In[ ]:


# read in our dat
suicide_attacks = pd.read_csv("../input/PakistanSuicideAttacks Ver 11 (30-November-2017).csv", 
                              encoding='Windows-1252')


# Now we're ready to get started! You can, as always, take a moment here to look at the data and get familiar with it. :)
# 
# 
# # Do some preliminary text pre-processing
# ___
# 
# For this exercise, I'm interested in cleaning up the "City" column to make sure there's no data entry inconsistencies in it. We could go through and check each row by hand, of course, and hand-correct inconsistencies when we find them. There's a more efficient way to do this though!

# In[ ]:


# get all the unique values in the 'City' column
cities = suicide_attacks['City'].unique()

# sort them alphabetically and then take a closer look
cities.sort()
cities


# Just looking at this, I can see some problems due to inconsistent data entry: 'Lahore' and 'Lahore ', for example, or 'Lakki Marwat' and 'Lakki marwat'.
# 
# The first thing I'm going to do is make everything lower case (I can change it back at the end if I like) and remove any white spaces at the beginning and end of cells. Inconsistencies in capitalizations and trailing white spaces are very common in text data and you can fix a good 80% of your text data entry inconsistencies by doing this.

# In[ ]:


# convert to lower case
suicide_attacks['City'] = suicide_attacks['City'].str.lower()
# remove trailing white spaces
suicide_attacks['City'] = suicide_attacks['City'].str.strip()


# Next we're going to tackle more difficult inconsistencies.

# In[ ]:


# Your turn! Take a look at all the unique values in the "Province" column. 
# Then convert the column to lowercase and remove any trailing white spaces


# # Use fuzzy matching to correct inconsistent data entry
# ___
# 
# Alright, let's take another look at the city column and see if there's any more data cleaning we need to do.

# In[ ]:


# get all the unique values in the 'City' column
cities = suicide_attacks['City'].unique()

# sort them alphabetically and then take a closer look
cities.sort()
cities


# It does look like there are some remaining inconsistencies: 'd. i khan' and 'd.i khan' should probably be the same. (I [looked it up](https://en.wikipedia.org/wiki/List_of_most_populous_cities_in_Pakistan) and 'd.g khan' is a seperate city, so I shouldn't combine those.) 
# 
# I'm going to use the [fuzzywuzzy](https://github.com/seatgeek/fuzzywuzzy) package to help identify which string are closest to each other. This dataset is small enough that we could probably could correct errors by hand, but that approach doesn't scale well. (Would you want to correct a thousand errors by hand? What about ten thousand? Automating things as early as possible is generally a good idea. Plus, it’s fun! :)
# 
# > **Fuzzy matching:** The process of automatically finding text strings that are very similar to the target string. In general, a string is considered "closer" to another one the fewer characters you'd need to change if you were transforming one string into another. So "apple" and "snapple" are two changes away from each other (add "s" and "n") while "in" and "on" and one change away (rplace "i" with "o"). You won't always be able to rely on fuzzy matching 100%, but it will usually end up saving you at least a little time.
# 
# Fuzzywuzzy returns a ratio given two strings. The closer the ratio is to 100, the smaller the edit distance between the two strings. Here, we're going to get the ten strings from our list of cities that have the closest distance to "d.i khan".

# In[ ]:


# get the top 10 closest matches to "d.i khan"
matches = fuzzywuzzy.process.extract("d.i khan", cities, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

# take a look at them
matches


# We can see that two of the items in the cities are very close to "d.i khan": "d. i khan" and "d.i khan". We can also see the "d.g khan", which is a seperate city, has a ratio of 88. Since we don't want to replace "d.g khan" with "d.i khan", let's replace all rows in our City column that have a ratio of > 90 with "d. i khan". 
# 
# To do this, I'm going to write a function. (It's a good idea to write a general purpose function you can reuse if you think you might have to do a specific task more than once or twice. This keeps you from having to copy and paste code too often, which saves time and can help prevent mistakes.)

# In[ ]:


# function to replace rows in the provided column of the provided dataframe
# that match the provided string above the provided ratio with the provided string
def replace_matches_in_column(df, column, string_to_match, min_ratio = 90):
    # get a list of unique strings
    strings = df[column].unique()
    
    # get the top 10 closest matches to our input string
    matches = fuzzywuzzy.process.extract(string_to_match, strings, 
                                         limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

    # only get matches with a ratio > 90
    close_matches = [matches[0] for matches in matches if matches[1] >= min_ratio]

    # get the rows of all the close matches in our dataframe
    rows_with_matches = df[column].isin(close_matches)

    # replace all rows with close matches with the input matches 
    df.loc[rows_with_matches, column] = string_to_match
    
    # let us know the function's done
    print("All done!")


# Now that we have a function, we can put it to the test!

# In[ ]:


# use the function we just wrote to replace close matches to "d.i khan" with "d.i khan"
replace_matches_in_column(df=suicide_attacks, column='City', string_to_match="d.i khan")


# And now let's can check the unique values in our City column again and make sure we've tidied up d.i khan correctly.

# In[ ]:


# get all the unique values in the 'City' column
cities = suicide_attacks['City'].unique()

# sort them alphabetically and then take a closer look
cities.sort()
cities


# Excellent! Now we only have "d.i khan" in our dataframe and we didn't have to change anything by hand. 

# In[ ]:


# Your turn! It looks like 'kuram agency' and 'kurram agency' should
# be the same city. Correct the dataframe so that they are.


# And that's it for today! If you have any questions, be sure to post them in the comments below or [on the forums](https://www.kaggle.com/questions-and-answers). 
# 
# Remember that your notebook is private by default, and in order to share it with other people or ask for help with it, you'll need to make it public. First, you'll need to save a version of your notebook that shows your current work by hitting the "Commit & Run" button. (Your work is saved automatically, but versioning your work lets you go back and look at what it was like at the point you saved it. It also lets you share a nice compiled notebook instead of just the raw code.) Then, once your notebook is finished running, you can go to the Settings tab in the panel to the left (you may have to expand it by hitting the [<] button next to the "Commit & Run" button) and setting the "Visibility" dropdown to "Public".
# 
# # More practice!
# ___
# 
# Do any other columns in this dataframe have inconsistent data entry? If you can find any, try to tidy them up.
# 
# You can also try reading in the `PakistanSuicideAttacks Ver 6 (10-October-2017).csv` file from this dataset and tidying up any inconsistent columns in that data file.
