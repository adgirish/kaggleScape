
# coding: utf-8

# New to data science? Need a quick refresher? This five day challenge will give you the guidance and support you need to kick-start your data science journey.
# 
# By the time you finish this challenge, you will:
# 
# - Read in and summarize data
# - Visualize both numeric and categorical data
# - Know when and how to use two foundational statistical tests (t-test and chi-squared)
# 
# The 5-Day Data Challenge originally ran from October 23, 2017 to October 27, 2017. I've collected all the material here so you can do it at your own pace. Feel free to ask questions on the forums or this notebook if you need a hand.
# 
# ____
# 
# # Table of Contents
# 
# * [Day 1: Reading data into a kernel](#Day-1:-Reading-data-into-a-kernel)
# * [Day 2: Plot a Numeric Variable with a Histogram](#Day-2:-Plot-a-Numeric-Variable-with-a-Histogram)
# * [Day 3: Perform a t-test](#Day-3:-Perform-a-t-test)
# * [Day 4: Visualize categorical data with a bar chart](#Day-4:-Visualize-categorical-data-with-a-bar-chart)
# * [Day 5: Using a Chi-Square Test](#Day-5:-Using-a-Chi-Square-Test)
# ____

# # Day 1: Reading data into a kernel
# 
# ### What are we doing today?
# We will read (or import) data into a kernel and summarize it using R or Python. If you're new to coding, go ahead and try the exercise in both languages so you can see what they're like. 
# 
# ### What data do you need?
# For today’s challenge, you’re going to need a dataset that has a .csv file in it. “CSV” stands for “comma separated values”, and is a way of storing a spreadsheet where each row is a new line and there’s a comma between each value. You can see if a dataset has a .csv in it by going to a dataset, clicking on the “Data” tab at the top (under the heading picture) and seeing if there are any files that have the .csv extension. If you’re having trouble finding a dataset, check out this blog post for some pointers.
# 
# ### Challenge Instructions
# 
# 1.** Find a Kaggle dataset that’s interesting to you and has at least one .csv file in it**. (You can find a list of some fun, beginner-friendly datasets [here](https://www.kaggle.com/rtatman/fun-beginner-friendly-datasets/). For today, it doesn't matter which dataset you pick.)
# 2. **Start a new kernel.** This is a quick way to see an overview of your dataset. You can do this by clicking the blue “New Kernel” button that shows up on the top left of any dataset. I’d recommend choosing a notebook kernel to get started. Give it a helpful title, like "5 Day Data Challenge: Day 1".
# 3. **Pick your language. **Kernels launch in Python 3, but you can also write kernels in R. Use the dropdown menu at the top of the notebook to change languages.
#     - **Python** is a general purpose programming language.
#     - **R**  is a programming language specifically for data analysis and visualization.
#     -Want a quick introduction to each language? Check out [this tutorial on R for complete beginners](https://www.kaggle.com/rtatman/getting-started-in-r-first-steps/) and [this beginner’s guide to Python](https://www.kaggle.com/sohier/whirlwind-tour-of-python-index).   
# 4.** Read in the libraries you’re going to use.** (Libraries are collections of useful functions that aren't included in the base programming language.) I'd recommend:
#     - Python: pandas (command: import pandas as pd)
#     - R: tidyverse (command: library(tidyverse))
# 5. **Read your data into a dataframe. **The filename, which you will need to put in the parentheses, will look like “../input/filename.csv”. Use the:
#     - **Python**: Read_csv() function from Pandas
#     - **R**: Read.csv() function built into R or the read_csv() function from the Tidyverse package
# 6. **Summarize your data.** One way to do this is by putting the read_csv or read.csv function you wrote above inside the parentheses of the functions below. Try the:
#     - **Python**: Describe() function from Pandas
#     - **R**: Summary() function built into R
# 7. **Optional:** If you want to share your analysis with friends or to ask for help, you’ll need to make it public so that other people can see it.
#     - Publish your kernel by hitting the big blue “publish” button. (This may take a second.)
#     - Change the visibility to “public” by clicking on the blue “Make Public” text (right above the “Fork Notebook” button).
# 
# ### Example kernels & additional resources: 
# 
# * [Day 1 (in R)](https://www.kaggle.com/rtatman/5-day-data-challenge-day-1-r)
# * [Day 1 (in Python)](https://www.kaggle.com/rtatman/5-day-data-challenge-day-1-python)
# * Recorded livestream of me doing the challenge (first in R, then in Python). The first day is broken into three videos becuase I was having technical problems with the stream. All other days are one video (and also don't' have freezing/buffering problems).
#     - [Part one](https://www.youtube.com/watch?v=qN9_z3vIp4U)
#     - [Part two](https://www.youtube.com/watch?v=ILqYEtQ_7G0)
#     - [Part three](https://www.youtube.com/watch?v=SFQ1ECXiUME)
#    
# ----

# # Day 2: Plot a Numeric Variable with a Histogram
# 
# ### What are we doing today?
# Today we’re going to be plotting a numeric variable using a histogram. Numeric data is any type of data where what you’re measuring can be represented with a number–like height, weight, income, distance, or number of clicks.
# A histogram is a type of chart where the x-axis (along the bottom) is the range of numeric values of the variable, chopped up into a series of bins. For example, if the range of a value is from 0 to 12, it might be split into four bins, the first one being 1 through 3, the second being 4 through 6, the third 7 through 8, and the fourth 9 through 12. The y-axis (along the side) is the count of how many observations fall within each bin.
#  
# ### What data do you need?
# For today’s challenge, you’re going to need a dataset with at least one numeric variable in it. (This just means that one of the columns in your dataframe will have numbers in it.)
# 
# ### Challenge Instructions
# 
# 1. **Find a dataset, start a kernel, load in your libraries and read your data into a dataframe** (just like we did yesterday). You can find a list of some datasets with at least one numeric variable here. Don’t forget to give your notebook a helpful title, like "5 Day Data Challenge: Day 2".
# 2. **Load in visualization libraries**. I'd recommend:
#     - **Python**: Matplotlib.pyplot (command: import matplotlib.pyplot as plt)
#     - **R**: ggplot2, which is included in the  package. If you’ve loaded in the tidyverse library (command: library(tidyverse)) you’ve already got access to ggplot2.
# 3. **Pick one column with numeric variables in it. ** (Hint: you can use the describe() function in Python or the summary() function in R to help figure out which columns are numeric.)
#     - **Python**: To get just one column of a dataframe, you can use the syntax dataframe[“columnName”]
#     - **R**: To get just one column of a dataframe, you can use the syntax dataframe\$columnName
# 4. **Plot a histogram of that column**. Try the:
#     -  **Python**: hist() function from Matplotlib
#     - **R**: geom_histogram() ggplot2 layer, which you will need to add to a blank plot generated using the ggplot() command
# 5. ** Don’t forget to add a title!** :) Use the:
#     -  **Python**: plt.title() command
#     - **R**: ggtitle() layer
# 6. ** Optional: **If you want to share your analysis with friends or to ask for help, you’ll need to make it public so that other people can see it.
#     - Publish your kernel by hitting the big blue “publish” button. (This may take a second.)
#     - Change the visibility to “public” by clicking on the blue “Make Public” text (right above the “Fork Notebook” button).
# 
# ### Example kernels & additional resources: 
# 
# * [Day 2 (in R)](https://www.kaggle.com/rtatman/5-day-data-challenge-day-2-r)
# * [Day 2 (in Python)](https://www.kaggle.com/rtatman/5-day-data-challenge-day-2-python) (Shout out to all the folks on the stream who helped out with this one! :)
# * [Recording of Day 2 livestream (first in R, then in Python)](https://www.youtube.com/watch?v=HXz6ZEvgorg)
# 
# ----

# # Day 3: Perform a t-test
# 
# ### What are we doing today?
# Today, we’re going to answer a question: is a numeric variable different between two groups? To answer this question, we’re going to use a t-test.
# 
# A t-test is a statistical test that can help us estimate whether the difference in a numerical measure between two groups is reliable. Because we’re comparing two groups to each other, we’re going to be using a special flavor of a t-test called an independent samples t-test. (You can do a t-test with only one sample, but you need to know what you’d expect the mean and standard deviation of the group you sampled it from to be.) If you want to compare more than two groups, you can use an extension of a t-test called an “Analysis of Variance” or “ANOVA”.
# 
# A t-test will return a p-value. If a p-value is very low (generally below 0.01) this is evidence that it’s unlikely that we would have drawn our second sample from the same distribution as the first just by chance. For more information on t-tests, I’d recommend reading Chapter Five of OpenIntro Statistics 3rd Edition, which you can [download for free](https://www.openintro.org/stat/textbook.php?stat_book=os).
# 
# ### What data do you need?
# For today’s challenge, you’re going to need a dataset that has at least one numerical variable it in that has been measured for two different groups. (If your dataset has more than two groups in it, you can always just pick two.)  Here are a couple of datasets that will work well for today’s challenge:
# 
# * [The cereals nutrition dataset](https://www.kaggle.com/crawford/80-cereals). You can look at whether there is the same amount of sugar or sodium in the two types of cereal (hot or cold).
# * [The Museums, Aquariums, and Zoos dataset](https://www.kaggle.com/imls/museum-directory). You can look at whether the revenue for zoos is different than all other types of museums combined. This dataset will require some cleaning.
# * [The Women's Shoe Price dataset](https://www.kaggle.com/datafiniti/womens-shoes-prices). Are pink shoes more expensive than other colors of shoes? This dataset will require some cleaning.
# 
# ### Challenge Instructions
# 
# 1. **You know the drill by now!** :) Find a dataset, start a kernel, load in your libraries, and read your data into a dataframe.
#     - **Python**: Import the ttest_ind() function from scipy.stats (command: from scipy.stats import ttest_ind)
#     - **R**: You don’t need to import anything. :) (R is a programming language specifically for statistics, so statistical methods are already built into it.)
# 2. **Figure out which column has your numeric variable in it** and which column has your group labels in it.
# 3. **Perform a t-test**. I'd recommend using:
#     - **Python**: The ttest_ind() function from scipy.stats. Note: I’d recommend using the argument “equal_var=False” with this function unless the standard deviation of your numeric variable is the same between the two groups. You can calculate this usig the std() function from numpy.
#     - **R**: The t.test function, which is built into R.
# 4. **Extra credit**: Plot two histograms of your data, one for each group you included in your t-test.
# 5. ** Optional: **If you want to share your analysis with friends or to ask for help, you’ll need to make it public so that other people can see it.
#     - Publish your kernel by hitting the big blue “publish” button. (This may take a second.)
#     - Change the visibility to “public”  by clicking on the blue “Make Public” text (right above the “Fork Notebook” button).
# 
# ### Example kernels & additional resources: 
# 
# * [Day 3 (in R)](https://www.kaggle.com/rtatman/5-day-data-challenge-day-3-r)
# * [Day 3 (in Python)](https://www.kaggle.com/rtatman/5-day-data-challenge-day-3-python)
# * [Recording of Day 3 livestream](https://www.youtube.com/watch?v=SFYjnqDUPSQ)
# 
# ---

# # Day 4: Visualize categorical data with a bar chart
# 
# ### What are we doing today?
# 
# Today, we’re going to take a break from numeric data and turn to categorical data.
# 
# Categorical data is a data type containing information that categorizes where other data points are from. Example categories are things like t-shirt size, zip code, dog breed, whether someone is a repeat customer, level of education or hair color.
# 
# We’re going to visualize categorical data using a bar chart. In a bar chart, each category is represented as a different bar, and the height of the bar indicates the count of items in that category.
# 
# ### What data do you need?
# 
# For this challenge, you’re going to need a dataset with a categorical variable in it. You can find a list of some datasets with at least one categorical variable [here](https://www.kaggle.com/rtatman/fun-beginner-friendly-datasets/).
# 
# ### Challenge Instructions
# 
# 1. We’ll start off with the usual stuff: **find a dataset, start a kernel, load in your libraries, and read your data into a dataframe**. You can find a list of some datasets with at least one categorical variable here. Don’t forget to give your notebook a helpful title, like "5 Day Data Challenge: Day 4". I'd recommend:
#     - **Python**: Seaborn (command: import seaborn as sns) and pandas (command: import pandas as pd)
#     - **R**: ggplot, which is included in the tidyverse library (command: library(tidyverse))
# 2. **Pick a column with a categorial variable in it**.
# 3. **Plot a bar-chart**. I'd recommend:
#     - **Python**: Using the sns.barplot() function from Seaborn
#     - **R**: Adding a geom_bar() layer to a ggplot
# 4. **Don’t forget to add a title!** :) Try the:
#     - **Python**: plt.title() command
#     - **R**: ggtitle() layer
# 5. **Extra credit**: Pick another visualization for your dataset and figure out how to do it in your language of choice. (I really like this interactive data visualization catalog for picking what type of chart or graph to use.)
# 5. **Optional**: If you want to share your analysis with friends or to ask for help, you’ll need to make it public so that other people can see it.
#     - Publish your kernel by hitting the big, blue “publish” button. (This may take a second.)
#     - Change the visibility to “public” by clicking on the blue “Make Public” text (right above the “Fork Notebook” button).
# 
# ### Example kernels & additional resources: 
# 
# * [Day 4 (in R)](https://www.kaggle.com/rtatman/5-day-data-challenge-day-4-r)
# * [Day 4 (in Python)](https://www.kaggle.com/rtatman/5-day-data-challenge-day-4-python) (This notebooks includes barcharts 
# made with both Matplotlib and Seaborn.)
# * [Recording of Day 4 livestream](https://www.youtube.com/watch?v=_FaOj9Ei5nI)
# _____

# # Day 5: Using a Chi-Square Test
# 
# 
# ### What are we doing today?
# Is the difference in the number of observations across different groups just the result of random variation? Or does it reflect an underlying difference between the two groups?
# 
# For example, we might have red and green apples, some of which are bruised and some which are not. Are apples of one color more likely to be bruised? We can figure out if one color of apple is more likely to bruise using the chi-square test (also written as Χ^2).
# 
# ### What data do you need?
# For this challenge, you’re going to need a dataset with at least two categorical variables in it. (Just like yesterday.)
# 
# ### Challenge Instructions
# 1. At this point you’re already a pro at this step. You should **find a dataset, start a kernel, load in your libraries, and read your data into a dataframe**. You can find a list of some datasets with at least one categorical variable [here](https://www.kaggle.com/rtatman/fun-beginner-friendly-datasets/). Don’t forget to give your notebook a helpful title, like "5 Day Data Challenge: Day 5". I'd recommend:
#     - **Python**: scipy.stats (command: import scipy.stats) and pandas (command: import pandas as pd)
#     - **R**: tidyverse (command: library(tidyverse))
# 2. **Pick two columns** which both have categorical variables in them. (These will probably be strings, characters or objects rather than numbers.)
# 3. **Calculate chi-square**. Try:
#     - **Python**: The chi2_contingency() function from scipy.stats
#     - **R**: Try chisq.test(), which is built into R
# 4. **Extra credit:** Visualize your dataset.
# 5. **Optional:** If you want to share your analysis with friends or to ask for help, you’ll need to make it public so that other people can see it.
#     - Publish your kernel by hitting the big blue “publish” button. (This may take a second.)
#     - Change the visibility to “public” by clicking on the blue “Make Public” text (right above the “Fork Notebook” button).
# 
# ### Example kernels and additional resources:
# 
# * [Day 5 (in R)](https://www.kaggle.com/rtatman/5-day-data-challenge-day-5-r)
# * [Day 5 (in Python)](https://www.kaggle.com/rtatman/5-day-data-challenge-day-5-python)
# made with both Matplotlib and Seaborn.)
# * [Recording of Day 5 livestream](https://www.youtube.com/watch?v=78kGMzoMUEE)
# _____

# # Congrats! You've completed the 5-Day Data Challenge
# 
# ## What now?
# 
# Now that you’ve got a basic set of tools, you’re ready to start exploring more datasets. You can always check out new datasets [here](https://www.kaggle.com/datasets?sortBy=updated&group=featured) or [upload your own](https://www.kaggle.com/datasets?modal=true).
#  
#  You can also check out some of the great learning resources and tutorials on Kaggle [here](https://www.kaggle.com/dansbecker/learning-materials-on-kaggle) if you're looking for more structure. 
#  
#  <center>
#  __Good luck and enjoy the rest of your data science journey!__
