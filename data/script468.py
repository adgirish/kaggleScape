
# coding: utf-8

# It is my first ever kernel, and in this notebook I tried to explore the categorical variables.
# 
# ###### Objective:
# + Sherlocks interpretation of the data by plotting them. In essence, I tried to understand the meaning of column by plotting them. Moreover, I have created stacked barplot function, which is reusable, using seaborn and matplotlib.
# 
# + Most of the exploration was covered by @SRK but this could further add to his kernel.
# 
# 
# *Skip to X3 and X4 for the essence of the kernel.*

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import seaborn as sns # plotting the charts
import matplotlib.pyplot as plt 

get_ipython().run_line_magic('matplotlib', 'inline')

color = sns.color_palette()
plt.style.use('seaborn-notebook')

pd.options.display.max_columns = 1000  # displaying all the columns on the screen
pd.options.mode.chained_assignment = None

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

print("train data shape {}".format(train_df.shape))
print("test data shape {}".format(test_df.shape))


# In[ ]:


train_df.head()


# ###### Observations :
#   + Both the train and test have exactly same number of records , why? 
#   + **Hypothesis**  :: Same car models but with different settings - since in the description of the competition, it was mentioned that there are different permutations of Mercedes-Benz car features. So, that could be possible.        

# In[ ]:


ids = pd.DataFrame({"train_ids": train_df.ID, "test_ids": test_df.ID})
ids.head(10)


# It looks like there were 8414 permutations of a single model or multiple models and they were split randomly into test and train dataset. Might lead to some interesting leaks if explored more.

# In[ ]:


# describing the 'y' variable
train_df["y"].describe()


# ## Categorical Exploration

# In[ ]:


# created a reusable function for plotting stacked bar charts

def plot_stack(col, train_color="green", test_color="#0000A3", sortby="total_val", ascending=False, title="categorical X"):
    test_x = dict(test_df[col].value_counts())
    train_x = dict(train_df[col].value_counts())
    test_xd = pd.DataFrame({"cols": list(test_x.keys()), "v_test": list(test_x.values())})
    train_xd = pd.DataFrame({"cols": list(train_x.keys()), "v_train": list(train_x.values())})

    
    total_xd = pd.merge(test_xd, train_xd, how="outer", on="cols")
    total_xd.fillna(0, inplace=True)
    total_xd["total_val"] = total_xd["v_test"] + total_xd["v_train"]
    
    total_xd.sort_values(by=sortby, inplace=True, ascending=ascending)
    
    # plotting the graph
    sns.set_style("darkgrid")
    sns.set_context({"figure.figsize": (20, 9)})
    
    sns.barplot(x = total_xd.cols, y = total_xd.total_val, color = train_color)
    bottom_plot = sns.barplot(x = total_xd.cols, y = total_xd.v_train, color = test_color)
    
    
    # adding legends
    topbar = plt.Rectangle((0,0),1,1,fc=train_color, edgecolor = 'none')
    bottombar = plt.Rectangle((0,0),1,1,fc=test_color,  edgecolor = 'none')
    l = plt.legend([bottombar, topbar], ['train', 'test'], loc=1, ncol = 2, prop={'size':16})
    l.draw_frame(False)
    
    sns.despine(left=True)
    bottom_plot.set_ylabel("frequency")
    bottom_plot.set_xlabel("category")
    bottom_plot.set_title(title, fontsize=15)
    
    for item in ([bottom_plot.xaxis.label, bottom_plot.yaxis.label] +
             bottom_plot.get_xticklabels() + bottom_plot.get_yticklabels()):
        item.set_fontsize(16)  
        
    plt.show()    
    print("  # of categories : {}".format(len(total_xd)))


# In[ ]:


# exploring variable X0 
plot_stack("X0", train_color="green", test_color="#0000A3", 
           title="variable 'X0' chart")


# ##### Observation:
# + test and train data are equally distributed
# + approching to the end, frequency falls to almost 1, hence they could be removed to decrease the # of categories during one hot encoding, as they are present in either test or in train but not in both
# + 53 categories **_do not signify_** anything as such

# In[ ]:


# exploring variable X1
plot_stack("X1", train_color="red", test_color="#1220A6", 
           sortby="total_val", 
           ascending=False, 
           title="variable 'X1' chart")


# ###### observations:
# + seems like a simple normal distribution, no inference.

# In[ ]:


# exploring variable X2
plot_stack("X2", train_color="yellow", test_color="purple", 
           title="variable 'X2' chart")


# ###### Observations:
# + huge difference in the highest and the lowest peak, 
# + we can just eliminate lot of categories same as in X0

# In[ ]:


# exploring variable X3
plot_stack("X3", train_color="aqua", test_color="olive", 
           sortby="cols", 
           title="variable 'X3' chart")


# ###### Observations:
# + 7, hmmm a peculiar number, on my research, I have learned that mercedes-benz has a [7 gear automatic transmission](https://en.wikipedia.org/wiki/Mercedes-Benz_7G-Tronic_transmission), I feel that this variable represents that.
# + or else it could just be as simple as day of week :)

# In[ ]:


# exploring variable X4
plot_stack("X4", train_color="white", test_color="black", 
           sortby="cols", 
           ascending=False, 
           title="variable 'X4' chart")


# ###### Observations:
# + only d category has significant frequency
# + so, based on the above column X4, automatic transmission has three modes n (neutral), p(parking), r(reverse), d(driving)  {[automatic transmission](https://en.wikipedia.org/wiki/Automatic_transmission)}. Since most of the tests are conducting in driving mode, d is highest compared to other categories

# In[ ]:


# exploring variable X5
plot_stack("X5", train_color="lightgreen", test_color="blue", 
           sortby="cols", 
           ascending=True, 
           title="variable 'X5' chart")


# In[ ]:


# exploring variable X6
plot_stack("X6", train_color="orange", test_color="darkblue", 
           sortby="cols", 
           ascending=True, 
           title="variable 'X6' chart")


# ###### Observations:
# + naive deductions suggest it is month of the year, if knows about cars please post in the comment section|.

# In[ ]:


# exploring variable X8, there exists no X7 just incase you missed
plot_stack("X8", train_color="hotpink", test_color="magenta", 
           sortby="cols", 
           ascending=True, 
           title="variable 'X8' chart")


# ## Conclusion:
# 
#  + I have tried to find the real world meaning of the data based on the distribution and # of categories but most of them were just normal distributions. 
# 
# **Coming up ......**
# 
#  +  [numerical variables {X9 - X385} - interaction in between the variables][1]
# 
#  + Performance of the sklearn algorithms and neural networks on the
#    dataset
# 
# 
#                                                             to be continued......
# 
# 
#   [1]: https://www.kaggle.com/remidi/sherlocks-exploration-season-02-e01-numerical

# Thank you, please upvote if you like the kernel.
# 
# 
# P.S:  Since this was my first ever kernel I might have missed something and made mistake. Please do provide your feedback, criticism, corrections or appreciation in the comments. 
