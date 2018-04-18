
# coding: utf-8

# Import pandas so we can use data frames. 

# In[ ]:


# import pandas
import pandas as pd


# Read in and print out our entire csv.

# In[ ]:


pd.read_csv("../input/archive.csv")


# Assign our data to a varaible named data and describe/summarize it. 

# In[ ]:


data = pd.read_csv("../input/archive.csv")
data.describe()

