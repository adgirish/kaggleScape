
# coding: utf-8

# In[ ]:


import pandas as pd
import plotly
plotly.__version__
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np
import os 
import sys

cwd=os.getcwd()
cwd
os.chdir('/kaggle/input/')
all_list=os.listdir()
if len(all_list)<5:
    os.chdir('/kaggle/input/earn-your-6-figure-prize/')



# In[ ]:


FTHT6 = pd.read_csv('FT_HT6.csv')
len(FTHT6)


# In[ ]:


ranks6 = pd.read_csv('ranks6.csv')


# In[ ]:


winrate6 = pd.read_csv('winrate6.csv')


# In[ ]:


country6 = pd.read_csv('country6.csv')


# In[ ]:


names6 = pd.read_csv('names6.csv')


# In[ ]:


fresults6 = pd.read_csv('fresults6.csv')


# In[ ]:


close6=fresults6[((ranks6.iloc[:,0]-ranks6.iloc[:,1]).abs()<7)]
x=close6.iloc[:,0]+close6.iloc[:,1]


# In[ ]:


mask1 =  (FTHT6.iloc[:,1]<0.49) & (FTHT6.iloc[:,0]<1.35) & ((ranks6.iloc[:,0]-ranks6.iloc[:,1]).abs()<7)


# In[ ]:


mask2 = ((winrate6<=65).all(1)) & ((ranks6>=8).all(1)) & (country6.iloc[:,0]!='England')


# In[ ]:


mask = mask1 & mask2


# In[ ]:


country6[mask]


# In[ ]:


names6[mask]

