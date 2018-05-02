
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from time import clock
from tqdm import tqdm
from subprocess import check_output
print(check_output(["ls", "../input/"]).decode("utf8"))


# In[ ]:


get_ipython().run_cell_magic('time', '', "child_prefs = pd.read_csv('../input/santa-gift-matching/child_wishlist.csv', header=None).drop(0, axis=1).values\ngift_prefs = pd.read_csv('../input/santa-gift-matching/gift_goodkids.csv', header=None).drop(0, axis=1).values\n# load sample sub\ndf = pd.read_csv('../input/santa-gift-matching/sample_submission_random.csv')\ndf2 = pd.read_csv('../input/085933376csv/0.85933376.csv')")


# In[ ]:


get_ipython().run_cell_magic('time', '', '# BUILD 2G lookup table\nchigif = np.full((1000000, 1000), -101,dtype=np.int16)\nVAL = (np.arange(20,0,-2)+1)*100\nfor c in tqdm(range(1000000)):\n    chigif[c,child_prefs[c]] += VAL \nVAL = (np.arange(2000,0,-2)+1)\nfor g in tqdm(range(1000)):\n    chigif[gift_prefs[g],g] += VAL ')


# In[ ]:


get_ipython().run_cell_magic('time', '', "# COMPUTE SCORE sample sub\nscore = np.sum(chigif[df.ChildId,df.GiftId])/2000000000\nprint('score',score)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# COMPUTE SCORE own sub\nscore = np.sum(chigif[df2.ChildId,df2.GiftId])/2000000000\nprint('score',score)")

