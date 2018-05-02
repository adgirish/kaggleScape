
# coding: utf-8

# **Typical full-time kaggler questions**
# * Ever wondered how much ranking points you can potentially gain by adding another team member?
# * May be you were interested if you should invest more time to gain several positions?
# * Does the number of competitors matter that much?
# 
# With my new tool you can finally answer all of those questions!

# Here is the table of ranking scores for competition with 3000 participants:

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
from pylab import rcParams
rcParams['figure.figsize'] = 25, 30

img = cv2.imread("../input/rank_3000.png")
plt.imshow(img)
plt.show()


# If you already have a big team adding one more member does not make much of a difference.
# 
# Here is the data for 1000 participants:

# In[ ]:


img = cv2.imread("../input/rank_1000.png")
plt.imshow(img)
plt.show()


# Medals don't matter but every position on the leaderboard does, especally on the top!
# 
# Small competition (300 contesters): 

# In[ ]:


img = cv2.imread("../input/rank_300.png")
plt.imshow(img)
plt.show()


# It's clear that number of participants does not matter that much. However, it is much easier to win in a smaller competition. 
# 
# You can play youself with the excel file attached (rank.xlsx). 
# 
# Happy kaggling!
