
# coding: utf-8

# It  seems to me this dataset has more garbage than data, let's wash it little bit

# In[ ]:


import numpy as np 
import pandas as pd 

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import matplotlib.pyplot as plt


# In[ ]:


train = pd.read_csv('../input/train.csv')
train.price_doc = train.price_doc/train.full_sq


# Let's look at heart of Moscow. Let's find flats closest to Kremlin

# In[ ]:


train.kremlin_km.min()


# 70 metres from Kremlin. What is it? Putin's cordyard, Lenin's Mausoleum?  Let's take a look on prices.

# In[ ]:


fig, ax = plt.subplots(figsize = (12,10))
ax.hist(train.loc[train.kremlin_km < 0.08, 'price_doc'], bins =200)
ax.set_title("Flats between Putin's courtyard and Lenin's Mausoleum" )
ax.set_xlim(0,500000)
ax.set_xlabel('PRICE for m2')
ax.set_ylabel('Number of observations')
plt.show()


# Most prices are lower than 100 000 Rub/m2. What?  Weekend in hotel in this area would cost this price.
# Probability to find this kind of prices is like to see elefants fly. 
# 
# It funny to think if dataset is right: we have flats with zero life square. Zero life square? Looks like a big piece of concrete. Or flats with build_year 1, right after Jesus's born.
# 
# And finally i've found some explanations for low prices in this area: 
#  
# - This flats made of paper
# - Hurricane grabbed  and threw some flats in the center of Moscow
# - We've found secret city under the ground
# - One more mistake in data
