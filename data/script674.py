
# coding: utf-8

# <img src="https://lh3.googleusercontent.com/-tNe1vwwd_w4/VZ_m9E44C7I/AAAAAAAAABM/5yqhpSyYcCUzwHi-ti13MwovCb_AUD_zgCJkCGAYYCw/w256-h86-n-no/Submarineering.png">

# This is  **EVEN better the public score** kernel in the competition until now. 
# I wanted strongly dedicate this kernel to that **scavenger colleagues** that spend the time tracking back the work of other members just to take chance and get any good idea only for his own benefit, but, however they don't share anything, just providing  bad comments for those who share with the best will and without breaking the rules.
# 
# If you do not feel alluded , **please VOTE me UP.**

# In[ ]:


import os
import numpy as np 
import pandas as pd 
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# First thing first@
# # Credits to the following awesome authors and kernels
# 
# Author: QuantScientist    
# File: sub_200_ens_densenet.csv     
# Link: https://www.kaggle.com/solomonk/pytorch-cnn-densenet-ensemble-lb-0-1538     
# 
# 
# Author: wvadim     
# File: sub_TF_keras.csv     
# Link: https://www.kaggle.com/wvadim/keras-tf-lb-0-18     
# 
# 
# Author: Ed Miller    
# File: sub_fcn.csv    
# Link: https://www.kaggle.com/bluevalhalla/fully-convolutional-network-lb-0-193     
# 
# 
# Author: Chia-Ta Tsai    
# File: sub_blend009.csv    
# Link: https://www.kaggle.com/cttsai/ensembling-gbms-lb-203    
# 
# 
# Author: DeveshMaheshwari    
# File: sub_keras_beginner.csv    
# Link: https://www.kaggle.com/devm2024/keras-model-for-beginners-0-210-on-lb-eda-r-d       
# 
# Author: Submarineering    
# 
# Files: submission38.csv , submission43.csv
# 
# Link : https://www.kaggle.com/submarineering/submission38-lb01448
# 
# ### Without their truly dedicated efforts, this notebook will not be possible.     

# # Data Load

# In[ ]:


sub_path = "../input/statoil-iceberg-submissions"
all_files = os.listdir(sub_path)
all_files = all_files[1:3]
all_files.append('submission38.csv')
all_files.append('submission43.csv')
all_files


# In[ ]:


# Read and concatenate submissions
out1 = pd.read_csv("../input/statoil-iceberg-submissions/sub_200_ens_densenet.csv", index_col=0)
out2 = pd.read_csv("../input/statoil-iceberg-submissions/sub_TF_keras.csv", index_col=0)
out3 = pd.read_csv("../input/submission38-lb01448/submission38.csv", index_col=0)
out4 = pd.read_csv("../input/submission38-lb01448/submission43.csv", index_col=0)
concat_sub = pd.concat([out1, out2, out3, out4], axis=1)
cols = list(map(lambda x: "is_iceberg_" + str(x), range(len(concat_sub.columns))))
concat_sub.columns = cols
concat_sub.reset_index(inplace=True)
concat_sub.head()


# In[ ]:


# check correlation
concat_sub.corr()


# In[ ]:


# get the data fields ready for stacking
concat_sub['is_iceberg_max'] = concat_sub.iloc[:, 1:6].max(axis=1)
concat_sub['is_iceberg_min'] = concat_sub.iloc[:, 1:6].min(axis=1)
concat_sub['is_iceberg_mean'] = concat_sub.iloc[:, 1:6].mean(axis=1)
concat_sub['is_iceberg_median'] = concat_sub.iloc[:, 1:6].median(axis=1)


# In[ ]:


# set up cutoff threshold for lower and upper bounds, easy to twist 
cutoff_lo = 0.8
cutoff_hi = 0.2


# # Mean Stacking

# In[ ]:


#concat_sub['is_iceberg'] = concat_sub['is_iceberg_mean']
#concat_sub[['id', 'is_iceberg']].to_csv('stack_mean.csv', index=False, float_format='%.6f')


# **LB 0.1698** , decent first try - still some gap comparing with our top-line model performance in stack.

# # Median Stacking

# In[ ]:


#concat_sub['is_iceberg'] = concat_sub['is_iceberg_median']
#concat_sub[['id', 'is_iceberg']].to_csv('stack_median.csv', index=False, float_format='%.6f')


# **LB 0.1575**, very close with our top-line model performance, but we want to see some improvement at least.

# # PushOut + Median Stacking 
# 
# Pushout strategy is a bit agressive given what it does...

# In[ ]:


#concat_sub['is_iceberg'] = np.where(np.all(concat_sub.iloc[:,1:6] > cutoff_lo, axis=1), 1, 
#                                    np.where(np.all(concat_sub.iloc[:,1:6] < cutoff_hi, axis=1),
#                                             0, concat_sub['is_iceberg_median']))
#concat_sub[['id', 'is_iceberg']].to_csv('stack_pushout_median.csv', 
#                                        index=False, float_format='%.6f')


# **LB 0.1940**, not very impressive results given the base models in the pipeline...

# # MinMax + Mean Stacking
# 
# MinMax seems more gentle and it outperforms the previous one given its peformance score.

# In[ ]:


#concat_sub['is_iceberg'] = np.where(np.all(concat_sub.iloc[:,1:6] > cutoff_lo, axis=1), 
#                                    concat_sub['is_iceberg_max'], 
#                                    np.where(np.all(concat_sub.iloc[:,1:6] < cutoff_hi, axis=1),
#                                             concat_sub['is_iceberg_min'], 
#                                             concat_sub['is_iceberg_mean']))
#concat_sub[['id', 'is_iceberg']].to_csv('stack_minmax_mean.csv', 
#                                        index=False, float_format='%.6f')


# **LB 0.1622**, need to stack with Median to see the results.

# # MinMax + Median Stacking 

# In[ ]:


#concat_sub['is_iceberg'] = np.where(np.all(concat_sub.iloc[:,1:6] > cutoff_lo, axis=1), 
#                                    concat_sub['is_iceberg_max'], 
#                                    np.where(np.all(concat_sub.iloc[:,1:6] < cutoff_hi, axis=1),
#                                             concat_sub['is_iceberg_min'], 
#                                             concat_sub['is_iceberg_median']))
#concat_sub['is_iceberg'] = np.clip(concat_sub['is_iceberg'].values, 0.001, 0.999)
#concat_sub[['id', 'is_iceberg']].to_csv('stack_minmax_median.csv', 
#                                       index=False, float_format='%.6f')


# **LB 0.1488** - **Great!** This is an improvement to our top-line model performance (LB 0.1538). But can we do better?

# # MinMax + BestBase Stacking

# In[ ]:


# load the model with best base performance
sub_base = pd.read_csv('../input/submission38-lb01448/submission43.csv')


# In[ ]:


concat_sub['is_iceberg_base'] = sub_base['is_iceberg']
concat_sub['is_iceberg'] = np.where(np.all(concat_sub.iloc[:,1:4] > cutoff_lo, axis=1), 
                                    concat_sub['is_iceberg_max'], 
                                    np.where(np.all(concat_sub.iloc[:,1:4] < cutoff_hi, axis=1),
                                             concat_sub['is_iceberg_min'], 
                                             concat_sub['is_iceberg_base']))
concat_sub['is_iceberg'] = np.clip(concat_sub['is_iceberg'].values, 0.001, 0.999)
concat_sub[['id', 'is_iceberg']].to_csv('submission54.csv', 
                                        index=False, float_format='%.6f')


# 
# Roboust model is always the key component, stacking only comes last with the promise to surprise, sometimes, in an unpleasant direction@. 
# 
# For more efficient models I highly recommend my engineering features extraction kernels: 
# 
# https://www.kaggle.com/submarineering/submarineering-size-matters-0-75-lb
# 
# https://www.kaggle.com/submarineering/submarineering-objects-isolation-0-75-lb
# 
# https://www.kaggle.com/submarineering/submarineering-what-about-volume-lb-0-45
# 
# Greeting, Subamrineering.
# 
# 
# 

# I hope these lines be useful for your. **Please vote up.**
