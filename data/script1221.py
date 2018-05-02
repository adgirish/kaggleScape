
# coding: utf-8

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
# ### Without their truly dedicated efforts, this notebook will not be possible.     

# # Data Load

# In[ ]:


sub_path = "../input/statoil-iceberg-submissions"
all_files = os.listdir(sub_path)

# Read and concatenate submissions
outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in all_files]
concat_sub = pd.concat(outs, axis=1)
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


concat_sub['is_iceberg'] = concat_sub['is_iceberg_mean']
concat_sub[['id', 'is_iceberg']].to_csv('stack_mean.csv', 
                                        index=False, float_format='%.6f')


# **LB 0.1698** , decent first try - still some gap comparing with our top-line model performance in stack.

# # Median Stacking

# In[ ]:


concat_sub['is_iceberg'] = concat_sub['is_iceberg_median']
concat_sub[['id', 'is_iceberg']].to_csv('stack_median.csv', 
                                        index=False, float_format='%.6f')


# **LB 0.1575**, very close with our top-line model performance, but we want to see some improvement at least.

# # PushOut + Median Stacking 
# 
# Pushout strategy is a bit agressive given what it does...

# In[ ]:


concat_sub['is_iceberg'] = np.where(np.all(concat_sub.iloc[:,1:6] > cutoff_lo, axis=1), 1, 
                                    np.where(np.all(concat_sub.iloc[:,1:6] < cutoff_hi, axis=1),
                                             0, concat_sub['is_iceberg_median']))
concat_sub[['id', 'is_iceberg']].to_csv('stack_pushout_median.csv', 
                                        index=False, float_format='%.6f')


# **LB 0.1940**, not very impressive results given the base models in the pipeline...

# # MinMax + Mean Stacking
# 
# MinMax seems more gentle and it outperforms the previous one given its peformance score.

# In[ ]:


concat_sub['is_iceberg'] = np.where(np.all(concat_sub.iloc[:,1:6] > cutoff_lo, axis=1), 
                                    concat_sub['is_iceberg_max'], 
                                    np.where(np.all(concat_sub.iloc[:,1:6] < cutoff_hi, axis=1),
                                             concat_sub['is_iceberg_min'], 
                                             concat_sub['is_iceberg_mean']))
concat_sub[['id', 'is_iceberg']].to_csv('stack_minmax_mean.csv', 
                                        index=False, float_format='%.6f')


# **LB 0.1622**, need to stack with Median to see the results.

# # MinMax + Median Stacking 

# In[ ]:


concat_sub['is_iceberg'] = np.where(np.all(concat_sub.iloc[:,1:6] > cutoff_lo, axis=1), 
                                    concat_sub['is_iceberg_max'], 
                                    np.where(np.all(concat_sub.iloc[:,1:6] < cutoff_hi, axis=1),
                                             concat_sub['is_iceberg_min'], 
                                             concat_sub['is_iceberg_median']))
concat_sub[['id', 'is_iceberg']].to_csv('stack_minmax_median.csv', 
                                        index=False, float_format='%.6f')


# **LB 0.1488** - **Great!** This is an improvement to our top-line model performance (LB 0.1538). But can we do better?

# # MinMax + BestBase Stacking

# In[ ]:


# load the model with best base performance
sub_base = pd.read_csv('../input/statoil-iceberg-submissions/sub_200_ens_densenet.csv')


# In[ ]:


concat_sub['is_iceberg_base'] = sub_base['is_iceberg']
concat_sub['is_iceberg'] = np.where(np.all(concat_sub.iloc[:,1:6] > cutoff_lo, axis=1), 
                                    concat_sub['is_iceberg_max'], 
                                    np.where(np.all(concat_sub.iloc[:,1:6] < cutoff_hi, axis=1),
                                             concat_sub['is_iceberg_min'], 
                                             concat_sub['is_iceberg_base']))
concat_sub[['id', 'is_iceberg']].to_csv('stack_minmax_bestbase.csv', 
                                        index=False, float_format='%.6f')


# **LB 0.1463** - **Yes!** This is a decent score given none of the models in our ensemble pipeline has achieved thus better. I am sure there are more twisted ways to boost the score further, so will keep updating or just leave to more Kagglers to discover!

# 
# ### P.S. As I wrote along this work, deeply I think, building strong & roboust model is always the key component, stacking only comes last with the promise to surprise, sometimes, in an unpleasant direction@ 
# 
# 
# 
