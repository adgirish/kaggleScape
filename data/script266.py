
# coding: utf-8

# In[ ]:


#import kaggle, os
#path = os.path.abspath('..') + '/input/'
#!kaggle config path -p "{path}"
#!kaggle competitions download -c jigsaw-toxic-comment-classification-challenge


# In[1]:


import pandas as pd
import numpy as np

sub1 = pd.read_csv('../input/blend-it-all/blend_it_all.csv')
sub2 = pd.read_csv('../input/toxic-avenger/submission.csv')

coly = [c for c in sub1.columns if c not in ['id','comment_text']]
sub2.columns = [x+'_' if x not in ['id'] else x for x in sub2.columns]
blend = pd.merge(sub1, sub2, how='left', on='id')
for c in coly:
    blend[c] = ((np.sqrt(blend[c] * blend[c+'_']) * 0.1) + ((((blend[c] * 4) + blend[c+'_'])/5) * 0.9))
    blend[c] = blend[c].clip(0+1e12, 1-1e12)
blend = blend[sub1.columns]
blend.to_csv('submission.csv', index=False)

