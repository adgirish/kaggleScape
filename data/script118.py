
# coding: utf-8

# Wei Wei has  [published a code][1] to normalize predictions distributions to match the estimated distribution of the test set used for public LB.  Here is a faster and more accurate code to do it.
# 
# It yields modest gains on my models, but it may make a difference on yours.  
# 
# 
#   [1]: https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/discussion/31067

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


interest_levels = ['low', 'medium', 'high']

tau_train = {
    'low': 0.694683, 
    'medium': 0.227529,
    'high': 0.077788, 
}

tau_test = {
    'low': 0.69195995, 
    'medium': 0.23108864,
    'high': 0.07695141, 
}

def correct(df, train=True, verbose=False):
    if train:
        tau = tau_train
    else:
        tau = tau_test
        
    df_sum = df[interest_levels].sum(axis=1)
    df_correct = df[interest_levels].copy()
    
    if verbose:
        y = df_correct.mean()
        a = [tau[k] / y[k]  for k in interest_levels]
        print( a)
    
    for c in interest_levels:
        df_correct[c] /= df_sum

    for i in range(20):
        for c in interest_levels:
            df_correct[c] *= tau[c] / df_correct[c].mean()

        df_sum = df_correct.sum(axis=1)

        for c in interest_levels:
            df_correct[c] /= df_sum
    
    if verbose:
        y = df_correct.mean()
        a = [tau[k] / y[k]  for k in interest_levels]
        print( a)

    return df_correct

