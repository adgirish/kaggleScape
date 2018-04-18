
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

import ml_metrics as metrics



# ##I am going to show how order matters in MAP@K when there is only 1 answer. 
# 
# ##This experiment is done by calculating AP@K, which gives 1 value. MAP@K is the average of AP@K. 
# 

# In[ ]:


actual = [1]

predicted = [1,2,3,4,5]

print('Answer=',actual,'predicted=',predicted)
print('AP@5 =',metrics.apk(actual,predicted,5) )

predicted = [2,1,3,4,5]
print('Answer=',actual,'predicted=',predicted)
print('AP@5 =',metrics.apk(actual,predicted,5) )

predicted = [3,2,1,4,5]
print('Answer=',actual,'predicted=',predicted)
print('AP@5 =',metrics.apk(actual,predicted,5) )

predicted = [4,2,3,1,5]
print('Answer=',actual,'predicted=',predicted)
print('AP@5 =',metrics.apk(actual,predicted,5) )

predicted = [4,2,3,5,1]
print('Answer=',actual,'predicted=',predicted)
print('AP@5 =',metrics.apk(actual,predicted,5) )


# As you see from the above example, the "earlier" you predict the correct answer 1, the higher your score. 
# 
# ### Next is an example of how MAP@K is calculated with a list of hotels
# I took the list of predictions from the previous AP@K example, and calculated the MAP@K. You can see the MAP@K is the average of AP@K
# 

# In[ ]:


metrics.mapk([[1],[1],[1],[1],[1]],[[1,2,3,4,5],[2,1,3,4,5],[3,2,1,4,5],[4,2,3,1,5],[4,2,3,5,1]], 5)

