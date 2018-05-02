
# coding: utf-8

# # INTRODUCTION
# * We want to multiply two matrices. Both are 10 million dimensional matrices.
# * The first way is **for loop** that is non-vectorized implementation.
# * The second way is **np.dot()** that is vectorized implementation.
# * The expected result is vectorized implementation becomes faster than non-vectorized implementation.
# * Why time is important? Because,in todays world data can be very huge and implementation time very important. Nobody wants to wait too much time.
# * What is too much time? While implementation process can take days in deep learning.
# * Lets try and comprare two implementations
# 

# In[ ]:


import numpy as np # linear algebra
import time
# Two matrices with 10 million dimensions
a = np.random.rand(10000000)
b = np.random.rand(10000000)
# Vectorized
tic = time.time()
c = np.dot(a,b)
toc = time.time()
print('Vectorized version calculation result: ',c)
print('Vectorized version process time: '+ str((toc-tic)*1000)+ ' ms')
# Non-vectorized
c = 0
tic = time.time()
for i in range(len(a)):
    c += a[i]*b[i]
toc = time.time()
print('Non-vectorized version calculation result: ',c)
print('Non-vectorized version process time: '+ str((toc-tic)*1000)+ ' ms')


# # Conclusion
# * When I use vectorized version, time was almost 5 ms
# * When I use non-vectorized version time was almost 5 s
# * Non-vectorized version is 1000 times longer than vectorized version.
# * Can I wait 5 s instead of 5 ms ? Yes, I can but  in deep learning, if the non vectorized version takes 5 hours, I prefer using vectorized version that takes 1 hour.
# * Another vectorized functions: np.exp(), np.log(), np.abs(), np.max() 
# * Avoid using explicit for loops.
# 
