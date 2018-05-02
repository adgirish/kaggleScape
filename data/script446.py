
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

# Any results you write to the current directory are saved as output.


# JSab and Michael Campos provided very detailed analysis on the time data.
# 
# Here I would like to share my method, and I think it confirms their conclusion that the unit of the time is minute in this dataset.
# 
# The method I used to figure out the time definition is through Fourier transform.

# In[ ]:


# Compute the histogram of the event time
time = df_train['time']
hist = hist = np.histogram(time,5000)

# To know the unit definition of Time
# we can look into the frequency structure of the histogram
hist_fft = np.absolute(np.fft.fft(hist[0]))
plt.plot(hist_fft)
plt.xlim([0,2500])
plt.ylim([0,1e6])
plt.title('FFT of event time histogram')
plt.xlabel('1/T')
plt.grid(True)
plt.show()


# The peaks in the FFT curve indicate strong periodic structure at that frequency.
# Let's zoom-in the see the numbers.

# In[ ]:


plt.plot(hist_fft)
plt.xlim([0,250])
plt.ylim([0,1e6])
plt.title('FFT of event time histogram')
plt.xlabel('1/T')
plt.grid(True)
plt.show()


# The first peak (fundamental frequency) is at 78
# which means the time histogram has a period of:

# In[ ]:


print(time.max()/78)


# The overall event histogram averages over all place_id, so I look at one of the most popular place_id to find more information.

# In[ ]:


time = df_train[df_train['place_id']==8772469670]['time']
hist = np.histogram(time,5000)
hist_fft = np.absolute(np.fft.fft(hist[0]))

plt.plot(hist_fft)
plt.xlim([0,2500])
plt.title('FFT of event time histogram')
plt.xlabel('1/T')
plt.grid(True)
plt.show()


# Small peak at 64 and large peak at 451, they conrespond to periods:

# In[ ]:


T1 = time.max()/64
T2 = time.max()/451
print('period T1:', T1)
print('period T2:', T2)


# T1 is the same period as the one found in overall histogram.
# 
# But the interesting part is that the ratio of T1 and T2 happen to be about 7, so they are very likely week and day.
# 
# And 1440 is the minute time for a day.

# In[ ]:


# Another place_id for confirmation
time = df_train[df_train['place_id']==4823777529]['time']
hist = np.histogram(time,5000)
hist_fft = np.absolute(np.fft.fft(hist[0]))

plt.plot(hist_fft)
plt.xlim([0,2500])
plt.title('FFT of event time histogram')
plt.xlabel('1/T')
plt.grid(True)
plt.show()


# In[ ]:


# peaks at 77 and 539, same periods: 10080 and 1440
T1 = time.max()/77
T2 = time.max()/539
print('period T1:', T1)
print('period T2:', T2)

