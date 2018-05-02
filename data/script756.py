
# coding: utf-8

# When Kobe Bryant declared his retirement, a Kaggler named Selfish Gene published this great script:
# https://www.kaggle.com/selfishgene/kobe-bryant-shot-selection/psychology-of-a-professional-athlete
# 
# His main story was as follows: Kobe's shot selection is influenced by his previous shot outcome, in a way that can be perceived as somewhat irrational: if he makes a shot, he tends to get further away from the rim in the next shot - possibly due to a confidence boost. If he misses, the opposite happens, and he gets closer to the hoop.  This can be explained by the "Hot Hand" theory - it is possible that his performance actually improves. it even sounds reasonable. however, Selfish Gene showed this is not the case - the hand stays cold (or normal body surface temperature at most), regardless of the previous shot outcome.
# 
# Inspired by this really great script, I was curios to see whether this effect can be seen in the general population of NBA Players.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('fivethirtyeight')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# import data
shots = pd.read_csv('../input/shot_logs.csv', header=0)
# Any results you write to the current directory are saved as output.


# In[ ]:


shots['previous'] = np.zeros(len(shots))
shots['dist_diff'] = np.zeros(len(shots))

for i,row in enumerate(shots[1:].iterrows()):
    if i>0:
        if shots.loc[i,'GAME_ID'] == shots.loc[i-1,'GAME_ID']:
            shots.loc[i,'previous'] = shots.loc[i-1,'SHOT_RESULT']
            shots.loc[i,'dist_diff'] = shots.loc[i,'SHOT_DIST'] - shots.loc[i-1,'SHOT_DIST']
            


# In[ ]:


after_made = shots[shots.previous == 'made']
after_miss = shots[shots.previous =='missed']

bins = np.arange(-30,30,0.5)
x = after_made.dist_diff
y = after_miss.dist_diff

h1 = np.histogram(after_made.dist_diff,bins)
h2 = np.histogram(after_miss.dist_diff,bins)
hist_1 = np.true_divide(h1[0],sum(h1[0]))
hist_2 = np.true_divide(h2[0],sum(h2[0]))
cumu_1 = []
cumu_1.append(0)
cumu_2 = []
cumu_2.append(0)

for i,item in enumerate(hist_1):
    if i>0:
        cumu_1.append(cumu_1[i-1] + hist_1[i])
        cumu_2.append(cumu_2[i-1] + hist_2[i])
        
        
plt.plot(bins[1:]*0.3,cumu_1)
plt.plot(bins[1:]*0.3,cumu_2)
plt.legend(['After made','After miss'], loc = 2)
plt.xlabel('Difference from previous shot [m]')
plt.ylabel('Cumulative Density Function')


#  Just like with Kobe, it seems that the average NBA Player gets closer to the basket after missing and the other way around after a successful attempt.
# 
# But again - can it simply be the actual effect of a hot hand? 

# In[ ]:


print('Success rate after a successful attempt...')
print(len(after_made[after_made.SHOT_RESULT == 'made'])/len(after_made))

print('Success rate after an unsuccessful attempt...')
print(len(after_miss[after_miss.SHOT_RESULT == 'made'])/len(after_miss))


# We can see that there is no evidence that players have a hot hand. It is true however that since after miss shots are on average more difficult (slightly further away on average), we need to control for shot difficulty before we can conclude that this is indeed pure irrationality

# Thanks to Selfish Gene I have some prior knowledge - making the last shot increases the chances of the following shot to be a 3 pointer. let's verify this theory (that was empirically proven to be correct in Kobe's case'):

# In[ ]:


print('% of 3 pointers out of all shots after a succesful attempt:')
print(len(after_made[after_made.PTS_TYPE == 3])/len(after_made))
print('% of 3 pointers out of all shots after an unsuccesful attempt:')
print(len(after_miss[after_miss.PTS_TYPE == 3])/len(after_miss))

print('% of "roughly in the paint shots" out of all shots after a succesful attempt:')
print(len(after_made[after_made.SHOT_DIST< 5])/len(after_made))
print('% of "roughly in the paint shots" out of all shots after an unsuccesful attempt:')
print(len(after_miss[after_miss.SHOT_DIST< 5])/len(after_miss))


# While the effect of increase in the number of 3-pointers after successful attempts doesn't seem significant, it does seem very obvious that after a failing attempt, the player is much more likely to get very close to the basket in his next attempt 
