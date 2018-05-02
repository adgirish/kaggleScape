
# coding: utf-8

# In[ ]:


#Ignore the seaborn warnings.
import warnings
warnings.filterwarnings("ignore");

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats


# In[ ]:


#Import data and see what states it has values for so far.
df = pd.read_csv('../input/primary_results.csv')
df.state.unique()


# In[ ]:


#Create a new dataframe that holds votes by state and the fraction of total votes(democrat + republican) that a candidate recieved and pare them down to only those that are still in the race as of 2 March.

votesByState = [[candidate, state] for candidate in df.candidate.unique() for state in df.state.unique()]
for i in votesByState:
    i.append(df[df.candidate == i[0]].party.unique()[0])
    i.append(df[(df.candidate == i[0]) & (df.state == i[1])].votes.sum())
    i.append(i[3]/df[(df.party == i[2]) & (df.state == i[1])].votes.sum())
    i.append(i[3]/df[df.state == i[1]].votes.sum())
vbs = pd.DataFrame(votesByState, columns = ['candidate', 'state', 'party', 'votes', 'partyFrac', 'totalFrac'])
vbs = vbs[vbs.candidate != ' Uncommitted']
vbs['state'] = vbs['state'].astype('category')
vbs = vbs[vbs.candidate.isin(['Hillary Clinton', 'Bernie Sanders', 'Donald Trump', 'Ben Carson', 'John Kasich', 'Ted Cruz', 'Marco Rubio'])]

#Add in a column with the order the primaries took place to easier visualize data.
count = 1
vbs['stateOrder'] = 0
for i in vbs.state.unique():
    vbs['stateOrder'][vbs.state == i] = count 
    count += 1


# In[ ]:


#Create a pair-wise list of candidates
canPairs = []
canPairsraw = [[i,j] for i in vbs.candidate.unique() for j in vbs.candidate.unique() if i != j]
for i in canPairsraw:
    if list(reversed(i)) in canPairs:
        continue
    canPairs.append(i)


# In[ ]:


#Calculate the Pearson correlation between each pair and finding the max and min.
corrVals = []
for i in canPairs:
    corrVals.append(['{} vs. {}'.format(i[0], i[1]), (scipy.stats.pearsonr(vbs[vbs.candidate == i[0]].totalFrac, vbs[vbs.candidate == i[1]].totalFrac)[0])**2])

max = ['',.5]
min = ['',.5]
for i in corrVals:
    if max[1] < i[1]:
        max = i
    if min[1] > i[1]:
        min = i

print('Max correlation: {}\nMin correlation: {}'.format(max, min))
#The higher the correlation, the less effect the candidate has on the other, the lower the correlation the more of an effect.  So, states that voted more for Trump, voted less for Carson.

#I made a mistake and should have been looking at r^2.  The closer to 0 the lower the effect, so that's fixed now.


# In[ ]:


g = sns.regplot(x='stateOrder', y = 'totalFrac', data = vbs[(vbs.candidate == 'Ted Cruz') | (vbs.candidate == 'Marco Rubio')])


# In[ ]:


g = sns.regplot(x='stateOrder', y = 'totalFrac', data = vbs[(vbs.candidate == 'Donald Trump') | (vbs.candidate == 'Ben Carson')])

