
# coding: utf-8

# ## Introduction
# 
# We will show how to use the  [Hungarian method](https://en.wikipedia.org/wiki/Hungarian_algorithm) to improve an existing solution. Let's forget the twins for a moment and look at the vast majority (99.6%) of the rest. Each kid should receive one gift so it is clearly an assigment problem and the objective function is linear.
# It is not even an NP hard problem we could use polinomial algorithms to solve it.
# 
# The only problem is that we have 10^6 points so an O(n^3) algorithm might not be feasible. Scipy.optimize has [linear_sum_assignment](https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html) let's use that on small random subsets of the large input space.
# 
# We will use [Vlad Golubev](https://www.kaggle.com/golubev)'s [submission](https://www.kaggle.com/golubev/c-submission) as the baseline submission to improve. You could start with a better solution with [Selfish Gene](https://www.kaggle.com/selfishgene)'s nice [heuristics](https://www.kaggle.com/selfishgene/nothing-fancy-just-some-heuristics-0-9372).

# In[ ]:


import os
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
import datetime as dt
from collections import defaultdict, Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime as dt
plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['font.size'] = 16
import seaborn as sns


# In[ ]:


N_CHILDREN = 1000000
N_GIFT_TYPE = 1000
N_GIFT_QUANTITY = 1000
N_GIFT_PREF = 1000
N_CHILD_PREF = 10
TWINS = int(0.004 * N_CHILDREN)

CHILD_PREF = pd.read_csv('../input/santa-gift-matching/child_wishlist.csv', header=None).drop(0, 1).values
GIFT_PREF = pd.read_csv('../input/santa-gift-matching/gift_goodkids.csv', header=None).drop(0, 1).values


# In[ ]:


GIFT_HAPPINESS = {}
for g in range(N_GIFT_TYPE):
    GIFT_HAPPINESS[g] = defaultdict(lambda: 1. / (2 * N_GIFT_PREF))
    for i, c in enumerate(GIFT_PREF[g]):
        GIFT_HAPPINESS[g][c] = -1. * (N_GIFT_PREF - i) / N_GIFT_PREF

CHILD_HAPPINESS = {}
for c in range(N_CHILDREN):
    CHILD_HAPPINESS[c] = defaultdict(lambda: 1. / (2 * N_CHILD_PREF))
    for i, g in enumerate(CHILD_PREF[c]):
        CHILD_HAPPINESS[c][g] = -1. * (N_CHILD_PREF - i) / N_CHILD_PREF

GIFT_IDS = np.array([[g] * N_GIFT_QUANTITY for g in range(N_GIFT_TYPE)]).flatten()


# In[ ]:


def my_avg_normalized_happiness(pred):
    total_child_happiness = 0
    total_gift_happiness = np.zeros(1000)
    for c, g in pred:
        total_child_happiness +=  -CHILD_HAPPINESS[c][g]
        total_gift_happiness[g] += -GIFT_HAPPINESS[g][c]
    nch = total_child_happiness / N_CHILDREN
    ngh = np.mean(total_gift_happiness) / 1000
    print('normalized child happiness', nch)
    print('normalized gift happiness', ngh)
    return nch + ngh


# In[ ]:


def optimize_block(child_block, current_gift_ids):
    gift_block = current_gift_ids[child_block]
    C = np.zeros((BLOCK_SIZE, BLOCK_SIZE))
    for i in range(BLOCK_SIZE):
        c = child_block[i]
        for j in range(BLOCK_SIZE):
            g = GIFT_IDS[gift_block[j]]
            C[i, j] = CHILD_HAPPINESS[c][g] + GIFT_HAPPINESS[g][c]
    row_ind, col_ind = linear_sum_assignment(C)
    return (child_block[row_ind], gift_block[col_ind])


# In[ ]:


BLOCK_SIZE = 400
INITIAL_SUBMISSION = '../input/c-submission/cpp_sub.csv'
N_BLOCKS = (N_CHILDREN - TWINS) / BLOCK_SIZE
print('Block size: {}, n_blocks {}'.format(BLOCK_SIZE, N_BLOCKS))


# In[ ]:


subm = pd.read_csv(INITIAL_SUBMISSION)
initial_anh = my_avg_normalized_happiness(subm[['ChildId', 'GiftId']].values.tolist())
print(initial_anh)
subm['gift_rank'] = subm.groupby('GiftId').rank() - 1
subm['gift_id'] = subm['GiftId'] * 1000 + subm['gift_rank']
subm['gift_id'] = subm['gift_id'].astype(np.int64)
current_gift_ids = subm['gift_id'].values


# ## Single iteration

# In[ ]:


start_time = dt.datetime.now()
for i in range(1):
    child_blocks = np.split(np.random.permutation(range(TWINS, N_CHILDREN)), N_BLOCKS)
    for child_block in tqdm(child_blocks[:500]):
        cids, gids = optimize_block(child_block, current_gift_ids=current_gift_ids)
        current_gift_ids[cids] = gids
    subm['GiftId'] = GIFT_IDS[current_gift_ids]
    anh = my_avg_normalized_happiness(subm[['ChildId', 'GiftId']].values.tolist())
    end_time = dt.datetime.now()
    print(i, anh, (end_time-start_time).total_seconds())
subm[['ChildId', 'GiftId']].to_csv('./submission_%i.csv' % int(anh * 10 ** 6), index=False)


# In[ ]:


print('Improvement {}'.format(anh - initial_anh))


# In[ ]:


'{:.1f} hours required to reach the top.'.format(((0.94253901 - 0.93421513) / (0.93421513 - initial_anh)) * 8)


# My current best submission used this method. It improved 0.0005 in 8 hours :) With the same speed I would  reach the top in a week :) Of course the improvement won't be linear and the top teams could improve as well...

# ## Happiness distribution

# In[ ]:


child_happiness = np.zeros(N_CHILDREN)
gift_happiness = np.zeros(N_CHILDREN)
for (c, g) in subm[['ChildId', 'GiftId']].values.tolist():
    child_happiness[c] += -CHILD_HAPPINESS[c][g]
    gift_happiness[c] += -GIFT_HAPPINESS[g][c]


# In[ ]:


plt.hist(gift_happiness, bins=20, color='r', normed=True, alpha=0.5, label='Santa happiness')
plt.hist(child_happiness, bins=20, color='g', normed=True, alpha=0.5, label='Child happiness')
plt.legend(loc=0)
plt.grid()
plt.xlabel('Happiness')
plt.title('The children will be happier than Santa!')
plt.show();


# ## Time complexity

# In[ ]:


result = []
for n in np.arange(100, 1600, 100):
    C = np.random.random((n, n))
    st = dt.datetime.now()
    linear_sum_assignment(C)
    et = dt.datetime.now()
    result.append([n, (et - st).total_seconds()])


# In[ ]:


result = np.array(result)
poly_estimate = np.polyfit(result[:, 0], result[:, 1], 3)


# In[ ]:


plt.scatter(result[:, 0], result[:, 1], c='y', s=500, marker='*', label='Run time')
plt.plot(result[:, 0], np.poly1d(poly_estimate)(result[:, 0]), c='g', lw=3, label='Polynomial Estimate')
plt.xlabel('Number of vertices')
plt.ylabel('Run time (s)')
plt.grid()
plt.title('Hungarian method - O(n^3) time complexity')
plt.legend(loc=0)
plt.show()

