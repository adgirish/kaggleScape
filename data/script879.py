
# coding: utf-8

# Almost everybody seems to drop features `'ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin'` in their kernels. 
# 
# At first, it seems logical to do this, as xgb.booster feature importances are low. But let's look, if we can find any structure in them.

# In[ ]:


# Load standard libraries

import numpy as np
RANDOM_SEED = 1337
np.random.seed(RANDOM_SEED)
import pandas as pd
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 70)

import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode()

get_ipython().run_line_magic('time', "df_train = pd.read_csv('../input/train.csv', na_values=-1)")

print('Loaded data and libraries.')


# I first assumed that these features are some kind of encoded (one-hot, binary, etc.). As I looked through the data, I see they are not one-hot, because there are more than one 1's in the same row.
# 
# Let's assume, they are binary encoded integers from a categorical value from 0 to 63 (as there are 6 columns, 2**6 = 64). Let's count some stats on these categories.

# In[ ]:


cols = ['ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin',
        'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin']

groupped = df_train.groupby(cols)['target'].agg(['sum', 'count', 'mean']).reset_index()
groupped


# If we plot count and sum columns, we can clearly see a pattern here.

# In[ ]:


py.iplot([go.Scatter(y=groupped['count'])])
py.iplot([go.Scatter(y=groupped['sum'])])
py.iplot([go.Scatter(y=groupped['mean'])])


# If we look at first two chart, there are jumps on multiples of 2 and 8. And, first half of the chart is on average greater than second half. The jumps seem to be related to our binary features.
# 
# We'll now sort these graphs as if they were smooth and try to find a pattern in permutation of binary representaton. First, calculate a category number from these bin features.

# In[ ]:


groupped['ps_calc_15_to_20']  = groupped['ps_calc_15_bin']*32 + groupped['ps_calc_16_bin']*16
groupped['ps_calc_15_to_20'] += groupped['ps_calc_17_bin']*8  + groupped['ps_calc_18_bin']*4
groupped['ps_calc_15_to_20'] += groupped['ps_calc_19_bin']*2  + groupped['ps_calc_20_bin']*1
groupped['ps_calc_15_to_20'].astype(np.uint8, inplace=True)
groupped


# In[ ]:


groupped = groupped.sort_values(by='count', ascending=False).reset_index()
groupped = groupped.drop('index', axis=1)
groupped


# Now we draw them on a chord chart. The symmetry in the graph is clearly visible.

# In[ ]:


# Draw Chord Graph
dots = groupped['ps_calc_15_to_20'].copy()
n = 64.0
# Dots start at 12 o'clock, and rotates clockwise.
x = np.sin((dots+0.5)/n*2*np.pi)
y = np.cos((dots+0.5)/n*2*np.pi)
edges = list(zip(x,y))
data = [go.Scattergl(
    x=x,
    y=y,
    text=['Dot %d'%c for c in dots],
    mode='lines+markers',
    marker={
        'color':'rgb(30,30,30)',
        'size':5
    },
    line={
        'color':'rgba(46, 147, 219, 0.5)',
        'width':5
    },
    hoverinfo='text'
)]
layout = go.Layout(autosize=False, width=500, height=500, showlegend=False)
py.iplot(go.Figure(data=data, layout=layout))


# Now let's stop here. I spent almost a day trying to figure out what this permutation might be. I've looked at cyclic groups, huffman codes, balanced gray codes, searched at OEIS etc. Then my brain stopped and I gave up! :)
# 
# It is what it is. Let's forget about the generation process and use it.
# 
# I noticed that the sum of the first and last indices after this sort operation is 63. This as a consequence of this symmetry situation. They may have first scrambled the 64 bin's indexes, then generated binary encoded features from that index. Whatever... Let's go on.

# In[ ]:


dots = pd.DataFrame(dots)
dots['ps_calc_15_to_20_reversed'] = dots['ps_calc_15_to_20'].iloc[::-1].values
dots['sum_of_them'] = dots['ps_calc_15_to_20'] + dots['ps_calc_15_to_20_reversed']
dots


# We see that there are some rows which sum is not equal to 63. As I said, they may have generated these by a rule (which I don't know yet) and we can fix our ordering according to this rule.

# In[ ]:


dots[dots['sum_of_them']!=63]


# If we switch rows 4-5 and 12-13, we'll have a nice pattern structure again.

# In[ ]:


new_index = list(range(4)) + [5,4] + list(range(6,12)) + [13,12] + list(range(14,64))
#new_index += [51,50] + list(range(52,58)) + [59,58] + list(range(60,64))
print(len(new_index))
np.array(new_index) # used np.array to display the list nicer.


# In[ ]:


dots2 = dots.copy()
dots2 = dots2.reindex(index=new_index).reset_index(drop=True)
dots2['ps_calc_15_to_20_reversed'] = dots2['ps_calc_15_to_20'].iloc[::-1].values
dots2['sum_of_them'] = dots2['ps_calc_15_to_20'] + dots2['ps_calc_15_to_20_reversed']

dots2


# In[ ]:


dots2[dots2['sum_of_them']!=63]


# Now we think we have a nice permutation of 0..63. Let's switch row 4-5 and 12-13 in first table and see if our graphs changed somehow.

# In[ ]:


groupped = groupped.reindex(index=new_index).reset_index(drop=True)
groupped


# In[ ]:


py.iplot([go.Scatter(y=groupped['count'])])
py.iplot([go.Scatter(y=groupped['sum'])])
py.iplot([go.Scatter(y=groupped['mean'])])


# That looks better and a little bit meaningfull now.
# 
# In summary, if you want to use this feature in your model, use:

# In[ ]:


# Columns -> binary decoded.

tmp  = df_train['ps_calc_15_bin'] * 32 + df_train['ps_calc_16_bin'] * 16 + df_train['ps_calc_17_bin'] * 8
tmp += df_train['ps_calc_18_bin'] * 4 + df_train['ps_calc_19_bin'] * 2 + df_train['ps_calc_20_bin'] * 1

tmp2 = [5, 22, 9, 32, 13, 38, 20, 47, 2, 19, 8, 30, 10, 35, 17, 45, 1,
        15, 4, 24, 7, 29, 14, 40, 0, 12, 3, 21, 6, 26, 11, 36, 27, 52,
        37, 57, 42, 60, 51, 63, 23, 49, 34, 56, 39, 59, 48, 62, 18, 46,
        28, 53, 33, 55, 44, 61, 16, 43, 25, 50, 31, 54, 41, 58]
tmp2 = pd.Series(tmp2)

df_train['ps_calc_15_16_17_18_19_20'] = tmp.map(tmp2)

# You may now drop the others peacefully.
#df_train.drop(['ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin',
#               'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin'], axis=1, inplace=True)


# Or use this, if you want to overfit a little bit :)

# Please, feel free to comment below. Especially, if you have any idea about the permutation method they have used, it would be appreciated.
# 
# And, please comment how much improvement did it make in your model.
# 
# Thank you!
