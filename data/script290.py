
# coding: utf-8

# ## Duplicates of Duplicates
# In this notebook, we will take a look at connected groups of questions that are marked as duplicates of each other.
# What we will find is that, at least in the training set, duplicates tend to occur in clusters of related questions.

# In[ ]:


import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
import warnings

TRAIN_PATH = "../input/train.csv"
tr = pd.read_csv(TRAIN_PATH)
pos = tr[tr.is_duplicate==1]


# We'll graph questions as nodes and place an edge between questions that are marked as duplicates. The connected components in this graph are the duplicates-of-duplicates.

# In[ ]:


g = nx.Graph()
g.add_nodes_from(pos.question1)
g.add_nodes_from(pos.question2)
edges = list(pos[['question1', 'question2']].to_records(index=False))
g.add_edges_from(edges)


# The number of nodes in this graph is equal to the number of unique questions in positive rows.

# In[ ]:


len(set(pos.question1) | set(pos.question2)), g.number_of_nodes()


# The number of edges in g is equal to the number of positive rows.

# In[ ]:


len(pos), g.number_of_edges()


# The mean degree is about 2. This means that for questions that ever occur as duplicates, the average number of times that they occur as duplicates is 2.

# In[ ]:


d = g.degree()
np.mean([d[k] for k in d])


# Here are a few medium-sized connected components:

# In[ ]:


cc = filter(lambda x : (len(x) > 3) and (len(x) < 10), 
            nx.connected_component_subgraphs(g))
g1 = next(cc)
g1.nodes()


# In[ ]:


# with block handles a deprecation warning that occurs inside nx.draw_networkx
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    nx.draw_circular(g1, with_labels=True, alpha=0.5, font_size=8)
    plt.show()


# In[ ]:


g1 = next(cc)
g1.nodes()


# In[ ]:


with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    nx.draw_circular(g1, with_labels=True, alpha=0.5, font_size=8)
    plt.show()


# In[ ]:


g1 = next(cc)
g1.nodes()


# In[ ]:


with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    nx.draw_circular(g1, with_labels=True, alpha=0.5, font_size=8)
    plt.show()


# So it looks like duplicates often occur in groups. The groups are not complete subgraphs, but they are pretty densely connected. Some of them are pretty big. Let's look at their sizes.

# In[ ]:


cc = nx.connected_component_subgraphs(g)
node_cts = list(sub.number_of_nodes() for sub in cc)
cc = nx.connected_component_subgraphs(g)
edge_cts = list(sub.number_of_edges() for sub in cc)
cts = pd.DataFrame({'nodes': node_cts, 'edges': edge_cts})
cts['mean_deg'] = 2 * cts.edges / cts.nodes
cts.nodes.clip_upper(10).value_counts().sort_index()


# In[ ]:


Most of the components have just 2 questions, the minimum possible. But there are also several thousand larger components.


# In[ ]:


cts.plot.scatter('nodes', 'edges')
plt.show()


# The largest components have over 100 nodes and 1000 edges. That means that the largest clusters of duplicates of duplicates occur on around 1% of the positive rows in train.

# In[ ]:


cts.plot.scatter('nodes', 'mean_deg')
plt.show()


# The visible straight diagonal edge on the top of the data are the components that are nearly complete subgraphs. 

# ### Conclusion (Work in Progress)
# 
# So at this point, I'm not quite sure what to make of this, but it seems pretty important. Recall that the nodes map to unique questions and the edges map to positive rows. There are about 149,000 positive rows in train and almost half of them involve questions that are in connected groups of 5 or more related (duplicate-of-duplicate) questions.

# In[ ]:


cts[cts.nodes >= 5].edges.sum()


# I may add more here later, but in the mean time, please add your thoughts in the comments.
