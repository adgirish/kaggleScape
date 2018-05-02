
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import io
import bson
import matplotlib.pyplot as plt
from skimage.data import imread
from tqdm import tqdm_notebook


# In[ ]:


categories = pd.read_csv('../input/category_names.csv', index_col='category_id')


# In[ ]:


prod_id = []
prod_category = []
prod_num_imgs = []

num_dicts = 7069896 # according to data page

# This will take about 02m15s to complete
with open('../input/train.bson', 'rb') as f, tqdm_notebook(total=num_dicts) as bar:
        
    data = bson.decode_file_iter(f)

    for c, d in enumerate(data):
        bar.update()
        prod_id.append(d['_id'])
        prod_category.append(d['category_id'])
        prod_num_imgs.append(len(d['imgs']))


# Create the dataframe

# In[ ]:


df_dict = {
    'category': prod_category,
    'num_imgs': prod_num_imgs
}
df = pd.DataFrame(df_dict, index=prod_id)
del df_dict # Free memory


# ### Number or images

# In[ ]:


df.num_imgs.value_counts().plot(kind='bar');


# ### Most common categories
# 
# Mos common on all lavels:

# In[ ]:


df.category.value_counts().to_frame().head(25).join(categories)


# Most common level 1

# In[ ]:


df.category.value_counts().to_frame().join(categories)     .groupby('category_level1')['category'].sum()     .sort_values(ascending=False).head(10).to_frame().reset_index()


# Most common level 2

# In[ ]:


df.category.value_counts().to_frame().join(categories)     .groupby(['category_level1', 'category_level2'])['category'].sum()     .sort_values(ascending=False).head(15).to_frame().reset_index()


# ### Least common categories

# In[ ]:


df.category.value_counts().to_frame().tail(15).join(categories)


# ### Is there any relation between id and category?

# In[ ]:


df_cat = df.category.sample(100000, random_state=42).reset_index()

fig, ax = plt.subplots(1,1, figsize=(10, 10))
ax.set_xlabel('index')
ax.set_ylabel('category')
ax.scatter(df_cat.index.values, df_cat.category.values, alpha=0.02);


# The category seens to be very unrelated to the identifier, which are good news (no leak here).

# ### Number of categories and accuracy
# 
# We may try to reduce the 5270 categories in order to ease the train. But what's the cost of accuracy for this approach? The chart bellow show this relation:

# In[ ]:


cum_sum = df.category.value_counts().cumsum().reset_index(drop=True)
cum_sum /= cum_sum.max()
ax = cum_sum.plot(figsize=(12, 8))
ax.grid()
ax.set_xlabel('Num. of Categories')
ax.set_ylabel('Max. Accuracy');


# And here is a table of some values.

# In[ ]:


max_accuracies = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0]
num_cat = map(lambda a: (cum_sum.values <= a).sum(), max_accuracies)

pd.DataFrame({'Num. Categories': list(num_cat), 'Max. Accuracy': max_accuracies})


# ### Most common category submission
# 
# This will score 0.01121 on LB

# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


submission['category_id'] = 1000018296

submission.to_csv('most_common_benchmark.csv.gz', compression='gzip', index=False)

