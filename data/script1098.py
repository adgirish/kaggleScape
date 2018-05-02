
# coding: utf-8

# # Non-Negative Matrix Factorization 

# I have been using matrix factorization and did not see a kernel running it, thought adding this would provide some interesting ideas. This kernel joins the data provided to get the order history of different users, creates an order counts matrix from that, and factors it into two lower dimension matrices. The new matrix could be used as additional feature columns and potentially improve LB scores. 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from collections import Counter


# In[2]:


order_prior = pd.read_csv("../input/order_products__prior.csv")
orders = pd.read_csv("../input/orders.csv")
train = pd.read_csv("../input/order_products__train.csv")
products = pd.read_csv("../input/products.csv")


# In[3]:


test = orders[orders['eval_set']=='test']
prior = orders[orders['eval_set']=='prior']
prior.tail()


# Match user_id of the prior part of the same orders table. That will give you the order history by user.

# In[4]:


test = pd.merge(test, prior[['user_id', 'order_id', 'order_number']], how = 'left', on = 'user_id')
print(test.shape)
test.head()


# Merging with order history from order_prior table, ran on 16gb memory, but timed out here on 8gb so I am running a subset of the data below.

# In[5]:


test = pd.merge(test, order_prior, left_on = 'order_id_y', right_on = 'order_id')
test['new_order_id'] = test['order_id_x']
test['prior_order_id'] = test['order_id_y']
test = test.drop(['order_id_x', 'order_id_y'], axis = 1)
del [orders, order_prior, train]


# In[6]:


lookup_aisles = test.product_id.map(products.set_index('product_id')['department_id'])
test['aisles'] = lookup_aisles


# In[7]:


test.head()


# Create the product list for each order with filtering in pandas. product_list represents reorders while all represents all items from an order.
# 

# In[8]:


product_list = test[test['reordered']==1].groupby(['user_id', 'order_number_x', 'new_order_id']).agg({'product_id': lambda x: tuple(x),
                                                                                                      'aisles': lambda x: tuple(x)})
product_list = pd.DataFrame(product_list.reset_index())
product_list['num_products_reordered'] = product_list.product_id.apply(len)
product_list.head(15)


#  The output below shows what one user's order history looks like.

# In[9]:


test[(test['user_id'] == 4)]


# To create the users x products count table loop through the prodcut ids data to as a sparse matrix (much more memory efficient), column position contains the product ids with position listed in a dict.
# 

# In[10]:


indptr = [0]
indices = []
data = []
column_position = {}
# input must be a list of lists
for order in product_list['product_id']:
    for product in order:
        index = column_position.setdefault(product, len(column_position))
        indices.append(index)
        data.append(1)
    indptr.append(len(indices))
    
prod_matrix = csr_matrix((data, indices, indptr), dtype=int)
#del(test)


# In[11]:


prod_matrix.shape


# The problem with using the sparse matrix is that it is gigantic,  there are a lot of users and a lot of products. Non-negative matrix decomposition is implemented in sklearn, use that to decompose the count matrix into two new matrices with considerably reduced dimensions. 

# In[12]:


from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize

nmf = NMF(n_components = 50, random_state = 42)
model = nmf.fit(prod_matrix)
H = model.components_
model.components_.shape


# What can I do with the results? For one, I could pick a random user and find users that have similar purchasing behavior. 

# In[13]:


W = model.transform(prod_matrix)
user_data = pd.DataFrame(normalize(W), index = product_list['user_id'])
idx = user_data.dot(user_data.iloc[0]).nlargest(5).index
user_data.dot(user_data.iloc[0]).nlargest(5)


# In[14]:


W.shape


# In[15]:


H.shape


# In[16]:


#tmp = np.dot(W[25], H).argmax()

def return_item(index):
    for key, value in column_position.items():
        if value == index:
            return(key)


# In[17]:


# is there a better way to return top values?
top_10 = pd.Series(np.matmul(W[35], H)).sort_values(ascending=False)[0:10]
top_10


# In[18]:


[return_item(idx) for idx in top_10.index]


# In[19]:


def prod_count(product_ids):

    prod_count = {}
    for item in product_ids:
        if item not in prod_count.keys():
            prod_count[item] = 1
        elif item in prod_count.keys():
            prod_count[item] += 1
    return prod_count


# In[20]:


prod_count(product_list.product_id[35])


# In[34]:


prod_matrix.shape


# In[119]:


np.sum(prod_matrix.sum(axis=0)[0,:] > 3286)


# In[ ]:


import seaborn as sns
sns.distplot(prod_matrix.sum(axis=0)[0,:], bins = 30000, kde=False)
plt.xlim(0, 100)

plt.show()


# In[92]:


prod_matrix.sum(axis=0)[:,580]


# In[39]:


column_position[21386]


# In[21]:


similar_users = product_list[product_list.user_id.isin(idx)]
similar_users


# In[22]:


overlap = set(similar_users.product_id.iloc[0]) & set(similar_users.product_id.iloc[2])
overlap


# In[23]:


counts = similar_users.product_id.apply(prod_count)


# Here are there order counts:

# In[24]:


def id_values(row, overlap):
    for key, value in row.items():
        if key in overlap:
            print(key, value)


# In[25]:


id_values(counts.iloc[0], overlap)


# In[26]:


id_values(counts.iloc[2], overlap)


# Another benefit is that the W matrix contains latent information regarding the amount of each item a user has ordered in the past, but has vastly reduced dimensions. In other words, it doesn't keep each product as a column of the original user x product count matrix. I have yet to test this in creating predictions, but it should be a helpful way to keep track of user order counts per product. 

# In[27]:


df = pd.concat([product_list.user_id, pd.DataFrame(W)], axis = 1)
df = pd.merge(test[0:10000], df, how = 'left').drop('eval_set', axis = 1)


# Using multi-nomial logistic regression with aisles instead of products to see if the W matrix columns have an impact on purchasing a product from an aisle, many of the p-values are significant at a 1% level. I ran the function on aisles because it is a way to group products together reducing the number of classes to compare against while having a clear relationship to products being purchased. 

# In[28]:


df.iloc[1:10, 12]


# In[29]:


x = df.iloc[:, 12:]
y = df.reordered.values
#del test, df


# In[30]:


from statsmodels import api 
x = api.add_constant(x, prepend = False)
mn_log = api.MNLogit(y, x)
model = mn_log.fit()


# In[31]:


model.summary()

