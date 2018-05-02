
# coding: utf-8

# Hi, Kagglers!
# <br>This is a quick Python-based version of Markov Chain Transitional Probabilities Calculation
# <br>(based on <a href="https://www.kaggle.com/mmotoki/markov-chain-tutorial">This R Kernel by Matt Motoki</a>)
# <br>See read it for the description / visualization and intuition behind this approach.
# 
# <br>This domain, as well as R language, are new to me, thus, the code might contain some technical/translational errors :)
# <br>This code doesn't implement multithreading, however, can be parallelized in a similar way to <a href="https://www.kaggle.com/mmueller/order-streaks-feature">This Script by Faron</a>
# <br>**Comments/Suggestions/Error Handlings are highly appreciated!**

# ## Library / Data Import

# In[ ]:


# libraries
import numpy as np
import pandas as pd
import gc

from pandas import isnull


# In[ ]:


# load data
input_dir = '../input/'

# prior orders
priors = pd.read_csv(input_dir + 'order_products__prior.csv',
                    dtype={
                        'order_id': np.int32,
                        'add_to_cart_order': np.uint8,
                        'reordered': np.int8,
                        'product_id': np.uint16
                    })

# order details
orders = pd.read_csv(input_dir + 'orders.csv',
                     dtype={
                         'order_dow': np.uint8,
                         'order_hour_of_day': np.uint8,
                         'order_number': np.uint8,
                         'order_id': np.uint32,
                         'user_id': np.uint32,
                         'days_since_prior_order': np.float16
                     })

# products, need just ids
products = pd.read_csv(input_dir + 'products.csv', 
                       dtype={
                           'aisle_id': np.uint8,
                           'department_id': np.uint8,
                           'product_id': np.uint16
                       }).drop(['product_name', 'aisle_id', 'department_id'], axis=1)

# set index
orders.set_index('order_id', drop=False, inplace=True)

# Append previous order info
orders['prev_order_id'] = orders.sort_values(['user_id', 'order_number']).groupby('user_id')['order_id'].shift().fillna(0).astype(np.uint32)


# ## Get Product Lists for Current / Previous Order

# In[ ]:


# get product list for all orders, except test :)
orders['prod_list'] = priors.groupby('order_id').aggregate(
    {'product_id':lambda x: set(x)})


# In[ ]:


# filter representative orders (those, whose previous order exists)
ords = orders[(orders.order_number > 1) & (orders.eval_set == 'prior')]

# to speed-up kernel execution - use only small subset of data
# comment it to get full results !!!
ords = orders[(orders.order_number > 1) & (orders.eval_set == 'prior')][:5000]


# In[ ]:


# get product list for all orders, except test :)
ords['prod_list'] = ords.order_id.map(orders.prod_list)


# In[ ]:


# get previous order's product list
ords['prev_prod_list'] = ords.prev_order_id.map(orders.prod_list)


# In[ ]:


# fill N/A values: na -> empty set
ords.loc[:, ['prod_list', 
               'prev_prod_list']] \
= ords.loc[:, ['prod_list', 
               'prev_prod_list']].applymap(lambda x: set() if isnull(x) else x)


# In[ ]:


# T11, products, bought in previous order and reordered in current
ords['T11'] = ords.apply(lambda r: r['prod_list'] & r['prev_prod_list'], axis=1)

# T01, products, ordered in current order and not ordered in previous
ords['T01'] = ords.apply(lambda r: r['prod_list'] - r['prev_prod_list'], axis=1)

# T10, products, ordered in previous order and not ordered in current
ords['T10'] = ords.apply(lambda r: r['prev_prod_list'] - r['prod_list'], axis=1)


# ## Transitions out of State 1

# In[ ]:


# product count -> # of bins needed
n_products = len(products)

# denominator
N = len(ords)

# N1 ----------------------------
# flatten list of sets
f = [val for sublist in [list(i) for i in ords.prev_prod_list.values] for val in sublist]
N1 = np.bincount(f, minlength=n_products+1)

# N11 ----------------------------
# flatten list of sets
f = [val for sublist in [list(i) for i in ords.T11.values] for val in sublist]
N11 = np.bincount(f, minlength=n_products+1)

# N10 ----------------------------
# flatten list of sets
f = [val for sublist in [list(i) for i in ords.T10.values] for val in sublist]
N10 = np.bincount(f, minlength=n_products+1)

del f
gc.collect()


# ## Transitions out of State 0

# In[ ]:


# N0 ----------------------------
N0 = N - N1

# N01 ----------------------------
# flatten list of sets
f = [val for sublist in [list(i) for i in ords.T01.values] for val in sublist]
N01 = np.bincount(f, minlength=n_products+1)

# N00 ----------------------------
N00 = N0 - N01

del f
gc.collect()


# ## Make DataFrame

# In[ ]:


product_probs = pd.DataFrame(
    data={
        'product_id': np.array(range(0, n_products+1)),
        'P0':   (N0+1) / (N+2),
        'P00': (N00+1) / (N0+2),
        'P01': (N01+1) / (N0+2),
        
        'P1':  (N1+1)  / (N+2), 
        'P10': (N10+1) / (N1+2),
        'P11': (N11+1) / (N1+2)
    }
)

# delete 1st row (for product_id) if you don't use None as a product, leave as is otherwise
product_probs = product_probs[1:]

product_probs.head()


# In[ ]:


# save to csv
product_probs.to_csv('transition-probabilities-by-product.csv', 
                     index=False, 
                     encoding='utf-8')

pass

