
# coding: utf-8

# # CV demonstration notebook
# (based on Fred Navruzov's "Dumb-and-the-Dumber-Baselines (PLB=0.3276826)" - https://www.kaggle.com/frednavruzov/dumb-and-the-dumber-baselines-plb-0-3276826.  Some of the code is refactored for less memory use, but the results are unchanged.)

# In[ ]:


import pandas as pd # dataframes
import numpy as np # algebra & calculus
import nltk # text preprocessing & manipulation
# from textblob import TextBlob
import matplotlib.pyplot as plt # plotting
import seaborn as sns # plotting

from functools import partial # to reduce df memory consumption by applying to_numeric

color = sns.color_palette() # adjusting plotting style
import warnings
warnings.filterwarnings('ignore') # silence annoying warnings


# In[ ]:


# aisles
aisles = pd.read_csv('../input/aisles.csv', engine='c')
print('Total aisles: {}'.format(aisles.shape[0]))

# departments
departments = pd.read_csv('../input/departments.csv', engine='c')
print('Total departments: {}'.format(departments.shape[0]))

# products
products = pd.read_csv('../input/products.csv', engine='c')
print('Total products: {}'.format(products.shape[0]))


# In[ ]:


# combine aisles, departments and products (left joined to products)
goods = pd.merge(left=pd.merge(left=products, right=departments, how='left'), right=aisles, how='left')
# to retain '-' and make product names more "standard"
goods.product_name = goods.product_name.str.replace(' ', '_').str.lower() 

# retype goods to reduce memory usage
goods.product_id = goods.product_id.astype(np.int32)
goods.aisle_id = goods.aisle_id.astype(np.int16)
goods.department_id = goods.department_id.astype(np.int8)


# In[ ]:


# load datasets

# train dataset
op_train = pd.read_csv('../input/order_products__train.csv', engine='c', 
                       dtype={'order_id': np.int32, 'product_id': np.int32, 
                              'add_to_cart_order': np.int16, 'reordered': np.int8})
print('Total ordered products(train): {}'.format(op_train.shape[0]))


# In[ ]:


# test dataset (submission)
test = pd.read_csv('../input/sample_submission.csv', engine='c')
print('Total orders(test): {}'.format(test.shape[0]))


# In[ ]:


#prior dataset
op_prior = pd.read_csv('../input/order_products__prior.csv', engine='c', 
                       dtype={'order_id': np.int32, 
                              'product_id': np.int32, 
                              'add_to_cart_order': np.int16, 
                              'reordered': np.int8})

print('Total ordered products(prior): {}'.format(op_prior.shape[0]))


# In[ ]:


# orders
orders = pd.read_csv('../input/orders.csv', engine='c', dtype={'order_id': np.int32, 
                                                           'user_id': np.int32, 
                                                           'order_number': np.int16,  # max 100, could use int8
                                                           'order_dow': np.int8, 
                                                           'order_hour_of_day': np.int8, 
                                                           'days_since_prior_order': np.float16})
print('Total orders: {}'.format(orders.shape[0]))

orders.eval_set = orders.eval_set.replace({'prior': 0, 'train': 1, 'test':2}).astype(np.int8)
orders.days_since_prior_order = orders.days_since_prior_order.fillna(-1).astype(np.int8)


# In[ ]:


from functools import partial

# merge train and prior together iteratively, to fit into 8GB kernel RAM
# split df indexes into parts
indexes = np.linspace(0, len(op_prior), num=10, dtype=np.int32)

# initialize it with train dataset
train_details = pd.merge(
                left=op_train,
                 right=orders, 
                 how='left', 
                 on='order_id'
        ).apply(partial(pd.to_numeric, errors='ignore', downcast='integer'))

# add order hierarchy
train_details = pd.merge(
                left=train_details,
                right=goods[['product_id', 
                             'aisle_id', 
                             'department_id']].apply(partial(pd.to_numeric, 
                                                             errors='ignore', 
                                                             downcast='integer')),
                how='left',
                on='product_id'
)

print(train_details.shape, op_train.shape)

# delete (redundant now) dataframes
#del op_train

#order_details.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', "# update by small portions\norder_details = pd.merge(left=pd.merge(\n                                left=op_prior,\n                                right=goods[['product_id', \n                                             'aisle_id', \n                                             'department_id' ]],\n                                how='left',\n                                on='product_id'\n                                ),\n                         right=orders, \n                         how='left', \n                         on='order_id')\n        \nprint('Datafame length: {}'.format(order_details.shape[0]))\nprint('Memory consumption: {:.2f} Mb'.format(sum(order_details.memory_usage(index=True, \n                                                                         deep=True) / 2**20)))\n# check dtypes to see if we use memory effectively\n#print(order_details.dtypes)\n\n# make sure we didn't forget to retain test dataset :D\n#test_orders = orders[orders.eval_set == 2]")


# In[ ]:


train_orders = orders[orders.eval_set == 1]


# In[ ]:


# switch to integer train indexes so .loc == .iloc
train_orders.index.name = 'raw_order'
train_orders.reset_index(inplace=True)


# In[ ]:


def get_last_orders_reordered(test_orders):
    test_history = order_details[(order_details.user_id.isin(test_orders.user_id))]
    last_orders = test_history.groupby('user_id')['order_number'].max()
    
    t = pd.merge(
        left=pd.merge(
                left=last_orders.reset_index(),
                right=test_history[test_history.reordered == 1],
                how='left',
                on=['user_id', 'order_number']
            )[['user_id', 'product_id']],
        right=test_orders[['user_id', 'order_id']],
        how='left',
        on='user_id'
    ).fillna(-1).groupby('order_id')['product_id'].apply(lambda x: ' '.join([str(int(e)) for e in set(x)]) 
                                              ).reset_index().replace(to_replace='-1', value='None')
    t.columns = ['order_id', 'products']
    
    # occasionally there is a bug where a line with order_id == -1 makes it through. doesn't *seem* to effect things
    return t[t.order_id > 0].set_index('order_id')


# ### Run the above function for 4 folds...
# 
# Strictly speaking, this model does not have any interdependance on the train set, but to provide a complete demonstration KFold is used anyway.

# In[ ]:


import sklearn.model_selection

cvpreds = []

kf = sklearn.model_selection.KFold(4, shuffle=True, random_state=0)
for train_index, test_index in kf.split(train_orders.index):
    cvpreds.append(get_last_orders_reordered(train_orders.iloc[test_index]))

df_cvpreds = pd.concat(cvpreds).sort_index()
df_cvpreds.head()


# #### Now to produce output (indentical to original notebook, so submission is not necessary!)

# In[ ]:


test_preds = get_last_orders_reordered(orders[orders.eval_set == 2])
test_preds.to_csv('cvtest-output.csv', encoding='utf-8')


# # CV F1 validation code begins here

# ### Produce an equivalent .csv + DataFrame to output with the train ground truth data

# In[ ]:


try:
    df_train_gt = pd.read_csv('train.csv', index_col='order_id')
except:
    train_gtl = []

    for uid, subset in train_details.groupby('user_id'):
        subset1 = subset[subset.reordered == 1]
        oid = subset.order_id.values[0]

        if len(subset1) == 0:
            train_gtl.append((oid, 'None'))
            continue

        ostr = ' '.join([str(int(e)) for e in subset1.product_id.values])
        # .strip is needed because join can have a padding space at the end
        train_gtl.append((oid, ostr.strip()))

    df_train_gt = pd.DataFrame(train_gtl)

    df_train_gt.columns = ['order_id', 'products']
    df_train_gt.set_index('order_id', inplace=True)
    df_train_gt.sort_index(inplace=True)
    
    df_train_gt.to_csv('train.csv')


# ### Now compare the ground truth and CV DataFrames

# In[ ]:


f1 = []
for gt, pred in zip(df_train_gt.sort_index().products, df_cvpreds.sort_index().products):
    lgt = gt.replace("None", "-1").split(' ')
    lpred = pred.replace("None", "-1").split(' ')
    
    rr = (np.intersect1d(lgt, lpred))
    precision = np.float(len(rr)) / len(lpred)
    recall = np.float(len(rr)) / len(lgt)

    denom = precision + recall
    f1.append(((2 * precision * recall) / denom) if denom > 0 else 0)

print(np.mean(f1))


# #### The original is .327, so we've got a good validation!
