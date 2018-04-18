
# coding: utf-8

# Work in progress:
# <br>-- add bf0 to data for all products NOT reordered to all orders after first ordered
# <br>--  add the exponential time weighting - for model memory loss
# <br>--  add new factors to model
# <br>--  try flat Prior where p(reorder) is same for all products
# 
# This file uses p(reordered|product_id) derived from order_products__prior data as a **Prior**. This is to be used in Bayesian Updating of our Prior: our_products_prior['prob_reordered']. Can also use a flat Prior.
# 
# The notion is that after calculating Bayes Factors for each test product purchase the final probability that a product will be reordered is the **Posterior** probability.  Beginning when a product is first purchased (say order k of n total orders) then the **Posterior = BFn x BFn-1 x ... x BFk x Prior**.
# 
# Many others here have noticed the correlation between reordered and add_to_cart_order and aisle. I have added an engineered factor I call reorder_count (or count of reordered items in a cart). Using these three variables, I have derived a simple Augmented Naive Bayesian Network as a model to calculate the Bayes Factors for updating.
# 
# ![Bayesian Network model of reordered][1]
# 
# Thanks to Kareem Eissa, Nick Sarris and Paul Nguyen for code and inspiration. Thank you smalllebowski and Sagar M for your corrections! You are very generous.
# 
# 
# 
#   [1]: http://elmtreegarden.com/wp-content/uploads/2017/07/Augmented-Naive-Bayesian-Network.png

# In[ ]:



import pandas as pd
import numpy as np
import operator

# special thanks to Nick Sarris who has written a similar notebook
# reading data
#mdf = 'c:/Users/John/Documents/Research/entropy/python/InstaCart/data/'
mdf = '../input/'
print('loading prior orders')
prior_orders = pd.read_csv(mdf + 'order_products__prior.csv', dtype={
        'order_id': np.int32,
        'product_id': np.int32,
        'add_to_cart_order': np.int16,
        'reordered': np.int8})
print('loading orders')
orders = pd.read_csv(mdf + 'orders.csv', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'eval_set': 'category',
        'order_number': np.int16,
        'order_dow': np.int8,
        'order_hour_of_day': np.int8,
        'days_since_prior_order': np.float32})
print('loading aisles info')
aisles = pd.read_csv(mdf + 'products.csv', engine='c',
                           usecols = ['product_id','aisle_id'],
                       dtype={'product_id': np.int32, 'aisle_id': np.int32})
pd.set_option('display.float_format', lambda x: '%.3f' % x)

prior_orders.shape
orders.shape


# In[ ]:


# removing all user_ids not in the test set from both files to save memory
# the test users present ample data to make models. (and saves space)
test  = orders[orders['eval_set'] == 'test' ]
user_ids = test['user_id'].values
order_ids = test['order_id'].values
orders = orders[orders['user_id'].isin(user_ids)]

#del test
test.shape


# In[ ]:



# Calculate the Prior : p(reordered|product_id)
prior = pd.DataFrame(prior_orders.groupby('product_id')['reordered']                     .agg([('number_of_orders',len),('sum_of_reorders','sum')]))
#prior['prior_p'] = (prior['sum_of_reorders']+1)/(prior['number_of_orders']+2) # Informed Prior
prior['prior_p'] = 1/2  # Flat Prior
prior.drop(['number_of_orders','sum_of_reorders'], axis=1, inplace=True)
print('Here is The Prior: our first guess of how probable it is that a product be reordered once it has been ordered.')

prior.head(3)


# In[ ]:


# merge everything into one dataframe and save any memory space

comb = pd.DataFrame()
comb = pd.merge(prior_orders, orders, on='order_id', how='right')
# slim down comb - 
comb.drop(['eval_set','order_dow','order_hour_of_day'], axis=1, inplace=True)
del prior_orders
del orders
comb = pd.merge(comb, aisles, on ='product_id', how = 'left')
del aisles
prior.reset_index(inplace = True)
comb = pd.merge(comb, prior, on ='product_id', how = 'left')
del prior
print('combined data in DataFrame comb')
comb.head(3)


# In[ ]:



# Build the factors needed for a model of probability of reordered. This model forms our
# hypothesis H and allows the calculation of each Bayes Factor: BF = p(e|H)/(1-p(e|H))
# where e is the test user product buying history. See DAG of model above.
# discretize reorder count into categories, 9 buckets, being sure to include 0 as bucket
# These bins maximize mutual information with ['reordered']. Done outside python
recount = pd.DataFrame()
recount['reorder_c'] = comb.groupby(comb.order_id)['reordered'].sum().fillna(0)
bins = [-0.1, 0, 2,4,6,8,11,14,19,71]
cat =  ['None','<=2','<=4','<=6','<=8','<=11','<=14','<=19','>19']
recount['reorder_b'] = pd.cut(recount['reorder_c'], bins, labels = cat)
recount.reset_index(inplace = True)
comb = pd.merge(comb, recount, how = 'left', on = 'order_id')
del recount

# discretize 'add_to_cart_order' (atco) into categories, 8 buckets
# These bins maximize mutual information with ['recount']. Done outside python
bins = [0,2,3,5,7,9,12,17,80]
cat = ['<=2','<=3','<=5','<=7','<=9','<=12','<=17','>17']
comb['atco1'] = pd.cut(comb['add_to_cart_order'], bins, labels = cat)
del comb['add_to_cart_order']
print('comb ')
comb.head(2)


# In[ ]:


# these are the children Nodes of reordered:atco, aisle, recount. Build occurrence tables
# first, then calculate probabilities. Then merge to add atco into comb.
# 
atco_fac = pd.DataFrame()
atco_fac = comb.groupby(['reordered', 'atco1'])['atco1'].agg(np.count_nonzero).unstack('atco1')
tot = pd.DataFrame()
tot = np.sum(atco_fac,axis=1)
atco_fac = atco_fac.iloc[:,:].div(tot, axis=0)
atco_fac = atco_fac.stack('atco1')
atco_fac = pd.DataFrame(atco_fac)
atco_fac.reset_index(inplace = True)
atco_fac.rename(columns = {0:'atco_fac_p'}, inplace = True)
comb = pd.merge(comb, atco_fac, how='left', on=('reordered', 'atco1'))

# calculate other two factors' probability tables, then probability
# and merge into comb

aisle_fac = pd.DataFrame()
aisle_fac = comb.groupby(['reordered', 'atco1', 'aisle_id'])['aisle_id']                .agg(np.count_nonzero).unstack('aisle_id')
tot = np.sum(aisle_fac,axis=1)
aisle_fac = aisle_fac.iloc[:,:].div(tot, axis=0)
aisle_fac = aisle_fac.stack('aisle_id')
aisle_fac = pd.DataFrame(aisle_fac)
aisle_fac.reset_index(inplace = True)
aisle_fac.rename(columns = {0:'aisle_fac_p'}, inplace = True)
comb = pd.merge(comb, aisle_fac, how = 'left', on = ('aisle_id','reordered','atco1'))
# last factor is reorder_count_factor   
    
recount_fac = pd.DataFrame()
recount_fac = comb.groupby(['reordered', 'atco1', 'reorder_b'])['reorder_b']                    .agg(np.count_nonzero).unstack('reorder_b')
tot = pd.DataFrame()
tot = np.sum(recount_fac,axis=1)
recount_fac = recount_fac.iloc[:,:].div(tot, axis=0)
recount_fac.stack('reorder_b')
recount_fac = pd.DataFrame(recount_fac.unstack('reordered').unstack('atco1')).reset_index()
recount_fac.rename(columns = {0:'recount_fac_p'}, inplace = True)
comb = pd.merge(comb, recount_fac, how = 'left', on = ('reorder_b', 'reordered', 'atco1'))

recount_fac.head(3)


# In[ ]:



# Use the factors in comb + the prior_p to update a posterior for each product purchased.
p = pd.DataFrame()
p = (comb.loc[:,'atco_fac_p'] * comb.loc[:,'aisle_fac_p'] * comb.loc[:,'recount_fac_p'])
p.reset_index()
comb['p'] = p

comb.head(3)


# In[ ]:



# work in progress on beta
# Use a test beta = 95% per month for memory retention function of users. Akin to Recency.


#split into three dataframes. Two are reordered == 1 and == 0
# add third group when order_number > first_order & reordered <> 1
# the trird group is when ordered=0 but we don't have data for order=0,
# so we make it.It must be appended to comb_last

# Calculate bf0 for products when first purchased aka reordered=0
comb0 = pd.DataFrame()
comb0 = comb[comb['reordered']==0]
comb0.loc[:,'first_order'] = comb0['order_number']
# now every product that was ordered has a posterior in usr.
comb0.loc[:,'beta'] = 1
comb0.loc[:,'bf'] = (comb0.loc[:,'prior_p'] * comb0.loc[:,'p']/(1 - comb0.loc[:,'p'])) # bf1
# Small 'slight of hand' here. comb0.bf is really the first posterior and second prior.

# Calculate beta and BF1 for the reordered products
comb1 = pd.DataFrame()
comb1 = comb[comb['reordered']==1]

comb1.loc[:,'beta'] = (1 - .05*comb1.loc[:,'days_since_prior_order']/30)
comb1.loc[:,'bf'] = (1 - comb1.loc[:,'p'])/comb1.loc[:,'p'] # bf0


comb_last = pd.DataFrame()
comb_last = pd.concat([comb0, comb1], axis=0).reset_index(drop=True)
comb_last = comb_last[['reordered','user_id','product_id','reorder_c','order_number',
                      'bf','beta','atco_fac_p', 'aisle_fac_p', 'recount_fac_p']]
comb_last = comb_last.sort_values((['user_id', 'order_number', 'bf']))

pd.set_option('display.float_format', lambda x: '%.6f' % x)
comb_last.head(3)


# In[ ]:


first_order = pd.DataFrame()
first_order = comb_last[comb_last.reordered == 0]
first_order.rename(columns = {'order_number':'first_o'}, inplace = True)
first_order.loc[:,'last_o'] = comb_last.groupby(['user_id'])['order_number'].transform(max)
first_order = first_order[['user_id','product_id','first_o','last_o']]
comb_last = pd.merge(comb_last, first_order, on = ('user_id', 'product_id'), how = 'left')

#com = pd.DataFrame()
#com = comb_last[(comb_last.user_id == 3) & (comb_last.first_o < comb_last.order_number)]
#com.groupby([('order_id', 'product_id', 'order_number')])['bf'].agg(np.sum).head(50)


# In[ ]:


# Calculate beta and bf0 for products not reordered after first order for all orders.
# must not occur until reordered==0 (aka: when first ordered)
# they do not exist in the data. there is no record of NOT Ordered.
# we must produce these records and calculate p, bf0 & beta for each
com = pd.DataFrame

# replace nan with bf0 if first_o < order_number (after product is first ordered)
com = pd.pivot_table(comb_last[(comb_last.user_id == 3) &                                (comb_last.first_o < comb_last.order_number)],
                     values = 'bf', index = ['user_id', 'product_id'],
                     columns = 'order_number', dropna=False)
temp = pd.DataFrame()
temp = com[(com.bf == 'nan')]
p = pd.DataFrame()
p.loc[:,'p'] = (temp.loc[:,'atco_fac_p'] * temp.loc[:,'aisle_fac_p'] * temp.loc[:,'recount_fac_p'])
p.reset_index()
temp.loc[:,'bf'] = (1 - temp.loc[:,p])/temp.loc[:,p]
comb_last = pd.merge(comb_last, temp, on =[('order_id', 'product_id',
                                            'order_number')]).reset_index()
temp = comb_last[comb_last.beta == 'nan']
temp.loc[:,'beta'] = (1 - .05*comb1.loc[:,'days_since_prior_order']/30)
comb_last = pd.merge(comb_last, temp, on = [('order_id', 'product_id',
                                             'order_number')]).reset_index()

# replace nan with 1 if first_o > order number (before product has been ordered)
com = pd.pivot_table(comb_last[(comb_last.user_id ==3) & (com.first_o < com.order_number)],
                     values = 'beta', index = ['user_id', 'product_id'], 
                     columns = 'order_number', dropna=False)
# 
temp = com[com.bf == 'nan']
temp.loc[:,'bf'] = 1
comb_last = pd.merge(comb_last, temp, on =[('order_id', 'product_id',
                                            'order_number')]).reset_index()
temp = comb_last[comb_last.beta == 'nan']
temp.loc[:,'beta'] = 1
comb_last = pd.merge(comb_last, temp, on = [('order_id', 'product_id',
                                             'order_number')]).reset_index()


pd.pivot_table(comb_last[comb_last.user_id ==3], values = 'bf',
               index = ['user_id', 'product_id'], columns = 'order_number', dropna=False).head(15)


# In[ ]:


# Find way to introduce beta to the update. ????
##  update = lambda bf(n) ,bf(n-1), beta(n): bf(n) * bf(n-1)**beta(n);

# finally, perform update of every product
# Calculate the posterior for every product a user has purchased
usr = pd.DataFrame()
usr = comb_last[comb_last.order_number >= comb_last.first_o].groupby(['user_id',
                                                                      'product_id'])['bf',
                                                                                     'beta']\
    .agg({['bf', 'beta']: lambda x,y: x**y}).reset_index() 

# Calculate the average number of reordered products per cart for each user
temp = pd.DataFrame()
temp = comb_last[comb_last.order_number > 1].groupby(['user_id'])['reorder_c']    .agg(np.mean).reset_index()
user = pd.merge(usr, temp, on = 'user_id', how = 'left')

user.head(5)


# In[ ]:


def f1(x):
    return ' '.join([str(int(a)) for a in x])
def f2(x):
    return 'None'

u = user.reset_index().sort_values(((['user_id','bf'])), ascending=False)
u['cumulative'] = u.groupby('user_id').cumcount()
uu = u[(round(u.reorder_c) > u.cumulative)].groupby('user_id').agg({'product_id': f1})
uu.reset_index(inplace=True)
uuu = u[round(u.reorder_c) == 0].groupby('user_id').agg({'product_id': f2})
uuu.reset_index(inplace=True)

uuuu = pd.concat([uu, uuu], axis=0).reset_index()
sub = pd.merge(uuuu, test, on='user_id', how ='left').sort_values('order_id')

sub.sort_values('order_id')
sub[['order_id', 'product_id']].to_csv('bayesian.csv', index=False)
sub[['order_id', 'product_id']].head(10)

