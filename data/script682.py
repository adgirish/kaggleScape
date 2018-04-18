
# coding: utf-8

# # My 15th solution
# 
# Though my rank is not so hight, I would like to share my solution 
# because I jumped 17 postions in finaly shakeup and I used BigQuery, which might be not familier in Kaggle well.
# 
# 

# # Why could I jump from 32nd to 15th?
# 
# I made my cv split whose ditribustion how many kind of items the customer bought is same. (I meand I made cv based on how many kind of items a customer bought.)
# Then, my solution did not make difference between my cv score, public score, and private score.
# 
# I think my cv splitting strategy might work well.

# # My features
# 
# I mainly made my features by using Google BigQuery.
# 
# (I will share the rest of my features later because the rest of my features are very complicated and made not much difference in improving my score.)

# ### Making my datamart ( joinining all data into one table)
# 
# I used Python for joinining all data into one table. Of course, you can use BigQuey.

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import pickle
import re
from IPython.core.display import display
from tqdm import tqdm_notebook as tqdm

get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.mode.chained_assignment = None  # default='warn'


# In[2]:


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[3]:


order_products_train_df = pd.read_csv("../input/order_products__train.csv")
order_products_prior_df = pd.read_csv("../input/order_products__prior.csv")
orders_df = pd.read_csv("../input/orders.csv")
products_df = pd.read_csv("../input/products.csv")
aisles_df = pd.read_csv("../input/aisles.csv")
departments_df = pd.read_csv("../input/departments.csv")


# In[ ]:


df_train = pd.merge(order_products_train_df, orders_df, how='left', on='order_id')
df_train = pd.merge(df_train, products_df, how='left', on='product_id')
df_train = pd.merge(df_train, aisles_df, how='left', on='aisle_id')
df_train = pd.merge(df_train, departments_df, how='left', on='department_id')
#df_train.to_csv('../input/df_train.csv', index=False) # if you want to use my feature, plz comment out.
df_train.head()


# In[ ]:


df_prior = pd.merge(order_products_prior_df, orders_df, how='left', on='order_id').head(10000) 
# if you want to use my feature, plz remove the ".head(10000)".
df_prior = pd.merge(df_prior, products_df, how='left', on='product_id')
df_prior = pd.merge(df_prior, aisles_df, how='left', on='aisle_id')
df_prior = pd.merge(df_prior, departments_df, how='left', on='department_id')
#df_prior.to_csv('../input/df_prior.csv', index=False) # if you want to use my feature, plz comment out.
df_prior.head()


# # Making User features
# 
# The both of **df_prior** and **df_train** are made from above scripts.  
# Then, I import these data into BigQuery, and ran below queries.

# ```
# # user_fund
# bq query --max_rows 1  --allow_large_results --destination_table "instacart.user_fund" --flatten_results --replace "
# SELECT
#   user_id,
#   count(1) as user_item_cnt,
#   EXACT_COUNT_DISTINCT(product_id) as user_prd_cnt,
#   EXACT_COUNT_DISTINCT(department_id) as user_depart_cnt,
#   EXACT_COUNT_DISTINCT(aisle_id) as user_aisle_cnt,
#   EXACT_COUNT_DISTINCT(order_id) as user_order_cnt,
#   EXACT_COUNT_DISTINCT(order_id) / count(1)  as user_order_rate,
#   MAX(order_number) as max_order_number,
#   AVG(days_since_prior_order) as avg_days_since_prior_order,
#   MAX(days_since_prior_order) as max_days_since_prior_order,
#   MIN(days_since_prior_order) as min_days_since_prior_order,
#   MAX(order_hour_of_day) as max_order_hour_of_day,
#   MIN(order_hour_of_day) as min_order_hour_of_day,
#   AVG(order_hour_of_day) as avg_order_hour_of_day,
#   AVG(reordered) as avg_reordered,
#   SUM(reordered) as sum_reordered,
#   AVG(order_dow) as avg_order_dow
# FROM
#   [instacart.df_prior]
# GROUP BY
#   user_id
# "
# 
# bq query --max_rows 1  --allow_large_results --destination_table "instacart.user_freq" --flatten_results --replace "
# SELECT
#   user_id,
#   count(1) as order_cnt,
#   AVG(days_since_prior_order) as avg_days_since_prior_order,
#   MAX(days_since_prior_order) as max_days_since_prior_order,
#   MIN(days_since_prior_order) as min_days_since_prior_order,
# FROM
#   [instacart.orders]
# WHERE
#   eval_set = 'prior'
# GROUP BY
#   user_id
# "
# 
# bq query --max_rows 1  --allow_large_results --destination_table "instacart.user_dow" --flatten_results --replace "
# SELECT
#   user_id,
#   sum(CASE WHEN order_dow = 0  THEN 1 ELSE 0 END) AS  order_dow_0,
#   sum(CASE WHEN order_dow = 1  THEN 1 ELSE 0 END) AS  order_dow_1,
#   sum(CASE WHEN order_dow = 2  THEN 1 ELSE 0 END) AS  order_dow_2,
#   sum(CASE WHEN order_dow = 3  THEN 1 ELSE 0 END) AS  order_dow_3,
#   sum(CASE WHEN order_dow = 4  THEN 1 ELSE 0 END) AS  order_dow_4,
#   sum(CASE WHEN order_dow = 5  THEN 1 ELSE 0 END) AS  order_dow_5,
#   sum(CASE WHEN order_dow = 6  THEN 1 ELSE 0 END) AS  order_dow_6,
# 
#   avg(CASE WHEN order_dow = 0  THEN reordered ELSE null END) AS  reorder_dow_0,
#   avg(CASE WHEN order_dow = 1  THEN reordered ELSE null END) AS  reorder_dow_1,
#   avg(CASE WHEN order_dow = 2  THEN reordered ELSE null END) AS  reorder_dow_2,
#   avg(CASE WHEN order_dow = 3  THEN reordered ELSE null END) AS  reorder_dow_3,
#   avg(CASE WHEN order_dow = 4  THEN reordered ELSE null END) AS  reorder_dow_4,
#   avg(CASE WHEN order_dow = 5  THEN reordered ELSE null END) AS  reorder_dow_5,
#   avg(CASE WHEN order_dow = 6  THEN reordered ELSE null END) AS  reorder_dow_6
# FROM
#   [instacart.df_prior]
# GROUP BY
#   user_id
# "
# 
# bq query --max_rows 1  --allow_large_results --destination_table "instacart.user_hour" --flatten_results --replace "
# SELECT
#   user_id,
#   sum(CASE WHEN order_hour_of_day = 0  THEN 1 ELSE 0 END) AS order_hour_of_day_0,
#   sum(CASE WHEN order_hour_of_day = 1  THEN 1 ELSE 0 END) AS order_hour_of_day_1,
#   sum(CASE WHEN order_hour_of_day = 2  THEN 1 ELSE 0 END) AS order_hour_of_day_2,
#   sum(CASE WHEN order_hour_of_day = 3  THEN 1 ELSE 0 END) AS order_hour_of_day_3,
#   sum(CASE WHEN order_hour_of_day = 4  THEN 1 ELSE 0 END) AS order_hour_of_day_4,
#   sum(CASE WHEN order_hour_of_day = 5  THEN 1 ELSE 0 END) AS order_hour_of_day_5,
#   sum(CASE WHEN order_hour_of_day = 6  THEN 1 ELSE 0 END) AS order_hour_of_day_6,
#   sum(CASE WHEN order_hour_of_day = 7  THEN 1 ELSE 0 END) AS order_hour_of_day_7,
#   sum(CASE WHEN order_hour_of_day = 8  THEN 1 ELSE 0 END) AS order_hour_of_day_8,
#   sum(CASE WHEN order_hour_of_day = 9  THEN 1 ELSE 0 END) AS order_hour_of_day_9,
#   sum(CASE WHEN order_hour_of_day = 10  THEN 1 ELSE 0 END) AS order_hour_of_day_10,
#   sum(CASE WHEN order_hour_of_day = 11  THEN 1 ELSE 0 END) AS order_hour_of_day_11,
#   sum(CASE WHEN order_hour_of_day = 12  THEN 1 ELSE 0 END) AS order_hour_of_day_12,
#   sum(CASE WHEN order_hour_of_day = 13  THEN 1 ELSE 0 END) AS order_hour_of_day_13,
#   sum(CASE WHEN order_hour_of_day = 14  THEN 1 ELSE 0 END) AS order_hour_of_day_14,
#   sum(CASE WHEN order_hour_of_day = 15  THEN 1 ELSE 0 END) AS order_hour_of_day_15,
#   sum(CASE WHEN order_hour_of_day = 16  THEN 1 ELSE 0 END) AS order_hour_of_day_16,
#   sum(CASE WHEN order_hour_of_day = 17  THEN 1 ELSE 0 END) AS order_hour_of_day_17,
#   sum(CASE WHEN order_hour_of_day = 18  THEN 1 ELSE 0 END) AS order_hour_of_day_18,
#   sum(CASE WHEN order_hour_of_day = 19  THEN 1 ELSE 0 END) AS order_hour_of_day_19,
#   sum(CASE WHEN order_hour_of_day = 20  THEN 1 ELSE 0 END) AS order_hour_of_day_20,
#   sum(CASE WHEN order_hour_of_day = 21  THEN 1 ELSE 0 END) AS order_hour_of_day_21,
#   sum(CASE WHEN order_hour_of_day = 22  THEN 1 ELSE 0 END) AS order_hour_of_day_22,
#   sum(CASE WHEN order_hour_of_day = 23  THEN 1 ELSE 0 END) AS order_hour_of_day_23
# FROM
#   [instacart.df_prior]
# GROUP BY
#   user_id
# "
# 
# bq query --max_rows 1  --allow_large_results --destination_table "instacart.user_depart" --flatten_results --replace "
# SELECT
#   user_id,
#   sum(CASE WHEN department_id = 1  THEN 1 ELSE 0 END) AS department_id_1,
#   sum(CASE WHEN department_id = 2  THEN 1 ELSE 0 END) AS department_id_2,
#   sum(CASE WHEN department_id = 3  THEN 1 ELSE 0 END) AS department_id_3,
#   sum(CASE WHEN department_id = 4  THEN 1 ELSE 0 END) AS department_id_4,
#   sum(CASE WHEN department_id = 5  THEN 1 ELSE 0 END) AS department_id_5,
#   sum(CASE WHEN department_id = 6  THEN 1 ELSE 0 END) AS department_id_6,
#   sum(CASE WHEN department_id = 7  THEN 1 ELSE 0 END) AS department_id_7,
#   sum(CASE WHEN department_id = 8  THEN 1 ELSE 0 END) AS department_id_8,
#   sum(CASE WHEN department_id = 9  THEN 1 ELSE 0 END) AS department_id_9,
#   sum(CASE WHEN department_id = 10  THEN 1 ELSE 0 END) AS department_id_10,
#   sum(CASE WHEN department_id = 11  THEN 1 ELSE 0 END) AS department_id_11,
#   sum(CASE WHEN department_id = 12  THEN 1 ELSE 0 END) AS department_id_12,
#   sum(CASE WHEN department_id = 13  THEN 1 ELSE 0 END) AS department_id_13,
#   sum(CASE WHEN department_id = 14  THEN 1 ELSE 0 END) AS department_id_14,
#   sum(CASE WHEN department_id = 15  THEN 1 ELSE 0 END) AS department_id_15,
#   sum(CASE WHEN department_id = 16  THEN 1 ELSE 0 END) AS department_id_16,
#   sum(CASE WHEN department_id = 17  THEN 1 ELSE 0 END) AS department_id_17,
#   sum(CASE WHEN department_id = 18  THEN 1 ELSE 0 END) AS department_id_18,
#   sum(CASE WHEN department_id = 19  THEN 1 ELSE 0 END) AS department_id_19,
#   sum(CASE WHEN department_id = 20  THEN 1 ELSE 0 END) AS department_id_20,
#   sum(CASE WHEN department_id = 21  THEN 1 ELSE 0 END) AS department_id_21
# FROM
#   [instacart.df_prior]
# GROUP BY
#   user_id
# "
# 
# bq query --max_rows 1  --allow_large_results --destination_table "instacart.dmt_user" --flatten_results --replace "
# SELECT
#   *
# FROM
#   [instacart.user_fund] as u1
# LEFT OUTER JOIN
#   [instacart.user_dow] as u2
# ON
#   u1.user_id = u2.user_id
# LEFT OUTER JOIN
#   [instacart.user_hour] as u3
# ON
#   u1.user_id = u3.user_id
# LEFT OUTER JOIN
#   [instacart.user_depart] as u4
# ON
#   u1.user_id = u4.user_id
# LEFT OUTER JOIN
#   [instacart.user_freq] as u5
# ON
#   u1.user_id = u5.user_id
# "
# ```

# # Making item features

# ```
# bq query --max_rows 1  --allow_large_results --destination_table "instacart.item_fund" --flatten_results --replace "  
# SELECT
#   product_id,
#   count(1) as item_user_cnt,
#   EXACT_COUNT_DISTINCT( user_id) as item_usr_cnt,
#   EXACT_COUNT_DISTINCT( department_id) as item_depart_cnt,
#   EXACT_COUNT_DISTINCT( aisle_id) as item_aisle_cnt,
#   EXACT_COUNT_DISTINCT( order_id) as item_order_cnt,
#   EXACT_COUNT_DISTINCT( order_id) / count(1) as item_order_rate,
#   AVG(days_since_prior_order) as avg_item_days_since_prior_order,
#   MIN(days_since_prior_order) as min_item_days_since_prior_order,
#   MAX(days_since_prior_order) as max_item_days_since_prior_order,
#   MAX(order_hour_of_day) as max_order_hour_of_day,
#   MIN(order_hour_of_day) as min_order_hour_of_day,
#   AVG(order_hour_of_day) as avg_order_hour_of_day,
#   AVG(reordered) as avg_item_reordered,
#   SUM(reordered) as sum_item_reordered,
#   AVG(order_dow) as avg_order_dow
# FROM
#   [instacart.df_prior]
# GROUP BY
#   product_id
# "
# 
# bq query --max_rows 1  --allow_large_results --destination_table "instacart.item_dow" --flatten_results --replace "  
# SELECT
#   product_id,
#   sum(CASE WHEN order_dow = 0  THEN 1 ELSE 0 END) AS  order_dow_0,
#   sum(CASE WHEN order_dow = 1  THEN 1 ELSE 0 END) AS  order_dow_1,
#   sum(CASE WHEN order_dow = 2  THEN 1 ELSE 0 END) AS  order_dow_2,
#   sum(CASE WHEN order_dow = 3  THEN 1 ELSE 0 END) AS  order_dow_3,
#   sum(CASE WHEN order_dow = 4  THEN 1 ELSE 0 END) AS  order_dow_4,
#   sum(CASE WHEN order_dow = 5  THEN 1 ELSE 0 END) AS  order_dow_5,
#   sum(CASE WHEN order_dow = 6  THEN 1 ELSE 0 END) AS  order_dow_6
# FROM
#   [instacart.df_prior]
# GROUP BY
#   product_id
# "
# 
# bq query --max_rows 1  --allow_large_results --destination_table "instacart.item_hour" --flatten_results --replace "  
# SELECT
#   product_id,
#   sum(CASE WHEN order_hour_of_day = 0  THEN 1 ELSE 0 END) AS order_hour_of_day_0,
#   sum(CASE WHEN order_hour_of_day = 1  THEN 1 ELSE 0 END) AS order_hour_of_day_1,
#   sum(CASE WHEN order_hour_of_day = 2  THEN 1 ELSE 0 END) AS order_hour_of_day_2,
#   sum(CASE WHEN order_hour_of_day = 3  THEN 1 ELSE 0 END) AS order_hour_of_day_3,
#   sum(CASE WHEN order_hour_of_day = 4  THEN 1 ELSE 0 END) AS order_hour_of_day_4,
#   sum(CASE WHEN order_hour_of_day = 5  THEN 1 ELSE 0 END) AS order_hour_of_day_5,
#   sum(CASE WHEN order_hour_of_day = 6  THEN 1 ELSE 0 END) AS order_hour_of_day_6,
#   sum(CASE WHEN order_hour_of_day = 7  THEN 1 ELSE 0 END) AS order_hour_of_day_7,
#   sum(CASE WHEN order_hour_of_day = 8  THEN 1 ELSE 0 END) AS order_hour_of_day_8,
#   sum(CASE WHEN order_hour_of_day = 9  THEN 1 ELSE 0 END) AS order_hour_of_day_9,
#   sum(CASE WHEN order_hour_of_day = 10  THEN 1 ELSE 0 END) AS order_hour_of_day_10,
#   sum(CASE WHEN order_hour_of_day = 11  THEN 1 ELSE 0 END) AS order_hour_of_day_11,
#   sum(CASE WHEN order_hour_of_day = 12  THEN 1 ELSE 0 END) AS order_hour_of_day_12,
#   sum(CASE WHEN order_hour_of_day = 13  THEN 1 ELSE 0 END) AS order_hour_of_day_13,
#   sum(CASE WHEN order_hour_of_day = 14  THEN 1 ELSE 0 END) AS order_hour_of_day_14,
#   sum(CASE WHEN order_hour_of_day = 15  THEN 1 ELSE 0 END) AS order_hour_of_day_15,
#   sum(CASE WHEN order_hour_of_day = 16  THEN 1 ELSE 0 END) AS order_hour_of_day_16,
#   sum(CASE WHEN order_hour_of_day = 17  THEN 1 ELSE 0 END) AS order_hour_of_day_17,
#   sum(CASE WHEN order_hour_of_day = 18  THEN 1 ELSE 0 END) AS order_hour_of_day_18,
#   sum(CASE WHEN order_hour_of_day = 19  THEN 1 ELSE 0 END) AS order_hour_of_day_19,
#   sum(CASE WHEN order_hour_of_day = 20  THEN 1 ELSE 0 END) AS order_hour_of_day_20,
#   sum(CASE WHEN order_hour_of_day = 21  THEN 1 ELSE 0 END) AS order_hour_of_day_21,
#   sum(CASE WHEN order_hour_of_day = 22  THEN 1 ELSE 0 END) AS order_hour_of_day_22,
#   sum(CASE WHEN order_hour_of_day = 23  THEN 1 ELSE 0 END) AS order_hour_of_day_23
# FROM
#   [instacart.df_prior]
# GROUP BY
#   product_id
# "
# 
# 
# bq query --max_rows 1  --allow_large_results --destination_table "instacart.item_depart" --flatten_results --replace "  
# SELECT
#   product_id,
#   sum(CASE WHEN department_id = 1  THEN 1 ELSE 0 END) AS department_id_1,
#   sum(CASE WHEN department_id = 2  THEN 1 ELSE 0 END) AS department_id_2,
#   sum(CASE WHEN department_id = 3  THEN 1 ELSE 0 END) AS department_id_3,
#   sum(CASE WHEN department_id = 4  THEN 1 ELSE 0 END) AS department_id_4,
#   sum(CASE WHEN department_id = 5  THEN 1 ELSE 0 END) AS department_id_5,
#   sum(CASE WHEN department_id = 6  THEN 1 ELSE 0 END) AS department_id_6,
#   sum(CASE WHEN department_id = 7  THEN 1 ELSE 0 END) AS department_id_7,
#   sum(CASE WHEN department_id = 8  THEN 1 ELSE 0 END) AS department_id_8,
#   sum(CASE WHEN department_id = 9  THEN 1 ELSE 0 END) AS department_id_9,
#   sum(CASE WHEN department_id = 10  THEN 1 ELSE 0 END) AS department_id_10,
#   sum(CASE WHEN department_id = 11  THEN 1 ELSE 0 END) AS department_id_11,
#   sum(CASE WHEN department_id = 12  THEN 1 ELSE 0 END) AS department_id_12,
#   sum(CASE WHEN department_id = 13  THEN 1 ELSE 0 END) AS department_id_13,
#   sum(CASE WHEN department_id = 14  THEN 1 ELSE 0 END) AS department_id_14,
#   sum(CASE WHEN department_id = 15  THEN 1 ELSE 0 END) AS department_id_15,
#   sum(CASE WHEN department_id = 16  THEN 1 ELSE 0 END) AS department_id_16,
#   sum(CASE WHEN department_id = 17  THEN 1 ELSE 0 END) AS department_id_17,
#   sum(CASE WHEN department_id = 18  THEN 1 ELSE 0 END) AS department_id_18,
#   sum(CASE WHEN department_id = 19  THEN 1 ELSE 0 END) AS department_id_19,
#   sum(CASE WHEN department_id = 20  THEN 1 ELSE 0 END) AS department_id_20,
#   sum(CASE WHEN department_id = 21  THEN 1 ELSE 0 END) AS department_id_21
# FROM
#   [instacart.df_prior]
# GROUP BY
#   product_id
# "
# 
# bq query --max_rows 1  --allow_large_results --destination_table "instacart.dmt_item" --flatten_results --replace "  
# SELECT
#   *
# FROM
#   [instacart.item_fund] as i1
# LEFT OUTER JOIN
#   [instacart.item_dow] as i2
# ON
#   i1.product_id = i2.product_id
# LEFT OUTER JOIN
#   [instacart.item_hour] as i3
# ON
#   i1.product_id = i3.product_id
# LEFT OUTER JOIN
#   [instacart.item_depart] as i4
# ON
#   i1.product_id = i4.product_id
# "
# ```

# # Making user and item features
# 
# These tables are my final datamart: 
# * dmt_train_only_rebuy
# * dmt_test_only_rebuy.
# 
# (But this is little old version, so I will update soon)

# ```
# bq query --max_rows 1  --allow_large_results --destination_table "work.tmp1" --flatten_results --replace "
# SELECT
#   user_id,
#   order_id,
#   eval_set,
#   order_number,
#   order_dow,
#   order_hour_of_day,  
#   days_since_prior_order,
#   SUM(days_since_prior_order) OVER (PARTITION BY user_id ORDER BY order_number ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_days
# FROM
#   [instacart.orders]
# "
# 
# bq query --max_rows 1  --allow_large_results --destination_table "instacart.cum_orders" --flatten_results --replace "
# SELECT
#   a.user_id user_id,
#   a.order_id order_id,
#   a.eval_set eval_set,
#   a.order_number order_number,
#   a.days_since_prior_order days_since_prior_order,
#   a.cum_days cum_days,
#   a.order_dow,
#   a.order_hour_of_day,  
#   b.max_cum_days max_cum_days,
#   b.max_cum_days - a.cum_days as last_buy
# FROM
#   [work.tmp1] as a
# LEFT OUTER JOIN
# (
# SELECT
#   user_id, max(cum_days) as max_cum_days
# FROM
#   [work.tmp1]
# GROUP BY
#   user_id
# ) as b
# ON
#   a.user_id = b.user_id
# "
# 
# bq query --max_rows 1  --allow_large_results --destination_table "instacart.only_rebuy" --flatten_results --replace "
# SELECT
#   a.user_id as user_id,
#   a.product_id as product_id
# FROM
#   [instacart.df_prior] as a
# GROUP BY
#   user_id, product_id
# "
# 
# bq query --max_rows 1  --allow_large_results --destination_table "instacart.only_rebuy_train" --flatten_results --replace "
# SELECT
#   r.user_id as user_id,
#   r.product_id as product_id,
#   o.order_id as order_id,
#   o.order_number as order_number,
#   o.order_dow as order_dow,
#   o.order_hour_of_day as order_hour_of_day,
#   o.days_since_prior_order as days_since_prior_order,
#   c.cum_days as cum_days
# FROM
#   [instacart.only_rebuy] as r
# INNER JOIN
#   (SELECT * FROM [instacart.orders] WHERE eval_set='train') as o
# ON
#   r.user_id = o.user_id
# LEFT OUTER JOIN
#   [instacart.cum_orders] as c
# ON
#   c.order_id = o.order_id
# "
# 
# bq query --max_rows 1  --allow_large_results --destination_table "instacart.only_rebuy_test" --flatten_results --replace "
# SELECT
#   r.user_id as user_id,
#   r.product_id as product_id,
#   o.order_id as order_id,
#   o.order_number as order_number,
#   o.order_dow as order_dow,
#   o.order_hour_of_day as order_hour_of_day,
#   o.days_since_prior_order as days_since_prior_order,
#   c.cum_days as cum_days
# FROM
#   [instacart.only_rebuy] as r
# INNER JOIN
#   (SELECT * FROM [instacart.orders] WHERE eval_set='test') as o
# ON
#   r.user_id = o.user_id
# LEFT OUTER JOIN
#   [instacart.cum_orders] as c
# ON
#   c.order_id = o.order_id
# "
# 
# ###
# 
# bq query --max_rows 1  --allow_large_results --destination_table "instacart.last_buy" --flatten_results --replace "
# SELECT
#   a.user_id as user_id,
#   a.product_id as product_id,
#   a.max_order_number as max_order_number,
#   a.max_reordered as max_reordered,
#   a.avg_reordered as avg_reordered,
#   o.order_id,
#   c.cum_days as cum_days,
#   c.last_buy as last_buy,
#   o.order_number,
#   o.order_dow,
#   o.order_hour_of_day,
#   o.days_since_prior_order,
#   o.order_number - c.order_number as order_number_diff,
#   o.days_since_prior_order - c.days_since_prior_order
# FROM
#   (
#   SELECT
#     user_id,
#     product_id,
#     max(reordered) as max_reordered,
#     avg(reordered) as avg_reordered,
#     MAX(order_number) as max_order_number
#   FROM
#     [instacart.df_prior]
#   GROUP BY
#     user_id,
#     product_id
#   ) as a
# LEFT OUTER JOIN
#   [instacart.orders] as o
# ON
#   a.user_id = o.user_id AND
#   a.max_order_number = o.order_number
# LEFT OUTER JOIN
#   [instacart.cum_orders] as c
# ON
#   c.order_id = o.order_id
# "
# 
# 
# bq query --max_rows 1  --allow_large_results --destination_table "instacart.last_buy_aisle" --flatten_results --replace "
# SELECT
#   a.user_id as user_id,
#   a.aisle_id as aisle_id,
#   a.max_order_number as max_order_number,
#   a.max_reordered as max_reordered,
#   a.avg_reordered as avg_reordered,
#   o.order_id,
#   c.cum_days as cum_days,
#   c.last_buy as last_buy,
#   o.order_number,
#   o.order_dow,
#   o.order_hour_of_day,
#   o.days_since_prior_order,
#   o.order_number - c.order_number as order_number_diff,
#   o.days_since_prior_order - c.days_since_prior_order
# FROM
#   (
#   SELECT
#     user_id,
#     aisle_id,
#     max(reordered) as max_reordered,
#     avg(reordered) as avg_reordered,
#     MAX(order_number) as max_order_number
#   FROM
#     [instacart.df_prior]
#   GROUP BY
#     user_id,
#     aisle_id
#   ) as a
# LEFT OUTER JOIN
#   [instacart.orders] as o
# ON
#   a.user_id = o.user_id AND
#   a.max_order_number = o.order_number
# LEFT OUTER JOIN
#   [instacart.cum_orders] as c
# ON
#   c.order_id = o.order_id
# "
# 
# bq query --max_rows 1  --allow_large_results --destination_table "instacart.last_buy_depart" --flatten_results --replace "
# SELECT
#   a.user_id as user_id,
#   a.department_id as department_id,
#   a.max_order_number as max_order_number,
#   a.max_reordered as max_reordered,
#   a.avg_reordered as avg_reordered,
#   o.order_id,
#   c.cum_days as cum_days,
#   c.last_buy as last_buy,
#   o.order_number,
#   o.order_dow,
#   o.order_hour_of_day,
#   o.days_since_prior_order,
#   o.order_number - c.order_number as order_number_diff,
#   o.days_since_prior_order - c.days_since_prior_order
# FROM
#   (
#   SELECT
#     user_id,
#     department_id,
#     max(reordered) as max_reordered,
#     avg(reordered) as avg_reordered,
#     MAX(order_number) as max_order_number
#   FROM
#     [instacart.df_prior]
#   GROUP BY
#     user_id,
#     department_id
#   ) as a
# LEFT OUTER JOIN
#   [instacart.orders] as o
# ON
#   a.user_id = o.user_id AND
#   a.max_order_number = o.order_number
# LEFT OUTER JOIN
#   [instacart.cum_orders] as c
# ON
#   c.order_id = o.order_id
# "
# 
# ###
# 
# 
# ###
# 
# bq query --max_rows 1  --allow_large_results --destination_table "instacart.last_buy_2" --flatten_results --replace "
# SELECT 
#   a.user_id as user_id,
#   a.product_id as product_id,
#   a.max_order_number as max_order_number,
#   a.max_reordered as max_reordered,
#   a.avg_reordered as avg_reordered,
#   o.order_id,
#   c.cum_days as cum_days,
#   c.last_buy as last_buy,
#   o.order_number,
#   o.order_dow,
#   o.order_hour_of_day,
#   o.days_since_prior_order,
#   o.order_number - c.order_number as order_number_diff,
#   o.days_since_prior_order - c.days_since_prior_order
# FROM
#   (
#   SELECT
#     user_id,
#     product_id,
#     max(reordered) as max_reordered,
#     avg(reordered) as avg_reordered,
#     MAX(order_number) - 1 as max_order_number
#   FROM
#     [instacart.df_prior]
#   GROUP BY
#     user_id,
#     product_id
#   ) as a
# LEFT OUTER JOIN
#   [instacart.orders] as o
# ON
#   a.user_id = o.user_id AND
#   a.max_order_number = o.order_number
# LEFT OUTER JOIN
#   [instacart.cum_orders] as c
# ON
#   c.order_id = o.order_id
# "
# 
# 
# bq query --max_rows 1  --allow_large_results --destination_table "instacart.last_buy_aisle_2" --flatten_results --replace "
# SELECT
#   a.user_id as user_id,
#   a.aisle_id as aisle_id,
#   a.max_order_number as max_order_number,
#   a.max_reordered as max_reordered,
#   a.avg_reordered as avg_reordered,
#   o.order_id,
#   c.cum_days as cum_days,
#   c.last_buy as last_buy,
#   o.order_number,
#   o.order_dow,
#   o.order_hour_of_day,
#   o.days_since_prior_order,
#   o.order_number - c.order_number as order_number_diff,
#   o.days_since_prior_order - c.days_since_prior_order
# FROM
#   (
#   SELECT
#     user_id,
#     aisle_id,
#     max(reordered) as max_reordered,
#     avg(reordered) as avg_reordered,
#     MAX(order_number) - 1 as max_order_number
#   FROM
#     [instacart.df_prior]
#   GROUP BY
#     user_id,
#     aisle_id
#   ) as a
# LEFT OUTER JOIN
#   [instacart.orders] as o
# ON
#   a.user_id = o.user_id AND
#   a.max_order_number = o.order_number
# LEFT OUTER JOIN
#   [instacart.cum_orders] as c
# ON
#   c.order_id = o.order_id
# "
# 
# bq query --max_rows 1  --allow_large_results --destination_table "instacart.last_buy_depart_2" --flatten_results --replace "
# SELECT
#   a.user_id as user_id,
#   a.department_id as department_id,
#   a.max_order_number as max_order_number,
#   a.max_reordered as max_reordered,
#   a.avg_reordered as avg_reordered,
#   o.order_id,
#   c.cum_days as cum_days,
#   c.last_buy as last_buy,
#   o.order_number,
#   o.order_dow,
#   o.order_hour_of_day,
#   o.days_since_prior_order,
#   o.order_number - c.order_number as order_number_diff,
#   o.days_since_prior_order - c.days_since_prior_order
# FROM
#   (
#   SELECT
#     user_id,
#     department_id,
#     max(reordered) as max_reordered,
#     avg(reordered) as avg_reordered,
#     MAX(order_number) - 1 as max_order_number
#   FROM
#     [instacart.df_prior]
#   GROUP BY
#     user_id,
#     department_id
#   ) as a
# LEFT OUTER JOIN
#   [instacart.orders] as o
# ON
#   a.user_id = o.user_id AND
#   a.max_order_number = o.order_number
# LEFT OUTER JOIN
#   [instacart.cum_orders] as c
# ON
#   c.order_id = o.order_id
# "
# 
# ###
# 
# 
# 
# 
# 
# bq query --max_rows 1  --allow_large_results --destination_table "instacart.dmt_user_item_30" --flatten_results --replace "
# SELECT
#   a.user_id as user_id,
#   a.product_id as product_id,
#   count(1) cnt_user_item,
#   EXACT_COUNT_DISTINCT(a.order_id) cnt_user_order,
#   avg(a.order_hour_of_day) avg_order_hour_of_day,
#   min(a.order_hour_of_day) min_order_hour_of_day,
#   max(a.order_hour_of_day) max_order_hour_of_day,
#   max(a.reordered) max_reordered,
#   sum(a.reordered) sum_reordered,
#   avg(a.reordered) avg_reordered,
#   AVG(a.order_dow) as avg_order_dow,
#   MIN(a.order_dow) as min_order_dow,
#   MAX(a.order_dow) as max_order_dow,
#   AVG(a.days_since_prior_order) as avg_days_since_prior_order,
#   MAX(a.days_since_prior_order) as max_days_since_prior_order,
#   MIN(a.days_since_prior_order) as min_days_since_prior_order,
#   sum(CASE WHEN a.order_dow = 0  THEN 1 ELSE 0 END) AS  order_dow_0,
#   sum(CASE WHEN a.order_dow = 1  THEN 1 ELSE 0 END) AS  order_dow_1,
#   sum(CASE WHEN a.order_dow = 2  THEN 1 ELSE 0 END) AS  order_dow_2,
#   sum(CASE WHEN a.order_dow = 3  THEN 1 ELSE 0 END) AS  order_dow_3,
#   sum(CASE WHEN a.order_dow = 4  THEN 1 ELSE 0 END) AS  order_dow_4,
#   sum(CASE WHEN a.order_dow = 5  THEN 1 ELSE 0 END) AS  order_dow_5,
#   sum(CASE WHEN a.order_dow = 6  THEN 1 ELSE 0 END) AS  order_dow_6,
#   avg(CASE WHEN a.order_dow = 0  THEN a.reordered ELSE null END) AS  reorder_dow_0,
#   avg(CASE WHEN a.order_dow = 1  THEN a.reordered ELSE null END) AS  reorder_dow_1,
#   avg(CASE WHEN a.order_dow = 2  THEN a.reordered ELSE null END) AS  reorder_dow_2,
#   avg(CASE WHEN a.order_dow = 3  THEN a.reordered ELSE null END) AS  reorder_dow_3,
#   avg(CASE WHEN a.order_dow = 4  THEN a.reordered ELSE null END) AS  reorder_dow_4,
#   avg(CASE WHEN a.order_dow = 5  THEN a.reordered ELSE null END) AS  reorder_dow_5,
#   avg(CASE WHEN a.order_dow = 6  THEN a.reordered ELSE null END) AS  reorder_dow_6
# FROM
#   [instacart.df_prior] as a
# LEFT OUTER JOIN
#   [instacart.cum_orders] as b
# ON
#   a.order_id = b.order_id
# WHERE
#   b.last_buy <= 30
# GROUP BY
#   user_id, product_id
# "
# 
# bq query --max_rows 1  --allow_large_results --destination_table "instacart.dmt_user_aisle_30" --flatten_results --replace "
# SELECT
#   a.user_id as user_id,
#   a.aisle_id as aisle_id,
#   count(1) cnt_user_aisle,
#   EXACT_COUNT_DISTINCT(a.order_id) cnt_aisle_order,
#   EXACT_COUNT_DISTINCT(a.product_id) cnt_product_order,
#   avg(a.order_hour_of_day) avg_order_hour_of_day,
#   min(a.order_hour_of_day) min_order_hour_of_day,
#   max(a.order_hour_of_day) max_order_hour_of_day,
#   AVG(a.days_since_prior_order) as avg_days_since_prior_order,
#   MAX(a.days_since_prior_order) as max_days_since_prior_order,
#   MIN(a.days_since_prior_order) as min_days_since_prior_order,
#   AVG(order_dow) as avg_order_dow,
#   MAX(order_dow) as max_order_dow,
#   MIN(order_dow) as min_order_dow,
#   max(reordered) max_reordered,
#   sum(reordered) sum_reordered,
#   avg(reordered) avg_reordered
# FROM
#   [instacart.df_prior] as a
# LEFT OUTER JOIN
#   [instacart.cum_orders] as b
# ON
#   a.order_id = b.order_id
# WHERE
#   b.last_buy <= 30
# GROUP BY
#   user_id, aisle_id
# "
# 
# bq query --max_rows 1  --allow_large_results --destination_table "instacart.dmt_user_depart_30" --flatten_results --replace "
# SELECT
#   a.user_id user_id,
#   a.department_id department_id,
#   count(1) cnt_user_depart,
#   EXACT_COUNT_DISTINCT(a.order_id) cnt_depart_order,
#   EXACT_COUNT_DISTINCT(a.product_id) cnt_product_order,
#   EXACT_COUNT_DISTINCT(a.aisle_id) cnt_aisle_order,
#   avg(a.order_hour_of_day) avg_order_hour_of_day,
#   min(a.order_hour_of_day) min_order_hour_of_day,
#   max(a.order_hour_of_day) max_order_hour_of_day,
#   AVG(a.days_since_prior_order) as avg_days_since_prior_order,
#   MAX(a.days_since_prior_order) as max_days_since_prior_order,
#   MIN(a.days_since_prior_order) as min_days_since_prior_order,
#   AVG(order_dow) as avg_order_dow,
#   MAX(order_dow) as max_order_dow,
#   MIN(order_dow) as min_order_dow,
#   max(reordered) max_reordered,
#   sum(reordered) sum_reordered,
#   avg(reordered) avg_reordered
# FROM
#   [instacart.df_prior] as a
# LEFT OUTER JOIN
#   [instacart.cum_orders] as b
# ON
#   a.order_id = b.order_id
# WHERE
#   b.last_buy <= 30
# GROUP BY
#   user_id, department_id
# "
# ###
# 
# 
# ###
# bq query --max_rows 1  --allow_large_results --destination_table "instacart.dmt_user_item" --flatten_results --replace "
# SELECT
#   user_id,
#   product_id,
#   count(1) cnt_user_item,
#   EXACT_COUNT_DISTINCT(order_id) cnt_user_order,
#   avg(order_hour_of_day) avg_order_hour_of_day,
#   min(order_hour_of_day) min_order_hour_of_day,
#   max(order_hour_of_day) max_order_hour_of_day,
#   max(reordered) max_reordered,
#   sum(reordered) sum_reordered,
#   avg(reordered) avg_reordered,
#   AVG(days_since_prior_order) as avg_days_since_prior_order,
#   MAX(days_since_prior_order) as max_days_since_prior_order,
#   MIN(days_since_prior_order) as min_days_since_prior_order,
#   AVG(order_dow) as avg_order_dow,
#   MAX(order_dow) as max_order_dow,
#   MIN(order_dow) as min_order_dow,
#   sum(CASE WHEN order_dow = 0  THEN 1 ELSE 0 END) AS  order_dow_0,
#   sum(CASE WHEN order_dow = 1  THEN 1 ELSE 0 END) AS  order_dow_1,
#   sum(CASE WHEN order_dow = 2  THEN 1 ELSE 0 END) AS  order_dow_2,
#   sum(CASE WHEN order_dow = 3  THEN 1 ELSE 0 END) AS  order_dow_3,
#   sum(CASE WHEN order_dow = 4  THEN 1 ELSE 0 END) AS  order_dow_4,
#   sum(CASE WHEN order_dow = 5  THEN 1 ELSE 0 END) AS  order_dow_5,
#   sum(CASE WHEN order_dow = 6  THEN 1 ELSE 0 END) AS  order_dow_6,
#   avg(CASE WHEN order_dow = 0  THEN reordered ELSE null END) AS  reorder_dow_0,
#   avg(CASE WHEN order_dow = 1  THEN reordered ELSE null END) AS  reorder_dow_1,
#   avg(CASE WHEN order_dow = 2  THEN reordered ELSE null END) AS  reorder_dow_2,
#   avg(CASE WHEN order_dow = 3  THEN reordered ELSE null END) AS  reorder_dow_3,
#   avg(CASE WHEN order_dow = 4  THEN reordered ELSE null END) AS  reorder_dow_4,
#   avg(CASE WHEN order_dow = 5  THEN reordered ELSE null END) AS  reorder_dow_5,
#   avg(CASE WHEN order_dow = 6  THEN reordered ELSE null END) AS  reorder_dow_6
# FROM
#   [instacart.df_prior]
# GROUP BY
#   user_id, product_id
# "
# 
# bq query --max_rows 1  --allow_large_results --destination_table "instacart.dmt_user_aisle" --flatten_results --replace "
# SELECT
#   user_id,
#   aisle_id,
#   count(1) cnt_user_aisle,
#   EXACT_COUNT_DISTINCT(order_id) cnt_aisle_order,
#   EXACT_COUNT_DISTINCT(product_id) cnt_product_order,
#   avg(order_hour_of_day) avg_order_hour_of_day,
#   min(order_hour_of_day) min_order_hour_of_day,
#   max(order_hour_of_day) max_order_hour_of_day,
#   AVG(days_since_prior_order) as avg_days_since_prior_order,
#   MAX(days_since_prior_order) as max_days_since_prior_order,
#   MIN(days_since_prior_order) as min_days_since_prior_order,
#   AVG(order_dow) as avg_order_dow,
#   MAX(order_dow) as max_order_dow,
#   MIN(order_dow) as min_order_dow,
#   max(reordered) max_reordered,
#   sum(reordered) sum_reordered,
#   avg(reordered) avg_reordered
# FROM
#   [instacart.df_prior]
# GROUP BY
#   user_id, aisle_id
# "
# 
# bq query --max_rows 1  --allow_large_results --destination_table "instacart.dmt_user_depart" --flatten_results --replace "
# SELECT
#   user_id,
#   department_id,
#   count(1) cnt_user_depart,
#   EXACT_COUNT_DISTINCT(order_id) cnt_depart_order,
#   EXACT_COUNT_DISTINCT(product_id) cnt_product_order,
#   EXACT_COUNT_DISTINCT(aisle_id) cnt_aisle_order,
#   avg(order_hour_of_day) avg_order_hour_of_day,
#   min(order_hour_of_day) min_order_hour_of_day,
#   max(order_hour_of_day) max_order_hour_of_day,
#   AVG(days_since_prior_order) as avg_days_since_prior_order,
#   MAX(days_since_prior_order) as max_days_since_prior_order,
#   MIN(days_since_prior_order) as min_days_since_prior_order,
#   AVG(order_dow) as avg_order_dow,
#   MAX(order_dow) as max_order_dow,
#   MIN(order_dow) as min_order_dow,
#   max(reordered) max_reordered,
#   sum(reordered) sum_reordered,
#   avg(reordered) avg_reordered
# FROM
#   [instacart.df_prior]
# GROUP BY
#   user_id, department_id
# "
# 
# bq query --max_rows 1  --allow_large_results --destination_table "instacart.user_cart_30" --flatten_results --replace "
# SELECT
#   user_id,
#   count(1) as cnt_order_num,
#   avg(cnt_order) as cnt_cart_num,
#   avg(cnt_item) as avg_cnt_item,
#   avg(cnt_aisle) as avg_cnt_aisle,
#   avg(cnt_depart) as avg_cnt_depart,
#   avg(sum_reordered) as avg_sum_reordered,
#   sum(sum_reordered) as sum_reordered,
#   avg(avg_reordered) as avg_reordered,
#   avg(order_dow) as order_dow,
#   avg(order_hour_of_day) as order_hour_of_day,
#   avg(days_since_prior_order) as days_since_prior_order
# FROM
# (
# SELECT
#   a.user_id as user_id,
#   a.order_id as order_id,
#   count(1) cnt_order,
#   EXACT_COUNT_DISTINCT(a.product_id) cnt_item,
#   EXACT_COUNT_DISTINCT(a.aisle_id) cnt_aisle,
#   EXACT_COUNT_DISTINCT(a.department_id) cnt_depart,
#   sum(a.reordered) sum_reordered,
#   avg(a.reordered) avg_reordered,
#   avg(a.order_dow) as order_dow,
#   avg(a.order_hour_of_day) as order_hour_of_day,
#   avg(a.days_since_prior_order) as days_since_prior_order
# FROM
#   [instacart.df_prior] as a
# LEFT OUTER JOIN
#   [instacart.cum_orders] as b
# ON
#   a.order_id = b.order_id
# WHERE
#   b.last_buy <= 30
# GROUP BY
#   user_id, order_id
# )
# GROUP BY
#   user_id
# "
# 
# ###
# bq query --max_rows 20  --allow_large_results --destination_table "instacart.diff_user_item_30" --flatten_results --replace "
# SELECT
#   user_id,
#   product_id,
#   CASE WHEN STDDEV(diffs - last_buy) is not NULL THEN STDDEV(diffs - last_buy) ELSE -1 END as std_diffs,
#   CASE WHEN NTH(501, QUANTILES(diffs - last_buy, 1001)) is not NULL THEN NTH(501, QUANTILES(diffs - last_buy, 1001)) ELSE -1 END as med_diffs,
#   CASE WHEN avg(diffs - last_buy) is not NULL THEN avg(diffs - last_buy) ELSE -1 END as avg_diffs,
#   CASE WHEN min(diffs - last_buy) is not NULL THEN min(diffs - last_buy) ELSE -1 END as min_diffs,
#   CASE WHEN max(diffs - last_buy) is not NULL THEN max(diffs - last_buy) ELSE -1 END as max_diffs  
# FROM
# (
# SELECT
#   a.user_id as user_id,
#   a.product_id as product_id,
#   LAG(b.last_buy, 1) OVER (PARTITION BY a.user_id, a.product_id ORDER BY a.order_number) as diffs,
#   b.last_buy as last_buy
# FROM
#   [instacart.df_prior] as a
# LEFT OUTER JOIN
#   [instacart.cum_orders] as b
# ON
#   a.order_id = b.order_id
# WHERE
#   b.last_buy <= 30
# ) as s
# GROUP BY
#   user_id, product_id
# "
# 
# 
# bq query --max_rows 20  --allow_large_results --destination_table "instacart.diff_user_item" --flatten_results --replace "
# SELECT
#   user_id,
#   product_id,
#   CASE WHEN STDDEV(diffs - last_buy) is not NULL THEN STDDEV(diffs - last_buy) ELSE -1 END as std_diffs,
#   CASE WHEN NTH(501, QUANTILES(diffs - last_buy, 1001)) is not NULL THEN NTH(501, QUANTILES(diffs - last_buy, 1001)) ELSE -1 END as med_diffs,
#   CASE WHEN avg(diffs - last_buy) is not NULL THEN avg(diffs - last_buy) ELSE -1 END as avg_diffs,
#   CASE WHEN min(diffs - last_buy) is not NULL THEN min(diffs - last_buy) ELSE -1 END as min_diffs,
#   CASE WHEN max(diffs - last_buy) is not NULL THEN max(diffs - last_buy) ELSE -1 END as max_diffs  
# FROM
# (
# SELECT
#   a.user_id as user_id,
#   a.product_id as product_id,
#   LAG(b.last_buy, 1) OVER (PARTITION BY a.user_id, a.product_id ORDER BY a.order_number) as diffs,
#   b.last_buy as last_buy
# FROM
#   [instacart.df_prior] as a
# LEFT OUTER JOIN
#   [instacart.cum_orders] as b
# ON
#   a.order_id = b.order_id
# ) as s
# GROUP BY
#   user_id, product_id
# "
# ###
# 
# ###
# bq query --max_rows 20  --allow_large_results --destination_table "instacart.diff_user_aisle_30" --flatten_results --replace "
# SELECT
#   user_id,
#   aisle_id,
#   CASE WHEN STDDEV(diffs - last_buy) is not NULL THEN STDDEV(diffs - last_buy) ELSE -1 END as std_diffs,
#   CASE WHEN NTH(501, QUANTILES(diffs - last_buy, 1001)) is not NULL THEN NTH(501, QUANTILES(diffs - last_buy, 1001)) ELSE -1 END as med_diffs,
#   CASE WHEN avg(diffs - last_buy) is not NULL THEN avg(diffs - last_buy) ELSE -1 END as avg_diffs,
#   CASE WHEN min(diffs - last_buy) is not NULL THEN min(diffs - last_buy) ELSE -1 END as min_diffs,
#   CASE WHEN max(diffs - last_buy) is not NULL THEN max(diffs - last_buy) ELSE -1 END as max_diffs  
# FROM
# (
# SELECT
#   a.user_id as user_id,
#   a.aisle_id as aisle_id,
#   LAG(b.last_buy, 1) OVER (PARTITION BY a.user_id, a.aisle_id ORDER BY a.order_number) as diffs,
#   b.last_buy as last_buy
# FROM
#   [instacart.df_prior] as a
# LEFT OUTER JOIN
#   [instacart.cum_orders] as b
# ON
#   a.order_id = b.order_id
# WHERE
#   b.last_buy <= 30
# ) as s
# GROUP BY
#   user_id, aisle_id
# "
# 
# 
# bq query --max_rows 20  --allow_large_results --destination_table "instacart.diff_user_aisle" --flatten_results --replace "
# SELECT
#   user_id,
#   aisle_id,
#   CASE WHEN STDDEV(diffs - last_buy) is not NULL THEN STDDEV(diffs - last_buy) ELSE -1 END as std_diffs,
#   CASE WHEN NTH(51, QUANTILES(diffs - last_buy, 101)) is not NULL THEN NTH(51, QUANTILES(diffs - last_buy, 101)) ELSE -1 END as med_diffs,
#   CASE WHEN avg(diffs - last_buy) is not NULL THEN avg(diffs - last_buy) ELSE -1 END as avg_diffs,
#   CASE WHEN min(diffs - last_buy) is not NULL THEN min(diffs - last_buy) ELSE -1 END as min_diffs,
#   CASE WHEN max(diffs - last_buy) is not NULL THEN max(diffs - last_buy) ELSE -1 END as max_diffs  
# FROM
# (
# SELECT
#   a.user_id as user_id,
#   a.aisle_id as aisle_id,
#   LAG(b.last_buy, 1) OVER (PARTITION BY a.user_id, a.aisle_id ORDER BY a.order_number) as diffs,
#   b.last_buy as last_buy
# FROM
#   [instacart.df_prior] as a
# LEFT OUTER JOIN
#   [instacart.cum_orders] as b
# ON
#   a.order_id = b.order_id
# ) as s
# GROUP BY
#   user_id, aisle_id
# "
# ###
# 
# ###
# bq query --max_rows 20  --allow_large_results --destination_table "instacart.diff_user_depart_30" --flatten_results --replace "
# SELECT
#   user_id,
#   department_id,
#   CASE WHEN STDDEV(diffs - last_buy) is not NULL THEN STDDEV(diffs - last_buy) ELSE -1 END as std_diffs,
#   CASE WHEN NTH(501, QUANTILES(diffs - last_buy, 1001)) is not NULL THEN NTH(501, QUANTILES(diffs - last_buy, 1001)) ELSE -1 END as med_diffs,
#   CASE WHEN avg(diffs - last_buy) is not NULL THEN avg(diffs - last_buy) ELSE -1 END as avg_diffs,
#   CASE WHEN min(diffs - last_buy) is not NULL THEN min(diffs - last_buy) ELSE -1 END as min_diffs,
#   CASE WHEN max(diffs - last_buy) is not NULL THEN max(diffs - last_buy) ELSE -1 END as max_diffs  
# FROM
# (
# SELECT
#   a.user_id as user_id,
#   a.department_id as department_id,
#   LAG(b.last_buy, 1) OVER (PARTITION BY a.user_id, a.department_id ORDER BY a.order_number) as diffs,
#   b.last_buy as last_buy
# FROM
#   [instacart.df_prior] as a
# LEFT OUTER JOIN
#   [instacart.cum_orders] as b
# ON
#   a.order_id = b.order_id
# WHERE
#   b.last_buy <= 30
# ) as s
# GROUP BY
#   user_id, department_id
# "
# 
# 
# bq query --max_rows 20  --allow_large_results --destination_table "instacart.diff_user_depart" --flatten_results --replace "
# SELECT
#   user_id,
#   department_id,
#   CASE WHEN STDDEV(diffs - last_buy) is not NULL THEN STDDEV(diffs - last_buy) ELSE -1 END as std_diffs,
#   CASE WHEN NTH(501, QUANTILES(diffs - last_buy, 1001)) is not NULL THEN NTH(501, QUANTILES(diffs - last_buy, 1001)) ELSE -1 END as med_diffs,
#   CASE WHEN avg(diffs - last_buy) is not NULL THEN avg(diffs - last_buy) ELSE -1 END as avg_diffs,
#   CASE WHEN min(diffs - last_buy) is not NULL THEN min(diffs - last_buy) ELSE -1 END as min_diffs,
#   CASE WHEN max(diffs - last_buy) is not NULL THEN max(diffs - last_buy) ELSE -1 END as max_diffs  
# FROM
# (
# SELECT
#   a.user_id as user_id,
#   a.department_id as department_id,
#   LAG(b.last_buy, 1) OVER (PARTITION BY a.user_id, a.department_id ORDER BY a.order_number) as diffs,
#   b.last_buy as last_buy
# FROM
#   [instacart.df_prior] as a
# LEFT OUTER JOIN
#   [instacart.cum_orders] as b
# ON
#   a.order_id = b.order_id
# ) as s
# GROUP BY
#   user_id, department_id
# "
# ###
# 
# bq query --max_rows 20  --allow_large_results --destination_table "instacart.diff_user_30" --flatten_results --replace "
# SELECT
#   user_id,
#   CASE WHEN STDDEV(diffs - last_buy) is not NULL THEN STDDEV(diffs - last_buy) ELSE -1 END as std_diffs,
#   CASE WHEN NTH(501, QUANTILES(diffs - last_buy, 1001)) is not NULL THEN NTH(501, QUANTILES(diffs - last_buy, 1001)) ELSE -1 END as med_diffs,
#   CASE WHEN avg(diffs - last_buy) is not NULL THEN avg(diffs - last_buy) ELSE -1 END as avg_diffs,
#   CASE WHEN min(diffs - last_buy) is not NULL THEN min(diffs - last_buy) ELSE -1 END as min_diffs,
#   CASE WHEN max(diffs - last_buy) is not NULL THEN max(diffs - last_buy) ELSE -1 END as max_diffs  
# FROM
# (
# SELECT
#   a.user_id as user_id,
#   a.product_id as product_id,
#   LAG(b.last_buy, 1) OVER (PARTITION BY a.user_id, a.product_id ORDER BY a.order_number) as diffs,
#   b.last_buy as last_buy
# FROM
#   [instacart.df_prior] as a
# LEFT OUTER JOIN
#   [instacart.cum_orders] as b
# ON
#   a.order_id = b.order_id
# WHERE
#   b.last_buy <= 30
# ) as s
# GROUP BY
#   user_id
# "
# 
# bq query --max_rows 20  --allow_large_results --destination_table "instacart.diff_user" --flatten_results --replace "
# SELECT
#   user_id,
#   CASE WHEN STDDEV(diffs - last_buy) is not NULL THEN STDDEV(diffs - last_buy) ELSE -1 END as std_diffs,
#   CASE WHEN NTH(501, QUANTILES(diffs - last_buy, 1001)) is not NULL THEN NTH(501, QUANTILES(diffs - last_buy, 1001)) ELSE -1 END as med_diffs,
#   CASE WHEN avg(diffs - last_buy) is not NULL THEN avg(diffs - last_buy) ELSE -1 END as avg_diffs,
#   CASE WHEN min(diffs - last_buy) is not NULL THEN min(diffs - last_buy) ELSE -1 END as min_diffs,
#   CASE WHEN max(diffs - last_buy) is not NULL THEN max(diffs - last_buy) ELSE -1 END as max_diffs  
# FROM
# (
# SELECT
#   a.user_id as user_id,
#   a.product_id as product_id,
#   LAG(b.last_buy, 1) OVER (PARTITION BY a.user_id, a.product_id ORDER BY a.order_number) as diffs,
#   b.last_buy as last_buy
# FROM
#   [instacart.df_prior] as a
# LEFT OUTER JOIN
#   [instacart.cum_orders] as b
# ON
#   a.order_id = b.order_id
# ) as s
# GROUP BY
#   user_id
# "
# 
# ###
# bq query --max_rows 20  --allow_large_results --destination_table "instacart.diff_item_30" --flatten_results --replace "
# SELECT
#   product_id,
#   CASE WHEN STDDEV(diffs - last_buy) is not NULL THEN STDDEV(diffs - last_buy) ELSE -1 END as std_diffs,
#   CASE WHEN NTH(501, QUANTILES(diffs - last_buy, 1001)) is not NULL THEN NTH(501, QUANTILES(diffs - last_buy, 1001)) ELSE -1 END as med_diffs,
#   CASE WHEN avg(diffs - last_buy) is not NULL THEN avg(diffs - last_buy) ELSE -1 END as avg_diffs,
#   CASE WHEN min(diffs - last_buy) is not NULL THEN min(diffs - last_buy) ELSE -1 END as min_diffs,
#   CASE WHEN max(diffs - last_buy) is not NULL THEN max(diffs - last_buy) ELSE -1 END as max_diffs  
# FROM
# (
# SELECT
#   a.user_id as user_id,
#   a.product_id as product_id,
#   LAG(b.last_buy, 1) OVER (PARTITION BY a.user_id, a.product_id ORDER BY a.order_number) as diffs,
#   b.last_buy as last_buy
# FROM
#   [instacart.df_prior] as a
# LEFT OUTER JOIN
#   [instacart.cum_orders] as b
# ON
#   a.order_id = b.order_id
# WHERE
#   b.last_buy <= 30
# ) as s
# GROUP BY
#   product_id
# "
# 
# 
# bq query --max_rows 20  --allow_large_results --destination_table "instacart.diff_item" --flatten_results --replace "
# SELECT
#   product_id,
#   CASE WHEN STDDEV(diffs - last_buy) is not NULL THEN STDDEV(diffs - last_buy) ELSE -1 END as std_diffs,
#   CASE WHEN NTH(501, QUANTILES(diffs - last_buy, 1001)) is not NULL THEN NTH(501, QUANTILES(diffs - last_buy, 1001)) ELSE -1 END as med_diffs,
#   CASE WHEN avg(diffs - last_buy) is not NULL THEN avg(diffs - last_buy) ELSE -1 END as avg_diffs,
#   CASE WHEN min(diffs - last_buy) is not NULL THEN min(diffs - last_buy) ELSE -1 END as min_diffs,
#   CASE WHEN max(diffs - last_buy) is not NULL THEN max(diffs - last_buy) ELSE -1 END as max_diffs  
# FROM
# (
# SELECT
#   a.user_id as user_id,
#   a.product_id as product_id,
#   LAG(b.last_buy, 1) OVER (PARTITION BY a.user_id, a.product_id ORDER BY a.order_number) as diffs,
#   b.last_buy as last_buy
# FROM
#   [instacart.df_prior] as a
# LEFT OUTER JOIN
#   [instacart.cum_orders] as b
# ON
#   a.order_id = b.order_id
# ) as s
# GROUP BY
#   product_id
# "
# ######
# 
# bq query --max_rows 20  --allow_large_results --destination_table "instacart.diff_aisle_30" --flatten_results --replace "
# SELECT
#   aisle_id,
#   CASE WHEN STDDEV(diffs - last_buy) is not NULL THEN STDDEV(diffs - last_buy) ELSE -1 END as std_diffs,
#   CASE WHEN NTH(501, QUANTILES(diffs - last_buy, 1001)) is not NULL THEN NTH(501, QUANTILES(diffs - last_buy, 1001)) ELSE -1 END as med_diffs,
#   CASE WHEN avg(diffs - last_buy) is not NULL THEN avg(diffs - last_buy) ELSE -1 END as avg_diffs,
#   CASE WHEN min(diffs - last_buy) is not NULL THEN min(diffs - last_buy) ELSE -1 END as min_diffs,
#   CASE WHEN max(diffs - last_buy) is not NULL THEN max(diffs - last_buy) ELSE -1 END as max_diffs
# FROM
# (
# SELECT
#   a.user_id as user_id,
#   a.aisle_id as aisle_id,
#   LAG(b.last_buy, 1) OVER (PARTITION BY a.user_id, a.aisle_id ORDER BY a.order_number) as diffs,
#   b.last_buy as last_buy
# FROM
#   [instacart.df_prior] as a
# LEFT OUTER JOIN
#   [instacart.cum_orders] as b
# ON
#   a.order_id = b.order_id
# WHERE
#   b.last_buy <= 30
# ) as s
# GROUP BY
#   aisle_id
# "
# 
# 
# bq query --max_rows 20  --allow_large_results --destination_table "instacart.diff_depart_30" --flatten_results --replace "
# SELECT
#   department_id,
#   CASE WHEN STDDEV(diffs - last_buy) is not NULL THEN STDDEV(diffs - last_buy) ELSE -1 END as std_diffs,
#   CASE WHEN NTH(501, QUANTILES(diffs - last_buy, 1001)) is not NULL THEN NTH(501, QUANTILES(diffs - last_buy, 1001)) ELSE -1 END as med_diffs,
#   CASE WHEN avg(diffs - last_buy) is not NULL THEN avg(diffs - last_buy) ELSE -1 END as avg_diffs,
#   CASE WHEN min(diffs - last_buy) is not NULL THEN min(diffs - last_buy) ELSE -1 END as min_diffs,
#   CASE WHEN max(diffs - last_buy) is not NULL THEN max(diffs - last_buy) ELSE -1 END as max_diffs
# FROM
# (
# SELECT
#   a.user_id as user_id,
#   a.department_id as department_id,
#   LAG(b.last_buy, 1) OVER (PARTITION BY a.user_id, a.department_id ORDER BY a.order_number) as diffs,
#   b.last_buy as last_buy
# FROM
#   [instacart.df_prior] as a
# LEFT OUTER JOIN
#   [instacart.cum_orders] as b
# ON
#   a.order_id = b.order_id
# WHERE
#   b.last_buy <= 30
# ) as s
# GROUP BY
#    department_id
# "
# 
# 
# bq query --max_rows 20  --allow_large_results --destination_table "instacart.diff_user_item_reordered_30" --flatten_results --replace "
# SELECT
#   user_id,
#   product_id,
#   CASE WHEN STDDEV(diffs - last_buy) is not NULL THEN STDDEV(diffs - last_buy) ELSE -1 END as std_diffs,
#   CASE WHEN NTH(501, QUANTILES(diffs - last_buy, 1001)) is not NULL THEN NTH(501, QUANTILES(diffs - last_buy, 1001)) ELSE -1 END as med_diffs,
#   CASE WHEN avg(diffs - last_buy) is not NULL THEN avg(diffs - last_buy) ELSE -1 END as avg_diffs,
#   CASE WHEN min(diffs - last_buy) is not NULL THEN min(diffs - last_buy) ELSE -1 END as min_diffs,
#   CASE WHEN max(diffs - last_buy) is not NULL THEN max(diffs - last_buy) ELSE -1 END as max_diffs
# FROM
# (
# SELECT
#   a.user_id as user_id,
#   a.product_id as product_id,
#   LAG(b.last_buy, 1) OVER (PARTITION BY a.user_id, a.product_id ORDER BY a.order_number) as diffs,
#   b.last_buy as last_buy
# FROM
#   [instacart.df_prior] as a
# LEFT OUTER JOIN
#   [instacart.cum_orders] as b
# ON
#   a.order_id = b.order_id
# WHERE
#   b.last_buy <= 30 and reordered = 1
# ) as s
# GROUP BY
#   user_id, product_id
# "
# 
# ###
# 
# bq query --max_rows 20  --allow_large_results --destination_table "instacart.user_item_recent_reordered" --flatten_results --replace "
# SELECT
#   a.user_id as user_id,
#   a.product_id as product_id,
#   AVG(CASE WHEN b.last_buy <=7 THEN reordered ELSE 0 END) as ui_under7,
#   AVG(CASE WHEN b.last_buy > 7 AND b.last_buy <= 14 THEN reordered ELSE 0 END) as ui_under14,
#   AVG(CASE WHEN b.last_buy > 14 AND b.last_buy <= 21 THEN reordered ELSE 0 END) as ui_under21,
#   AVG(CASE WHEN b.last_buy > 21 AND b.last_buy <= 28 THEN reordered ELSE 0 END) as ui_under28,
#   AVG(CASE WHEN b.last_buy > 28 THEN reordered ELSE 0 END) as ui_over28
# FROM
#   [instacart.df_prior] as a
# LEFT OUTER JOIN
#   [instacart.cum_orders] as b
# ON
#   a.order_id = b.order_id
# GROUP BY
#   user_id, product_id
# "
# ####
# bq query --max_rows 20  --allow_large_results --destination_table "instacart.user_aisle_recent_reordered" --flatten_results --replace "
# SELECT
#   a.user_id as user_id,
#   a.aisle_id as aisle_id,
#   AVG(CASE WHEN b.last_buy <=7 THEN reordered ELSE 0 END) as ui_under7,
#   AVG(CASE WHEN b.last_buy > 7 AND b.last_buy <= 14 THEN reordered ELSE 0 END) as ui_under14,
#   AVG(CASE WHEN b.last_buy > 14 AND b.last_buy <= 21 THEN reordered ELSE 0 END) as ui_under21,
#   AVG(CASE WHEN b.last_buy > 21 AND b.last_buy <= 28 THEN reordered ELSE 0 END) as ui_under28,
#   AVG(CASE WHEN b.last_buy > 28 THEN reordered ELSE 0 END) as ui_over28
# FROM
#   [instacart.df_prior] as a
# LEFT OUTER JOIN
#   [instacart.cum_orders] as b
# ON
#   a.order_id = b.order_id
# GROUP BY
#   user_id, aisle_id
# "
# 
# 
# bq query --max_rows 20  --allow_large_results --destination_table "instacart.user_depart_recent_reordered" --flatten_results --replace "
# SELECT
#   a.user_id as user_id,
#   a.department_id as department_id,
#   AVG(CASE WHEN b.last_buy <=7 THEN reordered ELSE 0 END) as ui_under7,
#   AVG(CASE WHEN b.last_buy > 7 AND b.last_buy <= 14 THEN reordered ELSE 0 END) as ui_under14,
#   AVG(CASE WHEN b.last_buy > 14 AND b.last_buy <= 21 THEN reordered ELSE 0 END) as ui_under21,
#   AVG(CASE WHEN b.last_buy > 21 AND b.last_buy <= 28 THEN reordered ELSE 0 END) as ui_under28,
#   AVG(CASE WHEN b.last_buy > 28 THEN reordered ELSE 0 END) as ui_over28
# FROM
#   [instacart.df_prior] as a
# LEFT OUTER JOIN
#   [instacart.cum_orders] as b
# ON
#   a.order_id = b.order_id
# GROUP BY
#   user_id, department_id
# "
# 
# bq query --max_rows 20  --allow_large_results --destination_table "instacart.aisle_recent_reordered" --flatten_results --replace "
# SELECT
#   a.aisle_id as aisle_id,
#   AVG(CASE WHEN b.last_buy <=7 THEN reordered ELSE 0 END) as ui_under7,
#   AVG(CASE WHEN b.last_buy > 7 AND b.last_buy <= 14 THEN reordered ELSE 0 END) as ui_under14,
#   AVG(CASE WHEN b.last_buy > 14 AND b.last_buy <= 21 THEN reordered ELSE 0 END) as ui_under21,
#   AVG(CASE WHEN b.last_buy > 21 AND b.last_buy <= 28 THEN reordered ELSE 0 END) as ui_under28,
#   AVG(CASE WHEN b.last_buy > 28 THEN reordered ELSE 0 END) as ui_over28
# FROM
#   [instacart.df_prior] as a
# LEFT OUTER JOIN
#   [instacart.cum_orders] as b
# ON
#   a.order_id = b.order_id
# GROUP BY
#   aisle_id
# "
# 
# 
# bq query --max_rows 20  --allow_large_results --destination_table "instacart.depart_recent_reordered" --flatten_results --replace "
# SELECT
#   a.department_id as department_id,
#   AVG(CASE WHEN b.last_buy <=7 THEN reordered ELSE 0 END) as ui_under7,
#   AVG(CASE WHEN b.last_buy > 7 AND b.last_buy <= 14 THEN reordered ELSE 0 END) as ui_under14,
#   AVG(CASE WHEN b.last_buy > 14 AND b.last_buy <= 21 THEN reordered ELSE 0 END) as ui_under21,
#   AVG(CASE WHEN b.last_buy > 21 AND b.last_buy <= 28 THEN reordered ELSE 0 END) as ui_under28,
#   AVG(CASE WHEN b.last_buy > 28 THEN reordered ELSE 0 END) as ui_over28
# FROM
#   [instacart.df_prior] as a
# LEFT OUTER JOIN
#   [instacart.cum_orders] as b
# ON
#   a.order_id = b.order_id
# GROUP BY
#   department_id
# "
# 
# 
# 
# ####
# 
# 
# bq query --max_rows 20  --allow_large_results --destination_table "instacart.item_recent_reordered" --flatten_results --replace "
# SELECT
#   a.product_id as product_id,
#   AVG(CASE WHEN b.last_buy <=7 THEN reordered ELSE 0 END) as i_under7,
#   AVG(CASE WHEN b.last_buy > 7 AND b.last_buy <= 14 THEN reordered ELSE 0 END) as i_under14,
#   AVG(CASE WHEN b.last_buy > 14 AND b.last_buy <= 21 THEN reordered ELSE 0 END) as i_under21,
#   AVG(CASE WHEN b.last_buy > 21 AND b.last_buy <= 28 THEN reordered ELSE 0 END) as i_under28,
#   AVG(CASE WHEN b.last_buy > 28 THEN reordered ELSE 0 END) as i_over28
# FROM
#   [instacart.df_prior] as a
# LEFT OUTER JOIN
#   [instacart.cum_orders] as b
# ON
#   a.order_id = b.order_id
# GROUP BY
#   product_id
# "
# 
# bq query --max_rows 20  --allow_large_results --destination_table "instacart.user_recent_reordered" --flatten_results --replace "
# SELECT
#   a.user_id as user_id,
#   AVG(CASE WHEN b.last_buy <=7 THEN reordered ELSE 0 END) as u_under7,
#   AVG(CASE WHEN b.last_buy > 7 AND b.last_buy <= 14 THEN reordered ELSE 0 END) as u_under14,
#   AVG(CASE WHEN b.last_buy > 14 AND b.last_buy <= 21 THEN reordered ELSE 0 END) as u_under21,
#   AVG(CASE WHEN b.last_buy > 21 AND b.last_buy <= 28 THEN reordered ELSE 0 END) as u_under28,
#   AVG(CASE WHEN b.last_buy > 28 THEN reordered ELSE 0 END) as u_over28
# FROM
#   [instacart.df_prior] as a
# LEFT OUTER JOIN
#   [instacart.cum_orders] as b
# ON
#   a.order_id = b.order_id
# GROUP BY
#   user_id
# "
# 
# 
# bq query --max_rows 1  --maximum_billing_tier 3 --allow_large_results --destination_table "instacart.dmt_train_only_rebuy" --flatten_results --replace "
# SELECT
#   CASE WHEN tr.reordered is not null THEN tr.reordered ELSE 0 END as target,
#   o.user_id,
#   p.aisle_id,
#   p.department_id,
#   o.product_id,
#   o.order_id,
#   o.order_number,
#   o.order_dow,
#   o.order_hour_of_day,
#   o.days_since_prior_order,
#   o.cum_days,
#   du.*,
#   dui3.*,
#   dd.*,
#   du3.*,
#   da3.*,
#   dd3.*,
#   udd.*,
#   udd3.*,
#   di.*,
#   di3.*,
#   ddi3.*,
#   ddd3.*,
#   dai3.*,
#   dddi3.*,
#   u.*,
#   u2.*,
#   uc.*,
#   i.*,
#   i2.*,
#   l.*,
#   la.*,
#   ld.*,
#   l2.*,
#   la2.*,
#   ld2.*,
#   ui.*,
#   ua.*,
#   ud.*,
#   ui3.*,
#   ua3.*,
#   ud3.*,
#   rui.*,
#   ru.*,
#   ri.*
# FROM
#   [instacart.only_rebuy_train] as o
# LEFT OUTER JOIN
#   [instacart.products] as p
# ON
#   o.product_id = p.product_id
# LEFT OUTER JOIN
#   [instacart.diff_user_item] as du
# ON
#   du.user_id = o.user_id AND  o.product_id = du.product_id
# LEFT OUTER JOIN
#   [instacart.diff_user_item_reordered_30] as dui3
# ON
#   dui3.user_id = o.user_id AND  o.product_id = dui3.product_id
# LEFT OUTER JOIN
#   [instacart.diff_user_aisle_reordered_30] as dai3
# ON
#   dai3.user_id = o.user_id AND p.aisle_id = dai3.aisle_id
# LEFT OUTER JOIN
#   [instacart.diff_user_depart_reordered_30] as dddi3
# ON
#   dddi3.user_id = o.user_id AND  p.department_id = dddi3.department_id
# LEFT OUTER JOIN
#   [instacart.diff_user_depart] as dd
# ON
#   dd.user_id = o.user_id AND  p.product_id = dd.department_id
# LEFT OUTER JOIN
#   [instacart.diff_user_item_30] as du3
# ON
#   du3.user_id = o.user_id AND  o.product_id = du3.product_id
# LEFT OUTER JOIN
#   [instacart.diff_user_aisle_30] as da3
# ON
#   da3.user_id = o.user_id AND  p.aisle_id = da3.aisle_id
# LEFT OUTER JOIN
#   [instacart.diff_user_depart_30] as dd3
# ON
#   dd3.user_id = o.user_id AND  p.product_id = dd3.department_id
# LEFT OUTER JOIN
#   [instacart.diff_user] as udd
# ON
#   udd.user_id = o.user_id
# LEFT OUTER JOIN
#   [instacart.diff_user_30] as udd3
# ON
#   udd3.user_id = o.user_id
# LEFT OUTER JOIN
#   [instacart.diff_item] as di
# ON
#   di.product_id = o.product_id
# LEFT OUTER JOIN
#   [instacart.diff_item_30] as di3
# ON
#   di3.product_id = o.product_id
# LEFT OUTER JOIN
#   [instacart.diff_aisle_30] as ddi3
# ON
#   ddi3.aisle_id = p.aisle_id
# LEFT OUTER JOIN
#   [instacart.diff_depart_30] as ddd3
# ON
#   ddd3.department_id = p.department_id
# LEFT OUTER JOIN
#   [instacart.dmt_user] as u
# ON
#   u.u1_user_id = o.user_id
# LEFT OUTER JOIN
#   [instacart.dmt_user2_30] as u2
# ON
#   u2.u1_user_id = o.user_id
# LEFT OUTER JOIN
#   [instacart.user_cart_30] as uc
# ON
#   uc.user_id = o.user_id
# LEFT OUTER JOIN
#   [instacart.dmt_item] as i
# ON
#   i.i1_product_id = o.product_id
# LEFT OUTER JOIN
#   [instacart.dmt_item2_30] as i2
# ON
#   i2.i1_product_id = o.product_id
# LEFT OUTER JOIN
#   [instacart.last_buy] as l
# ON
#   l.user_id = o.user_id AND l.product_id = o.product_id
# LEFT OUTER JOIN
#   [instacart.last_buy_aisle] as la
# ON
#   la.user_id = o.user_id AND la.aisle_id = p.aisle_id
# LEFT OUTER JOIN
#   [instacart.last_buy_depart] as ld
# ON
#   ld.user_id = o.user_id AND ld.department_id = p.department_id
# LEFT OUTER JOIN
#   [instacart.last_buy_2] as l2
# ON
#   l2.user_id = o.user_id AND l2.product_id = o.product_id
# LEFT OUTER JOIN
#   [instacart.last_buy_aisle_2] as la2
# ON
#   la2.user_id = o.user_id AND la2.aisle_id = p.aisle_id
# LEFT OUTER JOIN
#   [instacart.last_buy_depart_2] as ld2
# ON
#   ld2.user_id = o.user_id AND ld2.department_id = p.department_id
# LEFT OUTER JOIN
#   [instacart.dmt_user_item] as ui
# ON
#   ui.user_id = o.user_id AND ui.product_id = o.product_id
# LEFT OUTER JOIN
#   [instacart.dmt_user_aisle] as ua
# ON
#   ua.user_id = o.user_id AND ua.aisle_id = p.aisle_id
# LEFT OUTER JOIN
#   [instacart.dmt_user_depart] as ud
# ON
#   ud.user_id = o.user_id AND ud.department_id = p.department_id
# LEFT OUTER JOIN
#   [instacart.dmt_user_item_30] as ui3
# ON
#   ui3.user_id = o.user_id AND ui3.product_id = o.product_id
# LEFT OUTER JOIN
#   [instacart.dmt_user_aisle_30] as ua3
# ON
#   ua3.user_id = o.user_id AND ua3.aisle_id = p.aisle_id
# LEFT OUTER JOIN
#   [instacart.dmt_user_depart_30] as ud3
# ON
#   ud3.user_id = o.user_id AND ud3.department_id = p.department_id
# LEFT OUTER JOIN
#   [instacart.user_item_recent_reordered] as rui
# ON
#   rui.user_id = o.user_id AND rui.product_id = o.product_id
# LEFT OUTER JOIN
#   [instacart.user_recent_reordered] as ru
# ON
#   ru.user_id = o.user_id
# LEFT OUTER JOIN
#   [instacart.item_recent_reordered] as ri
# ON
#   ri.product_id = o.product_id
# LEFT OUTER JOIN
#   [instacart.df_train] as tr
# ON
#   tr.user_id = o.user_id AND tr.product_id = o.product_id AND tr.order_id = o.order_id
# "
# 
# 
# 
# bq query --maximum_billing_tier 2 --max_rows 1  --allow_large_results --destination_table "instacart.dmt_test_only_rebuy" --flatten_results --replace "
# SELECT
#   o.user_id,
#   p.aisle_id,
#   p.department_id,
#   o.product_id,
#   o.order_id,
#   o.order_number,
#   o.order_dow,
#   o.order_hour_of_day,
#   o.days_since_prior_order,
#   o.cum_days,
#   du.*,
#   dui3.*,
#   dd.*,
#   du3.*,
#   da3.*,
#   dd3.*,
#   udd.*,
#   udd3.*,
#   di.*,
#   di3.*,
#   ddi3.*,
#   ddd3.*,
#   dai3.*,
#   dddi3.*,
#   u.*,
#   u2.*,
#   uc.*,
#   i.*,
#   i2.*,
#   l.*,
#   la.*,
#   ld.*,
#   l2.*,
#   la2.*,
#   ld2.*,
#   ui.*,
#   ua.*,
#   ud.*,
#   ui3.*,
#   ua3.*,
#   ud3.*,
#   rui.*,
#   ru.*,
#   ri.*
# FROM
#   [instacart.only_rebuy_test] as o
# LEFT OUTER JOIN
#   [instacart.products] as p
# ON
#   o.product_id = p.product_id
# LEFT OUTER JOIN
#   [instacart.diff_user_item] as du
# ON
#   du.user_id = o.user_id AND  o.product_id = du.product_id
# LEFT OUTER JOIN
#   [instacart.diff_user_item_reordered_30] as dui3
# ON
#   dui3.user_id = o.user_id AND  o.product_id = dui3.product_id
# LEFT OUTER JOIN
#   [instacart.diff_user_aisle_reordered_30] as dai3
# ON
#   dai3.user_id = o.user_id AND p.aisle_id = dai3.aisle_id
# LEFT OUTER JOIN
#   [instacart.diff_user_depart_reordered_30] as dddi3
# ON
#   dddi3.user_id = o.user_id AND  p.department_id = dddi3.department_id
# LEFT OUTER JOIN
#   [instacart.diff_user_depart] as dd
# ON
#   dd.user_id = o.user_id AND  p.product_id = dd.department_id
# LEFT OUTER JOIN
#   [instacart.diff_user_item_30] as du3
# ON
#   du3.user_id = o.user_id AND  o.product_id = du3.product_id
# LEFT OUTER JOIN
#   [instacart.diff_user_aisle_30] as da3
# ON
#   da3.user_id = o.user_id AND  p.aisle_id = da3.aisle_id
# LEFT OUTER JOIN
#   [instacart.diff_user_depart_30] as dd3
# ON
#   dd3.user_id = o.user_id AND  p.product_id = dd3.department_id
# LEFT OUTER JOIN
#   [instacart.diff_user] as udd
# ON
#   udd.user_id = o.user_id
# LEFT OUTER JOIN
#   [instacart.diff_user_30] as udd3
# ON
#   udd3.user_id = o.user_id
# LEFT OUTER JOIN
#   [instacart.diff_item] as di
# ON
#   di.product_id = o.product_id
# LEFT OUTER JOIN
#   [instacart.diff_item_30] as di3
# ON
#   di3.product_id = o.product_id
# LEFT OUTER JOIN
#   [instacart.diff_aisle_30] as ddi3
# ON
#   ddi3.aisle_id = p.aisle_id
# LEFT OUTER JOIN
#   [instacart.diff_depart_30] as ddd3
# ON
#   ddd3.department_id = p.department_id
# LEFT OUTER JOIN
#   [instacart.dmt_user] as u
# ON
#   u.u1_user_id = o.user_id
# LEFT OUTER JOIN
#   [instacart.dmt_user2_30] as u2
# ON
#   u2.u1_user_id = o.user_id
# LEFT OUTER JOIN
#   [instacart.user_cart_30] as uc
# ON
#   uc.user_id = o.user_id
# LEFT OUTER JOIN
#   [instacart.dmt_item] as i
# ON
#   i.i1_product_id = o.product_id
# LEFT OUTER JOIN
#   [instacart.dmt_item2_30] as i2
# ON
#   i2.i1_product_id = o.product_id
# LEFT OUTER JOIN
#   [instacart.last_buy] as l
# ON
#   l.user_id = o.user_id AND l.product_id = o.product_id
# LEFT OUTER JOIN
#   [instacart.last_buy_aisle] as la
# ON
#   la.user_id = o.user_id AND la.aisle_id = p.aisle_id
# LEFT OUTER JOIN
#   [instacart.last_buy_depart] as ld
# ON
#   ld.user_id = o.user_id AND ld.department_id = p.department_id
# LEFT OUTER JOIN
#   [instacart.last_buy_2] as l2
# ON
#   l2.user_id = o.user_id AND l2.product_id = o.product_id
# LEFT OUTER JOIN
#   [instacart.last_buy_aisle_2] as la2
# ON
#   la2.user_id = o.user_id AND la2.aisle_id = p.aisle_id
# LEFT OUTER JOIN
#   [instacart.last_buy_depart_2] as ld2
# ON
#   ld2.user_id = o.user_id AND ld2.department_id = p.department_id
# LEFT OUTER JOIN
#   [instacart.dmt_user_item] as ui
# ON
#   ui.user_id = o.user_id AND ui.product_id = o.product_id
# LEFT OUTER JOIN
#   [instacart.dmt_user_aisle] as ua
# ON
#   ua.user_id = o.user_id AND ua.aisle_id = p.aisle_id
# LEFT OUTER JOIN
#   [instacart.dmt_user_depart] as ud
# ON
#   ud.user_id = o.user_id AND ud.department_id = p.department_id
# LEFT OUTER JOIN
#   [instacart.dmt_user_item_30] as ui3
# ON
#   ui3.user_id = o.user_id AND ui3.product_id = o.product_id
# LEFT OUTER JOIN
#   [instacart.dmt_user_aisle_30] as ua3
# ON
#   ua3.user_id = o.user_id AND ua3.aisle_id = p.aisle_id
# LEFT OUTER JOIN
#   [instacart.dmt_user_depart_30] as ud3
# ON
#   ud3.user_id = o.user_id AND ud3.department_id = p.department_id
# LEFT OUTER JOIN
#   [instacart.df_train] as tr
# ON
#   tr.user_id = o.user_id AND tr.product_id = o.product_id AND tr.order_id = o.order_id
# LEFT OUTER JOIN
#   [instacart.user_item_recent_reordered] as rui
# ON
#   rui.user_id = o.user_id AND rui.product_id = o.product_id
# LEFT OUTER JOIN
#   [instacart.user_recent_reordered] as ru
# ON
#   ru.user_id = o.user_id
# LEFT OUTER JOIN
#   [instacart.item_recent_reordered] as ri
# ON
#   ri.product_id = o.product_id
# "
# 
# gsutil -m rm -r gs://kaggle-instacart-takami/data/
# 
# bq extract --compression GZIP instacart.dmt_train_only_rebuy gs://kaggle-instacart-takami/data/dmt_train_only_rebuy/data*.csv.gz
# bq extract --compression GZIP instacart.dmt_test_only_rebuy gs://kaggle-instacart-takami/data/dmt_test_only_rebuy/data*.csv.gz
# 
# rm -rf ../data/
# gsutil -m cp -r gs://kaggle-instacart-takami/data/ ../
# 
# ```
