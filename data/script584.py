
# coding: utf-8

# We know that for the private leaderboard, there will be a Stage 2 where all of our kernels are re-run with ~5x more test data that we have never seen before. Some, including me, have become concerned about run times for their kernels... will we be able to run our kernel in under an hour with 5x more test data? And, more importantly, how long can our Stage 1 kernel run for to be safe for Stage 2?
# 
# Here, I take the approach of creating a version of the Stage 2 test by duplicating the known Stage 1 test data 5x and inserting novel words to simulate the effect on time and memory load that Stage 2 will have. I find that pre-processing time approximately doubles, training time is (obviously) unaffected, and that predict times go up by ~6x.

# In[ ]:


# First let's look at the shapes of the data.

import pandas as pd

train = pd.read_table('../input/train.tsv', engine='c')
stage_1_test = pd.read_table('../input/test.tsv', engine='c')

print("Train shape: ", train.shape)
print("Stage 1 Test shape: ", stage_1_test.shape)


# In[ ]:


# Now let's simulate Stage 2!

import random
import string

# Make a stage 2 test by copying test five times...
test1 = stage_1_test.copy()
test2 = stage_1_test.copy()
test3 = stage_1_test.copy()
test4 = stage_1_test.copy()
test5 = stage_1_test.copy()
stage_2_test = pd.concat([test1, test2, test3, test4, test5], axis=0)

# ...then introduce random new words
def introduce_new_unseen_words(desc):
    desc = desc.split(' ')
    if random.randrange(0, 10) == 0: # 10% chance of adding an unseen word
        new_word = ''.join(random.sample(string.ascii_letters, random.randrange(3, 15)))
        desc.insert(0, new_word)
    return ' '.join(desc)
stage_2_test.item_description = stage_2_test.item_description.apply(introduce_new_unseen_words)

# ...this should be a dataframe that roughly mimics the real Stage 2 in impact on TFIDF
# and other functions.

print("Stage 2 Test shape (guess): ", stage_2_test.shape)
stage_2_test[["name", "item_description"]].head(20)


# In[ ]:


# How large in memory are these objects?
# We can use `sys.getsizeof` to return size in bytes.

import sys

def object_size_in_mb(obj):
    size_in_bytes = sys.getsizeof(obj)
    size_in_kb = size_in_bytes / 1024
    size_in_mb = size_in_kb / 1024
    return size_in_mb

print("Train size: %.2f mb" % object_size_in_mb(train))
print("Stage 1 Test size: %.2f mb" % object_size_in_mb(stage_1_test))
print("Stage 2 Test size (guess): %.2f mb" % object_size_in_mb(stage_2_test))


# So we can see that Stage 2 is guessed to be a little over 5x in memory usage, but should still definitely fit in memory on a 16GB machine. How does this impact code run time? To check, I take [a popular well-scoring kernel](https://www.kaggle.com/tunguz/more-effective-ridge-lgbm-script-lb-0-44341-2/code) by Bojan (with help from others) and run a portion of it (the `Ridge` model) with both the stage 1 test data and the guess from stage 2.

# In[ ]:


# Note: Code adapted from https://www.kaggle.com/tunguz/more-effective-ridge-lgbm-script-lb-0-44341-2/code

import gc
import time
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer

NUM_CATS = 4000
NAME_MIN_DF = 10
MAX_FEATURES_ITEM_DESCRIPTION = 45000

def handle_missing_inplace(dataset):
    dataset['category_name'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)

def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category = dataset['category_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATS]
    dataset.loc[~dataset['category_name'].isin(pop_category), 'category_name'] = 'missing'

def to_categorical(dataset):
    dataset['category_name'] = dataset['category_name'].astype('category')
    dataset['brand_name'] = dataset['brand_name'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')

start_time = time.time()
nrow_train = train.shape[0]
y = np.log1p(train["price"])
merge: pd.DataFrame = pd.concat([train, stage_1_test])
submission: pd.DataFrame = stage_1_test[['test_id']]
handle_missing_inplace(merge)
print('[{}] Finished to handle missing'.format(time.time() - start_time))
cutting(merge)
print('[{}] Finished to cut'.format(time.time() - start_time))
to_categorical(merge)
print('[{}] Finished to convert categorical'.format(time.time() - start_time))
cv = CountVectorizer(min_df=NAME_MIN_DF)
X_name = cv.fit_transform(merge['name'])
print('[{}] Finished count vectorize `name`'.format(time.time() - start_time))
cv = CountVectorizer()
X_category = cv.fit_transform(merge['category_name'])
print('[{}] Finished count vectorize `category_name`'.format(time.time() - start_time))

tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION,
                     ngram_range=(1, 3),
                     stop_words='english')
X_description = tv.fit_transform(merge['item_description'])
print('[{}] Finished TFIDF vectorize `item_description`'.format(time.time() - start_time))

lb = LabelBinarizer(sparse_output=True)
X_brand = lb.fit_transform(merge['brand_name'])
print('[{}] Finished label binarize `brand_name`'.format(time.time() - start_time))
X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],
                                      sparse=True).values)
print('[{}] Finished to get dummies on `item_condition_id` and `shipping`'.format(time.time() - start_time))

sparse_merge = hstack((X_dummies, X_description, X_brand, X_category, X_name)).tocsr()
print('[{}] Finished to create sparse merge'.format(time.time() - start_time))


# In[ ]:


# Same code as above, but now we use our guess at `stage_2_test` and check differences in timing.

start_time = time.time()
nrow_train = train.shape[0]
y = np.log1p(train["price"])
merge_2: pd.DataFrame = pd.concat([train, stage_2_test])
submission_2: pd.DataFrame = stage_2_test[['test_id']]
handle_missing_inplace(merge_2)
print('[{}] Finished to handle missing 2'.format(time.time() - start_time))
cutting(merge_2)
print('[{}] Finished to cut 2'.format(time.time() - start_time))
to_categorical(merge_2)
print('[{}] Finished to convert categorical 2'.format(time.time() - start_time))
cv = CountVectorizer(min_df=NAME_MIN_DF)
X_name_2 = cv.fit_transform(merge_2['name'])
print('[{}] Finished count vectorize `name` 2'.format(time.time() - start_time))
cv = CountVectorizer()
X_category_2 = cv.fit_transform(merge_2['category_name'])
print('[{}] Finished count vectorize `category_name` 2'.format(time.time() - start_time))

tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION,
                     ngram_range=(1, 3),
                     stop_words='english')
X_description_2 = tv.fit_transform(merge_2['item_description'])
print('[{}] Finished TFIDF vectorize `item_description` 2'.format(time.time() - start_time))

lb = LabelBinarizer(sparse_output=True)
X_brand_2 = lb.fit_transform(merge_2['brand_name'])
print('[{}] Finished label binarize `brand_name` 2'.format(time.time() - start_time))
X_dummies_2 = csr_matrix(pd.get_dummies(merge_2[['item_condition_id', 'shipping']],
                                        sparse=True).values)
print('[{}] Finished to get dummies on `item_condition_id` and `shipping` 2'.format(time.time() - start_time))

sparse_merge_2 = hstack((X_dummies_2, X_description_2, X_brand_2, X_category_2, X_name_2)).tocsr()
print('[{}] Finished to create sparse merge 2'.format(time.time() - start_time))


# Here we can see that Stage 1 took 5min52sec (352s), but Stage 2 is estimated to take 10min21sec (621s), which is a little under twice as long, instead of 5x as long.
# 
# Here are the individual times for each step:
# 
# **Handle missing** - **1s** for Stage 1, **2s** for Stage 2 (2x longer)
# 
# **Cut** - **6s** for Stage 1, **15s** for Stage 2 (2.5x longer)
# 
# **Convert categorical** - **1s** for Stage 1, **2s** for Stage 2 (2x longer)
# 
# **Count vectorize `name`** - **13s** for Stage 1, **29s** for Stage 2 (2.2x longer)
#  
# **Count vectorize `category_name`** - **11s** for Stage 1, **26s** for Stage 2 (2.3x longer)
# 
# **TFIDF `item_description`** - **300s** for Stage 1, **500s** for Stage 2 (1.7x longer)
# 
# **Label binarize `brand_name`** - **10s** for Stage 1, **25s** for Stage 2 (2.5x longer)
# 
# **Get dummies** - **5s** for Stage 1, **10s** for Stage 2 (2x longer)
# 
# **Create sparse merge** - **6s** for Stage 1, **13s** for Stage 2 (2.2x longer)
# 
# **Overall** - **352s** for Stage 1, **621s** for Stage 2 (1.8x longer)

# In[ ]:


# How big is the resulting sparse matrix?

print("Stage 1 Sparse Matrix Merge Shape: ", sparse_merge.shape)
print("Stage 2 Sparse Matrix Merge Shape: ", sparse_merge_2.shape)

# It looks like `sys.getsizeof` does not work for sparse matrices, but this does.
# Thanks StackOverflow!
def sparse_df_size_in_mb(sparse_df):
    size_in_bytes = sparse_df.data.nbytes
    size_in_kb = size_in_bytes / 1024
    size_in_mb = size_in_kb / 1024
    return size_in_mb

print("Stage 1 Sparse Matrix Merge Size: %.2f mb" % sparse_df_size_in_mb(sparse_merge))
print("Stage 2 Sparse Matrix Merge Size (guess): %.2f mb" % sparse_df_size_in_mb(sparse_merge_2))


# Another important aspect of the script is not just the pre-processing, but also the model scoring. How does that change in Stage 2?

# In[ ]:


from sklearn.linear_model import Ridge

X = sparse_merge[:nrow_train]
X_test = sparse_merge[nrow_train:]

X_2 = sparse_merge_2[:nrow_train]
X_test_2 = sparse_merge_2[nrow_train:]

start_time = time.time()
model = Ridge(solver="sag", fit_intercept=True, random_state=205, alpha=3)
model.fit(X, y)
fit_finished = time.time()
print("Finished Stage 1 Fit in %.1f sec" % (fit_finished - start_time))
preds = model.predict(X_test)
pred_finished = time.time()
print("Finished Stage 1 Predict in %.1f sec" % (pred_finished - fit_finished))
print("Stage 1 Total in %.1f sec" % (pred_finished - start_time))

start_time_2 = time.time()
model_2 = Ridge(solver="sag", fit_intercept=True, random_state=205, alpha=3)
model_2.fit(X_2, y)
fit_finished_2 = time.time()
print("Finished Stage 2 Fit (guess) in %.1f sec" % (fit_finished_2 - start_time_2))
preds_2 = model_2.predict(X_test_2)
pred_finished_2 = time.time()
print("Finished Stage 2 Predict (guess) in %.1f sec" % (pred_finished_2 - fit_finished_2))
print("Stage 2 Total (guess) in %.1f sec" % (pred_finished_2 - start_time_2))


# The training time is obviously the same for both Stage 1 and Stage 2, because the train data is unchanged. The predict time is 6x as long, but because the `Ridge` model predicts so quickly (under 1sec) this barely matters for our kernel.
# 
# Based on this, I think assuming your pre-processing will roughly double to triple in time and that your predict time will roughly 6x should give you a good benchmark for your script may be safe. You should also assume your `to_csv` for submission will take longer too, though I'm not sure by how much.
# 
# Looking back at [Bojan's kernel](https://www.kaggle.com/tunguz/more-effective-ridge-lgbm-script-lb-0-44341-2/code), the logs show pre-processing takes 5.4min and that predicting takes 0.1sec for the first Ridge, 0.1sec for the second Ridge, 2min18sec for the first LGB, and 1min15sec for the second LGB. Based on my analysis here, I would expect Stage 2 to add an additional ~6min in pre-processing and an extra 21min19sec for predicting, for a total of ~27min added. This would put the completion time at 71min22sec, which is too long for Stage 2.
# 
# This is not to pick on Bojan (thanks for the great kernel!), but to help guide people toward being more successful in Stage 2. Feel free to try to use the `stage_2_test` object from this kernel to run your own experiments or to improve upon this kernel's accuracy!
