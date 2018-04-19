
# coding: utf-8

# In this simple notebook, we will use CatBoost to predict the price using only categorical features.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import catboost as cboost

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df_train = pd.read_csv('../input/train.tsv', sep='\t', index_col='train_id')
df_test = pd.read_csv('../input/test.tsv', sep='\t', index_col='test_id')


# In[ ]:


# Sneak peek on data
df_train.sample(10, random_state=42)


# In[ ]:


# Log price distribution
(df_train.price + 1).apply(np.log10).hist(bins=50);


# In[ ]:


# We only use categorical features in this naive approach
categorical_features = ['item_condition_id', 'category_name', 'brand_name', 'shipping']

df_x_train = df_train[categorical_features].copy()
df_x_test = df_test[categorical_features].copy()
df_y_log = np.log(df_train['price']+1)


# In[ ]:


# Factorize both train and test (avoid unseen categories in train)
# def factorize(train, test, col, min_count):
#     cat_ids = sorted(set(train[col].dropna().unique()) | set(test[col].dropna().unique()))

#     cat_ids = {k:i for i, k in enumerate(cat_ids)}
#     cat_ids[np.nan] = -1

#     train[col] = train[col].map(cat_ids)
#     test[col]  = test[col].map(cat_ids)
def factorize(train, test, col, min_count):
    train_cat_count = train[col].value_counts()
    test_cat_count = test[col].value_counts()
    
    train_cat = set(train_cat_count[(train_cat_count >= min_count)].index)

    cat_ids = {k:i for i, k in enumerate(sorted(train_cat))}
    cat_ids[np.nan] = -1
    
    train[col] = train[col].map(cat_ids)
    train[col] = train[col].fillna(len(cat_ids))  # Create 'other' category

    test[col] = test[col].map(cat_ids)
    test[col] = test[col].fillna(len(cat_ids))

# Factorize string columns
factorize(df_x_train, df_x_test, 'category_name', min_count=50)
factorize(df_x_train, df_x_test, 'brand_name', min_count=50)


# In[ ]:


df_x_train.nunique()


# In[ ]:


# Create train and test Pool of train
ptrain = cboost.Pool(df_x_train, df_y_log, cat_features=np.arange(len(categorical_features)),
                     column_description=categorical_features)

ptest = cboost.Pool(df_x_test, cat_features=np.arange(len(categorical_features)),
                     column_description=categorical_features)

# Add subsample of train for cross-validation speed
# sub_idx = np.random.choice(len(df_x_train), int(len(df_x_train) * 0.5), replace=False)
# ptrain_sub = cboost.Pool(df_x_train.iloc[sub_idx], df_y_log.iloc[sub_idx],
#                      cat_features=np.arange(len(categorical_features)),
#                      column_description=categorical_features)


# In[ ]:


# Tune your parameters here!
cboost_params = {
    'nan_mode': 'Min',
    'loss_function': 'RMSE',  # Try 'LogLinQuantile' as well
    'iterations': 200,
    'learning_rate': 1.0,
    'depth': 11,
    'verbose': True
}

best_iter = cboost_params['iterations']  # Initial 'guess' it not using CV

# cv_result = cboost.cv(cboost_params, ptrain_sub, fold_count=3)

# df_cv_result = pd.DataFrame({'train': cv_result['RMSE_train_avg'],
#                              'valid': cv_result['RMSE_test_avg']})

# # Best results
# print('Best results:')
# best_iter = df_cv_result.valid.argmin()+1
# df_cv_bestresult = df_cv_result.iloc[best_iter-1]
# print(df_cv_bestresult)

# fig, ax = plt.subplots(1, 2, figsize=(15, 6))
# df_cv_result.plot(ax=ax[0])

# ax[1].plot(df_cv_result.train, df_cv_result.valid, 'o-')
# ax[1].scatter([df_cv_bestresult['train']], [df_cv_bestresult['valid']], c='red')
# ax[1].set_xlabel('train')
# ax[1].set_ylabel('valid')


# In[ ]:


# Train model on full data
model = cboost.CatBoostRegressor(**dict(cboost_params, verbose=False, iterations=best_iter))

fit_model = model.fit(ptrain)


# In[ ]:


# Predict test and save to .csv
df_test['price_log'] = fit_model.predict(ptest).clip(0)  # Avoid negative prices

df_test['price'] = np.exp(df_test['price_log'])-1

df_test[['price']].round(5).to_csv('submission.csv', index=True)


# In[ ]:


get_ipython().system('head submission.csv')

