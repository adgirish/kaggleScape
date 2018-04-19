
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('time', '', 'import numpy as np\nimport pandas as pd\nfrom sklearn import *\nimport lightgbm as lgb\nimport random\n\ntrain = pd.read_json("../input/statoil-iceberg-classifier-challenge/train.json").fillna(-1.0).replace(\'na\', -1.0)\ntest = pd.read_json("../input/statoil-iceberg-classifier-challenge/test.json").fillna(-1.0).replace(\'na\', -1.0)\ntrain[\'angle_l\'] = train[\'inc_angle\'].apply(lambda x: len(str(x))) <= 7\ntest[\'angle_l\'] = test[\'inc_angle\'].apply(lambda x: len(str(x))) <= 7\ntrain[\'null_angle\'] = (train[\'inc_angle\']==-1).values\ntest[\'null_angle\'] = (test[\'inc_angle\']==-1).values\nx1 = train[train[\'inc_angle\']!= -1.0]\nx2 = train[train[\'inc_angle\']== -1.0]\ndel train;\nprint(x1.values.shape, x2.values.shape)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'pca_b1 = decomposition.PCA(n_components=50, whiten=False, random_state=12)\npca_b2 = decomposition.PCA(n_components=50, whiten=False, random_state=13)\netc = ensemble.ExtraTreesRegressor(n_estimators=200, max_depth=7, n_jobs=-1, random_state=14)\n\nband1 = [np.array(band).astype(np.float32).flatten() for band in x1["band_1"]]\nband2 = [np.array(band).astype(np.float32).flatten() for band in x1["band_2"]]\nband1 = pd.DataFrame(pca_b1.fit_transform(band1))\nband1.columns = [str(c)+\'_1\' for c in band1.columns]\nband2 = pd.DataFrame(pca_b2.fit_transform(band2))\nband2.columns = [str(c)+\'_2\' for c in band2.columns]\nfeatures = pd.concat((band1, band2), axis=1, ignore_index=True)\netc.fit(features, x1.inc_angle)\n\nband1 = [np.array(band).astype(np.float32).flatten() for band in x2["band_1"]]\nband2 = [np.array(band).astype(np.float32).flatten() for band in x2["band_2"]]\nband1 = pd.DataFrame(pca_b1.transform(band1))\nband1.columns = [str(c)+\'_1\' for c in band1.columns]\nband2 = pd.DataFrame(pca_b2.fit_transform(band2))\nband2.columns = [str(c)+\'_2\' for c in band2.columns]\nfeatures = pd.concat((band1, band2), axis=1, ignore_index=True)\nx2[\'inc_angle\'] = etc.predict(features)\n\ntrain = pd.concat((x1, x2), axis=0, ignore_index=True).reset_index(drop=True)\ndel x1; del x2;\nprint(train.values.shape)\ntrain.head()')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'pca_b1 = decomposition.PCA(n_components=50, whiten=True, random_state=15)\npca_b2 = decomposition.PCA(n_components=50, whiten=True, random_state=16)\npca_b3 = decomposition.PCA(n_components=50, whiten=True, random_state=17)\npca_b4 = decomposition.PCA(n_components=50, whiten=True, random_state=18)\n\nband1 = [np.array(band).astype(np.float32).flatten() for band in train["band_1"]]\nband2 = [np.array(band).astype(np.float32).flatten() for band in train["band_2"]]\npd_band1 = pd.DataFrame(band1)\npd_band2 = pd.DataFrame(band2)\npd_band3 = pd.DataFrame(np.dot(np.diag(train[\'inc_angle\'].values), ((pd_band1 + pd_band2) / 2)))\npd_band4 = pd.DataFrame(np.dot(np.diag(train[\'inc_angle\'].values), ((pd_band1 - pd_band2) / 2)))\nband1 = pd.DataFrame(pca_b1.fit_transform(pd_band1))\nband1.columns = [str(c)+\'_1\' for c in band1.columns]\nband2 = pd.DataFrame(pca_b2.fit_transform(pd_band2))\nband2.columns = [str(c)+\'_2\' for c in band2.columns]\nband3 = pd.DataFrame(pca_b3.fit_transform(pd_band3.values))\nband3.columns = [str(c)+\'_3\' for c in band3.columns]\nband4 = pd.DataFrame(pca_b4.fit_transform(pd_band4.values))\nband4.columns = [str(c)+\'_4\' for c in band4.columns]\nfeatures = pd.concat((band1, band2, band3, band4), axis=1, ignore_index=True).reset_index(drop=True)\nfeatures[\'inc_angle\'] = train[\'inc_angle\']\nfeatures[\'angle_l\'] = train[\'angle_l\']\nfeatures[\'null_angle\'] = train[\'null_angle\']\nfeatures[\'band1_min\'] = pd_band1.min(axis=1, numeric_only=True)\nfeatures[\'band2_min\'] = pd_band2.min(axis=1, numeric_only=True)\nfeatures[\'band3_min\'] = pd_band3.min(axis=1, numeric_only=True)\nfeatures[\'band4_min\'] = pd_band4.min(axis=1, numeric_only=True)\nfeatures[\'band1_max\'] = pd_band1.max(axis=1, numeric_only=True)\nfeatures[\'band2_max\'] = pd_band2.max(axis=1, numeric_only=True)\nfeatures[\'band3_max\'] = pd_band3.max(axis=1, numeric_only=True)\nfeatures[\'band4_max\'] = pd_band4.max(axis=1, numeric_only=True)\nfeatures[\'band1_med\'] = pd_band1.median(axis=1, numeric_only=True)\nfeatures[\'band2_med\'] = pd_band2.median(axis=1, numeric_only=True)\nfeatures[\'band3_med\'] = pd_band3.median(axis=1, numeric_only=True)\nfeatures[\'band4_med\'] = pd_band4.median(axis=1, numeric_only=True)\nfeatures[\'band1_mea\'] = pd_band1.mean(axis=1, numeric_only=True)\nfeatures[\'band2_mea\'] = pd_band2.mean(axis=1, numeric_only=True)\nfeatures[\'band3_mea\'] = pd_band3.mean(axis=1, numeric_only=True)\nfeatures[\'band4_mea\'] = pd_band4.mean(axis=1, numeric_only=True)\ndel pd_band1; del pd_band2; del pd_band3; del pd_band4\nfeatures1 = features.copy()\nfeatures.tail()')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'band1 = [np.array(band).astype(np.float32).flatten() for band in test["band_1"]]\nband2 = [np.array(band).astype(np.float32).flatten() for band in test["band_2"]]\npd_band1 = pd.DataFrame(band1)\npd_band2 = pd.DataFrame(band2)\npd_band3 = pd.DataFrame(np.dot(np.diag(test[\'inc_angle\'].values), ((pd_band1 + pd_band2) / 2)))\npd_band4 = pd.DataFrame(np.dot(np.diag(test[\'inc_angle\'].values), ((pd_band1 - pd_band2) / 2)))\nband1 = pd.DataFrame(pca_b1.transform(pd_band1))\nband1.columns = [str(c)+\'_1\' for c in band1.columns]\nband2 = pd.DataFrame(pca_b2.transform(pd_band2))\nband2.columns = [str(c)+\'_2\' for c in band2.columns]\nband3 = pd.DataFrame(pca_b3.transform(pd_band3.values))\nband3.columns = [str(c)+\'_3\' for c in band3.columns]\nband4 = pd.DataFrame(pca_b4.fit_transform(pd_band4.values))\nband4.columns = [str(c)+\'_4\' for c in band4.columns]\nfeatures = pd.concat((band1, band2, band3, band4), axis=1, ignore_index=True).reset_index(drop=True)\nfeatures[\'inc_angle\'] = test[\'inc_angle\']\nfeatures[\'angle_l\'] = test[\'angle_l\']\nfeatures[\'null_angle\'] = test[\'null_angle\']\nfeatures[\'band1_min\'] = pd_band1.min(axis=1, numeric_only=True)\nfeatures[\'band2_min\'] = pd_band2.min(axis=1, numeric_only=True)\nfeatures[\'band3_min\'] = pd_band3.min(axis=1, numeric_only=True)\nfeatures[\'band4_min\'] = pd_band4.min(axis=1, numeric_only=True)\nfeatures[\'band1_max\'] = pd_band1.max(axis=1, numeric_only=True)\nfeatures[\'band2_max\'] = pd_band2.max(axis=1, numeric_only=True)\nfeatures[\'band3_max\'] = pd_band3.max(axis=1, numeric_only=True)\nfeatures[\'band4_max\'] = pd_band4.max(axis=1, numeric_only=True)\nfeatures[\'band1_med\'] = pd_band1.median(axis=1, numeric_only=True)\nfeatures[\'band2_med\'] = pd_band2.median(axis=1, numeric_only=True)\nfeatures[\'band3_med\'] = pd_band3.median(axis=1, numeric_only=True)\nfeatures[\'band4_med\'] = pd_band4.median(axis=1, numeric_only=True)\nfeatures[\'band1_mea\'] = pd_band1.mean(axis=1, numeric_only=True)\nfeatures[\'band2_mea\'] = pd_band2.mean(axis=1, numeric_only=True)\nfeatures[\'band3_mea\'] = pd_band3.mean(axis=1, numeric_only=True)\nfeatures[\'band4_mea\'] = pd_band4.mean(axis=1, numeric_only=True)\ndel pd_band1; del pd_band2; del pd_band3\nfeatures2 = features.copy()\nfeatures.tail()')


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nlgb_models = []\n#xgb_models = []\ntest['is_iceberg'] = 0.\nfold = 5\nfor i in range(fold):\n    np.random.seed(i)\n    random.seed(i)\n    x1, x2, y1, y2 = model_selection.train_test_split(features1.astype(float), train['is_iceberg'].values, test_size=0.2, random_state=i)\n\n    #print('XGB...', i)\n    #params = {'eta': 0.02, 'max_depth': 4, 'objective': 'multi:softprob', 'eval_metric': 'mlogloss', 'num_class': 2, 'seed': i, 'silent': True}\n    #watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]\n    #xgb_models.append(xgb.train(params, xgb.DMatrix(x1, y1), 2000,  watchlist, verbose_eval=500, early_stopping_rounds=200))\n\n    print('LightGBM...', i)\n    params = {'learning_rate': 0.02, 'max_depth': 7, 'boosting_type': 'gbdt', 'objective': 'multiclass', 'metric' : 'multi_logloss', 'is_training_metric': True, 'num_class': 2, 'seed': i}\n    lgb_models.append(lgb.train(params, lgb.Dataset(x1, label=y1), 2000, lgb.Dataset(x2, label=y2), verbose_eval=500, early_stopping_rounds=200))\n    \n    #test['is_iceberg'] += xgb_models[i].predict(xgb.DMatrix(features2), ntree_limit=xgb_models[i].best_ntree_limit)[:, 1]\n    test['is_iceberg'] += lgb_models[i].predict(features2, num_iteration=lgb_models[i].best_iteration)[:, 1]")


# In[ ]:


test['is_iceberg'] = test['is_iceberg'].clip(0.+1e-15,1.-1e-15)
test[['id','is_iceberg']].to_csv("submission.csv", index=False)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame({'imp': lgb_models[0].feature_importance(importance_type='gain'), 'col':features2.columns})
df = df.sort_values(['imp','col'], ascending=[True, False])[:30]
_ = df.plot(kind='barh', x='col', y='imp', figsize=(7,12))


# In[ ]:


df1 = pd.read_csv('submission.csv')
df2 = pd.read_csv('../input/explore-stacking-another-hi-lo-and-clip-probs/stack_minmax_bestbase.csv')
df2.columns = [x+'_' if x not in ['id'] else x for x in df2.columns]
blend = pd.merge(df1, df2, how='left', on='id')
for c in df1.columns:
    if c != 'id':
        blend[c] = (blend[c] * 0.5)  + (blend[c+'_'] * 0.5)
blend = blend[df1.columns]
blend['is_iceberg'] = blend['is_iceberg'].clip(0.+1e-15,1.-1e-15)
blend.to_csv('blend1.csv', index=False)


# In[ ]:


df1 = pd.read_csv('blend1.csv')
df2 = pd.read_csv('../input/explore-stacking-another-hi-lo-and-clip-probs/stack_minmax_bestbase.csv')
df2.columns = [x+'_' if x not in ['id'] else x for x in df2.columns]
blend = pd.merge(df1, df2, how='left', on='id')
for c in df1.columns:
    if c != 'id':
        blend[c] = (blend[c]  + blend[c+'_'])/2 + np.sqrt(blend[c] * blend[c+'_'])
blend = blend[df1.columns]
blend['is_iceberg'] = blend['is_iceberg'].clip(0.+1e-15,1.-1e-15)
blend.to_csv('blend2.csv', index=False)

