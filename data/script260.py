
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

with pd.HDFStore("../input/train.h5", "r") as train:
    df = train.get('train')


# # 缺失值统计 missing value statistics

# In[ ]:


m, n = df.shape
miss_count = []
for col in df.columns:
    x = df[col].isnull().sum()
    miss_count.append(x)
miss_count_rate = np.array(miss_count) / m


# In[ ]:


fig, ax = plt.subplots(figsize=(8, 25))
ind = np.arange(n)
ax.barh(ind, miss_count_rate, color='y')
plt.yticks(ind+0.4, df.columns)
ax.set_xlabel('miss_count_rate in each col')
ax.set_title('miss_count_rate in each col')


# In[ ]:


df = df.drop(df.columns[miss_count_rate > 0.3], axis=1)


# # 异常值处理 Exception value processing

# In[ ]:


# 通过箱线图去除异常值之后，查看分布 observe hist of data within boxplot range
m, n = df.shape
col = df.columns
plt.figure(figsize=(8, 50))
k = 0
for i in range(2, n):
    k += 1
    col_ = df[col[i]][df[col[i]].notnull()]
    q_high = col_.quantile(0.75)
    q_low = col_.quantile(0.25)
    iqr = (q_high - q_low) * 1.5
    high = q_high + iqr
    low = q_low -iqr
    col_ = col_[(col_ < high) & (col_ > low)]
    plt.subplot(25, 4, k)
    plt.hist(col_, bins=100)
    plt.xticks()
    plt.title(str(i) + ' ' + col[i])
    plt.tight_layout(pad=0)


# ## 以下特征形态较好 The following is features with fine hist
# ### continuoues value[ 2, 6, 8, 13, 19, 20, 26, 27, 28, 30, 32, 34, 38, 41, 45, 49, 51, 52, 56, 59, 61, 62, 74, 76, 78, 79, 80, 81, 84, 86, 88, 89, 94, 97] 
# ### category0 has two class[60, 63, 66, 67, 70, 72,  82,  96] 
# ### category1 has three class [77, 87]

# In[ ]:


df = df.ix[:, [0, 1, 2, 6, 8, 13, 19, 20, 26, 27, 28, 30, 32, 34, 38, 41, 45, 49, 51,
       52, 56, 59, 61, 62, 74, 76, 78, 79, 80, 81, 84, 86, 88, 89, 94, 97, 60, 
               63, 66, 67, 70, 72, 82, 96, 77, 87, 98]]


# ## 处理cate0特征下的异常值 processing exception value in cate0

# In[ ]:


cate0 = range(36, 44)
col = df.columns
for i in cate0:
    df.ix[:, i] = np.where(df.ix[:, i] < -1, -2, df.ix[:, i])
    df.ix[:, i] = np.where(df.ix[:, i] >= -1, 0, df.ix[:, i])


# ## 去除y的异常值 drop samples which have exception value in y

# In[ ]:


q_high = df.y.quantile(0.75)
q_low = df.y.quantile(0.25)
iqr = (q_high - q_low) * 1.5
high = q_high + iqr
low = q_low -iqr
df = df.drop(df[df.y > high].index)
df = df.drop(df[df.y < low].index)


# In[ ]:


hist_ = plt.hist(df.ix[df['timestamp'] == 0, 'y'], bins=100)


# In[ ]:


hist_ = plt.hist(df.ix[df['id'] == 438, 'y'], bins=100)


# In[ ]:


## 同一timestamp或用一id对应的y分布都是对称的。


# # missing value fill 填充

# In[ ]:


df = df.sort_values(by='y')


# In[ ]:


df = df.fillna(method='ffill')
df = df.fillna(method='bfill') # 防止第一个值为nan


# # 模型 model

# ## 划分测试集和训练集 Divide the test set and the training set

# In[ ]:


test = {'x': [], 'y': [], 'timestamp': []}
time = range(1812, 1802, -1)
for i in range(10):
    df_ = df.ix[df['timestamp']==time[i], :]
    test['x'].append(df_.drop(['y', 'id', 'timestamp'], axis=1))
    test['y'].append(df_['y'])
    test['timestamp'].append(time[i])
df_ = df[df['timestamp'] < 1803]
X = df_.drop(['y', 'id', 'timestamp'], axis=1)
y = df_['y']


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf0 = RandomForestRegressor(n_jobs=-1, verbose=1)
rf0.fit(test['x'][0], test['y'][0])


# In[ ]:


col_ = X.columns
plt.figure(figsize=(8, 16))
ind = np.arange(len(col_))
plt.barh(ind, rf0.feature_importances_)
plt.yticks(ind+0.4, col_)


# ## 类别变量的feature_importances_比较小，是否说明这几个特征没有用？
# ## Feature_importances of features in cate0 and cate1 is small. Does it mean they are less important?

# In[ ]:


rf0.score(test['x'][0], test['y'][0])


# In[ ]:


rf0.score(test['x'][1], test['y'][1])


# ## 严重过拟合？试一下减少特征和增大训练集
# ## Over-fitting?Try to use less features or use more samples to fit modle.

# In[ ]:


# 只采用连续变量进行fit
# Train model with continuous value.
rf1 = RandomForestRegressor(n_jobs=-1, verbose=1)
rf1.fit(test['x'][0].ix[:, range(34)], test['y'][0])
rf1.score(test['x'][0].ix[:, range(34)], test['y'][0])


# In[ ]:


rf1.score(test['x'][1].ix[:, range(34)], test['y'][1])


# In[ ]:


# 采用前10个特征fit. Fit with 10 features.
rf2 = RandomForestRegressor(n_jobs=-1, verbose=1)
rf2.fit(test['x'][0].ix[:, range(10)], test['y'][0])
rf2.score(test['x'][0].ix[:, range(10)], test['y'][0])


# In[ ]:


rf2.score(test['x'][1].ix[:, range(10)], test['y'][1])


# In[ ]:


# 采用第一个特征fit. Fit with one feature
rf3 = RandomForestRegressor(n_jobs=-1, verbose=1)
rf3.fit(test['x'][0].ix[:, range(1)], test['y'][0])
rf3.score(test['x'][0].ix[:, range(1)], test['y'][0])


# In[ ]:


rf3.score(test['x'][1].ix[:, range(1)], test['y'][1])


# In[ ]:


plt.hist(test['y'][0])


# In[ ]:


plt.hist(test['y'][1])


# ### 两个timestamp对于的y分布近似
# ### y hist with defferent timestamps are similar.

# In[ ]:


df_ = df[df['timestamp'] < 100]
X2 = df_.drop(['y', 'id', 'timestamp'], axis=1)
y2 = df_['y']
rf4 = RandomForestRegressor() 
rf4.fit(X2, y2)
rf4.score(X2, y2)


# In[ ]:


score_ = []
for i in range(10):
    score_.append(rf4.score(test['x'][i], test['y'][i]))
plt.plot(range(10), score_)


# ## 还是过拟合 Still over fitting!

# plt.hist(test['x'][0].ix[test['x'][0].ix[:, 0]>-9, 0],bins=100)

# df_id = df.groupby('id')['y'].mean()

# sns.boxplot(df_id)

# df_ = df.ix[:, ['id', 'technical_35', 'derived_0']]
# y_ = df['y']
# rf_ = RandomForestRegressor()
# rf_.fit(df_, y_)
# rf_.score(df_, y_)

# ind = np.arange(3)
# plt.bar(ind, rf_.feature_importances_)
# plt.xticks(ind+0.4, ['id', 'technical_35', 'derived_0'])

# df_ = df.ix[:, ['technical_35', 'derived_0']]
# y_ = df['y']
# rf_ = RandomForestRegressor(n_jobs=-1)
# rf_.fit(df_, y_)
# rf_.score(df_, y_)

# ## id列对y的影响很不确定。

# ## 以上，主要工作是做了特征选取，异常值处理，missing value填充
# ## 选择随机森林模型进行拟合，过拟合现象比较严重。
# ## 欢迎指教。
# ## In this notebook, the main work is to do the feature selection, exception handling, missing value padding.
# ## Using RandomForest to fit data, i find a serious over-fitting phenomenon.
# ## Please give me some suggestions for the whole process. Happy kaggle！
