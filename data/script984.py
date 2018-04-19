
# coding: utf-8

# There are four clusters when one plots train 'y' with train prediction. In Scirpus' [__original script__][1] this is pretty clear, yet the color-coding for y-values doesn't show all that well.
# 
# 
#   [1]: https://www.kaggle.com/scirpus/four-blob-tsne "original script"

# In[ ]:


import numpy as np 
import pandas as pd 
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
get_ipython().run_line_magic('matplotlib', 'inline')


# Different color map.
# 

# In[ ]:


cm = plt.cm.get_cmap('RdYlBu')


# Five new features were added: X29, X48, X232, X236 and X263. All of them were found by genetic programming, just like the original set of Scirpus' features. I think this is justified as the scores will be better in the end. __X48 and X236 were subsequently removed.__

# In[ ]:


features = ['X118',
            'X127',
            'X47',
            'X315',
            'X311',
            'X179',
            'X314',
### added by Tilii
            'X232',
            'X29',
            'X263',
###
            'X261']


# In Scirpus' [__original script__][1] the whole y-range is used, so the color-coding gets stretched because of the >250 outlier. Therefore, most of the y-values end up in the bottom half of the color range and the whole plot is just various shades of blue that are difficult to tell apart. In this script I clip y-values so that everything above 130 will be the same shade of BLUE.
# 
# 
#   [1]: https://www.kaggle.com/scirpus/four-blob-tsne "original script"

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
y_clip = np.clip(train['y'].values, a_min=None, a_max=130)


# In[ ]:


tsne = TSNE(random_state=2016,perplexity=50,verbose=2)
x = tsne.fit_transform(pd.concat([train[features],test[features]]))


# In[ ]:


plt.figure(figsize=(12,10))
# plt.scatter(x[train.shape[0]:,0],x[train.shape[0]:,1], cmap=cm, marker='.', s=15, label='test')
cb = plt.scatter(x[:train.shape[0],0],x[:train.shape[0],1], c=y_clip, cmap=cm, marker='o', s=15, label='train')
plt.colorbar(cb)
plt.legend(prop={'size':15})
#plt.title('t-SNE embedding of train & test data', fontsize=20)
plt.title('t-SNE embedding of train data', fontsize=20)
plt.savefig('four-blob-tSNE-01.png')


# Pretty sure that the plot in this notebook will not look the same as my local plot. Not being paranoid - this is based on my experience with t-SNE Kaggle implementation from this [__script__][1]. Anyway, this is how my local plot looks like using the same script as in this notebook.
# 
# ![__t-SNE on raw data__][2]
# 
# Given that this t-SNE embedding was done on raw data, I think it compares quite nicely with the t-SNE embedding below which was done after neural network training. Four clusters should be obvious in both plots. One could even argue that there is a small 5th cluster.
# 
# ![__t-SNE after neural network training__][3]
# 
# 
#   [1]: https://www.kaggle.com/tilii7/you-want-outliers-we-got-them-outliers
#   [2]: https://i.imgur.com/JpmDztu.png
#   [3]: https://i.imgur.com/wRuOZkO.png

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold


# In[ ]:


score = 0
splits = 5
kf = KFold(n_splits=splits)
y = train.y.ravel()
for train_index, test_index in kf.split(range(train.shape[0])):
    blind = x[:train.shape[0]][test_index]
    vis = x[:train.shape[0]][train_index]
    knn = KNeighborsRegressor(n_neighbors=80,weights='uniform',p=2)
    knn.fit(vis,y[train_index])
    score +=(r2_score(y[test_index],(knn.predict(blind))))
print(score/splits)


# In[ ]:


score = 0
splits = 5
kf = KFold(n_splits=splits)
y = train.y.ravel()
for train_index, test_index in kf.split(range(train.shape[0])):
    blind = train[features].loc[test_index]
    vis = train[features].loc[train_index]
    knn = KNeighborsRegressor(n_neighbors=80,weights='uniform',p=2)
    knn.fit(vis,y[train_index])
    score +=(r2_score(y[test_index],(knn.predict(blind))))
print(score/splits)

