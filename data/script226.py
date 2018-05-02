
# coding: utf-8

# There has been an ongoing discussion about the feature usefulness (or lack thereof) - see [__here__](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/41487). We also have many feature importance plots to choose from that have been provided by other Kagglers. A tried and true approach to this is [__recursive feature elimination__](https://en.wikipedia.org/wiki/Feature_selection), where we remove N features (1 <= N < total features) at a time and see how it affects our predictions. If the score goes up we toss those features, or keep them if the score gets worse. Here I use sklearn's [__recursive feature elimination wrapped with cross-validation__](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html) because in my experience it provides a very unbiased estimate of feature importance. Random Forest classifier is used primarily for speed, but you can substitute in there any tree-based method that provides information about feature importance either through a coef attribute or through a feature_importances attribute. 
# 
# Sorry about the click-baiting title - the inspiration was [__this famous line__](https://www.youtube.com/watch?v=XT8hE7_8BCY) that just came to me as I was thinking about feature elimination.

# In[ ]:


__author__ = 'Tilii: https://kaggle.com/tilii7'

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        tmin, tsec = divmod((datetime.now() - start_time).total_seconds(), 60)
        print('\n Time taken: %i minutes and %s seconds.' % (tmin, round(tsec, 2)))

train = pd.read_csv('../input/train.csv', dtype={'id': np.int32, 'target': np.int8})
X = train.drop(['id', 'target'], axis=1).values
y = train['target'].values
test = pd.read_csv('../input/test.csv', dtype={'id': np.int32})
X_test = test.drop(['id'], axis=1).values

all_features = [x for x in train.drop(['id', 'target'], axis=1).columns]


# Here we define Random Forest classifier and RFECV parameters. To test the features properly, it is probably a good idea to change n_estimators to 200 and max_depth=20 (or remove max_depth). It will take longer, on the order of 2 hours, if you choose to do so.
# 
# Yet another important parameter is **step**, which specifies how many features are removed at a time. Setting it to 2-5 usually works well, but set it to 1 if you want to be thorough.
# 
# Note that I am specifying n_jobs=4 because Kaggle provides 4 CPUs per job. You may wish to set that to -1 so that all CPUs on your system are used. Also, the whole countdown will go 5 times because we are doing 5-fold cross-validation.

# In[ ]:


folds = 5
step = 2

rfc = RandomForestClassifier(n_estimators=100, max_features='sqrt', max_depth=10, n_jobs=4)

rfecv = RFECV(
              estimator=rfc,
              step=step,
              cv=StratifiedKFold(
                                 n_splits=folds,
                                 shuffle=False,
                                 random_state=1001).split(X,y),
              scoring='roc_auc',
              n_jobs=1,
              verbose=2)


# We estimate the feature importance and time the whole process.

# In[ ]:


starttime = timer(None)
start_time = timer(None)
rfecv.fit(X, y)
timer(start_time)


# Let's summarize the output.

# In[ ]:


print('\n Optimal number of features: %d' % rfecv.n_features_)
sel_features = [f for f, s in zip(all_features, rfecv.support_) if s]
print('\n The selected features are {}:'.format(sel_features))


# Plot number of features vs. CV scores.

# In[ ]:


plt.figure(figsize=(12, 9))
plt.xlabel('Number of features tested x 2')
plt.ylabel('Cross-validation score (AUC)')
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.savefig('Porto-RFECV-01.png', dpi=150)
plt.show()


# Save sorted feature rankings.

# In[ ]:


ranking = pd.DataFrame({'Features': all_features})
ranking['Rank'] = np.asarray(rfecv.ranking_)
ranking.sort_values('Rank', inplace=True)
ranking.to_csv('Porto-RFECV-ranking-01.csv', index=False)


# Make a prediction. This is only a proof-of-principle as the prediction will likely be poor until more optimal parameters are used above.

# In[ ]:


score = round((np.max(rfecv.grid_scores_) * 2 - 1), 5)
test['target'] = rfecv.predict_proba(X_test)[:,1]
test = test[['id', 'target']]
now = datetime.now()
sub_file = 'submission_5fold-RFECV-RandomForest-01_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
print("\n Writing submission file: %s" % sub_file)
test.to_csv(sub_file, index=False)
timer(starttime)

