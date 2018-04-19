
# coding: utf-8

# In this notebook, I tried to explore various dimensionality reduction techniques available. 
# 
# ### Motivation: ###
# 
#  -  I have seen in many kernels people using various dimensionality reduction techniques(DR's) to improve the score (CV/LB). 
# So have created this kernel to using various DR techniques and ran them on multiple regression algorithms like xgboost, lightgbm, ElasticNet, and DecisionTree. 
# 
# Dimensionality reduction techniques used:
# 
#  -  [Principal Component Analysis \[PCA\]][1]
#  -  [Independent Component Analysis \[ICA\]][2]
#  -  [Truncated SVD \[TSVD\]][3]
#  -  [Gaussian Random Projection \[GRP\]][4]
#  -  [Sparse Random Projection \[SRP\]][5]
#  -  [Non-negative Matrix factorization \[NMF\]][6]
#  -  [Feature Agglomeration \[FAG\]][7]
# 
# 
#   [1]: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
#   [2]: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html
#   [3]: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
#   [4]: http://scikit-learn.org/stable/modules/generated/sklearn.random_projection.GaussianRandomProjection.html
#   [5]: http://scikit-learn.org/stable/modules/generated/sklearn.random_projection.SparseRandomProjection.html
#   [6]: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
#   [7]: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.FeatureAgglomeration.html

# In[ ]:


# importing all the necessary modules

import pandas as pd
import numpy as np
import random
from itertools import combinations

import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
import lightgbm as lgb

from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.cluster import FeatureAgglomeration

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge

from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import pickle

import warnings
warnings.filterwarnings('ignore')

random.seed(1729)


# ### Loading and preparing the data ###

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# removing the outlier
train = train.loc[train['y'] < 170, :]

# seperating label and features
y_train = train['y']
train = train.drop('y', axis=1)

# label encoding the categorical features for dimension reduction
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))


# ### Creating the components using various dimensionality reduction techniques ###

# In[ ]:


n_comp = 12

# tSVD
tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
tsvd_results_train = tsvd.fit_transform(train)
tsvd_results_test = tsvd.transform(test)

# PCA
pca = PCA(n_components=n_comp, random_state=420)
pca2_results_train = pca.fit_transform(train)
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp, random_state=420)
ica2_results_train = ica.fit_transform(train)
ica2_results_test = ica.transform(test)

# GRP
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
grp_results_train = grp.fit_transform(train)
grp_results_test = grp.transform(test)

# SRP
srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
srp_results_train = srp.fit_transform(train)
srp_results_test = srp.transform(test)

# NMF
nmf = NMF(n_components=n_comp, init='nndsvdar', random_state=420)
nmf_results_train = nmf.fit_transform(train)
nmf_results_test = nmf.transform(test)

# FAG
fag = FeatureAgglomeration(n_clusters=n_comp, linkage='ward')
fag_results_train = fag.fit_transform(train)
fag_results_test = fag.transform(test)


# ### Filtering the most significant components and inserting in a Dataframe ###

# In[ ]:


dim_reds = list()
train_pca = pd.DataFrame()
test_pca = pd.DataFrame()

train_ica = pd.DataFrame()
test_ica = pd.DataFrame()

train_tsvd = pd.DataFrame()
test_tsvd = pd.DataFrame()

train_grp = pd.DataFrame()
test_grp = pd.DataFrame()

train_srp = pd.DataFrame()
test_srp = pd.DataFrame()

train_nmf = pd.DataFrame()
test_nmf = pd.DataFrame()

train_fag = pd.DataFrame()
test_fag = pd.DataFrame()


for i in range(1, n_comp + 1):
    train_pca['pca_' + str(i)] = pca2_results_train[:, i - 1]
    test_pca['pca_' + str(i)] = pca2_results_test[:, i - 1]

    train_ica['ica_' + str(i)] = ica2_results_train[:, i - 1]
    test_ica['ica_' + str(i)] = ica2_results_test[:, i - 1]

    train_tsvd['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
    test_tsvd['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

    train_grp['grp_' + str(i)] = grp_results_train[:, i - 1]
    test_grp['grp_' + str(i)] = grp_results_test[:, i - 1]

    train_srp['srp_' + str(i)] = srp_results_train[:, i - 1]
    test_srp['srp_' + str(i)] = srp_results_test[:, i - 1]
    
    train_nmf['nmf_' + str(i)] = nmf_results_train[:, i - 1]
    test_nmf['nmf_' + str(i)] = nmf_results_test[:, i - 1]
    
    train_fag['fag_' + str(i)] = fag_results_train[:, i - 1]
    test_fag['fag_' + str(i)] = fag_results_test[:, i - 1]
    
dim_reds.append(('pca', train_pca, test_pca))
dim_reds.append(('ica', train_ica, test_ica))
dim_reds.append(('tsvd', train_tsvd, test_tsvd))
dim_reds.append(('grp', train_grp, test_grp))
dim_reds.append(('srp', train_srp, test_srp))
dim_reds.append(('nmf', train_nmf, test_nmf))
dim_reds.append(('fag', train_fag, test_fag))


# ### Creating combinations and running models on them###

# In[ ]:


# creating the combination from the '7' sets of reduced components
combs = [combinations(dim_reds, i+1) for i in range(0, len(dim_reds))]

dr_scores = list()
for c1 in combs:
    for c2 in c1:   
        train_, test_, id_ = list(), list(), list()
        for k in c2:
            train_.append(k[1])
            test_.append(k[2])
            id_.append(k[0])
               
        train_x = train.reset_index(drop=True)
        train_.append(train_x)
        test_.append(test)
            
        train_ = pd.concat(train_, axis=1)
        test_ = pd.concat(test_, axis=1)
        
        
        # training and scoring the models with a particular combination
        
        
# ============================ DecisionTree Model =======================  
#         model = DecisionTreeRegressor(max_depth=3, min_samples_split=11, presort=False, random_state=1729)
#         model.fit(train_, y_train)
#         c_score = r2_score(y_train, model.predict(train_))

# ============================ ElasticNet model =======================
        model = ElasticNet(alpha=0.014, tol=0.11, l1_ratio=0.99999999, 
                           normalize=True, fit_intercept=False, warm_start=True, 
                          copy_X=True, precompute=False, positive=False, max_iter=60)
        model.fit(train_, y_train)
        c_score = r2_score(y_train, model.predict(train_))
        
# ============================ Ridge model =============================
#         model = Ridge()
#         model.fit(train_, y_train)
#         c_score = r2_score(y_train, model.predict(train_))
        
# ================================ lightgbm model =======================
#         lgb_params = {
#         'num_iterations': 200,
#         'learning_rate': 0.045,
#         'max_depth': 3,
#         'bagging_fraction': 0.93,
#         'metric': 'l2_root',
#         }

#         dtrain = lgb.Dataset(train_, y_train)
#         num_round = 1200
#         model = lgb.train(lgb_params, dtrain, num_round)
#         c_score = r2_score(y_train, model.predict(train_))

# ================================= xgboost model ============================
#         xgb_params = {
#         'n_trees': 520, 
#         'eta': 0.0045,
#         'max_depth': 4,
#         'subsample': 0.93,
#         'objective': 'reg:linear',
#         'eval_metric': 'rmse',
#         'base_score': np.mean(y_train),
#         }

#         dtrain = xgb.DMatrix(train_, y_train)

#         num_boost_rounds = 1250
#         model = xgb.train(xgb_params, dtrain, num_boost_round=num_boost_rounds)
#         c_score = r2_score(y_train, model.predict(dtrain))
        
        dr_scores.append((','.join(id_), c_score))


# In[ ]:


# baseline scoring for comparision
model = ElasticNet(alpha=0.014, tol=0.11, l1_ratio=0.99999999, 
                           normalize=True, fit_intercept=False, warm_start=True, 
                          copy_X=True, precompute=False, positive=False, max_iter=60)
model.fit(train, y_train)
full_score = r2_score(y_train, model.predict(train))

dr_scores.append(('baseline', full_score))


# ### Plotting the graph ###
# 
# >  Please open the plots in a separate tab for better labels and clarity.

# In[ ]:


x_axis = [c[0] for c in dr_scores]
y_axis = [c[1] for c in dr_scores]
fig, ax = plt.subplots(figsize=(22, 10))
plt.plot(y_axis)
ax.set_xticks(range(len(x_axis)))
ax.set_xticklabels(x_axis, rotation='vertical')
plt.show()


# ### Plots for other models ###
# 
# > I was unable to run all models due to kernel time constraint so I ran the models on my local and I am giving the results directly. Please download and run this kernel for any model above by just uncommenting the necessary model code.

# In[ ]:


# scores for xgboost on the DR components
xgb_dr_scores = [('pca', 0.67711803427267436), ('ica', 0.67586920454698141), ('tsvd', 0.6773481959197214), ('grp', 0.66224169180347559), ('srp', 0.66096135085216134), ('nmf', 0.67108736044682338), ('fag', 0.65904803827641634), ('pca,ica', 0.68245377669904106), ('pca,tsvd', 0.68279088835428392), ('pca,grp', 0.6787867994366088), ('pca,srp', 0.67792025638275744), ('pca,nmf', 0.67824271112992407), ('pca,fag', 0.67760115601615789), ('ica,tsvd', 0.6835942695318834), ('ica,grp', 0.67714387407053067), ('ica,srp', 0.67588373704965066), ('ica,nmf', 0.67972571931753245), ('ica,fag', 0.67610154710950088), ('tsvd,grp', 0.67832326077226301), ('tsvd,srp', 0.67772724917614169), ('tsvd,nmf', 0.68019973796858046), ('tsvd,fag', 0.67644006589507888), ('grp,srp', 0.6639663861228402), ('grp,nmf', 0.67298350211656932), ('grp,fag', 0.66285185530531221), ('srp,nmf', 0.67115462237246204), ('srp,fag', 0.66067639917962073), ('nmf,fag', 0.67045554585744727), ('pca,ica,tsvd', 0.68654114499013574), ('pca,ica,grp', 0.6845591569950219), ('pca,ica,srp', 0.68341568938247299), ('pca,ica,nmf', 0.68297606561083835), ('pca,ica,fag', 0.68308499474673812), ('pca,tsvd,grp', 0.68378339313413505), ('pca,tsvd,srp', 0.68356776992801049), ('pca,tsvd,nmf', 0.68352377920860341), ('pca,tsvd,fag', 0.68279271098255034), ('pca,grp,srp', 0.67966645954138882), ('pca,grp,nmf', 0.67943469252666477), ('pca,grp,fag', 0.67866040661124249), ('pca,srp,nmf', 0.67956276141791694), ('pca,srp,fag', 0.67791893370687739), ('pca,nmf,fag', 0.67810864643697355), ('ica,tsvd,grp', 0.6836705557289231), ('ica,tsvd,srp', 0.68444852137165446), ('ica,tsvd,nmf', 0.68499119061570446), ('ica,tsvd,fag', 0.68291988532496473), ('ica,grp,srp', 0.67826068686261565), ('ica,grp,nmf', 0.6803462107453162), ('ica,grp,fag', 0.67798837067985895), ('ica,srp,nmf', 0.67967578214024238), ('ica,srp,fag', 0.67733790019653739), ('ica,nmf,fag', 0.67949754170543986), ('tsvd,grp,srp', 0.67923008622903458), ('tsvd,grp,nmf', 0.68049437195624363), ('tsvd,grp,fag', 0.67884475754668472), ('tsvd,srp,nmf', 0.68040112239110107), ('tsvd,srp,fag', 0.67760402916832718), ('tsvd,nmf,fag', 0.67878244082676908), ('grp,srp,nmf', 0.67330304619738812), ('grp,srp,fag', 0.66499311514336668), ('grp,nmf,fag', 0.67272305180560688), ('srp,nmf,fag', 0.67060795522667327), ('pca,ica,tsvd,grp', 0.6879912555299923), ('pca,ica,tsvd,srp', 0.68824790746797515), ('pca,ica,tsvd,nmf', 0.6874142163999073), ('pca,ica,tsvd,fag', 0.6874916336682666), ('pca,ica,grp,srp', 0.68495477428381735), ('pca,ica,grp,nmf', 0.68444395334924479), ('pca,ica,grp,fag', 0.68445968450240868), ('pca,ica,srp,nmf', 0.68359817404064549), ('pca,ica,srp,fag', 0.68427438373791882), ('pca,ica,nmf,fag', 0.68372534635518956), ('pca,tsvd,grp,srp', 0.68522627464232233), ('pca,tsvd,grp,nmf', 0.68482997343399277), ('pca,tsvd,grp,fag', 0.68402325359793226), ('pca,tsvd,srp,nmf', 0.68451110170905172), ('pca,tsvd,srp,fag', 0.68376345836481944), ('pca,tsvd,nmf,fag', 0.68398845947727116), ('pca,grp,srp,nmf', 0.68057096050452826), ('pca,grp,srp,fag', 0.67936781899772836), ('pca,grp,nmf,fag', 0.67993008733246818), ('pca,srp,nmf,fag', 0.67880641218391258), ('ica,tsvd,grp,srp', 0.68585503201750986), ('ica,tsvd,grp,nmf', 0.68589078222575661), ('ica,tsvd,grp,fag', 0.68442858274152085), ('ica,tsvd,srp,nmf', 0.68625869472926981), ('ica,tsvd,srp,fag', 0.68459204233889182), ('ica,tsvd,nmf,fag', 0.68543928703685419), ('ica,grp,srp,nmf', 0.68184531802094139), ('ica,grp,srp,fag', 0.67948113256494302), ('ica,grp,nmf,fag', 0.68125384869666428), ('ica,srp,nmf,fag', 0.68044986028224808), ('tsvd,grp,srp,nmf', 0.68173625535126525), ('tsvd,grp,srp,fag', 0.67963365320087199), ('tsvd,grp,nmf,fag', 0.68106208115398559), ('tsvd,srp,nmf,fag', 0.68015617202501455), ('grp,srp,nmf,fag', 0.67436036079206474), ('pca,ica,tsvd,grp,srp', 0.68874171502676662), ('pca,ica,tsvd,grp,nmf', 0.68836418911595743), ('pca,ica,tsvd,grp,fag', 0.68823671222257321), ('pca,ica,tsvd,srp,nmf', 0.68718360137081413), ('pca,ica,tsvd,srp,fag', 0.68784143657029118), ('pca,ica,tsvd,nmf,fag', 0.68702839094409329), ('pca,ica,grp,srp,nmf', 0.68542880395930061), ('pca,ica,grp,srp,fag', 0.68580831435591105), ('pca,ica,grp,nmf,fag', 0.68520397140448863), ('pca,ica,srp,nmf,fag', 0.6838153499053613), ('pca,tsvd,grp,srp,nmf', 0.68475418776952801), ('pca,tsvd,grp,srp,fag', 0.68416888261697117), ('pca,tsvd,grp,nmf,fag', 0.68439534926539514), ('pca,tsvd,srp,nmf,fag', 0.68369054844962107), ('pca,grp,srp,nmf,fag', 0.68109487520163592), ('ica,tsvd,grp,srp,nmf', 0.68703175972715047), ('ica,tsvd,grp,srp,fag', 0.68605627429688421), ('ica,tsvd,grp,nmf,fag', 0.68625025013109198), ('ica,tsvd,srp,nmf,fag', 0.68508823305152089), ('ica,grp,srp,nmf,fag', 0.6818608839534277), ('tsvd,grp,srp,nmf,fag', 0.68121711681246844), ('pca,ica,tsvd,grp,srp,nmf', 0.68899654938766397), ('pca,ica,tsvd,grp,srp,fag', 0.68892201689977894), ('pca,ica,tsvd,grp,nmf,fag', 0.68839400442292953), ('pca,ica,tsvd,srp,nmf,fag', 0.68827193462597602), ('pca,ica,grp,srp,nmf,fag', 0.6851006771786381), ('pca,tsvd,grp,srp,nmf,fag', 0.68488975768813187), ('ica,tsvd,grp,srp,nmf,fag', 0.68648750730778907), ('pca,ica,tsvd,grp,srp,nmf,fag', 0.68823319136312078)]


# In[ ]:


x_axis = [c[0] for c in xgb_dr_scores]
y_axis = [c[1] for c in xgb_dr_scores]
fig, ax = plt.subplots(figsize=(22, 10))
plt.plot(y_axis)
ax.set_xticks(range(len(x_axis)))
ax.set_xticklabels(x_axis, rotation='vertical')
plt.show()


# ###Conclusions:###

# In[ ]:


sorted_id = np.argsort(y_axis)
print("Combinations which has the lowest score: {}".format(np.array(x_axis)[sorted_id[:7]]))
print("Combinations which has the highest score: {}".format(np.array(x_axis)[sorted_id[-7:]]))

print("\n\nBest Score: {}".format(np.array(x_axis)[sorted_id[-1]]))


# So,  feature agglomeration is really not helping. PCA and ICA are playing the main roles.

# ###Thanks for the viewing the kernel. Please upvote if you like it :) ###
# 
# **Previous kernels:**
# 
#  - [Categorical exploration  - python notebook][1]
#  - [Numerical variable exploration  - treemaps  - R notebook][2] 
# 
# **Coming up:**
# 
#  - Autoencoder and t-SNE - dimensionality reduction and denoising
# 
# ..........
# to be continued
# 
# **Update:** 
# 
#  - Added baseline score
#  - Added conclusion
# 
# 
# 
# 
#   [1]: https://www.kaggle.com/remidi/sherlock-s-exploration-season-01-categorical
#   [2]: https://www.kaggle.com/remidi/sherlocks-exploration-season-02-e01-numerical
