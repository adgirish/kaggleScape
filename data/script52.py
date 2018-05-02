
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
print(check_output(["ls", "../input/porto-seguros-safe-driver-noisy-features"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# This notebook displays the results of a search for noisy features. This search has been carried out using Light GBM in RandomForest mode (to avoid the hassle of *how many rounds fo I need to run ?* )
# 
# The file noisy_feature_check_results.csv contains the average importances of each feature and their corresponding *shadows* over 30 runs. Standard deviation of the importances is also available.
# 
# Shadows are simply shuffled copies of real features. Comparing features to their shadows is an easy way to assess their genuine forecasting power. This is extensively used in Boruta packages (python and R)
# 
# For information, Python Boruta packages has selected the following features (the rest is considered noise !!!):
# - ps_ind_01
# - ps_ind_03
# - ps_ind_05_cat
# - ps_ind_07_bin
# - ps_ind_15
# - ps_ind_16_bin
# - ps_reg_01
# - ps_reg_02
# - ps_reg_03
# - ps_car_01_cat
# - ps_car_03_cat
# - ps_car_07_cat
# - ps_car_12
# - ps_car_13
# - ps_car_14
# - ps_car_15
# 
# The classifier used for the task is a LGBMClassifier with the following parameters:
# * boosting_type="rf",
# * num_leaves=1024,
# * max_depth=6,
# * n_estimators=500,
# * subsample=.623,
# * colsample_bytree=.5
# 
# Now let's review some results

# In[ ]:


results = pd.read_csv("../input/porto-seguros-safe-driver-noisy-features/noisy_feature_check_results.csv")


# Show the best scoring features

# In[ ]:


results.sort_values(by="importance_mean", ascending=False, inplace=True)
results.dropna(axis=0, inplace=True)
results.head(10)


# In[ ]:


good_to_go = []
doubt = []
suspicious = []
rejected = []
for feature in results.feature.unique():
    sha_mean, sha_dev = results.loc[(results["feature"] == feature) 
                                    & (results["process"] == "Shadow"), ["importance_mean", "importance_std"]].values[0]
    id_mean, id_dev = results.loc[(results["feature"] == feature) 
                                    & (results["process"] == "Identity"), ["importance_mean", "importance_std"]].values[0]
    if sha_mean >= id_mean:
        rejected.append((feature, id_mean, sha_mean))
    elif sha_mean + sha_dev >= id_mean:
        suspicious.append((feature, id_mean, sha_mean))
    elif sha_mean + sha_dev >= id_mean - id_dev:
        doubt.append((feature, id_mean, sha_mean))
    else:
        good_to_go.append((feature, id_mean, sha_mean))

print("Good features (%d)" % len(good_to_go))
for f, score, sha in good_to_go:
    print("\t%-20s : %7.2f / shadow %7.2f" % (f, score, sha))
print("Doubts (%d)" % len(doubt))
for f, score, sha in doubt:
    print("\t%-20s : %7.2f / shadow %7.2f" % (f, score, sha))
print("Suspicious features (%d)" % len(suspicious))
for f, score, sha in suspicious:
    print("\t%-20s : %7.2f / shadow %7.2f" % (f, score, sha))
print("Rejected features (%d)" % len(rejected))
for f, score, sha in rejected:
    print("\t%-20s : %7.2f / shadow %7.2f" % (f, score, sha))
        


# Features kept by Boruta are also kept by my feature selection process.
# 
# I you find this note useful please upvote.
