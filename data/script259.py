
# coding: utf-8

# # Statoil CSV PyTorch ensemble LB 0.1690 (Now 0.1520)
# 
# This is a CSV file with an ensemble of around ~50 Resnet modles which I am sharing for the benefit of others. 
# 
# ### The Pytorch code for generating the models and single CSV file is here (I will update it soon):
# 
# - https://github.com/QuantScientist/Deep-Learning-Boot-Camp/blob/master/Kaggle-PyTorch/iceberg/statoil-iceberg-classifier-challenge-cnn-ver1.py
# 
# 
# Learning curve:
# ![logo](https://github.com/QuantScientist/Deep-Learning-Boot-Camp/raw/master/Kaggle-PyTorch/curve.png)
# 
# ![logo](https://github.com/QuantScientist/Deep-Learning-Boot-Camp/raw/master/day02-PyTORCH-and-PyCUDA/PyTorch/IceResNet_2017-11-27_02-33-15.png)
# 
# ### The Pytorch code for generating the CSV ensemble is below:
# 
# 

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

# Any results you write to the current directory are saved as output.


def ensemble():
    stacked_1 = pd.read_csv('./pth/kfold/' + 'ResNetLike_0.166586_submission.csv')
    stacked_2 = pd.read_csv('./pth/kfold/' + 'ResNetLike_0.167542_submission.csv')
    stacked_3 = pd.read_csv('./pth/kfold/' + 'ResNetLike_0.167621_submission.csv')
    stacked_4 = pd.read_csv('./pth/kfold/' + 'ResNetLike_0.166586_submission.csv')
    stacked_5 = pd.read_csv('./pth/kfold/' + 'ResNetLike_0.168133_submission.csv')
    stacked_6 = pd.read_csv('./pth/kfold/' + 'ResNetLike_0.170522_submission.csv')
    stacked_7 = pd.read_csv('./pth/kfold/' + 'ResNetLike_0.180278_submission.csv')
    stacked_8 = pd.read_csv('./pth/kfold/' + 'ResNetLike_0.185876_submission.csv')
    sub = pd.DataFrame()
    sub['id'] = stacked_1['id']
    sub['is_iceberg'] = np.exp(np.mean(
        [
            stacked_1['is_iceberg'].apply(lambda x: np.log(x)), \
            stacked_2['is_iceberg'].apply(lambda x: np.log(x)), \
            stacked_3['is_iceberg'].apply(lambda x: np.log(x)), \
            stacked_4['is_iceberg'].apply(lambda x: np.log(x)), \
            stacked_5['is_iceberg'].apply(lambda x: np.log(x)), \
            stacked_6['is_iceberg'].apply(lambda x: np.log(x)), \
            stacked_7['is_iceberg'].apply(lambda x: np.log(x)), \
            stacked_8['is_iceberg'].apply(lambda x: np.log(x)), \
            ], axis=0))
    sub.to_csv('./pth/kfold/' + 'ensamble.csv', index=False, float_format='%.6f')

