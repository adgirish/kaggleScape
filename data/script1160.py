
# coding: utf-8

# > This notebook is based on the prior work done by Roberto Ruiz in https://www.kaggle.com/robertoruiz/sberbank-russian-housing-market/dealing-with-multicollinearity. As such, I won't go into explanation of the theory because they have already done a fantastic job of doing so. If you have any questions on the theory then please feel free to ask, but I'd definitely suggest that everyone read Roberto's work as an introduction.

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

# Extra imports necessary for the code

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer

from statsmodels.stats.outliers_influence import variance_inflation_factor


# Based on the work set out by Roberto, I've created a scikit-learn transformer class that can be used to remove columns that have a high VIF factor (in short, they have high colinearity with other columns within the dataset and as such should probably be removed).
# 
# This class is based on a standard scikit-learn transformer and also uses the statsmodel library for calculating the VIF number. Information on scikit-learn transformers can be found [here](http://scikit-learn.org/stable/data_transforms.html) whilst the docs for the statsmodel function can be found [here](http://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html)

# In[ ]:


X = pd.read_csv('../input/train.csv', index_col=0, parse_dates=['timestamp'])
y = X.pop('price_doc')
X.head()


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

class ReduceVIF(BaseEstimator, TransformerMixin):
    def __init__(self, thresh=5.0, impute=True, impute_strategy='median'):
        # From looking at documentation, values between 5 and 10 are "okay".
        # Above 10 is too high and so should be removed.
        self.thresh = thresh
        
        # The statsmodel function will fail with NaN values, as such we have to impute them.
        # By default we impute using the median value.
        # This imputation could be taken out and added as part of an sklearn Pipeline.
        if impute:
            self.imputer = Imputer(strategy=impute_strategy)

    def fit(self, X, y=None):
        print('ReduceVIF fit')
        if hasattr(self, 'imputer'):
            self.imputer.fit(X)
        return self

    def transform(self, X, y=None):
        print('ReduceVIF transform')
        columns = X.columns.tolist()
        if hasattr(self, 'imputer'):
            X = pd.DataFrame(self.imputer.transform(X), columns=columns)
        return ReduceVIF.calculate_vif(X, self.thresh)

    @staticmethod
    def calculate_vif(X, thresh=5.0):
        # Taken from https://stats.stackexchange.com/a/253620/53565 and modified
        dropped=True
        while dropped:
            variables = X.columns
            dropped = False
            vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]
            
            max_vif = max(vif)
            if max_vif > thresh:
                maxloc = vif.index(max_vif)
                print(f'Dropping {X.columns[maxloc]} with vif={max_vif}')
                X = X.drop([X.columns.tolist()[maxloc]], axis=1)
                dropped=True
        return X


# In[ ]:


transformer = ReduceVIF()

# Only use 10 columns for speed in this example
X = transformer.fit_transform(X[X.columns[-10:]], y)

X.head()


# Things to bear in mind:
# 
#  - It's really slow, especially at the beginning when it's calculating VIF for all columns. As it drops more columns it will speed up, but there's a reason I've only used 10 columns for the example...
#  - If two or more columns "tie" (in the case when multiple can be inf...) then it'll drop the first column it finds to start with.
#  - The idea of imputing so much data makes me uneasy. I'd be tempted to modify it so you use the imputed data only for dropping columns but then continue analysis with all of the NaNs present.
