
# coding: utf-8

# ## Yoann Boj
# ### 8th November 2016

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


# In[ ]:


df = pd.read_csv("../input/train.csv")
print(df.describe())


# In[ ]:


df_test = pd.read_csv("../input/test.csv")
print(df_test.describe())


# In[ ]:


import seaborn as sns
sns.set()

sns.pairplot(df,hue="type")


# In[ ]:


y = df["type"]
indexes_test = df_test["id"]

df = df.drop(["type","color","id"],axis=1)
df_test = df_test.drop(["color","id"],axis=1)


# In[ ]:


df = pd.get_dummies(df)
df_test = pd.get_dummies(df_test)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=0)


# In[ ]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty='l2',C=1000000)
lr.fit(X_train,y_train)
y_pred= lr.predict(X_test) 

print(classification_report(y_pred,y_test))


# In[ ]:


y_pred = lr.predict(df_test)

Y = pd.DataFrame()
Y["id"] = indexes_test
Y["type"] = y_pred
Y.to_csv("submission.csv",index=False)

