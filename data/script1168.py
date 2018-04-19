
# coding: utf-8

#  Feature engineering in this competition is quite challenging as the column names have been masked. I have tried many approaches of adding, multiplying etc. which all have failed. The number of combinations that we can make using the given columns is too huge. To tackle this and to reduce the number of combinations I have created the following strategy for feature engineering. I hope it helps.

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

import warnings
warnings.filterwarnings('ignore')
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import r2_score


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


# Remove the outlier
train=train[train.y<250]


# ### ** The strategy **
# #### First we split the data into two parts. Using the cross validated results try finding out which range of values have max error. In this example I take 100.

# In[ ]:


# Check no. of rows greater than equal to 100
len(train['y'][(train.y>=100)])


# In[ ]:


# Check no. of rows less than 100
len(train['y'][(train.y<100)])


# #### Now we convert the training set into a classification problem. Create a new field for class.

# In[ ]:


train['y_class'] = train.y.apply(lambda x: 0 if x<100  else 1 )


# In[ ]:


# Concat the datasets
data = pd.concat([train,test])


# In[ ]:


# Removing object type vars as I am more interested in binary ones
data = data.drop(data.select_dtypes(include = ['object']).columns,axis=1)


# In[ ]:


feat = list(data.drop(['y','y_class'],axis=1).columns.values)


# In[ ]:


train_df = (data[:train.shape[0]])
test_df = (data[train.shape[0]:])


# In[ ]:


# I have not removed zero valued columns for now
len(feat)


# In[ ]:


# Remove ID as we want some honest features :)
feat.remove('ID')


# In[ ]:


from sklearn.metrics import f1_score as f1


# In[ ]:


# Calculating CV score
def cv_score(model):
    return cross_val_score(model,train_df[feat],train_df['y_class'],cv=10,scoring = 'f1').mean()


# ### Now, the interesting part. 
# > Decision trees are the basic entities that make up complex algos like XGB. But to understand the rules on the which splitting happens is not possible in XGB. Here, we build decision trees to understand the rules and build features for the same. Lets go!

# In[ ]:


from sklearn.tree import DecisionTreeClassifier as DTC


# In[ ]:


model = DTC(max_depth = 5,min_samples_split=200) # We don't want to overfit


# > Its important to notice that there is no sense in keeping high depth values. Since we need strong features, the rules should have considerably large sample size in the leaves. For small sample size values, the feature may not be that strong. 

# In[ ]:


cv_score(model) 


# > F1 looks good! But sometimes it may not. Doesn't matter, as we want the branches with good gini scores and sample size

# In[ ]:


model.fit(train_df[feat],train_df.y_class)


# #### > To visualize the tree we use graphviz

# In[ ]:


# Graphviz is used to build decision trees
from sklearn.tree import export_graphviz
from sklearn import tree


# In[ ]:


# This statement builds a dot file.
tree.export_graphviz(model, out_file='tree.dot',feature_names  = feat)  


# After building the tree dot file you can convert it into a png using :
# > **dot  -Tpng tree.dot -o tree.png**
# 
# in command line (first cd to dot directory)

# In[ ]:


# This will bring the image to the notebook (or you can view it locally)
from IPython.display import Image
#Image("tree.png") # Uncomment if you are trying this on local
# Can't read the image in kernal. Anyone know how? Will try and add image in comments


# EDIT : Here is the link to the image - [Tree][1]
# 
# 
#   [1]: https://ibb.co/jvxQkv

# > Now, if you start traversing the tree. We see that when, X314 = 0 and X315 = 0 and X47=0 then the gini score is .2183 with good sample size!!
# This can be a good feature as it gives us a good separation between the classes.

# Experiment more with this notebook and if you are generous enough, keep adding good features in comments below ;D

# ### Thanks! :)
