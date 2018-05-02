
# coding: utf-8

# # Try Different ML Methods in Python

# ## Load Data

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')
train.head()


# In[ ]:


train.shape


# High dimension data always results in a good training score, yet the test score is not necessarily good as the overfiting  problem may occur. Therefore, before learning, we use the feature selection techniques built in sklearn package to select the useful features. The tutorial of feature selection can be found at the link below.
# 
# http://scikit-learn.org/stable/modules/feature_selection.html

# ## Feature Selection

# ### Tree_based feature selection

# In[ ]:


import sklearn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
features = train.iloc[:,0:562]
label = train['Activity']
clf = ExtraTreesClassifier()
clf = clf.fit(features, label)
model = SelectFromModel(clf, prefit=True)
New_features = model.transform(features)
print(New_features.shape)


# ### L1-based feature selection

# In[ ]:


from sklearn.svm import LinearSVC
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(features, label)
model_2 = SelectFromModel(lsvc, prefit=True)
New_features_2 = model_2.transform(features)
print(New_features_2.shape)


# The L1-based feature selection will keep more features within the training set.  

# ## Fitting Classifiers

# **Load Models**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
Classifiers = [DecisionTreeClassifier(),RandomForestClassifier(n_estimators=200),GradientBoostingClassifier(n_estimators=200)]


# ### Without feature selection

# In[ ]:


from sklearn.metrics import accuracy_score
import timeit
test_features= test.iloc[:,0:562]
Time_1=[]
Model_1=[]
Out_Accuracy_1=[]
for clf in Classifiers:
    start_time = timeit.default_timer()
    fit=clf.fit(features,label)
    pred=fit.predict(test_features)
    elapsed = timeit.default_timer() - start_time
    Time_1.append(elapsed)
    Model_1.append(clf.__class__.__name__)
    Out_Accuracy_1.append(accuracy_score(test['Activity'],pred))


# ### Tree-based feature selection

# In[ ]:


test_features= model.transform(test.iloc[:,0:562])
Time_2=[]
Model_2=[]
Out_Accuracy_2=[]
for clf in Classifiers:
    start_time = timeit.default_timer()
    fit=clf.fit(New_features,label)
    pred=fit.predict(test_features)
    elapsed = timeit.default_timer() - start_time
    Time_2.append(elapsed)
    Model_2.append(clf.__class__.__name__)
    Out_Accuracy_2.append(accuracy_score(test['Activity'],pred))


# ### L1-Based feature selection

# In[ ]:


test_features= model_2.transform(test.iloc[:,0:562])
Time_3=[]
Model_3=[]
Out_Accuracy_3=[]
for clf in Classifiers:
    start_time = timeit.default_timer()
    fit=clf.fit(New_features_2,label)
    pred=fit.predict(test_features)
    elapsed = timeit.default_timer() - start_time
    Time_3.append(elapsed)
    Model_3.append(clf.__class__.__name__)
    Out_Accuracy_3.append(accuracy_score(test['Activity'],pred))


# ## Evaluation

# In the final chapter, we will evaluate the feature selections based on **running time and accuracy.** The running time is somehow determinant to the scala-bility of the model while the accuracy is gonna to tell us whether shrinking the dimension of the data may hugely jeopardize the performance of the model.

# ### Accuracy

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
ind =  np.arange(3)   # the x locations for the groups
width = 0.1       # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(ind, Out_Accuracy_1, width, color='r')
rects2 = ax.bar(ind + width, Out_Accuracy_2, width, color='y')
rects3 = ax.bar(ind + width + width ,Out_Accuracy_3, width, color='b')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy by Models and Selection Process')
ax.set_xticks(ind + width)
ax.set_xticklabels(Model_3,rotation=45)
plt.show()


# **The legend may cover the main part of bar so I remove them in this plot. For the meanings of colors, you can refer to the next plot.**

# ### Running Time

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
ind =  np.arange(3)   # the x locations for the groups
width = 0.1       # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(ind, Time_1, width, color='r')
rects2 = ax.bar(ind + width, Time_2, width, color='y')
rects3 = ax.bar(ind + width + width ,Time_3, width, color='b')
ax.set_ylabel('Running Time')
ax.set_title('Time by Models and Selection Process')
ax.set_xticks(ind + width)
ax.set_xticklabels(Model_3,rotation=45)
ax.legend((rects1[0], rects2[0],rects3[0]), ('No Selection', 'Tree_Based','L1_Based'))

plt.show()


# ## Conclusion

# 1. The feature selection can hugely decrease the running time of complicated model, without obviously jeopardizing the performance of model.
# 
# 2. The overall accuracy of the model will not be necessarily compromised by shrinking the size of the data set. The main reason is that good feature selection may prevent over-fitting to some extents.  
