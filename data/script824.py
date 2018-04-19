
# coding: utf-8

# Here I am going to apply Principal component analysis on the given dataset using Scikit-learn and find out the dimensions(also known as components) with maximum variance(where the data is spread out).Features with little variance in the data are then projected into new lower dimension. Then the models are  trained on transformed dataset to apply machine learning models.Then I have applied  Random forest Regressor on old and the transformed datasets and compared them.
# If you want to know the basic concept behind Principal Component Analysis check this out.
# (https://www.kaggle.com/nirajvermafcb/d/ludobenistant/hr-analytics/principal-component-analysis-explained)

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


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv('../input/data.csv') #Replace it with your path where the data file is stored
df.head()


# In[ ]:


df.describe()


# In[ ]:


df.corr()


# Let us find if there is any relationship between temperature and apparent_temperature

# In[ ]:


x=df['temperature']
y=df['apparent_temperature']
colors=('r','b')
plt.xlabel('Temperature')
plt.ylabel('Apparent_temperature')
plt.scatter(x,y,c=colors)


# The temperture given here is in fahrenheit.We will convert it into Celsius using the formula 
# **Celsius=(Fahrenheit-32)* (5/9)**

# In[ ]:


Fahrenheit=df['temperature']


# Converting it into the list so we can apply lambda function

# In[ ]:


F=Fahrenheit.tolist()


# Applying Lambda function

# In[ ]:


C= map(lambda x: (float(5)/9)*(x-32),F)
Celsius=(list(C))


# Converting list to series

# In[ ]:


temperature_celsius=pd.Series(Celsius)


# Applying the series to temperature column

# In[ ]:


df['temperature']= temperature_celsius
df['temperature']
df.head()


# Thus we have converted the temperature column from fahrenheit to degree celsius.Similarly we are now converting apparent_temperature to degree celsius.

# In[ ]:


at_fahrenheit=df['apparent_temperature']
at_F=at_fahrenheit.tolist()
at_C= map(lambda x: (float(5)/9)*(x-32),at_F)
at_Celsius=(list(C))
at_celsius=pd.Series(at_Celsius)
at_celsius


# In[ ]:


apparent_temperature_celsius=pd.Series(at_Celsius)
print(apparent_temperature_celsius)


# In[ ]:


df['apparent_temperature']= temperature_celsius
df['apparent_temperature']
df.head()


# In[ ]:


X = df.iloc[:,1:8]  # all rows, all the features and no labels
y = df.iloc[:, 0]  # all rows, label only
#X
#y


# In[ ]:


df.corr()


# In[ ]:


correlation = df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='viridis')

plt.title('Correlation between different fearures')


# Standardising data

# In[ ]:


# Scale the data to be between -1 and 1
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X=scaler.fit_transform(X)
X


# In[ ]:


from sklearn.decomposition import PCA
pca = PCA()
pca.fit_transform(X)


# In[ ]:


pca.get_covariance()


# In[ ]:


explained_variance=pca.explained_variance_ratio_
explained_variance


# In[ ]:


with plt.style.context('dark_background'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(7), explained_variance, alpha=0.5, align='center',
            label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()


# **Thus we can see from the above plot that  first two components constitute almost 55% of the variance.Third,fourth and fifth components has 42% of the data sprad.The last component has less than 5% of the variance.Hence we can drop the fifth component  **

# In[ ]:


pca=PCA(n_components=5)
X_new=pca.fit_transform(X)
X_new


# In[ ]:


pca.get_covariance()


# In[ ]:


explained_variance=pca.explained_variance_ratio_
explained_variance


# In[ ]:


with plt.style.context('dark_background'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(5), explained_variance, alpha=0.5, align='center',
            label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train.shape


# In[ ]:


# Establish model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()


# In[ ]:


# Try different numbers of n_estimators - this will take a minute or so
estimators = np.arange(10, 200, 10)
scores = []
for n in estimators:
    model.set_params(n_estimators=n)
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))
print(scores)    


# In[ ]:


plt.title("Effect of n_estimators")
plt.xlabel("n_estimator")
plt.ylabel("score")
plt.plot(estimators, scores)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=1)
X_train.shape


# In[ ]:


# Establish model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()


# In[ ]:


# Try different numbers of n_estimators - this will take a minute or so
estimators = np.arange(10, 200, 10)
scores = []
for n in estimators:
    model.set_params(n_estimators=n)
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))
print(scores)    


# In[ ]:


plt.title("Effect of n_estimators")
plt.xlabel("n_estimator")
plt.ylabel("score")
plt.plot(estimators, scores)


# You can find my notebook on Github:
# ("https://github.com/nirajvermafcb/Data-Science-with-python")
