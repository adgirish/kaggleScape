
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

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/train.csv")


# **Lets look at the histogram to check price distribution**

# In[ ]:


df.hist(column='price_doc', bins=100)
plt.show();


# In[ ]:


df['date_column'] = pd.to_datetime(df['timestamp'])
df['mnth_yr'] = df['date_column'].apply(lambda x: x.strftime('%B-%Y'))
df1=df[["price_doc","mnth_yr"]]
df2=df1.groupby('mnth_yr')['price_doc'].mean()
df2=pd.DataFrame(df2)
df2.reset_index(inplace=True)
df2['mnth_yr'] = pd.to_datetime(df2['mnth_yr'])
df2.sort_values('mnth_yr')
df2.plot(x='mnth_yr', y='price_doc')
plt.figure();
plt.show();


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="whitegrid", color_codes=True)


# In[ ]:


sns.countplot(x="build_year", data=df[df.build_year>1900.0].sort_index(), palette="Greens_d");


# In[ ]:


sns.barplot(x="build_year", y="price_doc", data=df[df.build_year>1900.0].sort_index(),palette="Greens_d");


# In[ ]:


def correlation_matrix(df):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 60)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Corelation Matrix visualization')
    labels=df1.columns
    ax1.set_xticklabels(labels,fontsize=4)
    ax1.set_yticklabels(labels,fontsize=4)
    fig.colorbar(cax, ticks=[-0.6,-0.3,0,0.3,0.6])
    plt.show()

correlation_matrix(df1);


# In[ ]:


corr_matrix=df.corr().as_matrix


# In[ ]:


corr_matrix[(corr_matrix>0.5)]

