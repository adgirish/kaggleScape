
# coding: utf-8

# How to calculate f1 score. Welcome feedback!

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


cv_labels_df = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


cv_labels_df['products'] = cv_labels_df['products'].astype(str)


# In[ ]:


cv_preds_df = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


cv_preds_df['products'] = cv_preds_df['products'].astype(str)


# In[ ]:


cv_labels_df = pd.merge(cv_labels_df, cv_preds_df, how='left', on='order_id')


# In[ ]:


cv_labels_df.head()


# In[ ]:


def eval_fun(labels, preds):
    labels = labels.split(' ')
    preds = preds.split(' ')
    rr = (np.intersect1d(labels, preds))
    precision = np.float(len(rr)) / len(preds)
    recall = np.float(len(rr)) / len(labels)
    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        return (precision, recall, 0.0)
    return (precision, recall, f1)


# In[ ]:


res = list()
for entry in cv_labels_df.itertuples():
    res.append(eval_fun(entry[2], entry[3]))


# In[ ]:


res = pd.DataFrame(np.array(res), columns=['precision', 'recall', 'f1'])


# In[ ]:


res.describe()

