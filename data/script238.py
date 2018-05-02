
# coding: utf-8

# === Please read also the comments below, in case Kagglers write updates to this post ===
# <br>
# <br>
# If you're new to Kaggle kernels, you may wonder how to create an output file. Perhaps you have already run in your notebook a function like .to_csv, but **you don't see your file anywhere**? I had the same problem. **You need to publish your notebook**. There is a <strike>Publish</strike> <strike>New Snapshot</strike> Commit & Run button in the top-right corner of your notebook page.
# <br>
# <br>To create a file from scratch, step-by-step, please read on.
# <br>
# <br>Let's say you started your first kernel based on Titanic dataset, by going to the <a href="https://www.kaggle.com/c/titanic/kernels">Kernels</a> tab and clicking New Kernel button. You would see something like this:

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


# You are going to read the test set input file, make a very rough prediction (a simple rule "all females survive, no males survive"), and create a simple dataframe with results you would like to submit.

# In[ ]:


test = pd.read_csv('../input/test.csv')
test['Survived'] = 0
test.loc[test['Sex'] == 'female','Survived'] = 1
data_to_submit = pd.DataFrame({
    'PassengerId':test['PassengerId'],
    'Survived':test['Survived']
})


# Now that you have your dataframe, you would like to export it as a csv file, like this:

# In[ ]:


data_to_submit.to_csv('csv_to_submit.csv', index = False)


# Everything runs smoothly, but the problem is you can't see your file anywhere in this page, nor in your Profile, Kernels tab, nowhere! This is because you haven't published your notebook yet. To do that, **click the** <strike>Publish</strike> <strike>New Snapshot</strike> **Commit & Run button** - as I write it, this is a light-blue button in the top-right corner of my notebook page, in the right pane. It may take a minute for the Kaggle server to publish your notebook.
# <br>
# <br>When this operation is done, you can go back by clicking '<<' button in the top-left corner. Then you should see your notebook with a top bar that has a few tabs: Notebook, Code, Data, **Output**, Comments, Log ... Edit Notebook.
# Click the Output tab. You should see your output csv file there, ready to download!
