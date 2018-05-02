
# coding: utf-8

# Here we use a simple linear model and article content with `CountVectorizer`.

# Import libraries.

# In[1]:


import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import json
from tqdm import tqdm_notebook
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_absolute_error


# The following code will help to throw away all HTML tags from an article content.

# In[2]:


from html.parser import HTMLParser

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


# In[3]:


PATH_TO_DATA = '../input/'


# Assume you have all data downloaded from competition's [page](https://www.kaggle.com/c/how-good-is-your-medium-article/data) in the PATH_TO_DATA folder and `.gz` files are ungzipped.

# In[4]:


get_ipython().system('ls -l $PATH_TO_DATA')


# Supplementary function to read a JSON line without crashing on escape characters. 

# In[5]:


def read_json_line(line=None):
    result = None
    try:        
        result = json.loads(line)
    except Exception as e:      
        # Find the offending character index:
        idx_to_replace = int(str(e).split(' ')[-1].replace(')',''))      
        # Remove the offending character:
        new_line = list(line)
        new_line[idx_to_replace] = ' '
        new_line = ''.join(new_line)     
        return read_json_line(line=new_line)
    return result


# This function takes a JSON and forms a txt file leaving only article content. When you resort to feature engineering and extract various features from articles, a good idea is to modify this function.

# In[6]:


def preprocess(path_to_inp_json_file):
    output_list = []
    with open(path_to_inp_json_file, encoding='utf-8') as inp_file:
        for line in tqdm_notebook(inp_file):
            json_data = read_json_line(line)
            content = json_data['content'].replace('\n', ' ').replace('\r', ' ')
            content_no_html_tags = strip_tags(content)
            output_list.append(content_no_html_tags)
    return output_list


# In[ ]:


get_ipython().run_cell_magic('time', '', "train_raw_content = preprocess(path_to_inp_json_file=os.path.join(PATH_TO_DATA, \n                                                                  'train.json'),)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "test_raw_content = preprocess(path_to_inp_json_file=os.path.join(PATH_TO_DATA, \n                                                                  'test.json'),)")


# We'll use a linear model (`Ridge`) with a very simple feature extractor – `CountVectorizer`, meaning that we resort to the Bag-of-Words approach. For now, we are leaving only 50k features. 

# In[ ]:


cv = CountVectorizer(max_features=50000)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'X_train = cv.fit_transform(train_raw_content)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'X_test = cv.transform(test_raw_content)')


# In[ ]:


X_train.shape, X_test.shape


# Read targets from file.

# In[ ]:


train_target = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_log1p_recommends.csv'), 
                           index_col='id')


# In[ ]:


train_target.shape


# In[ ]:


y_train = train_target['log_recommends'].values


# Make a 30%-holdout set. 

# In[ ]:


train_part_size = int(0.7 * train_target.shape[0])
X_train_part = X_train[:train_part_size, :]
y_train_part = y_train[:train_part_size]
X_valid =  X_train[train_part_size:, :]
y_valid = y_train[train_part_size:]


# Now we are ready to fit a linear model.

# In[ ]:


from sklearn.linear_model import Ridge


# In[ ]:


ridge = Ridge(random_state=17)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'ridge.fit(X_train_part, y_train_part);')


# In[ ]:


ridge_pred = ridge.predict(X_valid)


# Let's plot predictions and targets for the holdout set. Recall that these are #recommendations (= #claps) of Medium articles with the `np.log1p` transformation.

# In[ ]:


plt.hist(y_valid, bins=30, alpha=.5, color='red', label='true', range=(0,10));
plt.hist(ridge_pred, bins=30, alpha=.5, color='green', label='pred', range=(0,10));
plt.legend();


# As we can see, the prediction is far from perfect, and we get MAE $\approx$ 1.3 that corresponds to $\approx$ 2.7 error in #recommendations.

# In[ ]:


valid_mae = mean_absolute_error(y_valid, ridge_pred)
valid_mae, np.expm1(valid_mae)


# Finally, train the model on the full accessible training set, make predictions for the test set and form a submission file. 

# In[ ]:


get_ipython().run_cell_magic('time', '', 'ridge.fit(X_train, y_train);')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'ridge_test_pred = ridge.predict(X_test)')


# In[ ]:


def write_submission_file(prediction, filename,
    path_to_sample=os.path.join(PATH_TO_DATA, 'sample_submission.csv')):
    submission = pd.read_csv(path_to_sample, index_col='id')
    
    submission['log_recommends'] = prediction
    submission.to_csv(filename)


# In[ ]:


write_submission_file(prediction=ridge_test_pred, 
                      filename='first_ridge.csv')


# With this, you'll get 1.91185 on [public leaderboard](https://www.kaggle.com/c/how-good-is-your-medium-article/leaderboard). This is much higher than our validation MAE. This indicates that the target distribution in test set somewhat differs from that of the training set (recent Medium articles are more popular). This shouldn't confuse us as long as we see a correlation between local improvements and improvements on the leaderboard. 

# Some ideas for improvement:
# - Engineer good features, this is the key to success. Some simple features will be based on publication time, authors, content length and so on
# - You may not ignore HTML and extract some features from there
# - You'd better experiment with your validation scheme. You should see a correlation between your local improvements and LB score
# - Try TF-IDF, ngrams, Word2Vec and GloVe embeddings
# - Try various NLP techniques like stemming and lemmatization
# - Tune hyperparameters. In our example, we've left only 50k features and used `C`=1 as a regularization parameter, this can be changed 
# - SGD and Vowpal Wabbit will learn much faster
# - In our course, we don't cover neural nets. But it's not obliged to use GRUs or LSTMs in this competition. 
