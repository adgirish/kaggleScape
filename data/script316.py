
# coding: utf-8

# ## <center> Ridge and LightGBM: simple blending 
# 
# In this competition, the metric is mean absolute error (MAE), so it's better to optimize it directly. We'll do it with LightGBM, a powerfull implementation of gradient boosting. Remember: with Ridge regression we optimize mean squared error (MSE), and it's not the same as optimizng MAE. 
#  
# Moreover, here we'll apply a very simple method of averaging model predictions: blending.
# 
# <img src='https://habrastorage.org/webt/gm/ns/jp/gmnsjpxmgabagmi-bgialqtuhqa.png' width=30%>

# Import libraries.

# In[ ]:


import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import json
from tqdm import tqdm_notebook
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
import lightgbm as lgb


# The following code will help to throw away all HTML tags from article content/title.

# In[ ]:


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


# Let's have two paths – to raw data (downloaded from competition's page and ungzipped) and to processed data. Change this if you'd like to.

# In[ ]:


PATH_TO_DATA = '../input/'


# Assume you have all data downloaded from competition's [page](https://www.kaggle.com/c/how-good-is-your-medium-article/data) in the PATH_TO_RAW_DATA folder and `.gz` files are ungzipped.

# Supplementary function to read a JSON line without crashing on escape characters. 

# In[ ]:


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


# This function takes a JSON and forms a txt file leaving only article titles. When you resort to feature engineering and extract various features from articles, a good idea is to modify this function.

# In[ ]:


def preprocess(path_to_inp_json_file, path_to_out_txt_file):
    with open(path_to_inp_json_file, encoding='utf-8') as inp_file, open(path_to_out_txt_file, 'w', encoding='utf-8') as out_file:
        for line in tqdm_notebook(inp_file):
            json_data = read_json_line(line)
            content = json_data['title'].replace('\n', ' ').replace('\r', ' ')
            content_no_html_tags = strip_tags(content)
            out_file.write(content_no_html_tags + '\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', "preprocess(path_to_inp_json_file=os.path.join(PATH_TO_DATA, 'train.json'),\n           path_to_out_txt_file='train_titles.txt')")


# In[ ]:


get_ipython().run_cell_magic('time', '', "preprocess(path_to_inp_json_file=os.path.join(PATH_TO_DATA, 'test.json'),\n           path_to_out_txt_file='test_titles.txt')")


# In[ ]:


get_ipython().system('wc -l *_titles.txt')


# We'll use a very simple feature extractor – `CountVectorizer`, meaning that we resort to the Bag-of-Words approach. For now, we are leaving only 50k features. 

# In[ ]:


cv = CountVectorizer(max_features=50000)


# In[ ]:


get_ipython().run_cell_magic('time', '', "with open('train_titles.txt', encoding='utf-8') as input_train_file:\n    X_train = cv.fit_transform(input_train_file)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "with open('test_titles.txt', encoding='utf-8') as input_test_file:\n    X_test = cv.transform(input_test_file)")


# In[ ]:


X_train.shape, X_test.shape


# Read targets from file.

# In[ ]:


train_target = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_log1p_recommends.csv'), 
                           index_col='id')


# In[ ]:


y_train = train_target['log_recommends'].values


# Target is still somewhat skewed, though it's allready `log1p`-transformed (#claps with `log1p` transformation). Yet, we'll apply `log1p` once more time.

# In[ ]:


plt.hist(y_train, bins=30, alpha=.5, color='red', label='original', range=(0,10));
plt.hist(np.log1p(y_train), bins=30, alpha=.5, color='green', label='log1p', range=(0,10));
plt.legend();


# Make a 30%-holdout set. 

# In[ ]:


train_part_size = int(0.7 * train_target.shape[0])
X_train_part = X_train[:train_part_size, :]
y_train_part = y_train[:train_part_size]
X_valid =  X_train[train_part_size:, :]
y_valid = y_train[train_part_size:]


# Now we are ready to fit a linear model.

# In[ ]:


ridge = Ridge(random_state=17)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'ridge.fit(X_train_part, np.log1p(y_train_part));')


# After `log1p`-transformation, we need to apply an inverse  `expm1`-trasformation to predictions.

# In[ ]:


ridge_pred = np.expm1(ridge.predict(X_valid))


# Then, we fit a LightGBM model with `mean_absolute_error` as objective (it's important!).

# In[ ]:


lgb_x_train_part = lgb.Dataset(X_train_part.astype(np.float32), 
                           label=np.log1p(y_train_part))


# In[ ]:


lgb_x_valid = lgb.Dataset(X_valid.astype(np.float32), 
                      label=np.log1p(y_valid))


# In[ ]:


param = {'num_leaves': 31, 'num_trees': 100, 'objective': 'mean_absolute_error',
        'metric': 'mae'}


# In[ ]:


num_round = 100
bst_lgb = lgb.train(param, lgb_x_train_part, num_round, valid_sets=[lgb_x_valid], early_stopping_rounds=20)


# In[ ]:


lgb_pred = np.expm1(bst_lgb.predict(X_valid.astype(np.float32), 
                                    num_iteration=bst_lgb.best_iteration))


# Let's plot predictions and targets for the holdout set. Recall that these are #recommendations (= #claps) of Medium articles with the `np.log1p` transformation.

# In[ ]:


plt.hist(y_valid, bins=30, alpha=.5, color='red', label='true', range=(0,10));
plt.hist(ridge_pred, bins=30, alpha=.5, color='green', label='Ridge', range=(0,10));
plt.hist(lgb_pred, bins=30, alpha=.5, color='blue', label='Lgbm', range=(0,10));
plt.legend();


# As we can see, the prediction is far from perfect, and we get MAE $\approx$ 1.3 that corresponds to $\approx$ 2.7 error in #recommendations.

# In[ ]:


ridge_valid_mae = mean_absolute_error(y_valid, ridge_pred)
ridge_valid_mae


# In[ ]:


lgb_valid_mae = mean_absolute_error(y_valid, lgb_pred)
lgb_valid_mae


# Now let's mix predictions. We's just pick up weights 0.6 for Lgbm and 0.4 for Ridge, but these are typically tuned via cross-validation. 

# In[ ]:


mean_absolute_error(y_valid, .6 * lgb_pred + .4 * ridge_pred)


# Finally, train both models on the full accessible training set, make predictions for the test set and form submission files. 

# In[ ]:


get_ipython().run_cell_magic('time', '', 'ridge.fit(X_train, np.log1p(y_train));')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'ridge_test_pred = np.expm1(ridge.predict(X_test))')


# In[ ]:


lgb_x_train = lgb.Dataset(X_train.astype(np.float32),
                          label=np.log1p(y_train))


# In[ ]:


num_round = 50
bst_lgb = lgb.train(param, lgb_x_train, num_round)


# In[ ]:


lgb_test_pred = np.expm1(bst_lgb.predict(X_test.astype(np.float32)))


# In[ ]:


mix_pred = .6 * lgb_test_pred + .4 * ridge_test_pred


# In[ ]:


def write_submission_file(prediction, filename,
                          path_to_sample=os.path.join(PATH_TO_DATA, 'sample_submission.csv')):
    submission = pd.read_csv(path_to_sample, index_col='id')
    
    submission['log_recommends'] = prediction
    submission.to_csv(filename)


# If you apply the hack with submitting all zeroes and thus correcting predictions (it's a part of Assignment 6 in [our course](https://github.com/Yorko/mlcourse_open) and thus is not disclosed), you'll get the following scores (MAEs) on public LB:
# 
# - Ridge – 1.71365
# - LGBM - 1.70211
# - Ridge-LGBM mix – 1.65761
