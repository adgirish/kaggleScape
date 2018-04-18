
# coding: utf-8

# # General information
# 
# This kernel is dedicated to extensive EDA of DonorsChoose.org Application Screening competition as well as feature engineering and modelling.

# In[64]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import datetime
import lightgbm as lgb
import pandas_profiling as pp
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
stop = set(stopwords.words('english'))

import xgboost as xgb
import lightgbm as lgb


# In[2]:


train = pd.read_csv('../input/train.csv', parse_dates=['project_submitted_datetime'])
test = pd.read_csv('../input/test.csv', parse_dates=['project_submitted_datetime'])
resources = pd.read_csv('../input/resources.csv')
submission = pd.read_csv('../input/sample_submission.csv')


# ## Data overview

# In[3]:


train.head()


# In[4]:


resources.head()


# pandas_profiling library can generate a report, which gives a fast analysis of all features and shows their statistics.

# In[5]:


pp.ProfileReport(resources[['quantity', 'price']])


# In[6]:


pp.ProfileReport(train[['teacher_id', 'teacher_prefix', 'school_state', 'project_grade_category', 'teacher_number_of_previously_posted_projects', 'project_is_approved']])


# In[7]:


train.info()


# We can see several things from this overview:
# - There is a variety of features: numerical, categorical, text and date;
# - Only teacher prefix column has missing values;
# - project_essay_3 and project_essay_4 columns also have missing values, but this is a special case, we will talk about it later;
# - Some texts have '\n' and '\r' symbols, so they'll need to be cleaned;
# - Several categorical features have a lot of unique values;
# - Some teachers have submitted more than 70 projects;
# - 10183 rows are duplicate. We'll need to drop them;

# ### Feature analysis
# 
# We saw a lot of information about features, so let's now analyze each of them in more details.

# ### Essays

# Starting on May 17, 2016, the number of essays was reduced from 4 to 2. Essentially old 1 and 2 became new 1 and old 3 and 4 became new 2. This means that columns 'project_essay_3' and 'project_essay_4' have NaN values since that date. Let's see the number of projects before and after this date.

# In[8]:


print('Projests before 2016-05-17:', np.sum(train.project_submitted_datetime.dt.date < datetime.date(2016, 5, 7)))
print('Projests after 2016-05-17:', np.sum(train.project_submitted_datetime.dt.date >= datetime.date(2016, 5, 7)))


# So only 3751 (or 2%) of all projects were submitted before this date. We have several options here: split dataset into two parts and train different models on them, drop samples before this date, combine 1 with 2 and 3 with 4 for old samples or do nothing. I prefer combining old samples.

# In[9]:


train.loc[train.project_submitted_datetime.dt.date < datetime.date(2016, 5, 7), 'project_essay_1'] = train.loc[train.project_submitted_datetime.dt.date < datetime.date(2016, 5, 7), 'project_essay_1'] + ' ' + train.loc[train.project_submitted_datetime.dt.date < datetime.date(2016, 5, 7), 'project_essay_2']
train.loc[train.project_submitted_datetime.dt.date < datetime.date(2016, 5, 7), 'project_essay_2'] = train.loc[train.project_submitted_datetime.dt.date < datetime.date(2016, 5, 7), 'project_essay_3'] + ' ' + train.loc[train.project_submitted_datetime.dt.date < datetime.date(2016, 5, 7), 'project_essay_4']
train.drop(['project_essay_3', 'project_essay_4'], axis=1, inplace=True)


# Let's clean and visualize information in these columns.

# In[10]:


# replacing symbols which appeared due to formatting
train['project_essay_1'] = train['project_essay_1'].apply(lambda x: x.replace('\\r', ' ').replace('\\n', ' ').replace('  ', ' '))
train['project_essay_2'] = train['project_essay_2'].apply(lambda x: x.replace('\\r', ' ').replace('\\n', ' ').replace('  ', ' '))


# In[ ]:


text = ' '.join(train['project_essay_1'].values)
wordcloud = WordCloud(max_font_size=None, stopwords=stop, background_color='white',
                      width=1200, height=1000).generate(text)
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud)
plt.title('Top words for project_essay_1')
plt.axis("off")
plt.show()


# It isn't surprising that this essay is about students, how they learn and want continue learning as well as about some problems which the schools face.

# In[ ]:


text = ' '.join(train['project_essay_2'].values)
wordcloud = WordCloud(max_font_size=None, stopwords=stop, background_color='white',
                      width=1200, height=1000).generate(text)
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud)
plt.title('Top words for project_essay_2')
plt.axis("off")
plt.show()


# Second essay is more about problems which students face, things which will help them and how exactly will they help.

# ### project_resource_summary

# In[ ]:


train.project_resource_summary[train.project_resource_summary.str.contains('My students need') == False].values


# It is quite interesting that there are 2 projects where teachers don't use phrase "My students need" and still both of them are for students.

# In[ ]:


train['project_resource_summary'] = train['project_resource_summary'].apply(lambda x: x.replace('\\r', ' ').replace('\\n', ' ').replace('  ', ' '))


# Let's see what words are used in summaries of approved and non-approved projects.

# In[ ]:


text = ' '.join(train.loc[train['project_is_approved'] == 1, 'project_resource_summary'].values)
wordcloud = WordCloud(max_font_size=None, stopwords=stop, background_color='white',
                      width=1200, height=1000).generate(text)
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud)
plt.title('Top words for approved projects')
plt.axis("off")
plt.show()


# In[ ]:


text = ' '.join(train.loc[train['project_is_approved'] == 0, 'project_resource_summary'].values)
wordcloud = WordCloud(max_font_size=None, stopwords=stop, background_color='white',
                      width=1200, height=1000).generate(text)
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud)
plt.title('Top words for non-approved projects')
plt.axis("off")
plt.show()


# I'd say that there is no real difference.

# ### project_title
# 
# Let's see what project titles are about.

# In[ ]:


train['project_title'] = train['project_title'].apply(lambda x: x.replace('\\r', ' ').replace('\\n', ' ').replace('  ', ' '))
text = ' '.join(train['project_title'].values)
text = [i for i in ngrams(text.split(), 3)]
print('Common trigrams.')
Counter(text).most_common(20)


# In[ ]:


print('Common titles.')
train['project_title'].value_counts().head(20)


# Teachers usually write about things which they need or make some exclamations. Let's see summaries of projects with interesting titles!

# In[ ]:


print('Title:', 'Wiggle While You Work')
for i in train.loc[train['project_title'] == "Wiggle While You Work", 'project_resource_summary'].values[:3]:
    print(i)
print()
print('Title:', 'Can You Hear Me Now?')
for i in train.loc[train['project_title'] == "Can You Hear Me Now?", 'project_resource_summary'].values[:3]:
    print(i)
print()
print('Title:', 'We Like to Move It, Move It!')
for i in train.loc[train['project_title'] == "We Like to Move It, Move It!", 'project_resource_summary'].values[:3]:
    print(i)
print()
print('Title:', 'Listen Up!')
for i in train.loc[train['project_title'] == "Listen Up!", 'project_resource_summary'].values[:3]:
    print(i)
print()
print('Title:', "Let's Get Moving!")
for i in train.loc[train['project_title'] == "Let's Get Moving!", 'project_resource_summary'].values[:3]:
    print(i)
print()
print('Title:', 'Read All About It!')
for i in train.loc[train['project_title'] == "Read All About It!", 'project_resource_summary'].values[:3]:
    print(i)
print()


# ### teacher_prefix

# In[ ]:


train.teacher_prefix.value_counts(dropna=False)


# In[ ]:


pd.crosstab(train.teacher_prefix, train.project_is_approved, dropna=False, normalize='index')


# We can see that there are much more female teachers than male, which is normal. And it seems that project approval rate is independent from teacher gender. If anything writing title could lead to lesser chances.

# In[ ]:


#Let's fill missing values with most common one.
train['teacher_prefix'].fillna('Mrs.', inplace=True)


# ### school_state

# In[ ]:


train.groupby('school_state').agg({'project_is_approved': ['mean', 'count']}).reset_index().sort_values([('project_is_approved', 'mean')], ascending=False).reset_index(drop=True)


# As we can see states have a similar approval rate ranging from 0.81 to 0.89. And this rate doesn't depend on the number of projects.

# ### project_submitted_datetime

# At first let's create new features based on project_submitted_datetime: date, weekday and day of month. 

# In[ ]:


train['date'] = train.project_submitted_datetime.dt.date
train['weekday'] = train.project_submitted_datetime.dt.weekday
train['day'] = train.project_submitted_datetime.dt.day
count_by_date = train.groupby('date')['project_is_approved'].count()
mean_by_date = train.groupby('date')['project_is_approved'].mean()


# It could be interesting to see trends:

# In[ ]:


fig, ax1 = plt.subplots(figsize=(16, 8))
plt.title("Trends of approval rates and number of projects")
count_by_date.rolling(window=12,center=False).mean().plot(ax=ax1, legend=False)
ax1.set_ylabel('Projects count', color='b')
plt.legend(['Projects count'])
ax2 = ax1.twinx()
mean_by_date.rolling(window=12,center=False).mean().plot(ax=ax2, color='g', legend=False)
ax2.set_ylabel('Approval rate', color='g')
plt.legend(['Approval rate'], loc=(0.875, 0.9))
plt.grid(False)


# As we can see, we don't have even one year of data, so it is diffucult to make good conclusions about long-term trends, but let's try to make some observations:
# - At the very beginning, when teachers had to write 4 essays, the number of projects was quite low, but the approval rate was the highest;
# - Then approval rate began rapidly declining. Maybe teachers weren't able to adapt to new rules fast enough, and as a result they weren't able to meet new criteria?
# - There are two huge spikes in the number of submitted projects: at the beginning of August and September. I suppose that teachers requested new supplies for the beginning of new school year;
# - And in 2017 we can see a downtrend both in the number of submitted projects and approval rates;

# In[ ]:


fig, ax1 = plt.subplots(figsize=(16, 8))
plt.title("Project count and approval rate by day of week.")
sns.countplot(x='weekday', data=train, ax=ax1)
ax1.set_ylabel('Projects count', color='b')
plt.legend(['Projects count'])
ax2 = ax1.twinx()
sns.pointplot(x="weekday", y="project_is_approved", data=train, ci=99, ax=ax2, color='black')
ax2.set_ylabel('Approval rate', color='g')
plt.legend(['Approval rate'], loc=(0.875, 0.9))
plt.grid(False)


# We can see that number of submissions is the highest on Wednesday and then lowers on weekends.

# ### project_grade_category

# In[ ]:


pd.crosstab(train.project_grade_category, train.project_is_approved, dropna=False, normalize='index')


# It seems that project approval is independent from project grade category.

# ### project_subject_categories and project_subject_subcategories
# Project categories and subcategories often have several subjects divided by ",". Let's see the number of real subjects.

# In[ ]:


psc = [i.split(', ') for i in train.project_subject_categories.values]
psc = [i for j in psc for i in j]
print('Common subject categories.')
Counter(psc).most_common()


# In[ ]:


pss = [i.split(', ') for i in train.project_subject_subcategories.values]
pss = [i for j in pss for i in j]
print('Common subject subcategories.')
Counter(pss).most_common()


# ### teacher_number_of_previously_posted_projects

# In[ ]:


train.groupby('teacher_number_of_previously_posted_projects')['project_is_approved'].mean().plot()


# We can see that the more projects were submitted by a techer, the higher are chances for approval.

# ### Resources

# Let's transform the data:
# - create cost column which will show total cost of the item (quantity * price);
# - group data by project id and show aggregate values;

# In[ ]:


resources['cost'] = resources['quantity'] * resources['price']
resources_aggregated = resources.groupby('id').agg({'description': ['nunique'], 'quantity': ['sum'], 'cost': ['mean', 'sum']})
resources_aggregated.columns = ['unique_items', 'total_quantity', 'mean_cost', 'total_cost']
resources_aggregated.reset_index(inplace=True)
resources_aggregated.head()


# In[ ]:


print('99 percentile is {0}.'.format(np.percentile(resources_aggregated.mean_cost, 99)))
plt.boxplot(resources_aggregated.mean_cost);


# We can see that prices are heavily skewed. Let's try using boxcox transformation.

# In[ ]:


resources_aggregated['mean_cost'] = stats.boxcox(resources_aggregated.mean_cost + 1)[0]
plt.hist(resources_aggregated.mean_cost);
plt.title('Transformed mean cost');


# As we can see boxcox transformation is very useful in such cases. Let's try to do the same for other columns.

# In[ ]:


resources_aggregated['unique_items'] = stats.boxcox(resources_aggregated.unique_items + 1)[0]
plt.hist(resources_aggregated.unique_items);
plt.title('Transformed number of unique items');


# In[ ]:


resources_aggregated['total_quantity'] = stats.boxcox(resources_aggregated.total_quantity + 1)[0]
plt.hist(resources_aggregated.total_quantity);
plt.title('Transformed total quantity');


# In[ ]:


resources_aggregated['total_cost'] = stats.boxcox(resources_aggregated.total_cost + 1)[0]
plt.hist(resources_aggregated.total_cost);
plt.title('Transformed total cost');


# Now we can merge train and test data with processed resources.

# In[52]:


resources_aggregated.head()


# In[ ]:


train = pd.merge(train, resources_aggregated, how='left', on='id')
test = pd.merge(test, resources_aggregated, how='left', on='id')


# ## Feature engineering

# In[ ]:


# Applying the same feature transformation to test.
test.loc[test.project_submitted_datetime.dt.date < datetime.date(2016, 5, 7), 'project_essay_1'] = test.loc[test.project_submitted_datetime.dt.date < datetime.date(2016, 5, 7), 'project_essay_1'] + ' ' + test.loc[test.project_submitted_datetime.dt.date < datetime.date(2016, 5, 7), 'project_essay_2']
test.loc[test.project_submitted_datetime.dt.date < datetime.date(2016, 5, 7), 'project_essay_2'] = test.loc[test.project_submitted_datetime.dt.date < datetime.date(2016, 5, 7), 'project_essay_3'] + ' ' + test.loc[test.project_submitted_datetime.dt.date < datetime.date(2016, 5, 7), 'project_essay_4']
test.drop(['project_essay_3', 'project_essay_4'], axis=1, inplace=True)

test['project_essay_1'] = test['project_essay_1'].apply(lambda x: x.replace('\\r', ' ').replace('\\n', ' ').replace('  ', ' '))
test['project_essay_2'] = test['project_essay_2'].apply(lambda x: x.replace('\\r', ' ').replace('\\n', ' ').replace('  ', ' '))

test['project_resource_summary'] = test['project_resource_summary'].apply(lambda x: x.replace('\\r', ' ').replace('\\n', ' ').replace('  ', ' '))
test['project_title'] = test['project_title'].apply(lambda x: x.replace('\\r', ' ').replace('\\n', ' ').replace('  ', ' '))
test['teacher_prefix'].fillna('Mrs.', inplace=True)

test['weekday'] = test.project_submitted_datetime.dt.weekday
test['day'] = test.project_submitted_datetime.dt.day

# Also dropping date from train.
train.drop('date', axis=1, inplace=True)


# ### Categorical data

# There are four columns with categorical data: teacher_prefix, project_grade_category, weekday, school_state. First three of them have little number of unique values, so we can use one hot encoding for them.

# In[ ]:


train = pd.concat([train,
                   pd.get_dummies(train['teacher_prefix'], drop_first=True),
                   pd.get_dummies(train['project_grade_category'], drop_first=True),
                   pd.get_dummies(train['weekday'], drop_first=True)], axis=1)
train.drop(['teacher_prefix', 'project_grade_category', 'weekday'], axis=1, inplace=True)

test = pd.concat([test,
                   pd.get_dummies(test['teacher_prefix'], drop_first=True),
                   pd.get_dummies(test['project_grade_category'], drop_first=True),
                   pd.get_dummies(test['weekday'], drop_first=True)], axis=1)
test.drop(['teacher_prefix', 'project_grade_category', 'weekday'], axis=1, inplace=True)


# But school_state has a lot of unique values. Using one hot encoding will create a lot of new features. But there is another way of dealing with categorical data - target encoding. 

# In[ ]:


def target_encode(trn_series=None, 
                  tst_series=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):
    """
    
    https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior  
    """ 
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return ft_trn_series, ft_tst_series


# In[ ]:


train['school_state'], test['school_state'] = target_encode(train['school_state'], test['school_state'], train['project_is_approved'])


# In[ ]:


train['school_state'].head(10)


# ### Text data

# We have several features with text data and they need to be processed in different ways. But at first let's create new features based on texts: length of text (symbols) and number of words.

# In[ ]:


train['len_project_subject_categories'] = train['project_subject_categories'].apply(lambda x: len(x))
train['words_project_subject_categories'] = train['project_subject_categories'].apply(lambda x: len(x.split()))
train['len_project_subject_subcategories'] = train['project_subject_subcategories'].apply(lambda x: len(x))
train['words_project_subsubject_categories'] = train['project_subject_subcategories'].apply(lambda x: len(x.split()))
train['len_project_title'] = train['project_title'].apply(lambda x: len(x))
train['words_project_title'] = train['project_title'].apply(lambda x: len(x.split()))
train['len_project_resource_summary'] = train['project_resource_summary'].apply(lambda x: len(x))
train['words_project_resource_summary'] = train['project_resource_summary'].apply(lambda x: len(x.split()))
train['len_project_essay_1'] = train['project_essay_1'].apply(lambda x: len(x))
train['words_project_essay_1'] = train['project_essay_1'].apply(lambda x: len(x.split()))
train['len_project_essay_2'] = train['project_essay_2'].apply(lambda x: len(x))
train['words_project_essay_2'] = train['project_essay_2'].apply(lambda x: len(x.split()))

test['len_project_subject_categories'] = test['project_subject_categories'].apply(lambda x: len(x))
test['words_project_subject_categories'] = test['project_subject_categories'].apply(lambda x: len(x.split()))
test['len_project_subject_subcategories'] = test['project_subject_subcategories'].apply(lambda x: len(x))
test['words_project_subsubject_categories'] = test['project_subject_subcategories'].apply(lambda x: len(x.split()))
test['len_project_title'] = test['project_title'].apply(lambda x: len(x))
test['words_project_title'] = test['project_title'].apply(lambda x: len(x.split()))
test['len_project_resource_summary'] = test['project_resource_summary'].apply(lambda x: len(x))
test['words_project_resource_summary'] = test['project_resource_summary'].apply(lambda x: len(x.split()))
test['len_project_essay_1'] = test['project_essay_1'].apply(lambda x: len(x))
test['words_project_essay_1'] = test['project_essay_1'].apply(lambda x: len(x.split()))
test['len_project_essay_2'] = test['project_essay_2'].apply(lambda x: len(x))
test['words_project_essay_2'] = test['project_essay_2'].apply(lambda x: len(x.split()))


# Now let's start transforming texts. Categories and subcategories have little number of unique words, so we can use default values for TfidfVectorizer (only add stopwords).

# In[ ]:


vectorizer=TfidfVectorizer(stop_words=stop)
vectorizer.fit(train['project_subject_categories'])
train_project_subject_categories = vectorizer.transform(train['project_subject_categories'])
test_project_subject_categories = vectorizer.transform(test['project_subject_categories'])

vectorizer.fit(train['project_subject_subcategories'])
train_project_subject_subcategories = vectorizer.transform(train['project_subject_subcategories'])
test_project_subject_subcategories = vectorizer.transform(test['project_subject_subcategories'])


# Titles and summaries have real texts, so we need to limit TfidfVectorizer.

# In[ ]:


vectorizer=TfidfVectorizer(stop_words=stop, ngram_range=(1, 2), max_df=0.9, min_df=5, max_features=2000)
vectorizer.fit(train['project_title'])
train_project_title = vectorizer.transform(train['project_title'])
test_project_title = vectorizer.transform(test['project_title'])

vectorizer.fit(train['project_resource_summary'])
train_project_resource_summary = vectorizer.transform(train['project_resource_summary'])
test_project_resource_summary = vectorizer.transform(test['project_resource_summary'])


# Essays, of course are even bigger, so we need a limit as well.

# In[ ]:


vectorizer=TfidfVectorizer(stop_words=stop, ngram_range=(1, 3), max_df=0.9, min_df=5, max_features=2000)
vectorizer.fit(train['project_essay_1'])
train_project_essay_1 = vectorizer.transform(train['project_essay_1'])
test_project_essay_1 = vectorizer.transform(test['project_essay_1'])

vectorizer.fit(train['project_essay_2'])
train_project_essay_2 = vectorizer.transform(train['project_essay_2'])
test_project_essay_2 = vectorizer.transform(test['project_essay_2'])


# Now let's normalize continuous variables.

# In[ ]:


cols_to_normalize = ['teacher_number_of_previously_posted_projects', 'len_project_subject_categories', 'words_project_subject_categories', 'len_project_subject_subcategories',
                     'words_project_subsubject_categories', 'len_project_title', 'words_project_title', 'len_project_resource_summary', 'words_project_resource_summary',
                     'len_project_essay_1', 'words_project_essay_1', 'len_project_essay_2', 'words_project_essay_2']
scaler = StandardScaler()
for col in cols_to_normalize:
    #print(col)
    scaler.fit(train[col].values.reshape(-1, 1))
    train[col] = scaler.transform(train[col].values.reshape(-1, 1))
    test[col] = scaler.transform(test[col].values.reshape(-1, 1))


# Now we need to drop unnecessary columns:

# In[ ]:


to_drop = ['teacher_id', 'school_state', 'project_submitted_datetime', 'project_subject_categories', 'project_subject_subcategories', 'project_title', 'project_essay_1', 'project_essay_2', 'project_resource_summary']
for col in to_drop:
    train.drop([col], axis=1, inplace=True)
    test.drop([col], axis=1, inplace=True)


# In[ ]:


X = train.drop(['id', 'project_is_approved'], axis=1)
y = train['project_is_approved']
X_test = test.drop('id', axis=1)


# Let's stack all features.

# In[58]:


X_full = csr_matrix(hstack([X.values, train_project_subject_categories, train_project_subject_subcategories, train_project_resource_summary, train_project_essay_1, train_project_essay_2]))
X_test_full = csr_matrix(hstack([X_test.values, test_project_subject_categories, test_project_subject_subcategories, test_project_resource_summary, test_project_essay_1, test_project_essay_2]))

X_train, X_valid, y_train, y_valid = train_test_split(X_full, y, test_size=0.20, random_state=42)


# In[61]:


X_train, X_valid, y_train, y_valid = train_test_split(X_full, y, test_size=0.20, random_state=42)


# In[59]:


# Delete unnecessary data to free memory.
del train_project_subject_categories
del train_project_subject_subcategories
del train_project_resource_summary
del train_project_essay_1
del train_project_essay_2
del test_project_subject_categories
del test_project_subject_subcategories
del test_project_resource_summary
del test_project_essay_1
del test_project_essay_2
del X_full


# ## Models
# When we think about common classification and regression problems, XGBoost and LightGBM are most commonly used models. I'll use both of them and the prediction will be their average.

# In[63]:


params = {'eta': 0.05, 'max_depth': 15, 'objective': 'binary:logistic', 'eval_metric': 'auc', 'seed': 42, 'silent': True, 'colsample':0.9}
watchlist = [(xgb.DMatrix(X_train, y_train), 'train'), (xgb.DMatrix(X_valid, y_valid), 'valid')]
model = xgb.train(params, xgb.DMatrix(X_train, y_train), 1000,  watchlist, verbose_eval=10, early_stopping_rounds=20)


# In[ ]:


submission['project_is_approved'] = model.predict(xgb.DMatrix(X_test_full), ntree_limit=model.best_ntree_limit)


# In[ ]:


params = {'learning_rate': 0.05, 'max_depth': 14, 'boosting': 'gbdt', 'objective': 'binary', 'metric': 'auc', 'is_training_metric': True, 'seed': 42}
model2 = lgb.train(params, lgb.Dataset(X_train, label=y_train), 1000, lgb.Dataset(X_valid, label=y_valid), verbose_eval=10, early_stopping_rounds=20)


# In[ ]:


submission['project_is_approved'] += model2.predict(X_test_full, num_iteration=model2.best_iteration)
submission['project_is_approved'] = submission['project_is_approved'] / 2
submission.to_csv('xgb_lgb.csv', index=False)


# ## Conclusions
# 
# In this kernel I did EDA, feature engineering and used xgboost with lightgbm to make predictions. Here are some ideas to improve the quality of predictions:
# - generate more features based on continuous variables, for example mean price and quantity of items in a project;
# - tune tf-idf vectorizer parameters;
# - make careful features selection;
# - tune model parameters and blend more models;
# - etc.
# 
# I hope that this notebook was interesting! Feedback is welcomed and appreciated!
