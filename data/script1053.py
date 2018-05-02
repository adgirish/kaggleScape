
# coding: utf-8

# # Hi, Welcome to my kernel. We will show you how we achieve 80+% in this challenge. 
# # If you find this helpful, <span style="color:#3">please support us!!!</span>
# ## We will focus on the rate of **<span style="color:red">REJECTED</span>** according to different factors. This may help you to design your model on features that truly matter.
# ![donor](https://i2.wp.com/speechisbeautiful.com/wp-content/uploads/2015/06/donorschoose_logo.png)
# ### DonorsChoose is an US organisation that provide funding to school teachers who wish to improve their education environment. They received approximately thounsands of project proposals every year and the challange they are facing is dealing with enormous of proposal with limited volunteers. We're writing this kernel to help the organizer and participants to filter out insignificant features when designing pre-screening algorithm.
# ### Let's see what we going to analyze in this kernel:
# 1.  Introduction of dataset
# > Importing the libraries<br/>
# > Data preparation  <br/>
# 2. Analysis
# > **Which state has the highest rate of rejected?**<br/>
# > **What is the relationship between funding amount and rate of rejected? ** <br/>
# > **Will grade categories affect the rate of rejected?** <br/>
# > **Which combination of category and sub-category has highest rate of rejected?** <br/>
# > **What is the relationship between essay sentiment and rate of rejected?** <br/>
# 3. Suggestion
# 4. LGBM + TFIDF
# 5. GRU-ATT
# 6. Results
# 
# 

# # 1.1 Importing the libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999


# # 1.2 Data Preparation

# In[ ]:


df = pd.read_csv('../input/donorschoose-application-screening/train.csv')
df.sort_values('project_submitted_datetime').head(3)


# ## In the [train.csv](https://www.kaggle.com/c/donorschoose-application-screening/data), we got total of 16 unique features:
# * id (unique id, no repeated)
# * teacher_id (unique id, repeated)
# * teacher_prefix (Mr. Ms. Mrs. Teacher)
# * school_state 
# * project_submitted_datetime
# * project_grade_categories (4 categories)
# * project_subject_categories (9 categories)
# * project_subject_subcategories (30 categories)
# * project_title
# * project_essay_1
# * project_essay_2
# * project_essay_3
# * project_essay_4
# * project_resource_summary
# * teacher_number_of_previously_posted_projects
# * project_is_approved (0 - Rejected , 1 - Approved)

# In[ ]:


resources_df = pd.read_csv('../input/donorschoose-application-screening/resources.csv')
resources_df.head(3)


# ## In the [resources.csv](https://www.kaggle.com/c/donorschoose-application-screening/data), we got total of 4 unique features:
# * id (refer to id in train.csv, can be multiple entries)
# * description
# * quantity
# * price

# ## Migrate the features from resource.csv to train.csv
# We found that the resources.csv is additional features for a project proposal, it includes the resources that request by teachers with details such as desriptions, quantity and price. What we can do is we calculate the total quantity multiply with the price to get the total sum that requested by a project proposal. This amount is then concatenate into the main dataset *df*.

# In[ ]:


resources_df['amount'] = resources_df['quantity']*resources_df['price']
amount_df = resources_df.groupby('id')['amount'].agg('sum').sort_values(ascending=False).reset_index()

resource_amount_map = {}
for i, row in amount_df.iterrows():
    resource_amount_map[row['id']] = row['amount']

df.insert(4,'total_amount',df['id'].map(resource_amount_map))
df.tail(3)


# # 2 Analysis
# This section we will focus on analyzing the data and find out the possible reason behind the rejected proposals. As people have uploaded comprehensive individual feature visualization, we will straight away move into deeper analysis. You can always check other's kernel, e.g [An Educated Guess - DonorsChoose EDA](https://www.kaggle.com/headsortails/an-educated-guess-donorschoose-eda). Also, we will focus on the rate of rejected rather than approved as the ratio of approved is 85:15. The small percentage of rejected proposal is exactly what we going to analyse.
# 
# # 2.1 Which state has the highest rate of rejected?
# ### Statistics of proposal submission from each state
# First of all, let's see what is the number of submission from each states, you can always **hover** to see more details. 

# In[ ]:


scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

state_df = df.groupby('school_state')['project_is_approved'].agg('sum').reset_index().sort_values('school_state').reset_index()
submit_df = df['school_state'].value_counts().reset_index()
submit_df.columns=['school_state','total_submit']
submit_df = submit_df.sort_values('school_state').reset_index()

data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = state_df['school_state'],
        z = submit_df['total_submit'].astype(int),
        locationmode = 'USA-states',
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            )
        ),
        colorbar = dict(
            title = "Number of Proposal"
        )
    ) ]

layout = dict(
        title = 'US DonorChoose Proposal Submission from States',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)',
        ),
    )

fig = dict(data=data, layout=layout)

url = py.iplot(fig, filename='donor-state-submit')


# ### Statistics of rate of rejected proposal from each state

# In[ ]:


scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

state_df = df.groupby('school_state')['project_is_approved'].agg('sum').reset_index().sort_values('school_state').reset_index()
submit_df = df['school_state'].value_counts().reset_index()
submit_df.columns=['school_state','total_submit']
submit_df = submit_df.sort_values('school_state').reset_index()

data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = submit_df['school_state'],
        z = (submit_df['total_submit'].astype(int)-state_df['project_is_approved'].astype(int))/submit_df['total_submit'].astype(int),
        locationmode = 'USA-states',
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            )
        ),
        colorbar = dict(
            title = "Rate of Rejected"
        )
    ) ]

layout = dict(
        title = 'US DonorChoose Approval in States',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)',
        ),
    )

fig = dict(data=data, layout=layout)

url = py.iplot(fig, filename='donor-state-reject')


# ### Observation:
# * We found that the state that has most submission is **California**(25.7k), follow by **Texas**(12.3k) and **New York**(12.15k).
# * There is no visible relationship between neighbours of states and number of submissions. 
# * Texas has the highest rate of rejected (18.43%) which above the average (~15%).
# * All the neighbours of Texas have relatively high rate of rejected.

# # 2.2 What is the relationship between funding amount and rate of rejected?
# We usually assume that people will reject proposal that request for excessive amount of money especially from non-profit organization. Will it be true in this case? Let'see.
# ### Average amount of a single funded proposal in states

# In[ ]:


state_df = df[df['project_is_approved']==1]
state_df = state_df.groupby('school_state')['project_is_approved'].agg('sum').reset_index().sort_values('school_state').reset_index()
a_df = df.groupby('school_state')['total_amount'].agg('sum').reset_index().sort_values('school_state').reset_index()
submit_df = df['school_state'].value_counts().reset_index()
submit_df.columns=['school_state','total_submit']
submit_df = submit_df.sort_values('school_state').reset_index()

data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = state_df['school_state'],
        z = a_df['total_amount'].astype(int)/state_df['project_is_approved'].astype(int),
        #z = (submit_df['total_submit'].astype(int)-state_df['project_is_approved'].astype(int))/submit_df['total_submit'].astype(int),
        locationmode = 'USA-states',
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            )
        ),
        colorbar = dict(
            title = "USD"
        )
    ) ]

layout = dict(
        title = 'US DonorChoose Average Funding in States',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)',
        ),
    )

fig = dict(data=data, layout=layout)

url = py.iplot(fig, filename='donor-state-funding')


# ### Statistic of amount of proposal's request fund

# In[ ]:


df['total_amount']=df['total_amount'].apply(lambda x: int(round(x,-2)))
fund_total_df = df.groupby('total_amount').count()['project_is_approved'].head(50)
fund_total_df = fund_total_df.reset_index()
data = [
    go.Bar(
        x=fund_total_df['total_amount'], # assign x as the dataframe column 'x'
        y=fund_total_df['project_is_approved']
    )
]

url = py.iplot(data, filename='fund-total-chart')


# ### Funding Amount v/s Rate of Reject

# In[ ]:


fund_df = ((df.groupby('total_amount').count()['project_is_approved']-df.groupby('total_amount')['project_is_approved'].agg('sum'))/df.groupby('total_amount').count()['project_is_approved']).head(50)
fund_df = fund_df.reset_index()
data = [
    go.Bar(
        x=fund_df['total_amount'], # assign x as the dataframe column 'x'
        y=fund_df['project_is_approved']
    )
]

url = py.iplot(data, filename='fund-reject-chart')


# ### Observation:
# * Most proposal request for fund in between 100USD to 500USD and the number of proposal drop significantly after 2000USD.
# * Rate of reject increase gradually after 500USD from ~15% to ~20%.
# * The rate of reject after 2000USD should be consider carefully as the lesser the data, the greater the uncertainties.

# # 2.3 Will grade categories affect the rate of rejected?
# Since we got 4 categories in grade (3-5, 6-8, 9-12 and PreK-2), let's figure out whether the grades will affect the rate of rejected? 
# ### Grades v/s Funding Amount

# In[ ]:


df['total_amount'] = df['total_amount'].clip(upper=3000)
grade_amount_df = df.groupby(['project_grade_category','total_amount']).count()['id'].unstack().clip(upper=2000)

plt.figure(figsize=(20,2))#You can Arrange The Size As Per Requirement
ax = sns.heatmap(grade_amount_df, cmap='viridis_r')
plt.title("Correlation between Grades v/s Funding Amount")


# In[ ]:


import cufflinks as cf
cf.set_config_file(offline=True, world_readable=True, theme='ggplot')

grade_amount_df = df.groupby('project_grade_category').count()['total_amount']
grade_amount_df.iplot(kind='bar', yTitle='Number of Proposal', title='Submission from Grade Categories',
             filename='Grade categorical-bar-chart')


# In[ ]:


grade_approval_df = df.groupby('project_grade_category')['project_is_approved'].agg('sum')
grade_total_df = df.groupby('project_grade_category').count()['id']
((grade_total_df-grade_approval_df)/grade_total_df).iplot(kind='bar', yTitle='Rate of Rejected', title='Rejected Rate of Grade Categories',
             filename='Grade categorical-bar-chart')


# ### Observation:
# * Grade 3-5 and PreK-2 have the considerably greater amount in proposal submission compare to grade 6-8 and grade 9-12.
# * However, they have very close rate of rejected which lies in between 14.6% to 16.7%.
# * This shows that the grade categories is a poor features to training which can be ignored.

# # 2.4 Which combination of category and sub-category has highest rate of rejected?

# In[ ]:


cat_df = df[['project_subject_categories','project_subject_subcategories','project_is_approved']]
new_cat,new_sub,new_approve = [],[],[]

for i, row in cat_df.iterrows():
    cats = row['project_subject_categories'].split(', ')
    subs = row['project_subject_subcategories'].split(', ')
    for j in range(len(cats)):
        for k in range(len(subs)):
            new_cat.append(cats[j])
            new_sub.append(subs[k])
            new_approve.append(row['project_is_approved'])
            
new_cat = pd.DataFrame({'project_subject_categories':new_cat})
new_sub = pd.DataFrame({'project_subject_subcategories':new_sub})
new_approve = pd.DataFrame({'project_is_approved':new_approve})

cat_total_df = pd.concat([new_cat,new_sub,new_approve],axis=1).reset_index()
cat_approval_df = cat_total_df.groupby(['project_subject_categories','project_subject_subcategories'])['project_is_approved'].agg('sum')
cat_all_df = cat_total_df.groupby(['project_subject_categories','project_subject_subcategories']).count()['project_is_approved']
cat_heat = (cat_approval_df/cat_all_df).unstack()

plt.figure(figsize=(18,5))#You can Arrange The Size As Per Requirement
ax = sns.heatmap(cat_heat, cmap='viridis_r')
plt.title("Aprroved rate for categories and sub-categories combination")


# ### Observation:
# * There are total 9x30 combinations of categories and sub-categories.
# * Top 3 categories that consistent the most in the rate of approved is Applied Learning, Literacy & Language and Math & Science.
# * Both Warmth and Care & Hunger have very extreme rate of approved and rejected on certain sub-categories.
# 

# # 2.5 What is the relationship between essay sentiment and rate of rejected?
# In this section, we going to analyze:
# * The relationship between length of sentences in essays and rate of rejected
# * The relationship between essay sentiment (Polarity and Subjectivity) and rate of rejected
# 
# ### Length of Sentences in Essays v/s Rate of Rejected

# In[ ]:


from nltk import sent_tokenize, word_tokenize

def count_sent(text):
    if text == "nan" or not text:
        return 0
    sents = sent_tokenize(text)
    return len(sents)

sent_df = df[['project_essay_1','project_is_approved']]
sent_df['sent_length'] = sent_df['project_essay_1'].apply(count_sent)

apr = sent_df.groupby('sent_length')['project_is_approved'].agg('sum').reset_index().sort_values('sent_length')
tot = sent_df.groupby('sent_length').count()['project_is_approved'].reset_index().sort_values('sent_length')
rat = (tot['project_is_approved']-apr['project_is_approved'])/tot['project_is_approved']
rat.iplot(kind='bar', yTitle='Number of Sentences', title='Length of Sentences in Essay_1 v/s Rate of Rejected',
             filename='Grade categorical-bar-chart')


# In[ ]:


sent_df = df[['project_essay_2','project_is_approved']]
sent_df['project_essay_2'] = sent_df['project_essay_2'].astype(str)
sent_df['sent_length'] = sent_df['project_essay_2'].apply(count_sent)

apr = sent_df.groupby('sent_length')['project_is_approved'].agg('sum').reset_index().sort_values('sent_length')
tot = sent_df.groupby('sent_length').count()['project_is_approved'].reset_index().sort_values('sent_length')
rat = (tot['project_is_approved']-apr['project_is_approved'])/tot['project_is_approved']
rat.iplot(kind='bar', yTitle='Number of Sentences', title='Length of Sentences in Essay_2 v/s Rate of Rejected',
             filename='Grade categorical-bar-chart')


# In[ ]:


sent_df = df[['project_essay_3','project_is_approved']]
sent_df['project_essay_3'] = sent_df['project_essay_3'].astype(str)
sent_df['sent_length'] = sent_df['project_essay_3'].apply(count_sent)

apr = sent_df.groupby('sent_length')['project_is_approved'].agg('sum').reset_index().sort_values('sent_length')
tot = sent_df.groupby('sent_length').count()['project_is_approved'].reset_index().sort_values('sent_length')
rat = (tot['project_is_approved']-apr['project_is_approved'])/tot['project_is_approved']
rat.iplot(kind='bar', yTitle='Number of Sentences', title='Length of Sentences in Essay_3 v/s Rate of Rejected',
             filename='Grade categorical-bar-chart')


# In[ ]:


sent_df = df[['project_essay_4','project_is_approved']]
sent_df['project_essay_4'] = sent_df['project_essay_4'].astype(str)
sent_df['sent_length'] = sent_df['project_essay_4'].apply(count_sent)

apr = sent_df.groupby('sent_length')['project_is_approved'].agg('sum').reset_index().sort_values('sent_length')
tot = sent_df.groupby('sent_length').count()['project_is_approved'].reset_index().sort_values('sent_length')
rat = (tot['project_is_approved']-apr['project_is_approved'])/tot['project_is_approved']
rat.iplot(kind='bar', yTitle='Number of Sentences', title='Length of Sentences in Essay_4 v/s Rate of Rejected',
             filename='Grade categorical-bar-chart')


# ### Observation :
# * There is no significant correlation in between length of sentences and rate of rejected.
# * Both essay_3 and essay_4 leave unfill by most of the project proposal.
# * High rate of rejected is observed in proposal with 0 sentences in essay_2 (~32%)

# # 2.5 What is the relationship between essay sentiment and rate of rejected?
# In this section we going to analyze how essay's sentiment (polarity and subjectivity) will affect rate of rejected. First of all, we concatenate all the essay_1 to essay_4. Then, textblob sentiment is used to get the level of polarity and subjectivity for each project proposal. Below is the plotted heatmap that shows what essay sentiment have highest chance to be rejected. 

# In[ ]:


from textblob import TextBlob
df["essay"] = df["project_essay_1"].map(str) + df["project_essay_2"].map(str) + df["project_essay_3"].map(str) + df["project_essay_4"].map(str)
def get_polarity(text):
    textblob = TextBlob(text)
    pol = textblob.sentiment.polarity
    return round(pol,2)

def get_subjectivity(text):
    textblob = TextBlob(text)
    subj = textblob.sentiment.subjectivity
    return round(subj,2)

pol_df = df[['essay','project_is_approved']]
pol_df['sent_polarity'] = pol_df['essay'].apply(get_polarity)
pol_df['sent_subjectivity'] = pol_df['essay'].apply(get_subjectivity)

pol_rj_df = pol_df[pol_df['project_is_approved']==0]
#sub_df = df[['project_essay_1','project_is_approved']]
#sub_df['project_essay_1'] = sub_df['project_essay_1'].astype(str)
#sub_df['sent_subjectivity'] = sub_df['project_essay_1'].apply(subjectivity)

pol_total_df = pol_df.groupby(['sent_polarity','sent_subjectivity']).count()['project_is_approved'].unstack().clip(upper=5000)
pol_rj_df = pol_rj_df.groupby(['sent_polarity','sent_subjectivity']).count()['project_is_approved'].unstack().clip(upper=5000)
pol_rat_df = pol_rj_df/pol_total_df
#pol_heat_df
plt.figure(figsize=(20,20))#You can Arrange The Size As Per Requirement
ax = sns.heatmap(pol_rat_df, cmap='viridis_r')
plt.title("Correlation between Polarity and Subjectivity for Rate of")
#sent_df
#apr = sent_df.groupby('sent_polarity')['project_is_approved'].agg('sum').reset_index()
#tot = sent_df.groupby('sent_polarity').count()['project_is_approved'].reset_index()
#rat = apr['project_is_approved']/tot['project_is_approved']
#rat


# ### Observation :
# * The essay in centre of sentiment heatmap has the lowest rate of rejected
# * They rejected whatever proposal that subjectivity(greater than 0.78 , less than 0.18) and polarity(less than0.08 , greater than 0.58)
# * The rate of rejected increase gradually from centre of heatmap to periphery of heatmap.
# 
# # 3 Suggestion
# * From this kernel, we shows that not all the features will fit your training model, some are considerably flat and might be redundant to your model.
# * Features that suggested is **Categories & Sub-categories** and ** Sentiment of Essay**.
# * DonorChoose.org can use these features as pre-screen to allocate volunteer effectively.
# 

# # 4 LightGBM + TFIDF
# First of all, we need to thank **Oleg Panichev ** for his [LightGBM and Tf-idf Starter](https://www.kaggle.com/opanichev/lightgbm-and-tf-idf-starter/code). We implement his model but we have modified it by adding and removing some features as mentioned in EDA above.
# 1. We removed the teacher_prefix and grade_categories
# 2. We included the sentiment analysis (Polarity and Subjectivity)
# 

# In[ ]:


'''
import gc
import numpy as np
import pandas as pd
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm
import lightgbm as lgb


# Load Data
dtype = {
    'id': str,
    'teacher_id': str,
    'teacher_prefix': str,
    'school_state': str,
    'project_submitted_datetime': str,
    'project_grade_category': str,
    'project_subject_categories': str,
    'project_subject_subcategories': str,
    'project_title': str,
    'project_essay_1': str,
    'project_essay_2': str,
    'project_essay_3': str,
    'project_essay_4': str,
    'project_resource_summary': str,
    'teacher_number_of_previously_posted_projects': int,
    'project_is_approved': np.uint8,
}
data_path = os.path.join('..', 'input')
train = pd.read_csv(os.path.join(data_path, 'train.csv'), dtype=dtype, low_memory=True)
test = pd.read_csv(os.path.join(data_path, 'test.csv'), dtype=dtype, low_memory=True)
res = pd.read_csv(os.path.join(data_path, 'resources.csv'))

print(train.head())
# print(test.head())
print(train.shape, test.shape)


# Preprocess data
train['project_essay'] = train.apply(lambda row: ' '.join([
    str(row['teacher_prefix']), 
    str(row['school_state']), 
    str(row['project_grade_category']), 
    str(row['project_subject_categories']), 
    str(row['project_subject_subcategories']), 
    str(row['project_essay_1']), 
    str(row['project_essay_2']), 
    str(row['project_essay_3']), 
    str(row['project_essay_4']),
    ]), axis=1)
test['project_essay'] = test.apply(lambda row: ' '.join([
    str(row['teacher_prefix']), 
    str(row['school_state']), 
    str(row['project_grade_category']), 
    str(row['project_subject_categories']), 
    str(row['project_subject_subcategories']), 
    str(row['project_essay_1']), 
    str(row['project_essay_2']), 
    str(row['project_essay_3']), 
    str(row['project_essay_4']),
    ]), axis=1)

def extract_features(df):
    df['project_title_len'] = df['project_title'].apply(lambda x: len(str(x)))
    df['project_essay_1_len'] = df['project_essay_1'].apply(lambda x: len(str(x)))
    df['project_essay_2_len'] = df['project_essay_2'].apply(lambda x: len(str(x)))
    df['project_essay_3_len'] = df['project_essay_3'].apply(lambda x: len(str(x)))
    df['project_essay_4_len'] = df['project_essay_4'].apply(lambda x: len(str(x)))
    df['project_resource_summary_len'] = df['project_resource_summary'].apply(lambda x: len(str(x)))
  
extract_features(train)
extract_features(test)

from textblob import TextBlob
def get_polarity(text):
    textblob = TextBlob(unicode(text, 'utf-8'))
    pol = textblob.sentiment.polarity
    return round(pol,3)

def get_subjectivity(text):
    textblob = TextBlob(unicode(text, 'utf-8'))
    subj = textblob.sentiment.subjectivity
    return round(subj,3)

train['polarity'] = train['project_essay'].apply(get_polarity)
train['subjectivity'] = train['project_essay'].apply(get_subjectivity)
test['polarity'] = test['project_essay'].apply(get_polarity)
test['subjectivity'] = test['project_essay'].apply(get_subjectivity)

train = train.drop([
    'project_essay_1', 
    'project_essay_2', 
    'project_essay_3', 
    'project_essay_4'], axis=1)
test = test.drop([
    'project_essay_1', 
    'project_essay_2', 
    'project_essay_3', 
    'project_essay_4'], axis=1)

df_all = pd.concat([train, test], axis=0)
gc.collect()

# Merge with resources
res = pd.DataFrame(res[['id', 'price']].groupby('id').price.agg(\
    [
        'count', 
        'sum', 
        'min', 
        'max', 
        'mean', 
        'std', 
        # 'median',
        lambda x: len(np.unique(x)),
    ])).reset_index()
print(res.head())
train = train.merge(res, on='id', how='left')
test = test.merge(res, on='id', how='left')
del res
gc.collect()

# Preprocess columns with label encoder
print('Label Encoder...')
cols = [
    'teacher_id', 
    'school_state', 
    'project_subject_categories', 
    'project_subject_subcategories'
]

for c in tqdm(cols):
    le = LabelEncoder()
    le.fit(df_all[c].astype(str))
    train[c] = le.transform(train[c].astype(str))
    test[c] = le.transform(test[c].astype(str))
del le
gc.collect()
print('Done.')


# Preprocess timestamp
print('Preprocessing timestamp...')

train['project_submitted_datetime'] = pd.to_datetime(train['project_submitted_datetime']).values.astype(np.int64)
test['project_submitted_datetime'] = pd.to_datetime(test['project_submitted_datetime']).values.astype(np.int64)
print('Done.')


# Preprocess text
print('Preprocessing text...')
cols = [
    'project_title', 
    'project_essay', 
    'project_resource_summary'
]
n_features = [
    400, 
    5000, 
    400
]

for c_i, c in tqdm(enumerate(cols)):
    tfidf = TfidfVectorizer(max_features=n_features[c_i], min_df=3)
    tfidf.fit(df_all[c])
    tfidf_train = np.array(tfidf.transform(train[c]).todense(), dtype=np.float16)
    tfidf_test = np.array(tfidf.transform(test[c]).todense(), dtype=np.float16)

    for i in range(n_features[c_i]):
        train[c + '_tfidf_' + str(i)] = tfidf_train[:, i]
        test[c + '_tfidf_' + str(i)] = tfidf_test[:, i]
        
    del tfidf, tfidf_train, tfidf_test
    gc.collect()
    
print('Done.')
del df_all
gc.collect()

# Prepare data
cols_to_drop = [
    'id',
    'project_title', 
    'project_essay', 
    'project_resource_summary',
    'project_is_approved',
]
X = train.drop(cols_to_drop, axis=1, errors='ignore')
y = train['project_is_approved']
X_test = test.drop(cols_to_drop, axis=1, errors='ignore')
id_test = test['id'].values
feature_names = list(X.columns)
print(X.shape, X_test.shape)

del train, test
gc.collect()

# Build the model
cnt = 0
p_buf = []
n_splits = 5
n_repeats = 1
kf = RepeatedKFold(
    n_splits=n_splits, 
    n_repeats=n_repeats, 
    random_state=0)
auc_buf = []   

for train_index, valid_index in kf.split(X):
    print('Fold {}/{}'.format(cnt + 1, n_splits))
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 16,
        'num_leaves': 31,
        'learning_rate': 0.025,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'verbose': 0,
        'num_threads': 1,
        'lambda_l2': 1,
        'min_gain_to_split': 0,
    }  

    
    model = lgb.train(
        params,
        lgb.Dataset(X.loc[train_index], y.loc[train_index], feature_name=feature_names),
        num_boost_round=10000,
        valid_sets=[lgb.Dataset(X.loc[valid_index], y.loc[valid_index])],
        early_stopping_rounds=100,
        verbose_eval=100,
    )

    if cnt == 0:
        importance = model.feature_importance()
        model_fnames = model.feature_name()
        tuples = sorted(zip(model_fnames, importance), key=lambda x: x[1])[::-1]
        tuples = [x for x in tuples if x[1] > 0]
        print('Important features:')
        print(tuples[:50])

    p = model.predict(X.loc[valid_index], num_iteration=model.best_iteration)
    auc = roc_auc_score(y.loc[valid_index], p)

    print('{} AUC: {}'.format(cnt, auc))

    p = model.predict(X_test, num_iteration=model.best_iteration)
    if len(p_buf) == 0:
        p_buf = np.array(p)
    else:
        p_buf += np.array(p)
    auc_buf.append(auc)

    cnt += 1
    if cnt > 0: # Comment this to run several folds
        break
    
    del model
    gc.collect

auc_mean = np.mean(auc_buf)
auc_std = np.std(auc_buf)
print('AUC = {:.6f} +/- {:.6f}'.format(auc_mean, auc_std))

preds = p_buf/cnt

subm = pd.DataFrame()
subm['id'] = id_test
subm['project_is_approved'] = preds
subm.to_csv('submission.csv', index=False)
'''


# # 5 GRU-ATT
# Then, we introduce another model called GRU-ATT network. It's come from our machine learning research and will get publish soon. This model is serve for text classification and it got state-of-art performance in some dataset. We comment all the code as it will exceed the running time allow by Kaggle notebook. Let's see what's the output if we apply this model to DonorChoose data.

# In[ ]:


#Author-Poon
'''
import numpy as np
import pandas as pd
import cPickle
from collections import defaultdict
import re

from bs4 import BeautifulSoup
import sys
import os

os.environ['KERAS_BACKEND']='theano'
import keras
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model
from keras.models import load_model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

MAX_SENT_LENGTH = 100
MAX_SENTS = 50
MAX_NB_WORDS = 50000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)    
    string = re.sub(r"\'", "", string)    
    string = re.sub(r"\"", "", string)   
    return string.strip().lower()

dtype = {
    'id': str,
    'teacher_id': str,
    'teacher_prefix': str,
    'school_state': str,
    'project_submitted_datetime': str,
    'project_grade_category': str,
    'project_subject_categories': str,
    'project_subject_subcategories': str,
    'project_title': str,
    'project_essay_1': str,
    'project_essay_2': str,
    'project_essay_3': str,
    'project_essay_4': str,
    'project_resource_summary': str,
    'teacher_number_of_previously_posted_projects': int,
    'project_is_approved': np.uint8,
}
#data_path = os.path.join('..', 'input')
train = pd.read_csv('train.csv', dtype=dtype, low_memory=True)
test = pd.read_csv( 'test.csv', dtype=dtype, low_memory=True)

test['project_is_approved'] = 1

train['text'] = train.apply(lambda row: ' '.join([
    str(row['project_title']), 
    str(row['project_resource_summary']), 
    str(row['project_essay_1']), 
    str(row['project_essay_2']), 
    str(row['project_essay_3']), 
    str(row['project_essay_4'])]), axis=1)
test['text'] = test.apply(lambda row: ' '.join([
    str(row['project_title']), 
    str(row['project_resource_summary']), 
    str(row['project_essay_1']), 
    str(row['project_essay_2']), 
    str(row['project_essay_3']), 
    str(row['project_essay_4'])]), axis=1)

train = train.drop([
    'teacher_id',
    'teacher_prefix',
    'school_state',
    'project_submitted_datetime',
    'project_grade_category',
    'project_subject_categories',
    'project_subject_subcategories',
    'project_title',
    'project_essay_1',
    'project_essay_2',
    'project_essay_3',
    'project_essay_4',
    'project_resource_summary',
    'teacher_number_of_previously_posted_projects'], axis=1)
test = test.drop([
    'teacher_id',
    'teacher_prefix',
    'school_state',
    'project_submitted_datetime',
    'project_grade_category',
    'project_subject_categories',
    'project_subject_subcategories',
    'project_title',
    'project_essay_1',
    'project_essay_2',
    'project_essay_3',
    'project_essay_4',
    'project_resource_summary',
    'teacher_number_of_previously_posted_projects'], axis=1)

data_train = pd.concat([train,test],axis = 0).reset_index()

import nltk
from nltk import tokenize

reviews = []
labels = []
texts = []
instance_inputs = []
comment_id = []

#Return dimension of data_train.review([0]=row)
for idx in range(data_train.text.shape[0]):
    sys.stdout.write("\rProcessing ---- %d"%idx)
    sys.stdout.flush()
    comment_id.append(data_train.id[idx])
    text = ''.join(data_train.text[idx])
    #parse the sentences into beautifulsoup object
    #print text
    text = BeautifulSoup(text)
    text = clean_str(text.get_text().encode('ascii','ignore'))
    #insert clear text into texts array
    texts.append(text)
    #Return a sentence-tokenized copy of text( divide string into substring by punkt)
    sentences = tokenize.sent_tokenize(text)
    reviews.append(sentences)
    labels.append(data_train.project_is_approved[idx])

#Class for vectorizing texts (Tokenizer)
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
#list of texts to train on
tokenizer.fit_on_texts(texts)

#New 3D array filled with zero with (length,15,100) length= num of char
data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
word_len = np.zeros(10000)
#enumerate produce a tuple(index)
for i, sentences in enumerate(reviews):
    for j, sent in enumerate(sentences):
	word_len[len(sentences)] +=1
        if j< MAX_SENTS:
	    #Split sentence into a list of words
            wordTokens = text_to_word_sequence(sent)
            k=0
            for _, word in enumerate(wordTokens):
                if k<MAX_SENT_LENGTH and tokenizer.word_index[word]<MAX_NB_WORDS:
		    #dictionary mapping word to their rank/index (int)
                    data[i,j,k] = tokenizer.word_index[word]
                    k=k+1                    
                    
word_index = tokenizer.word_index
print('Total %s unique tokens.' % len(word_index))

#Converts a class vector (integers) to binary class matrix
labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

nb_validation_samples = 78035
#split training and validation set
x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]
comment_id = comment_id[-nb_validation_samples:]

print('Number of positive and negative reviews in traing and validation set')
print y_train.sum(axis=0)
print y_val.sum(axis=0)

GLOVE_DIR = "../../Glove"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    #split the vector of 100d
    values = line.split()
    #word at values[0]
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SENT_LENGTH,
                            trainable=True)

class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        self.W = self.add_weight(name='kernel', 
                                  shape=(input_shape[-1],),
                                  initializer='normal',
                                  trainable=True)
        super(AttLayer, self).build(input_shape) 

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))     
        ai = K.exp(eij)
        weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')
        weighted_input = x*weights.dimshuffle(0,1,'x')
        return weighted_input.sum(axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = {}
        base_config = super(AttLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

print('Shape of data tensor:', data.shape)
sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32', name='main_input')
embedded_sequences = embedding_layer(sentence_input)
l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
l_dense = TimeDistributed(Dense(200))(l_lstm)
l_att = AttLayer()(l_dense)
sentEncoder = Model(sentence_input, l_att)
review_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
review_encoder = TimeDistributed(sentEncoder)(review_input)
l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(review_encoder)
l_dense_sent = TimeDistributed(Dense(200))(l_lstm_sent)
l_att_sent = AttLayer()(l_dense_sent)
preds = Dense(2, activation='softmax')(l_att_sent)
model = Model(review_input, preds)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("model fitting - Hierachical attention network")
print model.summary()
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=3, batch_size=100, verbose=2)

score = model.evaluate(data, labels, batch_size = 100, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
yFit = model.predict(data, batch_size = 100, verbose=2)
yFit = yFit[-nb_validation_samples:,1]
val_pre = pd.DataFrame({'project_is_approved':yFit})
val_id = pd.DataFrame({'id':comment_id})
data_test_merged = pd.concat([val_id,val_pre], axis=1)
data_test_merged.to_csv('GRU.csv', encoding='utf-8', index = False)
'''


# # Result

# In[ ]:


LGBM = pd.read_csv('../input/submit/LGBMTFIDF.csv')
GRU = pd.read_csv('../input/submit/GRU.csv')
combine = pd.read_csv('../input/submit/submission_combine7.csv')

res_df = pd.concat([LGBM['project_is_approved'],GRU['project_is_approved'],combine['project_is_approved']],axis=1)
res_df.columns=['LGBMTDIDF','GRU','Combined']

data = []
for col in res_df.columns:
    data.append(  go.Box( y=res_df[col], name=col, showlegend=False ) )
    
data.append( go.Scatter( x = res_df.columns, y = res_df.mean(), mode='lines', name='mean' ) )

# IPython notebook
# py.iplot(data, filename='pandas-box-plot')

url = py.iplot(data, filename='pandas-box-plot')


# From the box plot we can see the result from LGBM-TFIDF has the higher mean compare to result from GRU which shows that GRU is more distributed in comparison. By combining result from both model with weights, we can have advantages from both model and a more balanced result.
# ## Result submission

# In[ ]:


output = pd.read_csv('../input/submit/submission_combine7.csv')
output.to_csv('submission.csv', index=False)


# ----
# # Thanks for watching. We are new to Kaggle, if you find this kernel is helpful, please **VOTE** for our kernel. Your support is Greatly appreciate!!! Also, please feel free to drop us any question or comment.
# 
# # To be continued..
