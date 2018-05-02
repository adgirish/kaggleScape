
# coding: utf-8

# # Battle of the languages
# ![][1]  
# (Featuring [@inversion][2] as the python user)
# 
# **Let's look at what StackOverflow's 2017 survey has to say about Python and R users**. 
# 
# 
#   [1]: https://i.imgur.com/mctHfUO.png?1
#   [2]: http://kaggle.com/inversion

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

data = pd.read_csv('../input/survey_results_public.csv')

# Get users that have worked in R or Python in the last year
data['r_user'] = data['HaveWorkedLanguage'].apply(lambda x: 'R' in str(x).split('; '))
data['python_user'] = data['HaveWorkedLanguage'].apply(lambda x: 'Python' in str(x).split('; '))


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
pal = sns.color_palette()

fig, ax = plt.subplots()
plt.xlabel('Language')
plt.ylabel('User count')
plt.title('Number of surveyed users')
plt.bar([0, 1], [data['r_user'].sum(), data['python_user'].sum()], color=pal[3])
ax.set_xticks([0, 1])
ax.set_xticklabels(('R', 'Python'))
print()


# In[ ]:


#data_select = data[data['r_user'] + data['python_user'] > 0]
def langs(row):
    row = [int(i) for i in row.values]
    if sum(row) == 2: return 'Both'
    elif row[0] == 1: return 'R'
    elif row[1] == 1: return 'Python'
    else: return 'Neither'
    
order = ['Neither', 'Python', 'R', 'Both']
    
data['lang'] = data[['r_user', 'python_user']].apply(langs, axis=1)

plt.figure(figsize=(12, 8))
sns.boxplot(x="lang", y="Salary", data=data, order=order)
plt.title('Salary versus programming language')
print('Mean salary by language:')
print(data.groupby('lang')['Salary'].mean())


# So on average, R users make quite a bit more than Python users ($68k vs $61k)

# In[ ]:


def parse_dates(date):
    date = str(date)
    if 'Noon' in date:
        return 12.
    elif 'AM' in date:
        return float(date.split(':')[0])
    elif 'PM' in date:
        return float(date.split(':')[0]) + 12
    else:
        return np.nan

plt.figure(figsize=(12, 8))
plt.title('Ideal start time for an 8 hour work day (24h clock)')
data['WorkStartProcessed'] = data['WorkStart'].apply(parse_dates)
sns.violinplot(x="lang", y="WorkStartProcessed", data=data, order=order)
print('Mean of ideal workday start time:')
print(data.groupby('lang')['WorkStartProcessed'].mean())


# R users also like to start working slightly earlier in the morning (9:18AM vs 9:34AM on average)
# 
# Now onto some more important questions:

# In[ ]:


data['gif_with_a_g'] = data['PronounceGIF'] == 'With a hard "g," like "gift"'
data['gif_with_a_j'] = data['PronounceGIF'] == 'With a soft "g," like "jiff"'

plt.title('Percentage of users that pronounce gif with a hard g (like gift)')
sns.barplot(x="lang", y="gif_with_a_g", data=data, order=order)
plt.ylim(0, 1)
plt.show()

plt.title('Percentage of users that pronounce gif with a soft g (like jiff)')
sns.barplot(x="lang", y="gif_with_a_j", data=data, order=order)
plt.ylim(0, 1)


# It's within the confidence interval, but R users seem to pronounce gif like 'jiff' more often. Truly barbaric.

# In[ ]:


data['ClickyKeysProcessed'] = data['ClickyKeys'].apply(lambda x: 1 if x == 'Yes' else 0 if x == 'No' else np.nan)

plt.figure(figsize=(10, 8))
sns.barplot(x="lang", y="ClickyKeysProcessed", data=data, order=order)
plt.ylim(0, 1)
plt.title('If two developers are sharing an office, is it OK for one of them to get a mechanical keyboard with loud "clicky" keys?')


# Seems that Python users are more okay with colleagues having mechanical keyboards, and I would probably agree (unless you have cherry blue keys, those are audible from the next building)

# In[ ]:


# I transform strongly disgree to strongly agree on a scale of 1-5, and then 
# take the mean for each language
data['KinshipDevelopers'].value_counts()
def transform_agree(x):
    responses = ['Strongly disagree', 'Disagree', 'Somewhat agree', 'Agree', 'Strongly agree']
    if x not in responses:
        return np.nan
    return responses.index(x) + 1

data['KinshipDevelopersProcessed'] = data['KinshipDevelopers'].apply(transform_agree)
plt.figure(figsize=(10, 8))
sns.barplot(x="lang", y="KinshipDevelopersProcessed", data=data, order=order)
plt.ylim(1, 5)
plt.title('"I feel a sense of kinship to other developers"')


# In[ ]:


data['WorkPayCareProcessed'] = data['WorkPayCare'].apply(transform_agree)
plt.figure(figsize=(10, 8))
sns.barplot(x="lang", y="WorkPayCareProcessed", data=data, order=order)
plt.ylim(1, 5)
plt.title("I don't really care what I work on, so long as I'm paid well")


# In[ ]:


data['BoringDetailsProcessed'] = data['BoringDetails'].apply(transform_agree)
plt.figure(figsize=(10, 8))
sns.barplot(x="lang", y="BoringDetailsProcessed", data=data, order=order)
plt.ylim(1, 5)
plt.title('"I tend to get bored by implementation details"')

