
# coding: utf-8

# An interesting look at metal bands generated per country and popular genres.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
world = pd.read_csv('../input/world_population_1960_2015.csv', encoding='latin-1')
bands = pd.read_csv('../input/metal_bands_2017.csv', encoding='latin-1')


# The first question I had is which country produces more metal bands?  This has actually been answered on Kaggle already, but it leads into some slightly different questions.

# In[ ]:


bands_country = bands['origin'].value_counts()
plt.title('Counts of bands per country')
sns.barplot(x=bands_country[:10].keys(), y=bands_country[:10].values)


# Unsurprisingly this yields the same results as other analysis, however I thought it was unfair that the large population of the U.S. would probably mean we generate more of everything.  So to try and see how many bands a country generates per capital I had to pull in the world population information.

# In[ ]:


# probably a better way to do this but I couldn't figure it out
band_count = [0 for i in range(len(world))]
for index, country in world.iterrows():
    if country['Country Name'] in bands_country:
        band_count[index] = bands_country[country['Country Name']]

# per capital = number of bands / population * arbitrary scalar
world['metal_pc'] = (pd.Series(band_count) / world['2015']) * 10000
world = world.sort_values(by = 'metal_pc', ascending=False)
plt.title('Bands created per capita')
sns.barplot(x=world[:10]['Country Name'], y=world[:10]['metal_pc'])


# Ah the Faroe Islands, not exactly the first place I think of when listening to metal, though it is off the coast of Denmark so it is in the right area. 
# 
# Next I was curious about which types of metal were more popular.  After I started digging into this I realized that many bands have multiple genres, due to this I wanted to see the popularity of pure single genre bands vs those of multiple types.  To do this I keep track of of three different values:  any reference to a genre, when a genre is the only type for a band, when a genre is not the only one listed for a band.

# In[ ]:


mixed_type = {}
pure_type = {}
all_type = {}
# count how many occurences of each genre we find
# make a special note if that band has only a single genre
for entry in list(bands['style'].to_dict().values()):
    subs = entry.split(',')
    for indv in subs:
        all_type[indv] = all_type.get(indv, 0) + 1
    if (len(subs) == 1):
        pure_type[subs[0]] = pure_type.get(subs[0], 0) + 1
    else:
        for indv in subs:
            mixed_type[indv] = mixed_type.get(indv, 0) + 1

# constructe a new df based on counts
type_df = pd.DataFrame()
type_df['mixed_counts'] = pd.Series(mixed_type)
type_df['pure_counts'] = pd.Series(pure_type)
type_df['all_type'] = pd.Series(all_type)
type_df = type_df.sort_values(by='mixed_counts', ascending=False)[:10]

# let seaborn handle the rest
sns.barplot(x=type_df.index, y=type_df['all_type'], color='#D17260')
sns.barplot(x=type_df.index, y=type_df['pure_counts'], color='#7FB7E0')
sns.plt.title('Counts of the top Metal Genres')
mixed_legend = mpatches.Patch(color='#D17260', label='All')
pure_legend = mpatches.Patch(color='#7FB7E0', label='Pure')
plt.legend(handles=[mixed_legend, pure_legend])


# I didn't have a supposition about what would be highest on the list so I can't say that I was surprised by the results (though I was hoping that melodic death or symphonic would be higher).  I thought it was interesting that 'hard rock' is rarely the only genre listed for a band while most of the others were just under half.
# 
# As suggested by user wuyanan, another way to look at this information would be to examine the bands that are pure against the ones that are only mixed, in the above graph I drew this slightly different so I could stack the bars instead of group them, here is what the data would look like done slightly different.

# In[ ]:


type_df[['pure_counts', 'mixed_counts']].plot(kind='bar', color=['#D17260', '#7FB7E0'])
sns.plt.title('Counts of the top Metal Genres')
mixed_legend = mpatches.Patch(color='#D17260', label='Pure')
pure_legend = mpatches.Patch(color='#7FB7E0', label='Mixed')
plt.legend(handles=[mixed_legend, pure_legend])


# The final question I had was about where the metal genre is in its life cycle, is it becoming more popular or is it dying out?  I only started listening to metal a few years ago so was I late to the party or not?

# In[ ]:


# find occurences of year start times
a = list(bands['formed'].to_dict().values())
b = list(bands['split'].to_dict().values())
year_counts = {i:a.count(i) for i in a if '-' not in i}
split_counts = {i:b.count(i) for i in b if '-' not in i}
year_df = pd.DataFrame()
year_df['band_counts'] = pd.Series(year_counts)
year_df['split_count'] = pd.Series(split_counts)

# seaborn
sns.plt.title('Bands Started Per Year 1979-2016')
ax = sns.barplot(x=year_df.index, y = year_df['band_counts'])
ax.set(xlabel='year', ylabel='band_count', xticklabels=[])
plt.show()


# When I first saw this I thought it was pretty depressing, looks like metal music is dead, there are hardly any new bands coming out... but then I realized that there are a lot of bands sticking around as well.  So lets take a look at bands starting vs bands quitting.

# In[ ]:


bands['formed'] = pd.to_numeric(bands['formed'], errors='coerce')
bands['split'] = pd.to_numeric(bands['split'], errors='coerce')
sns.violinplot(data=bands[['formed', 'split']], inner='quartile')


# Great!  At least the rate of bands breaking up has dramatically decreased too.  This might mean the eventual heat death of the metal genre however...though maybe a resurgence could inject some new life into the genre (as was encountered in the early 2000s).  So until the eventual heat occurs I wanted to at least see how many bands are currently active, this might give us a better indication of life.

# In[ ]:


year_df.fillna(0, inplace=True)
bands_alive = {}
prev_value = 0
for index, row in year_df.iterrows():
    new_total = prev_value + row['band_counts'] - row['split_count']
    bands_alive[index] = new_total
    prev_value = new_total
year_df['active_count'] = pd.Series(bands_alive)

plt.title('Active Bands Per Year 1970-2016')
ax = sns.barplot(x=year_df.index, y = year_df['active_count'])
ax.set(xlabel='year', ylabel='active_count', xticklabels=[])


# So for now at least the amount of metal bands is hitting a plateau a little over 2500 with a slight decline showing.
# 
# So this was a pretty fun exercise.  I think I could have used a few dataframe tricks to make the data modeling code smaller, but I just used the first method that came to mind and happened to work.
