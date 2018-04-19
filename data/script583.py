
# coding: utf-8

# # Impact of Famous Athletes on Baby Names
# 
# Big name athletes have a major impact on our culture, but do they influence the names of our babies? Let's find out. 
# 
# - Babe Ruth
# - Cassius Clay (Muhammad Ali)
# - Nolan Ryan
# - Greg Maddox
# - Michael Jordan
# - Tiger Woods
# - Mia Hamm
# - Serena Williams
# - Kobe Bryant
# - Lebron James

# In[ ]:


# Imports
import numpy as np
import pandas as pd
from bokeh.plotting import figure, show, output_notebook
from bokeh.palettes import brewer
colors = brewer['Spectral'][10]
output_notebook()

# Read the data
national_data = pd.read_csv("../input/NationalNames.csv")
print("Baby names loaded!")



# In[ ]:


# Create a dict with some of the world's top athletes with significant names. 
# Note: Active athletes have a retirement_year of 2014. 

athletes = [
    { 'name': 'Babe', 'full_name': 'Babe Ruth', 'rookie_year': 1914, 'retired_year': 1934, 'sex': 'M'},
    { 'name': 'Cassius', 'full_name': 'Cassius Clay', 'rookie_year': 1960, 'retired_year': 1979, 'sex': 'M'},
    { 'name': 'Nolan', 'full_name': 'Nolan Ryan', 'rookie_year': 1966, 'retired_year': 1993, 'sex': 'M'},
    { 'name': 'Maddux', 'full_name': 'Greg Maddux', 'rookie_year': 1986, 'retired_year': 2008, 'sex': 'M' },
    { 'name': 'Jordan', 'full_name': 'Michael Jordan', 'rookie_year': 1984, 'retired_year': 2003, 'sex': 'M' },
    { 'name': 'Mia', 'full_name': 'Mia Hamm', 'rookie_year': 1991, 'retired_year': 2004, 'sex': 'F' },
    { 'name': 'Tiger', 'full_name': 'Tiger Woods', 'rookie_year': 1996, 'retired_year': 2014, 'sex': 'M' },
    { 'name': 'Serena', 'full_name': 'Serena Williams', 'rookie_year': 1995, 'retired_year': 2014, 'sex': 'F' },
    { 'name': 'Kobe', 'full_name': 'Kobe Bryant', 'rookie_year': 1996, 'retired_year': 2014, 'sex': 'M' },
    { 'name': 'Lebron', 'full_name': 'Lebron James', 'rookie_year': 2003, 'retired_year': 2014, 'sex': 'M' }    
]          

# Utility method to print interesting stats for each athlete

def analyze_athlete(i, athlete): 
    # set variables for readability
    name, full_name, rookie_year, retired_year, sex  = athlete['name'], athlete['full_name'], athlete['rookie_year'], athlete['retired_year'], athlete['sex']
    
    # set DataFrame
    df = national_data[national_data['Name'] == name ]
    df = df[df['Gender'] == sex ]
    
    # calculate ranges for athletes career and prior generation
    career_years = range(rookie_year, retired_year)
    prior_generation_years = range(rookie_year-25, rookie_year)
    total_years = range(rookie_year-25, retired_year+25)
    
    # calculate interesting factoids
    rookie_name_count = df[df['Year'] == rookie_year]['Count'].sum()
    retired_name_count = df[df['Year'] == retired_year]['Count'].sum()
    career_name_count = df[df['Year'].isin(career_years)]['Count'].sum() 
    prior_name_count = df[df['Year'].isin(prior_generation_years)]['Count'].sum() 
    career_name_avg = round(df[df['Year'].isin(career_years)]['Count'].mean(), 2)
    prior_name_avg = round(df[df['Year'].isin(prior_generation_years)]['Count'].mean(), 2)
    influence_rate = round(100*(career_name_avg - prior_name_avg) / prior_name_avg, 2)
    influence_count = int(career_name_count - (prior_name_count*1.10)) # add 10% for approx 25y population growth
    
    
    # plot the trends
    p = figure(title=(full_name+' Inspired Trends'), x_axis_label='Year', y_axis_label='Count')
    
    df = df[df['Year'].isin(total_years)]
    x = df['Year'].tolist()
    y = df['Count'].tolist()  
    
    p.line(x, y, legend=name, line_width=2, line_color=colors[i], line_cap='round')
    p.quad(top=[(max(y))], bottom=[0], left=[rookie_year],
       right=[retired_year], color="#B3DE69", alpha=0.3, legend="Active Career")
    show(p)
    
    
    # print the some factoids
    print("Here's how {0} influenced for the name '{1}':".format(full_name, name))
    print("{0} babies named {1} were born in {2}, {3}'s rookie year.".format(rookie_name_count, name, rookie_year, full_name))
    print("{0} babies named {1} were born in {2}, {3}'s last active year.".format(retired_name_count, name, retired_year, full_name))
    print("On average, {0} babies named {1} were born per year while athlete's career was active.".format(career_name_avg, name))
    print("Comparatively, babies named {1} averaged {0} in the preceeding generation (25 years).".format(prior_name_avg, name))
    print("The frequency of babies named {0} changed by {1}% after the athlete's rookie debut.".format(name, influence_rate))
    print("{0} probably inspired the names of {1} babies.".format(full_name, influence_count))


# ## Visualizing the Impact of Athletes on Baby Names

# In[ ]:


for i, athlete in enumerate(athletes): 
    analyze_athlete(i, athlete)

