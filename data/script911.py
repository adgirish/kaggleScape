
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# # How Much Sugar Do We Eat?
# 
# After watching [That Sugar Film](http://www.imdb.com/title/tt3892434/) and getting more into cooking and food in general, I thought it would be interesting to see how much of particular ingredients the people of certain countries eat in their food.
# 
# ## Sugar
# The first check was how much sugar a number of countries take in. Companies have been putting more and more sugar into the products we eat for a number of decades now, particularly in products like soft drinks/sodas, which isn't great for our bodies. There are some stereotypical guesses one could make about the countries that consume the most sugar, but doing some data analysis is generally more informative. 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

world_food_facts = pd.read_csv('../input/FoodFacts.csv')
world_food_facts.countries = world_food_facts.countries.str.lower()
    
def mean(l):
    return float(sum(l)) / len(l)

world_sugars = world_food_facts[world_food_facts.sugars_100g.notnull()]

def return_sugars(country):
    return world_sugars[world_sugars.countries == country].sugars_100g.tolist()
    
# Get list of sugars per 100g for some countries
fr_sugars = return_sugars('france') + return_sugars('en:fr')
za_sugars = return_sugars('south africa')
uk_sugars = return_sugars('united kingdom') + return_sugars('en:gb')
us_sugars = return_sugars('united states') + return_sugars('en:us') + return_sugars('us')
sp_sugars = return_sugars('spain') + return_sugars('españa') + return_sugars('en:es')
nd_sugars = return_sugars('netherlands') + return_sugars('holland')
au_sugars = return_sugars('australia') + return_sugars('en:au')
cn_sugars = return_sugars('canada') + return_sugars('en:cn')
de_sugars = return_sugars('germany')

countries = ['FR', 'ZA', 'UK', 'US', 'ES', 'ND', 'AU', 'CN', 'DE']
sugars_l = [mean(fr_sugars), 
            mean(za_sugars), 
            mean(uk_sugars), 
            mean(us_sugars), 
            mean(sp_sugars), 
            mean(nd_sugars),
            mean(au_sugars),
            mean(cn_sugars),
            mean(de_sugars)]
            
y_pos = np.arange(len(countries))
    
plt.bar(y_pos, sugars_l, align='center', alpha=0.5)
plt.title('Average total sugar content per 100g')
plt.xticks(y_pos, countries)
plt.ylabel('Sugar/100g')
    
plt.show()


# ## Which countries eat the most sugar?
# 
# Interesting results, although for a number of countries the amount of data is a lot less (particularly countries like South Africa), and thus the data can be skewed. Another interesting note is the lack of any data on total sugars for Asian countries such as Japan and China. There are not many data entries for these countries either, but there are enough to make me wonder why there is no data on their sugar intake.
# 
# I'm making an assumption that food in the database, tied to a country, is consumed on a regular basis by its citizens, or at least often enough to keep these products on the shelves. The ranking of the countries analysed was this:
# 
# 1. **Netherlands**
# 2. **Canada**
# 3. **UK/USA**
# 4. **Australia**
# 5. **France**
# 6. **Germany**
# 7. **Spain**
# 8. **South Africa**

# # How Much Salt Do We Eat?
# 
# The other main ingredient that is criticised for its effects on health is salt, contained primarily in the sodium of a product's ingredient list. Here there was data on China and Japan, so I included them in the analysis, although again there is a much smaller dataset for these countries than for France or the UK.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

world_food_facts = pd.read_csv('../input/FoodFacts.csv')
world_food_facts.countries = world_food_facts.countries.str.lower()
    
def mean(l):
    return float(sum(l)) / len(l)

world_sodium = world_food_facts[world_food_facts.sodium_100g.notnull()]

def return_sodium(country):
    return world_sodium[world_sodium.countries == country].sodium_100g.tolist()
    
# Get list of sodium per 100g for some countries
fr_sodium = return_sodium('france') + return_sodium('en:fr')
za_sodium = return_sodium('south africa')
uk_sodium = return_sodium('united kingdom') + return_sodium('en:gb')
us_sodium = return_sodium('united states') + return_sodium('en:us') + return_sodium('us')
sp_sodium = return_sodium('spain') + return_sodium('españa') + return_sodium('en:es')
ch_sodium = return_sodium('china')
nd_sodium = return_sodium('netherlands') + return_sodium('holland')
au_sodium = return_sodium('australia') + return_sodium('en:au')
jp_sodium = return_sodium('japan') + return_sodium('en:jp')
de_sodium = return_sodium('germany')

countries = ['FR', 'ZA', 'UK', 'USA', 'ES', 'CH', 'ND', 'AU', 'JP', 'DE']
sodium_l = [mean(fr_sodium), 
            mean(za_sodium), 
            mean(uk_sodium), 
            mean(us_sodium), 
            mean(sp_sodium), 
            mean(ch_sodium),
            mean(nd_sodium),
            mean(au_sodium),
            mean(jp_sodium),
            mean(de_sodium)]

y_pos = np.arange(len(countries))
    
plt.bar(y_pos, sodium_l, align='center', alpha=0.5)
plt.title('Average sodium content per 100g')
plt.xticks(y_pos, countries)
plt.ylabel('Sodium/100g')
    
plt.show()


# ## Which countries eat the most salt?
# 
# Same as before, based on the same assumptions. Also of interest is that the amount of sodium in food, on average per 100g, is far, far less than the amount of sugar. 
# 
# The countries analysed were ranked like so:
# 
# 1. **China**
# 2. **Netherlands**
# 3. **USA**
# 4. **Australia**
# 5. **Spain**
# 6. **France**
# 7. **Germany**
# 8. **UK**
# 9. **South Africa**
# 10. **Japan**

# # How Many Additives Are In Our Food?
# 
# Next is how many additives are in our food on average. 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

world_food_facts = pd.read_csv('../input/FoodFacts.csv')
world_food_facts.countries = world_food_facts.countries.str.lower()
    
def mean(l):
    return float(sum(l)) / len(l)

world_additives = world_food_facts[world_food_facts.additives_n.notnull()]

def return_additives(country):
    return world_additives[world_additives.countries == country].additives_n.tolist()
    
# Get list of additives amounts for some countries
fr_additives = return_additives('france') + return_additives('en:fr')
za_additives = return_additives('south africa')
uk_additives = return_additives('united kingdom') + return_additives('en:gb')
us_additives = return_additives('united states') + return_additives('en:us') + return_additives('us')
sp_additives = return_additives('spain') + return_additives('españa') + return_additives('en:es')
ch_additives = return_additives('china')
nd_additives = return_additives('netherlands') + return_additives('holland')
au_additives = return_additives('australia') + return_additives('en:au')
jp_additives = return_additives('japan') + return_additives('en:jp')
de_additives = return_additives('germany')

countries = ['FR', 'ZA', 'UK', 'US', 'ES', 'CH', 'ND', 'AU', 'JP', 'DE']
additives_l = [mean(fr_additives), 
            mean(za_additives), 
            mean(uk_additives), 
            mean(us_additives), 
            mean(sp_additives), 
            mean(ch_additives),
            mean(nd_additives),
            mean(au_additives),
            mean(jp_additives),
            mean(de_additives)]

y_pos = np.arange(len(countries))
    
plt.bar(y_pos, sodium_l, align='center', alpha=0.5)
plt.title('Average amount of additives')
plt.xticks(y_pos, countries)
plt.ylabel('Amount of additives')
    
plt.show()


# ## Which countries' food contains the most additives?
# 
# I don't really worry about how many additives are in my food, but the results here are interesting again, mostly because it seems like there aren't any countries who consistently have the loweset/highest averages for these supposedly unhealthy ingredients.
# 
# The possible exceptions to this would be China, the Netherlands, and the USA. South Africa and Japan consistently have lower amounts, but they also have only a few entries in the dataset in general, which means their results could be biased in relation to other countries.
# 
# Ranking of countries by the amount of additives in their food:
# 
# 1. **China**
# 2. **Netherlands**
# 3. **USA**
# 4. **France**
# 5. **Australia**
# 6. **Spain**
# 7. **Germany**
# 8. **UK**
# 9. **South Africa**
# 10. **Japan**
