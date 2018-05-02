
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# #Comparing European habits
# What kind of food do Europeans consume? Do their food habits vary based on the region they belong to?
# 
# I created three regions according to the geographical place of the European countries: 
# North Europe (United Kingdom, Denmark, Sweden and Norway),
# Central Europe (France, Belgium, Germany, Switzerland and Netherlands) and
# South Europe (Portugal, Greece, Italy, Spain, Croatia and Albania).
# ##Food Categories
# Keeping the most popular categories across the dataset, one can see in the following chart how food habits vary 
# from north to south.
# 
# 

# In[ ]:


# coding=utf8
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

#load food data from file
food = pd.read_csv('../input/FoodFacts.csv', encoding='utf8')

#labels: list of Strings with country names
#returns the rows that have as country name any of the names in labels
def get_rows_country(labels):
    return food[food.countries.isin(labels)]

#labels: list of Strings with country names
#returns the categories and the percentage that appear in the rows of a particular country
def get_categories_counts(labels):
    rows = get_rows_country(labels)
    food_country = rows.main_category_en.value_counts() / len(rows)*100
    return food_country

#France
labels_france = ['France', 'en:FR', 'France,Europe','Belgique,France']
french_food = get_categories_counts(labels_france)

#Spain
labels_spain = ['España','en:ES','Espagne','Spain']
spanish_food = get_categories_counts(labels_spain)

#Germany
labels_germany = ['Deutschland','Germany','Allemagne','en:DE']
german_food = get_categories_counts(labels_germany)

#United Kingdom
labels_uk = ['en:UK','United Kingdom','en:GB','uk','UK']
uk_food = get_categories_counts(labels_uk)

#Belgium
labels_belgium = ['Belgique','en:BE','Belgique,France','Belgium','Belgique,France']
belgium_food = get_categories_counts(labels_belgium)

#Italia
labels_italia = ['Italia','en:IT','Italie']
italian_food = get_categories_counts(labels_italia)

#Switzerland
labels_switzerland = ['Suisse','Switzerland']
suisse_food = get_categories_counts(labels_switzerland)

#Netherlands
labels_netherlands = ['Netherlands', 'Holland']
holland_food = get_categories_counts(labels_netherlands)

#Denmark
labels_denmark = ['Denmark','Dänemark']
dannish_food = get_categories_counts(labels_denmark)

#Portugal
labels_portugal = ['Portugal','en:portugal']
portuguese_food = get_categories_counts(labels_portugal)

#Greece
labels_greece = ['Greece','en:GR','Grèce','en:greece']
greek_food = get_categories_counts(labels_greece)

#Sweden
labels_sweden = ['Sweden','en:SE','en:sweden']
swedish_food = get_categories_counts(labels_sweden)

#Norway
labels_norway = ['Norway','en:NO','en:norway']
norwegian_food = get_categories_counts(labels_norway)

#Croatia
labels_croatia = ['Croatia','en:HR','en:croatia']
croatian_food = get_categories_counts(labels_croatia)

#Albania
labels_albania = ['Albania','en:AL','en:albania']
albanian_food = get_categories_counts(labels_albania)

#convert each Seried to a dataframe
french_df = pd.DataFrame({'Category':french_food.index, 'Percentage':french_food.values})
spanish_df = pd.DataFrame({'Category':spanish_food.index, 'Percentage':spanish_food.values})
german_df = pd.DataFrame({'Category':german_food.index, 'Percentage':german_food.values})
uk_df = pd.DataFrame({'Category':uk_food.index, 'Percentage':uk_food.values})
belgium_df = pd.DataFrame({'Category':belgium_food.index, 'Percentage':belgium_food.values})
italia_df = pd.DataFrame({'Category':italian_food.index, 'Percentage':italian_food.values})
suisse_df = pd.DataFrame({'Category':suisse_food.index, 'Percentage':suisse_food.values})
holland_df = pd.DataFrame({'Category':holland_food.index, 'Percentage':holland_food.values})
dannish_df = pd.DataFrame({'Category':dannish_food.index, 'Percentage':dannish_food.values})
portuguese_df = pd.DataFrame({'Category':portuguese_food.index, 'Percentage':portuguese_food.values})
greek_df = pd.DataFrame({'Category':greek_food.index, 'Percentage':greek_food.values})
swedish_df = pd.DataFrame({'Category':swedish_food.index, 'Percentage':swedish_food.values})
norwegian_df = pd.DataFrame({'Category':norwegian_food.index, 'Percentage':norwegian_food.values})
croatian_df = pd.DataFrame({'Category':croatian_food.index, 'Percentage':croatian_food.values})
albanian_df = pd.DataFrame({'Category':albanian_food.index, 'Percentage':albanian_food.values})


#merge data frames per region
#North Europe
north_countries = [dannish_df, swedish_df, norwegian_df]
#set the first element of the merged frame
north = uk_df
for country in north_countries:
    north = pd.merge(left=north, right=country, on='Category', how='outer') #ensure outer join to keep lines with nan values
north.loc[:,'mean'] = north.mean(axis=1)

#Central Europe
central_countries = [holland_df, german_df, french_df, suisse_df]
central = belgium_df
for country in central_countries:
    central = pd.merge(left=central, right=country, on='Category', how='outer')
central.loc[:,'mean'] = central.mean(axis=1)

#South Europe
south_countries = [portuguese_df, greek_df, italia_df, croatian_df, albanian_df]
south = spanish_df
for country in south_countries:
    south = pd.merge(left=south, right=country, on='Category', how='outer')
south.loc[:,'mean'] = south.mean(axis=1)


frames_to_merge = [central, south]
merged = north[['Category','mean']]
for frame in frames_to_merge:
    merged = pd.merge(left=merged, right=frame[['Category','mean']], on='Category', how='outer')

merged.columns = ['Category', 'mean_north', 'mean_central', 'mean_south']

result =  merged.head(28)

fig, ax = plt.subplots(figsize=(20,10))
pos = list(range(len(result)))
width = 0.25

#plot
plt.barh(pos, result['mean_north'], width, color='#4957DF', edgecolor='w', label='North Europe')
plt.barh([p + width for p in pos], result['mean_central'], width, color='#ADD8B3', edgecolor='w', label='Central Europe')
plt.barh([p + width*2 for p in pos], result['mean_south'], width, color='#FF999F', edgecolor='w', label='South Europe')
#x labels
ax.xaxis.set_label_position('top')
ax.yaxis.set_ticks_position('none')
ax.xaxis.set_ticks_position('none')
ax.xaxis.tick_top()
plt.xlabel('Amount consumed (%)', fontsize=20)
#y labels
ax.set_yticklabels(result['Category'], alpha=0.7, fontsize=40)
plt.tick_params(labelsize=20)
ax.set_yticks([p + 1.5 * width for p in pos])
ax.invert_yaxis()
#background
ax.patch.set_facecolor('white')
ax.grid(False)
#legend
plt.legend(loc='center right',prop={'size':20})

sns.despine()
#x axis on top
ax.xaxis.tick_top()

plt.tight_layout()
plt.show()


# ##Ingredients
# What about the fat, protein and carbohydrate per 100g consumption across Europe?
# 
# It seems that for fat and protein the quantity falls from north to south. Also, Central Europe is 
# appeared to consume food with high amount of carbohydrate.

# In[ ]:


#define some String variables
fat_string = 'fat_100g'
protein_string = 'proteins_100g'
carbohydr_string = 'carbohydrates_100g'

###North Europe###
uk_rows = get_rows_country(labels_uk)
dannish_rows = get_rows_country(labels_denmark)
norwegian_rows = get_rows_country(labels_norway)
swedish_rows = get_rows_country(labels_sweden)
#define the first element of the north lists
north_fat = uk_rows[fat_string]
north_protein = uk_rows[protein_string]
north_carbohydr = uk_rows[carbohydr_string]
#define which countries should go in which region
north_fat_list = [dannish_rows[fat_string], norwegian_rows[fat_string], swedish_rows[fat_string]]
north_protein_list = [dannish_rows[protein_string], norwegian_rows[protein_string], swedish_rows[protein_string]]
north_carbohydr_list = [dannish_rows[carbohydr_string], norwegian_rows[carbohydr_string], swedish_rows[carbohydr_string]]
#append the lists in the corresponding region
for l in north_fat_list:
    north_fat.append(l)
for l in north_protein_list:
    north_protein.append(l)
for l in north_carbohydr_list:
    north_carbohydr.append(l)

###Central Europe###
belgium_rows = get_rows_country(labels_belgium)
holland_rows = get_rows_country(labels_netherlands)
german_rows = get_rows_country(labels_germany)
french_rows = get_rows_country(labels_france)
suisse_rows = get_rows_country(labels_switzerland)
#same as before
central_fat = belgium_rows[fat_string]
central_protein = belgium_rows[protein_string]
central_carbohydr = belgium_rows[carbohydr_string]

central_fat_list = [holland_rows[fat_string], german_rows[fat_string], french_rows[fat_string], suisse_rows[fat_string]]
central_protein_list = [holland_rows[protein_string], german_rows[protein_string], french_rows[protein_string], suisse_rows[protein_string]]
central_carbohydr_list = [holland_rows[carbohydr_string], german_rows[carbohydr_string], french_rows[carbohydr_string], suisse_rows[carbohydr_string]]
for l in central_fat_list:
    central_fat.append(l)
for l in central_protein_list:
    central_protein.append(l)
for l in central_carbohydr_list:
    central_carbohydr.append(l)

###South Europe###
spanish_rows = get_rows_country(labels_spain)
greek_rows = get_rows_country(labels_greece)
portuguese_rows = get_rows_country(labels_portugal)
italian_rows = get_rows_country(labels_italia)
croatian_rows = get_rows_country(labels_croatia)
albanian_rows = get_rows_country(labels_albania)
#same as before
south_fat = spanish_rows[fat_string]
south_protein = spanish_rows[protein_string]
south_carbohydr = spanish_rows[carbohydr_string]

south_fat_list = [greek_rows[fat_string], portuguese_rows[fat_string], italian_rows[fat_string], croatian_rows[fat_string], albanian_rows[fat_string]]
south_protein_list = [greek_rows[protein_string], portuguese_rows[protein_string], italian_rows[protein_string], croatian_rows[protein_string], albanian_rows[protein_string]]
south_carbohydr_list = [greek_rows[carbohydr_string], portuguese_rows[carbohydr_string], italian_rows[carbohydr_string], croatian_rows[carbohydr_string], albanian_rows[carbohydr_string]]
for li in south_fat_list:
    south_fat.append(li)
for li in south_protein_list:
    south_protein.append(li)
for li in south_carbohydr_list:
    south_carbohydr.append(li)

#create a frame with columns (Region, Fat, Protein, Carbohydrate) to keep the info together
elements_df = pd.DataFrame(['North Europe', np.mean(north_fat), np.mean(north_protein), np.mean(north_carbohydr)]).T
elements_df = pd.concat([elements_df, pd.DataFrame(['Central Europe', np.mean(central_fat), np.mean(central_protein), np.mean(central_carbohydr)]).T], ignore_index=True)
elements_df = pd.concat([elements_df, pd.DataFrame(['South Europe', np.mean(south_fat), np.mean(south_protein), np.mean(south_carbohydr)]).T], ignore_index=True)
elements_df.columns = ['Region', 'Fat', 'Protein', 'Carbohydrate']

#plot
fig, ax = plt.subplots(figsize=(20,10))
#index
pos = np.arange(3)
#plot title
ax.set_title('Average quantity per 100g', fontsize=20)

#auxiliary var for the colors in rgb format
const = float(255)
ax.plot(pos, elements_df['Fat'],  marker='o', markeredgecolor=(84/const, 170/const, 118/const), label='Fat',markersize=10)
ax.plot(pos, elements_df['Protein'], marker='o',markeredgecolor=(84/const, 170/const, 118/const), label='Protein',markersize=10)
ax.plot(pos, elements_df['Carbohydrate'], marker='o',markeredgecolor=(84/const, 170/const, 118/const), label='Carbohydrate',markersize=10)

#plot limits
ax.set_ylim(0,30)
ax.set_xlim(-1,3)
#x ticks adjustment
ax.set_xticks(pos)
ax.set_xticklabels(elements_df['Region'], alpha=0.7, fontsize=18)
#legend
plt.legend(loc='best',prop={'size':22})
plt.tight_layout()
plt.show()

