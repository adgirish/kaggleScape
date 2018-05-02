
# coding: utf-8

# # Principal Component Analysis of Pokemon Data
# This is a quick look at the Pokemon data, but instead at the variables themselves, we perform a PCA to get a reduced number of variables and examine the results. 

# In[ ]:


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Loading data
df = pd.read_csv('../input/Pokemon.csv')

# Renaming one column for clarity
columns = df.columns.tolist()
columns[0] = 'id'
df.columns = columns

# Selecting columns to consider
cols = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']

df.head()


# Before performing the PCA, I scale the data so that the distribution of HP, Attack, etc is centered around 0 with a standard deviation of 1. I do not consider the Total column as it's a sum of the following ones.

# In[ ]:


scaler = StandardScaler().fit(df[cols])
df_scaled = scaler.transform(df[cols])

print(df_scaled[:,0].mean())  # zero (or very close)
print(df_scaled[:,0].std())  # 1 (or very close)


# I opt to use as many principal components as necessary to explain 80% of the variance in the original dataset.

# In[ ]:


pca = PCA(n_components=0.8)  # consider enough components to explain 80% of the variance
pca.fit(df_scaled)
pcscores = pd.DataFrame(pca.transform(df_scaled))
pcscores.columns = ['PC'+str(i+1) for i in range(len(pcscores.columns))]
loadings = pd.DataFrame(pca.components_, columns=cols)
loadings.index = ['PC'+str(i+1) for i in range(len(pcscores.columns))]


# What the PCA does is construct new variables (or principal components) that explain most of the variance or scatter of the original dataset. Each component is a linear combination of all the variables and is perpendicular to every other component. Each variable in each component is multiplied by set of factors, the loading factors, which transforms the original data into this new component space. These loading factors are constrained so that the square of the sum is equal to 1, hence they can serve as weights to see which parameters are most important for a particular principal component.
# 
# Let's look at that in more detail with some figures.

# In[ ]:


load_sqr = loadings**2
ax = sns.heatmap(load_sqr.transpose(), linewidths=0.5, cmap="BuGn", annot=True)
ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=0, fontsize=8)
ax.set_yticklabels(ax.yaxis.get_majorticklabels(), rotation=0, fontsize=8)


# The darkest shades in the plot above indicate which parameters are the most important. For example, the loading factors for PC4 show that HP is the most dominant parameter. That is, Pokemon with high HP will have high absolute values of PC4.
# 
# Let's look at the actual values of the loading factors now:

# In[ ]:


ax = sns.heatmap(loadings.transpose(), center=0, linewidths=0.5, 
                 cmap="RdBu", vmin=-1, vmax=1, annot=True)
ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=0, fontsize=8)
ax.set_yticklabels(ax.yaxis.get_majorticklabels(), rotation=0, fontsize=8)


# Here you can see some more trends. For example, a Pokemon with high Defense or low Speed will have a positive value of PC2. On the other hand, things like Attack or Sp. Defense will control what value a Pokemon will have for PC3.

# Yet another way to look at this is to examine the data with a biplot, which is a scatter plot with vectors indicating what direction a datapoint will take in the PCA given its underlying parameters. For fun, I will color-code the Pokemon by Type to see if there is any obvious trends.

# In[ ]:


# Create labels based on Type 1
labels = set(df['Type 1'])
df['type'] = df['Type 1']
lab_dict = dict()
for i, elem in enumerate(labels):
    lab_dict[elem] = i
df = df.replace({'type' : lab_dict})

pc_types = pcscores.copy()
pc_types['Type'] = df['Type 1']

# Biplots
def make_plot(pcscores, loadings, xval=0, yval=1, max_arrow=0.2, alpha=0.4):
    n = loadings.shape[1]
    scalex = 1.0 / (pcscores.iloc[:, xval].max() - pcscores.iloc[:, xval].min())  # Rescaling to be from -1 to +1
    scaley = 1.0 / (pcscores.iloc[:, yval].max() - pcscores.iloc[:, yval].min())

    pcscores.iloc[:, xval] = pcscores.iloc[:, xval] * scalex
    pcscores.iloc[:, yval] = pcscores.iloc[:, yval] * scaley

    g = sns.lmplot(x='PC{}'.format(xval + 1), y='PC{}'.format(yval + 1), hue='Type', data=pcscores,
                   fit_reg=False, size=6, palette='muted')

    for i in range(n):
        # Only plot the longer ones
        length = sqrt(loadings.iloc[xval, i] ** 2 + loadings.iloc[yval, i] ** 2)
        if length < max_arrow:
            continue

        plt.arrow(0, 0, loadings.iloc[xval, i], loadings.iloc[yval, i], color='k', alpha=0.9)
        plt.text(loadings.iloc[xval, i] * 1.15, loadings.iloc[yval, i] * 1.15,
                 loadings.columns.tolist()[i], color='k', ha='center', va='center')

    g.set(ylim=(-1, 1))
    g.set(xlim=(-1, 1))


# In[ ]:


# Actually make a biplot (PC3 vs PC4)
make_plot(pc_types, loadings, 2, 3, max_arrow=0.3)


# Above, you can see that Pokemon are primarily centrally distributed; that is, their stats are fairly balanced. There don't appear to be any obvious trends with type. There are some outliers, for example, 2 normal type Pokemon with high values of PC4. If you recall, PC4's loading factors indicated that HP was the dominant parameter. 

# In[ ]:


best = pc_types.sort_values(by='PC4', ascending=False)[:2]
df.loc[best.index]


# In[ ]:


# Top HP Pokemon:
df.sort_values(by='HP', ascending=False)[:2]


# Indeed, the Pokemons with the highest HP also have the highest PC4, as expected.

# Let's have a look at all the PC combinations with a Seaborn pairplot:

# In[ ]:


g = sns.pairplot(pc_types, hue='Type', palette='muted')


# Again, we can't see a strong trend with Pokemon type. Out of curiosity, what's the Bug-type pokemon with high PC2 and PC3 values?

# In[ ]:


print(pc_types.sort_values(by='PC2', ascending=False)[:1])
print(pc_types.sort_values(by='PC3', ascending=False)[:1])


# In[ ]:


df.loc[230]


# This is Shuckle, a bug-type Pokemon with very high Defense and Sp. Def at the cost of Attack attributes. Indeed, the PC2-PC3 biplot reveals how it can be used to select high-defense Pokemon. These will be located towards the top right (high PC2 and PC3 values).

# In[ ]:


make_plot(pc_types, loadings, 1, 2, max_arrow=0.3)


# # Concluding Remarks
# One powerful thing of the PCA is the dimensionality reduction aspect of it. We went from having 6 variables to consider to only having 4. From this, it may be possible to examine fits or create classification models. That's something I'll look into in the future, but for now this concludes this notebook.
