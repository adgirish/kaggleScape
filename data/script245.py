
# coding: utf-8

# # Pikachu vs Bulbasaur Matchup
# 
# *Press "Fork" at the top-right of this screen and set the `your_attack_pokemon`, `their_defense_pokemon`, and `mode` parameters to run this notebook yourself. Use `mode = 'ORIGINAL'` to include all Pokémon or `mode = 'GO'` to only include Pokémon from the Pokémon Go game.*
# 
# This notebook enables you to find the best Pokémon to use to attack a defending Pokémon. It also visualizes the stats between both your attacking Pokémon and their defending Pokémon.

# In[ ]:


your_attack_pokemon = 'Pikachu'
their_defense_pokemon = 'Bulbasaur'
mode = 'GO' # GO or ORIGINAL


# Import the libraries    

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO


# Load Pokémon stats (first 151 for Pokemon Go)

# In[ ]:


pokemon = pd.read_csv('../input/Pokemon.csv')
pokemon = pokemon[pokemon['#'] < 151 if mode == 'GO' else 10000]


# Add Pokémon type matchup multipliers

# In[ ]:


types = pd.read_csv(StringIO("""Attacking,Normal,Fire,Water,Electric,Grass,Ice,Fighting,Poison,Ground,Flying,Psychic,Bug,Rock,Ghost,Dragon,Dark,Steel,Fairy
Normal,1,1,1,1,1,1,1,1,1,1,1,1,0.5,0,1,1,0.5,1
Fire,1,0.5,0.5,1,2,2,1,1,1,1,1,2,0.5,1,0.5,1,2,1
Water,1,2,0.5,1,0.5,1,1,1,2,1,1,1,2,1,0.5,1,1,1
Electric,1,1,2,0.5,0.5,1,1,1,0,2,1,1,1,1,0.5,1,1,1
Grass,1,0.5,2,1,0.5,1,1,0.5,2,0.5,1,0.5,2,1,0.5,1,0.5,1
Ice,1,0.5,0.5,1,2,0.5,1,1,2,2,1,1,1,1,2,1,0.5,1
Fighting,2,1,1,1,1,2,1,0.5,1,0.5,0.5,0.5,2,0,1,2,2,0.5
Poison,1,1,1,1,2,1,1,0.5,0.5,1,1,1,0.5,0.5,1,1,0,2
Ground,1,2,1,2,0.5,1,1,2,1,0,1,0.5,2,1,1,1,2,1
Flying,1,1,1,0.5,2,1,2,1,1,1,1,2,0.5,1,1,1,0.5,1
Psychic,1,1,1,1,1,1,2,2,1,1,0.5,1,1,1,1,0,0.5,1
Bug,1,0.5,1,1,2,1,0.5,0.5,1,0.5,2,1,1,0.5,1,2,0.5,0.5
Rock,1,2,1,1,1,2,0.5,1,0.5,2,1,2,1,1,1,1,0.5,1
Ghost,0,1,1,1,1,1,1,1,1,1,2,1,1,2,1,0.5,1,1
Dragon,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,0.5,0
Dark,1,1,1,1,1,1,0.5,1,1,1,2,1,1,2,1,0.5,1,0.5
Steel,1,0.5,0.5,0.5,1,2,1,1,1,1,1,1,2,1,1,1,0.5,2
Fairy,1,0.5,1,1,1,1,2,0.5,1,1,1,1,1,1,2,2,0.5,1"""))


# Merge Pokémon stats with type multipliers

# In[ ]:


pokemon_attack = pokemon.merge(types, left_on='Type 1', right_on='Attacking')


# Find the defending Pokémon's type and multiplier to calculate the adjusted attack

# In[ ]:


opponent_type = pokemon[pokemon['Name'] == their_defense_pokemon]['Type 1'].iloc[0]
opponent_multiplier = pokemon_attack[opponent_type]
adjusted_attack = pokemon_attack['Total'] * opponent_multiplier


# Normalize the total attack against the defending Pokémon to a number between 0 and 100

# In[ ]:


pokemon_attack['Adjusted Attack'] = (adjusted_attack - adjusted_attack.min()) / (adjusted_attack.max() - adjusted_attack.min()) * 100


# Sort by adjusted attack and plot

# In[ ]:


pokemon_attack.sort_values('Adjusted Attack', inplace=True)
pokemon_attack.tail(n=20).plot(kind='barh', x='Name', y='Adjusted Attack', figsize=(10, 7), title='Best 20 Pokemon to Attack %s' % their_defense_pokemon)


# Define a `pokeplot` method that creates a scatterplot

# In[ ]:


def pokeplot(x, y):
    f = sns.FacetGrid(pokemon, hue='Type 1', size=8)        .map(plt.scatter, x, y, alpha=0.5)        .add_legend()
    plt.subplots_adjust(top=0.9)
    f.fig.suptitle('%s vs. %s' % (x, y))
    f.ax.set_xlim(0,)
    f.ax.set_ylim(0,)

    attack_pokemon  = pokemon[pokemon['Name']==your_attack_pokemon]
    defense_pokemon = pokemon[pokemon['Name']==their_defense_pokemon]

    plt.scatter(attack_pokemon[x],attack_pokemon[y], s=100, c='#f46d43')
    plt.text(attack_pokemon[x]+6,attack_pokemon[y]-3, your_attack_pokemon, 
             fontsize=16, weight='bold', color='#f46d43')

    plt.scatter(defense_pokemon[y],defense_pokemon[y], s=100, c='#74add1')
    plt.text(defense_pokemon[x]+6,defense_pokemon[y]-3, their_defense_pokemon, 
             fontsize=16, weight='bold', color='#74add1')


# ## Attack vs Defense
# Compare the Pokémons' Attack and Defense stats using a scatterplot

# In[ ]:


pokeplot('Attack', 'Defense')


# ## Speed vs HP
# Compare the Pokémons' Speed and HP stats using a scatterplot

# In[ ]:


pokeplot('Speed', 'HP')


# ## Base Stats Comparison
# Compare the Pokémon's base stats using a radar chart

# In[ ]:


# Taken from https://www.kaggle.com/wenxuanchen/d/abcsds/pokemon/pokemon-visualization-radar-chart-t-sne

TYPE_LIST = ['Grass','Fire','Water','Bug','Normal','Poison',
            'Electric','Ground','Fairy','Fighting','Psychic',
            'Rock','Ghost','Ice','Dragon','Dark','Steel','Flying']

COLOR_LIST = ['#8ED752', '#F95643', '#53AFFE', '#C3D221', '#BBBDAF', '#AD5CA2', 
              '#F8E64E', '#F0CA42', '#F9AEFE', '#A35449', '#FB61B4', '#CDBD72', 
              '#7673DA', '#66EBFF', '#8B76FF', '#8E6856', '#C3C1D7', '#75A4F9']

# The colors are copied from this script: https://www.kaggle.com/ndrewgele/d/abcsds/pokemon/visualizing-pok-mon-stats-with-seaborn
# The colors look reasonable in this map: For example, Green for Grass, Red for Fire, Blue for Water...
COLOR_MAP = dict(zip(TYPE_LIST, COLOR_LIST))


# A radar chart example: http://datascience.stackexchange.com/questions/6084/how-do-i-create-a-complex-radar-chart
def _scale_data(data, ranges):
    (x1, x2), d = ranges[0], data[0]
    return [(d - y1) / (y2 - y1) * (x2 - x1) + x1 for d, (y1, y2) in zip(data, ranges)]

class RaderChart():
    def __init__(self, fig, variables, ranges, n_ordinate_levels = 6):
        angles = np.arange(0, 360, 360./len(variables))

        axes = [fig.add_axes([0.1,0.1,0.8,0.8],polar = True, label = "axes{}".format(i)) for i in range(len(variables))]
        _, text = axes[0].set_thetagrids(angles, labels = variables)
        
        for txt, angle in zip(text, angles):
            txt.set_rotation(angle - 90)
        
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.xaxis.set_visible(False)
            ax.grid('off')
        
        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i], num = n_ordinate_levels)
            grid_label = ['']+[str(int(x)) for x in grid[1:]]
            ax.set_rgrids(grid, labels = grid_label, angle = angles[i])
            ax.set_ylim(*ranges[i])
        
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]

    def plot(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

    def fill(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

    def legend(self, *args, **kw):
        self.ax.legend(*args, **kw)

# select display colors according to Pokemon's Type 1
def select_color(types):
    colors = [None] * len(types)
    used_colors = set()
    for i, t in enumerate(types):
        curr = COLOR_MAP[t]
        if curr not in used_colors:
            colors[i] = curr
            used_colors.add(curr)
    unused_colors = set(COLOR_LIST) - used_colors
    for i, c in enumerate(colors):
        if not c:
            try:
                colors[i] = unused_colors.pop()
            except:
                raise Exception('Attempt to visualize too many pokemons. No more colors available.')
    return colors

df = pd.read_csv('../input/Pokemon.csv')

# In this order, 
# HP, Defense and Sp. Def will show on left; They represent defense abilities
# Speed, Attack and Sp. Atk will show on right; They represent attack abilities
# Attack and Defense, Sp. Atk and Sp. Def will show on opposite positions
use_attributes = ['Speed', 'Sp. Atk', 'Defense', 'HP', 'Sp. Def', 'Attack']
# choose the Pokemons you like
use_pokemons = [their_defense_pokemon, your_attack_pokemon]

df_plot = df[df['Name'].map(lambda x:x in use_pokemons)==True]
datas = df_plot[use_attributes].values 
ranges = [[2**-20, df_plot[attr].max()] for attr in use_attributes]
colors = select_color(df_plot['Type 1']) # select colors based on pokemon Type 1 

fig = plt.figure(figsize=(10, 10))
radar = RaderChart(fig, use_attributes, ranges)
for data, color, pokemon in zip(datas, colors, use_pokemons):
    radar.plot(data, color = color, label = pokemon)
    radar.fill(data, alpha = 0.1, color = color)
    radar.legend(loc = 1, fontsize = 'small')
plt.title('Base Stats of '+(', '.join(use_pokemons[:-1])+' and '+use_pokemons[-1] if len(use_pokemons)>1 else use_pokemons[0]))

