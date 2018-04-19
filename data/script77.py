
# coding: utf-8

# # Pistol Round Analysis
# 
# Hello! I am the creator of this data-set and just wanted to provide a sample analysis for anyone interested in looking at CS.  I look at mainly the pistol round here but many of the techniques can be applied to all types of rounds.  There are many ways to analyze this dataset so I hope you can go off and answer interesting questions for yourself :)
# 
# In this notebook, I will analyze pistol round outcomes and damages done on average to improve player decision making on pistol rounds.  The analysis I provide in this notebook is very minimal, I'm only here to show code, however, you can find [the full analysis that I put on reddit with more in-depth talk about the numbers](https://www.reddit.com/r/GlobalOffensive/comments/72fkl7/mm_analytics_some_pistol_round_statistics_and/). 
# 
# The following questions will be answered in this notebook:
# 
# 1. What are the most common pistol round buys?
# - What is the ADR by each pistol on pistol rounds?
# - What sites do bomb get planted the most on pistol rounds?
# - After bomb gets planted at A/B Site, for all XvX situation, what is the win Probability for Ts?
# - In a 1v1, 1v2, 2v1, 2v2, should players play out of site/in-site or one-in one-out to deal the most damage while receiving the least?

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from scipy.misc import imread
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import matplotlib.colors as colors
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/mm_master_demos.csv', index_col=0)
map_bounds = pd.read_csv('../input/map_data.csv', index_col=0)
df.head()


# ### Data Prep
# 
# Let's first only isolate for active duty maps as they are the maps that most competitive players really care about.  I also want to first convert the in-game coordinates to overhead map coordinates.

# In[ ]:


active_duty_maps = ['de_cache', 'de_cbble', 'de_dust2', 'de_inferno', 'de_mirage', 'de_overpass', 'de_train']
df = df[df['map'].isin(active_duty_maps)]
df = df.reset_index(drop=True)
md = map_bounds.loc[df['map']]
md[['att_pos_x', 'att_pos_y', 'vic_pos_x', 'vic_pos_y']] = (df.set_index('map')[['att_pos_x', 'att_pos_y', 'vic_pos_x', 'vic_pos_y']])
md['att_pos_x'] = (md['ResX']*(md['att_pos_x']-md['StartX']))/(md['EndX']-md['StartX'])
md['att_pos_y'] = (md['ResY']*(md['att_pos_y']-md['StartY']))/(md['EndY']-md['StartY'])
md['vic_pos_x'] = (md['ResX']*(md['vic_pos_x']-md['StartX']))/(md['EndX']-md['StartX'])
md['vic_pos_y'] = (md['ResY']*(md['vic_pos_y']-md['StartY']))/(md['EndY']-md['StartY'])
df[['att_pos_x', 'att_pos_y', 'vic_pos_x', 'vic_pos_y']] = md[['att_pos_x', 'att_pos_y', 'vic_pos_x', 'vic_pos_y']].values


# In[ ]:


print("Total Number of Rounds: %i" % df.groupby(['file', 'round'])['tick'].first().count())


# ### Pistol Round Buys
# 
# Let's first start by taking only pistol rounds and count the number of rounds

# In[ ]:


avail_pistols = ['USP', 'Glock', 'P2000', 'P250', 'Tec9', 'FiveSeven', 'Deagle', 'DualBarettas', 'CZ']

df_pistol = df[(df['round'].isin([1,16])) & (df['wp'].isin(avail_pistols))]
print("Total Number of Pistol Rounds: %i" % df_pistol.groupby(['file', 'round'])['tick'].first().count())


# Let's first start by looking at pistol round buys.  We infer this from the damage dealt by pistols each round.  There is a bias here where if you did 0 damage with that pistol you had, then it doesn't get counted.  The potential bias is that aim punch will make most weapons get undercounted but I don't think it's a large issue.

# In[ ]:


pistol_buys = df_pistol.groupby(['file', 'round', 'att_side', 'wp'])['hp_dmg'].first()
(pistol_buys.groupby(['wp']).count()/pistol_buys.groupby(['wp']).count().sum())*100.


# Looks like Glock/USP trumps over most pistols.
# 
# ---
# 
# ### Heatmaps of Frequency of Pistol Damage
# 
# Next we can look at what are the most frequent spots when attacking as a T.  To keep it short, I will just do it on dust2 but changing `smap` will work on any map within `active_duty_maps`

# In[ ]:


smap = 'de_dust2'

bg = imread('../input/'+smap+'.png')
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(18,16))
ax1.grid(b=True, which='major', color='w', linestyle='--', alpha=0.25)
ax2.grid(b=True, which='major', color='w', linestyle='--', alpha=0.25)
ax1.imshow(bg, zorder=0, extent=[0.0, 1024, 0., 1024])
ax2.imshow(bg, zorder=0, extent=[0.0, 1024, 0., 1024])
plt.xlim(0,1024)
plt.ylim(0,1024)

plot_df = df_pistol.loc[(df_pistol.map == smap) & (df_pistol.att_side == 'Terrorist')]
sns.kdeplot(plot_df['att_pos_x'], plot_df['att_pos_y'], cmap='YlOrBr', bw=15, ax=ax1)
ax1.set_title('Terrorists Attacking')

plot_df = df_pistol.loc[(df_pistol.map == smap) & (df_pistol.att_side == 'CounterTerrorist')]
sns.kdeplot(plot_df['att_pos_x'], plot_df['att_pos_y'], cmap='Blues', bw=15, ax=ax2)
ax2.set_title('Counter-Terrorists Attacking')


# 
# ---
# 
# ### ADR by Pistols
# 
# Next let's take a look at the average damage per round dealt by a player given their pistol.  Note that if they had picked up a pistol during the round, it does get counted separately.  However, given that most pistol kills are headshots, it shouldn't skew the statistic that much (especially for USPS).

# In[ ]:


df_pistol.groupby(['file', 'round', 'wp', 'att_id'])['hp_dmg'].sum().groupby('wp').agg(['count', 'mean']).sort_values(by='mean')


# __Deagle has a massive advantage in damage__
# 
# ---
# 
# ### Bomb Site Plants
# 
# Let's now look at the Number of bomb plants by site.  This statistic tells us the T's preferences for deciding which site to take during the round.  Although the possibility of rotates are always there, it gives us a good idea of what to expect.

# In[ ]:


df_pistol[~df_pistol['bomb_site'].isnull()].groupby(['file', 'map', 'round', 'bomb_site'])['tick']         .first().groupby(['map', 'bomb_site']).count().unstack('bomb_site')


# ---
# 
# ### Post-plant Win Probabilities by Advantages
# 
# This one could be further disseminated but we want to be able to look at the win probabilities post plant given the context of how many Ts and CTs are alive at that time.  First, we can look at overall statistic:

# In[ ]:


bomb_prob_overall = df_pistol[~df_pistol['bomb_site'].isnull()].groupby(['file', 'round', 'map', 'bomb_site', 'winner_side'])['tick'].first().groupby(['map', 'bomb_site', 'winner_side']).count()
bomb_prob_overall_pct = bomb_prob_overall.groupby(level=[0,1]).apply(lambda x: 100 * x / float(x.sum()))
bomb_prob_overall_pct.unstack('map')


# Next we can first find for each round the post-plant situation (if it was planted at all) and calculate advantages.  I've given two options (XvX) or more generally by differences (e.g 5 Ts - 3 CTs = 2).

# In[ ]:


df_pistol['XvX'] = np.nan
for k,g in df_pistol.groupby(['file', 'round']):
    if((g['is_bomb_planted'] == True).any() == False):
        continue
    else:
        post_plant_survivors = 5-np.floor((g[~g['is_bomb_planted']].groupby('vic_side')['hp_dmg'].sum()/100.))
       # df_pistol.loc[g.index, 'XvX'] = "%iv%i" % (post_plant_survivors.get('Terrorist', 5), post_plant_survivors.get('CounterTerrorist', 5) )
        df_pistol.loc[g.index, 'XvX'] = post_plant_survivors.get('Terrorist', 5) - post_plant_survivors.get('CounterTerrorist', 5)


# Now we can calculate the Win probabilities by advantages, note that I isolate for just Terrorist because having CT columns (which is redundant), muddles the table.

# In[ ]:


bomb_prob = df_pistol[~df_pistol['XvX'].isnull()].groupby(['file', 'round', 'map', 'bomb_site', 'XvX', 'winner_side'])['tick'].first().groupby(['XvX', 'map', 'bomb_site', 'winner_side']).count()
bomb_prob_pct = bomb_prob.groupby(level=[0,1,2]).apply(lambda x: 100 * x / float(x.sum()))
bomb_prob_pct.xs('Terrorist', level=3).unstack(['XvX', 'bomb_site']).fillna(0)


# ---
# 
# ### In/Out-of-site ADR
# 
# I've always wondered if it's better to play post-plants in-site our out-of-site during a one-man up/down or equal situation. In-site has the advantage of peeking when the CTs are clearing outer-site spots but the con of being in a spot where you are forced to duel the CTs. Outer-site has pro of baiting shots and playing time but having to peek into the CT when he is defusing. Let's isolate for only 1v1, 2v1, 1v2, 2v2 situations and look at average ADR differential when you play inner site or outer site.
# 
# Before we do that though, we have to define what is considered Inner/Outer site.  Using some basic rectangles, I can draw sites on the map and then define them via simple top-left, bottom-right coordinates.

# In[ ]:


callouts = {
    'de_cache': {
        'B inner': [[310,782,413,865]],
        'A inner': [[278,165,388,320]]
    },
    'de_cbble': {
        'B inner': [[625,626,720,688]],
        'A inner': [[134,746,225,861]]
    },
    'de_train': {
        'B inner': [[405,754,607,812]],
        'A inner': [[582,462,713,539]]
    },
    'de_dust2': {
        'B inner': [[162,99,256,199]],
        'A inner': [[786,182,846,239]]
    },
    'de_mirage': {
        'B inner': [[188,245,286,345]],
        'A inner': [[498,737,610,835]]
    },
    'de_inferno': {
        'B inner': [[410,115,548,320]],
        'A inner': [[783,638,877,765]]
    },
    'de_overpass': {
        'B inner': [[686,294,745,359]],
        'A inner': [[452,174,560,272]]
    },
}

def find_callout(x,y,m,buffer=10): 
    callout = 'N/A'
    for c,coord in callouts[m].items():
        for box in coord:
            if ((box[2]+buffer >= x >= box[0]-buffer) & 
                (buffer+(1024-box[1]) >= y >= (1024-box[3])-buffer)):
                    callout = c
    return callout


# Let's see an example of this on dust2

# In[ ]:


smap = 'de_dust2'

def calc_plot_coord(l):
    tx,ty,bx,by = l
    by = 1024-by; ty=1024-ty;
    w = bx-tx; h= ty-by;
    return (tx,by,w,h)

bg = imread('../input/'+smap+'.png')
fig, ax = plt.subplots(figsize=(10,10))
ax.grid(b=True, which='major', color='w', linestyle='--', alpha=0.25)
ax.imshow(bg, zorder=0, extent=[0.0, 1024, 0., 1024])
plt.xlim(0,1024)
plt.ylim(0,1024)
patches = []
for k,coords in callouts[smap].items():
    for c in coords:
        x,y,w,h = calc_plot_coord(c)
        patches.append(mpatches.Rectangle((x,y),w,h))
        plt.text(x+w/2.3,y+h/2.3, s=k, size= 8, color='w')
colors = np.linspace(0, 1, len(patches))
collection = PatchCollection(patches, cmap=plt.cm.hsv, alpha=0.4)
collection.set_array(np.array(colors))
ax.add_collection(collection)


# Now let's convert the coordinates of T attackers or victims to callouts (either N/A: outer site or Inner A/Inner B)

# In[ ]:


bomb_dist = df_pistol[(df_pistol['XvX'].isin([-1, 0, 1]))
                     &(~df_pistol['bomb_site'].isnull())
                     &((df_pistol['vic_side'] == 'Terrorist') | (df_pistol['att_side'] == 'Terrorist'))]

bomb_dist['att_callout'] = bomb_dist.apply(lambda x: find_callout(x['att_pos_x'], x['att_pos_y'], x['map'], buffer=5), axis=1)
bomb_dist['vic_callout'] = bomb_dist.apply(lambda x: find_callout(x['vic_pos_x'], x['vic_pos_y'], x['map'], buffer=5), axis=1)


# Now we can calculate ADR by site

# In[ ]:


bomb_dist_total_dmg_att = bomb_dist.groupby(['file', 'map', 'round', 'att_callout', 'att_id'])['hp_dmg'].sum()
bomb_dist_total_dmg_vic = bomb_dist.groupby(['file', 'map', 'round', 'vic_callout', 'vic_id'])['hp_dmg'].sum()
dmg_dealt = bomb_dist_total_dmg_att.groupby(['map', 'att_callout']).agg(['count', 'mean'])
dmg_rec = bomb_dist_total_dmg_vic.groupby(['map', 'vic_callout']).agg(['count', 'mean'])
dmg_diff = dmg_dealt['mean'] - dmg_rec['mean']
dmg_diff.unstack('att_callout')

