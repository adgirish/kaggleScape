
# coding: utf-8

# Let's start by importing the database. 
# The row factory setting allows to use column names instead of integer when exploring a resultset.

# In[ ]:


import sqlite3

database = '../input/database.sqlite'
conn = sqlite3.connect(database)
conn.row_factory = sqlite3.Row
cur = conn.cursor()


# Next we query a match. I chose a home game of my favourite team: Olympique Lyonnais!

# In[ ]:


match_api_id = 1989903
sql = 'SELECT * From MATCH WHERE match_api_id=?'
cur.execute(sql, (match_api_id,))
match = cur.fetchone()


# We retrieve the x,y coordinates and players api

# In[ ]:


home_players_api_id = list()
away_players_api_id = list()
home_players_x = list()
away_players_x = list()
home_players_y = list()
away_players_y = list()

for i in range(1,12):
    home_players_api_id.append(match['home_player_%d' % i])
    away_players_api_id.append(match['away_player_%d' % i])
    home_players_x.append(match['home_player_X%d' % i])
    away_players_x.append(match['away_player_X%d' % i])
    home_players_y.append(match['home_player_Y%d' % i])
    away_players_y.append(match['away_player_Y%d' % i])

print('Example, home players api id: ')
print(home_players_api_id)


# Next, we get the players last names from the table Player. I filter out the None values (if any) from the query and add them back later to the players_names list.
# I try to keep the name in the same order as the other lists, so as to later map the names to the x,y coordinates

# In[ ]:


#Fetch players'names 
players_api_id = [home_players_api_id,away_players_api_id]
players_api_id.append(home_players_api_id) # Home
players_api_id.append(away_players_api_id) # Away
players_names = [[None]*11,[None]*11]

for i in range(2):
    players_api_id_not_none = [x for x in players_api_id[i] if x is not None]
    sql = 'SELECT player_api_id,player_name FROM Player'
    sql += ' WHERE player_api_id IN (' + ','.join(map(str, players_api_id_not_none)) + ')'
    cur.execute(sql)
    players = cur.fetchall()
    for player in players:
        idx = players_api_id[i].index(player['player_api_id'])
        name = player['player_name'].split()[-1] # keep only the last name
        players_names[i][idx] = name

print('Home team players names:')
print(players_names[0])
print('Away team players names:')
print(players_names[1])


# Next we need to rework the x coordinate a little bit, replacing 1 (the goal keeper) with 5. You will understand why when we do the plot.

# In[ ]:


home_players_x = [5 if x==1 else x for x in home_players_x]
away_players_x = [5 if x==1 else x for x in away_players_x]


# Finally let's plot the lineup with a top-down view of the pitch. 
# You should clearly see the differences between the two squad formations.
# Lyon plays in 4-3-1-2 shape while St Etienne (the away team) uses a 4-2-3-1 formation.

# In[ ]:


import matplotlib.pyplot as plt

# Home team (in blue)
plt.subplot(2, 1, 1)
plt.rc('grid', linestyle="-", color='black')
plt.rc('figure', figsize=(12,20))
plt.gca().invert_yaxis() # Invert y axis to start with the goalkeeper at the top
for label, x, y in zip(players_names[0], home_players_x, home_players_y):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (-20, 20),
        textcoords = 'offset points', va = 'bottom')
plt.scatter(home_players_x, home_players_y,s=480,c='blue')
plt.grid(True)

# Away team (in red)
plt.subplot(2, 1, 2)
plt.rc('grid', linestyle="-", color='black')
plt.rc('figure', figsize=(12,20))
plt.gca().invert_xaxis() # Invert x axis to have right wingers on the right
for label, x, y in zip(players_names[1], away_players_x, away_players_y):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (-20, 20),
        textcoords = 'offset points', va = 'bottom')
plt.scatter(away_players_x, away_players_y,s=480,c='red')
plt.grid(True)


ax = [plt.subplot(2,2,i+1) for i in range(0)]
for a in ax:
    a.set_xticklabels([])
    a.set_yticklabels([])
plt.subplots_adjust(wspace=0, hspace=0)


plt.show()


# We can also buil a string with the formations and print it:

# In[ ]:


from collections import Counter

players_y = [home_players_y,away_players_y]
formations = [None] * 2
for i in range(2):
    formation_dict=Counter(players_y[i]);
    sorted_keys = sorted(formation_dict)
    formation = ''
    for key in sorted_keys[1:-1]:
        y = formation_dict[key]
        formation += '%d-' % y
    formation += '%d' % formation_dict[sorted_keys[-1]] 
    formations[i] = formation


print('Home team formation: ' + formations[0])
print('Away team formation: ' + formations[1])

