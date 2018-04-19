
# coding: utf-8

# In[ ]:


import math
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import datetime as dt
import datetime
import sqlalchemy
from numpy.random import random
from sqlalchemy import create_engine
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


conn = sqlite3.connect('../input/database.sqlite')
c = conn.cursor()


# In[ ]:


import datetime
def get_zodiac_of_date(date):
    zodiacs = [(120, 'Cap'), (218, 'Aqu'), (320, 'Pis'), (420, 'Ari'), (521, 'Tau'),
           (621, 'Gem'), (722, 'Can'), (823, 'Leo'), (923, 'Vir'), (1023, 'Lib'),
           (1122, 'Sco'), (1222, 'Sag'), (1231, 'Cap')]
    date_number = int("".join((str(date.month), '%02d' % date.day)))
    for z in zodiacs:
        if date_number <= z[0]:
            return z[1]
def get_zodiac_for_football_players(x):
    date  =  x.split(" ")[0]
    date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    return get_zodiac_of_date(date)
def get_age_for_football_players(x):
    date  =  x.split(" ")[0]
    today = datetime.datetime.strptime("2015-01-01", "%Y-%m-%d").date()
    born = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
def get_overall_rating(x):
    global c
    all_rating = c.execute("""SELECT overall_rating FROM Player_Stats WHERE player_api_id = '%d' """ % (x)).fetchall()
    all_rating = np.array(all_rating,dtype=np.float)[:,0]
    mean_rating = np.nanmean(all_rating)
    return mean_rating
def get_current_team_and_country(x):
    global c
    all_rating = c.execute("""SELECT overall_rating FROM Player_Stats WHERE player_api_id = '%d' """ % (x)).fetchall()
    all_rating = np.array(all_rating,dtype=np.float)[:,0]
    rating = np.nanmean(all_rating)
    if (rating>1): 
        all_football_nums = reversed(range(1,12))
        for num in all_football_nums:
            all_team_id = c.execute("""SELECT home_team_api_id, country_id FROM Match WHERE home_player_%d = '%d'""" % (num,x)).fetchall()
            if len(all_team_id) > 0:
                number_unique_teams = len(np.unique(np.array(all_team_id)[:,0]))
                last_team_id = all_team_id[-1][0]
                last_country_id = all_team_id[-1][1]
                last_country = c.execute("""SELECT name FROM Country WHERE id = '%d'""" % (last_country_id)).fetchall()[0][0]
                last_team = c.execute("""SELECT team_long_name FROM Team WHERE team_api_id = '%d'""" % (last_team_id)).fetchall()[0][0]
                return last_team, last_country, number_unique_teams
    return None, None, 0
def get_position(x):
    global c
    all_rating = c.execute("""SELECT overall_rating FROM Player_Stats WHERE player_api_id = '%d' """ % (x)).fetchall()
    all_rating = np.array(all_rating,dtype=np.float)[:,0]
    rating = np.nanmean(all_rating)
    if (rating>1): 
        all_football_nums = reversed(range(1,12))
        for num in all_football_nums:
            all_y_coord = c.execute("""SELECT home_player_Y%d FROM Match WHERE home_player_%d = '%d'""" % (num,num,x)).fetchall()
            if len(all_y_coord) > 0:
                Y = np.array(all_y_coord,dtype=np.float)
                mean_y = np.nanmean(Y)
                if (mean_y >= 10.0):
                    return "for"
                elif (mean_y > 5):
                    return "mid"
                elif (mean_y > 1):
                    return "def"
                elif (mean_y == 1.0):
                    return "gk"
    return None


# In[ ]:


with sqlite3.connect('../input/database.sqlite') as con:
    sql = "SELECT * FROM Player"
    max_players_to_analyze = 1000
    players_data = pd.read_sql_query(sql, con)
    players_data = players_data.iloc[0:max_players_to_analyze]
    players_data["zodiac"] = np.vectorize(get_zodiac_for_football_players)(players_data["birthday"])
    players_data["rating"] = np.vectorize(get_overall_rating)(players_data["player_api_id"])
    players_data["age"] = np.vectorize(get_age_for_football_players)(players_data["birthday"])
    players_data["team"], players_data["country"], players_data["num_uniq_team"] = np.vectorize(get_current_team_and_country)(players_data["player_api_id"])
    players_data["position"] = np.vectorize(get_position)(players_data["player_api_id"])
players_data.head()


# In[ ]:


countries_rating = players_data.groupby("country")["rating"].mean()
del countries_rating["None"]
countries_rating= countries_rating.reset_index()
countries_rating
min_rating = countries_rating["rating"].min()
countries_coef = np.vectorize(lambda x :x)(countries_rating["rating"] - min_rating + 5)
countries_rating["rating"] = countries_coef
countries_rating
final_ratings = {}
for i in countries_rating.values:
    final_ratings[i[0]] = i[1]
final_ratings


# In[ ]:


import mpl_toolkits.basemap as bm
countries = {}
countries["England"] = [-0.12, 51.5, 20.0]
countries["Belgium"] = [4.34, 50.85, 20.0]
countries["France"] = [2.34, 48.86, 20.0]
countries["Germany"] = [13.4, 52.52, 20.0]
countries["Italy"] = [12.49, 41.89, 20.0]
countries["Netherlands"] =[4.89, 52.37, 20.0]
countries["Poland"] = [21.01, 52.23, 20.0]
countries["Portugal"] = [-9.14, 38.73, 20.0]
countries["Scotland"] = [-4.25, 55.86, 20.0]
countries["Spain"] = [-3.70, 40.41, 20.0]
countries["Switzerland"] = [6.14, 46.2, 20.0]
for i in final_ratings.keys():
    countries[i][2] = 3*final_ratings[i]
plt.figure(figsize=(12,12))

m = bm.Basemap(projection='cyl', llcrnrlat=35, urcrnrlat=58, llcrnrlon=-10, urcrnrlon=22, resolution='f')

m.drawcountries(linewidth=0.2)
m.fillcontinents(color='lavender', lake_color='#907099')
m.drawmapboundary(linewidth=0.2, fill_color='#000040')
m.drawparallels(np.arange(-90,90,30),labels=[0,0,0,0], color='white', linewidth=0.5)
m.drawmeridians(np.arange(0,360,30),labels=[0,0,0,0], color='white', linewidth=0.5)
for i in countries.keys():
    m.plot(countries[i][0], countries[i][1], 'bo', markersize = countries[i][2], color='r')
for label, xpt, ypt in zip(list(countries.keys()), np.array(list(countries.values()))[:,0],                           np.array(list(countries.values()))[:,1]):
    plt.text(xpt - 0.85, ypt, label, fontsize = 20, color="purple")
plt.show()


# In[ ]:


def plot_beatiful_quantiles(players_data):
    points = np.linspace(0.00,1.0, 100)
    quantiles = []
    for i in points:
        quantiles.append(players_data["rating"].quantile(i))
    fig = plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
    subplot = fig.add_subplot(111)
    subplot.tick_params(axis='both', which='major', labelsize=22)
    subplot.plot(quantiles,points, color="red", linewidth=2.5, linestyle="-")
    plt.xlabel('Fifa rating', fontsize=30)
    plt.ylabel('Percent of football players', fontsize=30)
    plt.show()
plot_beatiful_quantiles(players_data)


# In[ ]:


calc_num_by_zodiac = players_data.groupby("zodiac").count()["rating"]
football_zodiac = (100.0 * calc_num_by_zodiac/(float(calc_num_by_zodiac.sum()))).to_dict()
data_from_statisticbrain = {}
data_from_statisticbrain["Aqu"] = 6.3
data_from_statisticbrain["Ari"] = 8.1
data_from_statisticbrain["Cap"] = 8.2
data_from_statisticbrain["Can"] = 8.5
data_from_statisticbrain["Gem"] = 9.3
data_from_statisticbrain["Leo"] = 7.1
data_from_statisticbrain["Lib"] = 8.8
data_from_statisticbrain["Pis"] = 9.1
data_from_statisticbrain["Sag"] = 7.3
data_from_statisticbrain["Sco"] = 9.6
data_from_statisticbrain["Tau"] = 8.3
data_from_statisticbrain["Vir"] = 9.4
def plot_beatiful_bar(football_data, statisticbrain_data):
    N = 12              # num of zodiac signs
    all_signs = football_data.keys()
    football_means = []
    statisticbrain_means = []
    for sign in all_signs:
        football_means.append(football_data[sign])
        statisticbrain_means.append(statisticbrain_data[sign])
    ind = np.arange(N)  # the x locations for the groups
    width = 0.4       # the width of the bars    
    fig, ax = plt.subplots(figsize=(12,12))
    rects1 = ax.bar(ind, football_means, width, color='r')
    rects2 = ax.bar(ind + width,statisticbrain_means, width, color='y')
    ax.set_ylabel('Percent', fontsize=25)
    ax.set_title('Percentage of population by zodiac sign', fontsize=35)
    ax.set_xticks(ind + width)
    ax.set_xticklabels(all_signs, fontsize=20)
    ax.legend((rects1[0], rects2[0]), ('Football players', 'Ordinary peoples'), fontsize=20)    
    plt.show()
plot_beatiful_bar(football_zodiac, data_from_statisticbrain)


# In[ ]:


def plot_beautiful_scatter_uniq_teams(players_data):
    temp = players_data[["rating", "num_uniq_team"]][players_data["num_uniq_team"] != 0].values
    score = temp[:, 0]
    num = temp[:,1]
    fig = plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
    subplot = fig.add_subplot(111)
    subplot.tick_params(axis='both', which='major', labelsize=22)
    subplot.scatter(score, num, s=100, c="red", alpha=0.75)
    plt.xlabel('Fifa rating', fontsize=30)
    plt.ylabel('Count of uniq teams', fontsize=30)
    plt.show()
plot_beautiful_scatter_uniq_teams(players_data)


# In[ ]:


def plot_beautiful_scatter_weight_and_height(players_data):
    colors = ['r', 'g', 'b', 'y']
    lbs_to_kg = 0.453592
    fig = plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
    def_data = players_data[players_data["position"] == "def"]
    forw_data = players_data[players_data["position"] == "for"]
    gk_data = players_data[players_data["position"] == "gk"]
    midf_data =  players_data[players_data["position"] == "mid"]
    def_heigh = (def_data["height"] + np.random.normal(loc=0.0, scale=3.0, size = len(def_data)))
    forw_heigh = forw_data["height"] + np.random.normal(loc=0.0, scale=3.0, size = len(forw_data))
    gk_heigh = gk_data["height"] + np.random.normal(loc=0.0, scale=3.0, size = len(gk_data))
    midf_heigh = midf_data["height"] + np.random.normal(loc=0.0, scale=3.0, size = len(midf_data))
    def_weight = (def_data["weight"] + np.random.normal(loc=0.0, scale=3.0, size = len(def_data)))*lbs_to_kg
    forw_weight = (forw_data["weight"] + np.random.normal(loc=0.0, scale=3.0, size = len(forw_data)))*lbs_to_kg
    gk_weight = (gk_data["weight"] + np.random.normal(loc=0.0, scale=3.0, size = len(gk_data)))*lbs_to_kg
    midf_weight = (midf_data["weight"] + np.random.normal(loc=0.0, scale=3.0, size = len(midf_data)))*lbs_to_kg
    subplot = fig.add_subplot(111)
    subplot.tick_params(axis='both', which='major', labelsize=22)
    midf  = subplot.scatter(midf_weight, midf_heigh, marker='o', color="r", alpha = 0.5, s=50)
    defend = subplot.scatter(def_weight, def_heigh, marker='o', color="g", alpha = 0.5, s=50)
    forw = subplot.scatter(forw_weight, forw_heigh, marker='o', color="b", alpha = 0.5, s=50)
    gk  = subplot.scatter(gk_weight, gk_heigh, marker='o', color="pink", alpha = 0.5, s=50)
    plt.xlabel('Weight (kilograms)', fontsize=30)
    plt.ylabel('Height (centimeters)', fontsize=30)
    plt.legend((defend, forw, gk, midf),
           ('Defender', 'Forward', 'Goalkeeper', 'Midfielder'),
           scatterpoints=1,
           loc='upper left',
           ncol=1,
           fontsize=20)
    plt.show()
plot_beautiful_scatter_weight_and_height(players_data)


# In[ ]:


def plot_age_dependence(players_data):
    def_data = players_data[players_data["position"] == "def"]
    forw_data = players_data[players_data["position"] == "for"]
    gk_data = players_data[players_data["position"] == "gk"]
    midf_data =  players_data[players_data["position"] == "mid"]
    from scipy.interpolate import interp1d
    fig = plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
    group = forw_data.groupby("age")["rating"].mean().reset_index()
    age_forw = group["age"]
    rating_forw = group["rating"]

    ff = interp1d(age_forw, rating_forw, kind='cubic')
    group = def_data.groupby("age")["rating"].mean().reset_index()
    age_def = group["age"]
    rating_def = group["rating"]
    fd = interp1d(age_def, rating_def, kind='cubic')

    group = gk_data.groupby("age")["rating"].mean().reset_index()
    age_gk = group["age"]
    rating_gk = group["rating"]
    fg = interp1d(age_gk, rating_gk, kind='cubic')


    group = midf_data.groupby("age")["rating"].mean().reset_index()
    age_midf = group["age"]
    rating_midf = group["rating"]
    fm = interp1d(age_midf, rating_midf, kind='cubic')

    agenew = np.linspace(17, 40, num=100, endpoint=True)
    subplot = fig.add_subplot(111)
    subplot.tick_params(axis='both', which='major', labelsize=22)
    plt.xlabel('Age', fontsize=30)
    plt.ylabel('Fifa rating', fontsize=30)
    plt.plot( agenew, ff(agenew), "-", agenew, fd(agenew), "-", agenew, fg(agenew), "-", agenew, fm(agenew), "-")
    plt.legend(['Forwards', "Defenders", "Goalkeepers", "Midfielders"], loc='best', fontsize = 20)
    plt.show()
plot_age_dependence(players_data)


# In[ ]:


positions = players_data["position"]
x_data = players_data[["rating", "height", "weight"]]
x_data["age"] = np.vectorize(get_age_for_football_players)(players_data["birthday"])
x_data.head()


# In[ ]:


def get_potential(x):
    global c
    all_rating = c.execute("""SELECT potential FROM Player_Stats WHERE player_api_id = '%d' """ % (x)).fetchall()
    all_rating = np.array(all_rating,dtype=np.float)[:,0]
    mean_rating = np.nanmean(all_rating)
    return mean_rating
def get_preferred_foot(x):
    global c
    all_rating = c.execute("""SELECT preferred_foot FROM Player_Stats WHERE player_api_id = '%d' """ % (x)).fetchall()
    if all_rating[0][0] == "right":
        return 0.0
    else:
        return 1.0
    return float("nan")
def get_attacking_work_rate(x):
    global c
    all_rating = c.execute("""SELECT attacking_work_rate FROM Player_Stats WHERE player_api_id = '%d' """ % (x)).fetchall()
    if all_rating[0][0] == "high":
        return 2.0
    if all_rating[0][0] == "medium":
        return 1.0
    if all_rating[0][0] == "low":
        return 0.0
    return float("nan")
def get_defensive_work_rate(x):
    global c
    all_rating = c.execute("""SELECT defensive_work_rate FROM Player_Stats WHERE player_api_id = '%d' """ % (x)).fetchall()
    if all_rating[0][0] == "high":
        return 2.0
    if all_rating[0][0] == "medium":
        return 1.0
    if all_rating[0][0] == "low":
        return 0.0
    return float("nan")
def get_anyone_statistic(x, col_name):
    global c
    all_rating = c.execute("""SELECT %s FROM Player_Stats WHERE player_api_id = '%d' """ % (col_name, x)).fetchall()
    all_rating = np.array(all_rating,dtype=np.float)[:,0]
    mean_rating = np.nanmean(all_rating)
    return mean_rating


# In[ ]:


train_x_data = x_data
train_x_data["potential"] = np.vectorize(get_potential)(players_data["player_api_id"])
train_x_data["defensive_work_rate"] = np.vectorize(get_defensive_work_rate)(players_data["player_api_id"])
train_x_data["attacking_work_rate"] = np.vectorize(get_attacking_work_rate)(players_data["player_api_id"])
train_x_data["preferred_foot"] = np.vectorize(get_attacking_work_rate)(players_data["player_api_id"])
all_columns = c.execute('PRAGMA TABLE_INFO(Player_stats)').fetchall()
for i in all_columns:
    if i[0] > 8:
        train_x_data[i[1]] = np.vectorize(get_anyone_statistic)(players_data["player_api_id"], i[1])
train_x_data.head()    


# In[ ]:


new_train_x = train_x_data.fillna(train_x_data.median()).values
def pos_to_num(x):
    if x =="for":
        return 4
    elif x == "mid":
        return 3
    elif x == "def":
        return "2"
    elif x == "gk":
        return 1
positions = pd.DataFrame(positions)
positions["nums"] = np.vectorize(pos_to_num)(positions)
y_train = positions.fillna(positions.median())
y_train.fillna(value=np.nan, inplace=True)
temp = pd.to_numeric(y_train["nums"], errors='coerce')
temp = temp.fillna(temp.median())
y_train = temp.values
x_train = new_train_x


# In[ ]:


from sklearn.preprocessing import normalize
x_data = normalize(x_train)
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
x_data, y_train, test_size=0.25, random_state=42)


# In[ ]:


from keras.layers import Input, Dense
from keras.models import Model
encoding_dim = 5
input_img = Input(shape=(41,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(41, activation='sigmoid')(encoded)
autoencoder = Model(input=input_img, output=decoded)
encoder = Model(input=input_img, output=encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train,
                nb_epoch=500,
                batch_size=10,
                shuffle=True,
                verbose = 0,
                validation_data=(x_test, x_test))


# In[ ]:


from sklearn.manifold import TSNE
encoded_data = encoder.predict(x_test)
tsne = TSNE(n_components=2, init='pca', random_state=0)
two_dims_data = tsne.fit_transform(encoded_data)
plt.figure(figsize=(12, 12))
plt.scatter(two_dims_data[:, 0],two_dims_data[:, 1], c=y_test + 1.0, cmap=plt.cm.plasma, s=50)
cbar =plt.colorbar()
cbar.ax.get_yaxis().set_ticks([])
for j, lab in enumerate(['$gk$','$def$','$mid$','$forw$']):
    cbar.ax.text(.5, (2 * j + 1) / 8.0, lab, ha='center', va='center', fontsize = 30)
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('football players positions', rotation=270, fontsize = 20)
plt.show()


# In[ ]:


new_train_x = train_x_data.fillna(train_x_data.median()).values
def pos_to_num(x):
    if x =="for":
        return 4
    elif x == "mid":
        return 3
    elif x == "def":
        return "2"
    elif x == "gk":
        return 1
positions = pd.DataFrame(positions)
positions["nums"] = np.vectorize(pos_to_num)(positions)
y_train = positions.fillna(positions.median())
y_train.fillna(value=np.nan, inplace=True)
temp = pd.to_numeric(y_train["nums"], errors='coerce')
temp = temp.fillna(temp.median())
y_train = temp.values
x_train = new_train_x
x_data = normalize(x_train)
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
x_data, y_train, test_size=0.25, random_state=42)


# In[ ]:


from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
batch_size = 5
original_dim = 41
latent_dim = 2
intermediate_dim = 15
nb_epoch = 150
epsilon_std = 0.01
x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              std=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)
def vae_loss(x, x_decoded_mean):
    xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss

vae = Model(x, x_decoded_mean)
vae.compile(optimizer='rmsprop', loss=vae_loss)

vae.fit(x_train, x_train,
        shuffle=True,
        nb_epoch=nb_epoch,
        batch_size=batch_size,
        validation_data=(x_test, x_test),
        verbose = 0)


# In[ ]:


encoder = Model(x, z_mean)
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(12, 12))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test, cmap=plt.cm.plasma, s=50)
cbar =plt.colorbar()
cbar.ax.get_yaxis().set_ticks([])
for j, lab in enumerate(['$gk$','$def$','$mid$','$forw$']):
    cbar.ax.text(.5, (2 * j + 1) / 8.0, lab, ha='center', va='center', fontsize = 30)
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('football players positions', rotation=270, fontsize = 20)
plt.show()


# In[ ]:


players_data["preferred_foot"] = np.vectorize(get_preferred_foot)(players_data["player_api_id"])
def zodiac_to_num(x):
    zodiacs = ['Cap', 'Aqu', 'Pis', 'Ari', 'Tau',
           'Gem', 'Can', 'Leo', 'Vir', 'Lib',
           'Sco', 'Sag', 'Cap']
    return zodiacs.index(x)
def position_to_num(x):
    positions = ['gk', 'def', 'mid', 'for']
    if x in positions:
        return 1.0*positions.index(x)
    else:
        return float("nan")
    
data_for_prediction = players_data[["height", "weight", "zodiac", "age", "preferred_foot", "position", "rating"]]
data_for_prediction["zodiac"] = np.vectorize(zodiac_to_num)(data_for_prediction["zodiac"])
data_for_prediction["position"] = np.vectorize(position_to_num)(data_for_prediction["position"])
data_for_prediction = data_for_prediction.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
rating = data_for_prediction["rating"]
good_players = ((rating > 70)*1).values
data_for_prediction = data_for_prediction.drop(["rating"], axis=1).values


# In[ ]:



from sklearn import tree
from sklearn.cross_validation import train_test_split
clf = tree.DecisionTreeClassifier(max_depth = 5, min_samples_leaf = 10)
x_train, x_test, y_train, y_test = train_test_split(
data_for_prediction, good_players, test_size=0.25, random_state=42)
clf = clf.fit(x_train ,y_train)
from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, clf.predict(x_test)))


# ![simpe][1]
# 
# 
#   [1]: https://pp.vk.me/c604626/v604626285/f24b/-PWxpr_deSw.jpg
