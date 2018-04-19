from haversine import haversine
import seaborn as sns
import pandas as pd
import numpy as np
import sqlite3
import math
import time

#%matplotlib inline
import matplotlib.pyplot as plt

# Otimization functions
north_pole = (90,0)
weight_limit = 1000.0

def bb_sort(ll):
    s_limit = 100
    optimal = False
    ll = [[0,north_pole,10]] + ll[:] + [[0,north_pole,10]] 
    while not optimal:
        optimal = True
        for i in range(1,len(ll) - 2):
            lcopy = ll[:]
            lcopy[i], lcopy[i+1] = ll[i+1][:], ll[i][:]
            if path_opt_test(ll[1:-1])[0] > path_opt_test(lcopy[1:-1])[0]:
                #print("Sort Swap")
                ll = lcopy[:]
                optimal = False
                s_limit -= 1
                if s_limit < 0:
                    optimal = True
                    break
    return ll[1:-1]

def prev_path_opt(curr,prev):
    curr = [[0,north_pole,10]] + curr[:] + [[0,north_pole,10]]
    prev = [[0,north_pole,10]] + prev[:] + [[0,north_pole,10]]
    for curr_ in range(1,len(curr) - 2):
        for prev_ in range(1,len(prev) - 2):
            lcopy_curr = curr[:]
            lcopy_prev = prev[:]
            lcopy_curr[curr_], lcopy_prev[prev_] = lcopy_prev[prev_][:], lcopy_curr[curr_][:]
            if ((path_opt_test(lcopy_curr[1:-1])[0] + path_opt_test(lcopy_prev[1:-1])[0]) < (path_opt_test(curr[1:-1])[0] + path_opt_test(prev[1:-1])[0])) and path_opt_test(lcopy_curr[1:-1])[2] <=1000 and path_opt_test(lcopy_prev[1:-1])[2] <= 1000:
                #print("Trip Swap")
                curr = lcopy_curr[:]
                prev = lcopy_prev[:]
    return [curr[1:-1], prev[1:-1]]

def prev_path_opt_s1(curr,prev):
    curr = [[0,north_pole,10]] + curr[:] + [[0,north_pole,10]]
    prev = [[0,north_pole,10]] + prev[:] + [[0,north_pole,10]]
    for curr_ in range(1,len(curr) - 1):
        for prev_ in range(1,len(prev) - 1):
            lcopy_curr = curr[:]
            lcopy_prev = prev[:]
            if len(lcopy_prev)-1 <= prev_:
                break
            lcopy_curr = lcopy_curr[:curr_+1][:] + [lcopy_prev[prev_]] + lcopy_curr[curr_+1:][:]
            lcopy_prev.pop(prev_)
            if ((path_opt_test(lcopy_curr[1:-1])[0] + path_opt_test(lcopy_prev[1:-1])[0]) <= (path_opt_test(curr[1:-1])[0] + path_opt_test(prev[1:-1])[0])) and path_opt_test(lcopy_curr[1:-1])[2] <=1000 and path_opt_test(lcopy_prev[1:-1])[2] <= 1000:
                #print("Trip Swap - Give to current")
                curr = lcopy_curr[:]
                prev = lcopy_prev[:]
    return [curr[1:-1], prev[1:-1]]

def split_trips(curr):
    prev = []
    curr = [[0,north_pole,10]] + curr[:] + [[0,north_pole,10]]
    for curr_ in range(1,len(curr) - 1):
        lcopy_curr = curr[:]
        if len(lcopy_curr)-1 <=curr_:
            break
        lcopy_prev = [[0,north_pole,10]] + [lcopy_curr[curr_]] + [[0,north_pole,10]]
        lcopy_curr.pop(curr_)
        if ((path_opt_test(lcopy_curr[1:-1])[0] + path_opt_test(lcopy_prev[1:-1])[0]) < (path_opt_test(curr[1:-1])[0])):
            #print("Trip Split")
            curr = lcopy_curr[:]
            prev = lcopy_prev[:]
    return [curr[1:-1], prev[1:-1]]

def path_opt_test(llo):
    f_ = 0.0
    d_ = 0.0
    we_ = 0.0
    l_ = north_pole
    for i in range(len(llo)):
        d_ += haversine(l_, llo[i][1])
        we_ += llo[i][2]
        f_ += d_ * llo[i][2]
        l_ = llo[i][1]
    d_ += haversine(l_, north_pole)
    f_ += d_ * 10
    return [f_,d_,we_]


# Slicing
gifts = pd.read_csv("../input/gifts.csv").fillna(" ")[:2000] # Lets take only the first 2,000
gifts['TripId']=0
gifts['i']=0
gifts['j']=0

for n in [1.26]: # Slicing Tuning Parameter
    i_ = 0
    j_ = 0
    for i in range(90,-90,int(-180/n)):
        i_ += 1
        j_ = 0
        for j in range(180,-180,int(-360/n)):
            j_ += 1
            gifts.loc[(gifts['Latitude']>(i-180/n))&(gifts['Latitude']<i)&(gifts['Longitude']>(j-360/n))&(gifts['Longitude']<(j)),"i"]=i_
            gifts.loc[(gifts['Latitude']>(i-180/n))&(gifts['Latitude']<i)&(gifts['Longitude']>(j-360/n))&(gifts['Longitude']<(j)),"j"]=j_
    for limit_ in [67]:  # Slicing Tuning Parameter
        trips=gifts[gifts['TripId']==0]
        trips=trips.sort_values(['i','j','Longitude','Latitude'])
        trips=trips[0:limit_]
        t_ = 0
        while len(trips.GiftId)>0:
            g = []
            t_ += 1
            w_ = 0.0
            for i in range(len(trips.GiftId)):
                    if (w_ + float(trips.iloc[i,3]))<= weight_limit:
                        w_ += float(trips.iloc[i,3])
                        g.append(trips.iloc[i,0])
            gifts.loc[gifts['GiftId'].isin(g),'TripId']=t_
            trips=gifts[gifts['TripId']==0]
            trips=trips.sort_values(['i','j','Longitude','Latitude'])
            trips=trips[0:limit_]
        ou_ = open("submission_opt" + str(limit_) + " " + str(n) + ".csv","w")
        ou_.write("TripId,GiftId\n")
        bm = 0.0
        for s_ in range(1,t_+1):
            trip=gifts[gifts['TripId']==s_]
            trip=trip.sort_values(['Latitude','Longitude'],ascending=[0,1])
            a = []
            for x_ in range(len(trip.GiftId)):
                a.append([trip.iloc[x_,0],(trip.iloc[x_,1],trip.iloc[x_,2]),trip.iloc[x_,3]])

            print("TripId",s_, path_opt_test(a)[0])
            bm += path_opt_test(a)[0]
            for y_ in range(len(a)):
                ou_.write(str(s_)+","+str(a[y_][0])+"\n")
                
        ou_.close()

        benchmark = 144525525772.40200
        if bm < benchmark:
            print(n, limit_, "Improvement", bm, bm - benchmark, benchmark)
        else:
            print(n, limit_, "Try again", bm, bm - benchmark, benchmark)

#Lets take a look at the output
## Credit to beluga

Submission = pd.read_csv('submission_opt67 1.26.csv') 
gifts = pd.read_csv("../input/gifts.csv").fillna(" ")[:2000]
print(Submission.head())
Submission = pd.merge(Submission, gifts, how='left', on=['GiftId'])
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = [16, 12]
fig = plt.figure()
plt.scatter(Submission['Longitude'], Submission['Latitude'], c=Submission['TripId'],  alpha=0.8, s=8, linewidths=0)
for t in Submission.TripId.unique():
    trip = Submission[Submission['TripId'] == t]
    plt.plot(trip['Longitude'], trip['Latitude'], 'k.-', alpha=0.1)
plt.colorbar()
plt.grid()
plt.title('Trips')
plt.tight_layout()
fig.savefig('Trips1.png', dpi=300)

fig = plt.figure()
plt.scatter(Submission['Longitude'].values, Submission['Latitude'].values, c='k', alpha=0.1, s=1, linewidths=0)
for t in Submission.TripId.unique():
    previous_location = north_pole
    trip = Submission[Submission['TripId'] == t]
    i = 0
    for _, gift in trip.iterrows():
        plt.plot([previous_location[1], gift['Longitude']], [previous_location[0], gift['Latitude']],
                 color=plt.cm.copper_r(i/90.), alpha=0.1)
        previous_location = tuple(gift[['Latitude', 'Longitude']])
        i += 1
    plt.scatter(gift['Longitude'], gift['Latitude'], c='k', alpha=0.5, s=20, linewidths=0)

plt.scatter(gift['Longitude'], gift['Latitude'], c='k', alpha=0.5, s=20, linewidths=0, label='TripEnds')
plt.legend(loc='upper right')
plt.grid()
plt.title('TripOrder')
plt.tight_layout()
fig.savefig('TripsinOrder1.png', dpi=300)



# Now Lets Optimize (Feeling Greedy)

ou_ = open("submission_v1.csv","w")
ou_.write("TripId,GiftId\n")
Submission = Submission[:2000] # Lets only work with a few Trips
bm = 0.0
d = {}
previous_trip = []
Submission['colFromIndex'] = Submission.index
Submission = Submission.sort_index(by=['TripId', 'colFromIndex'], ascending=[True, True])
uniq_trips = Submission.TripId.unique()

for s_ in range(len(uniq_trips)):
    trip = Submission[(Submission['TripId']==(uniq_trips[s_]))].copy()
    trip = trip.reset_index()

    b = []
    for x_ in range(len(trip.GiftId)):
        b.append([trip.GiftId[x_],(trip.Latitude[x_],trip.Longitude[x_]),trip.Weight[x_]])
    d[str(s_+1)] = [path_opt_test(b)[0],b[:]]

for s_ in range(len(uniq_trips)):
    key = str(s_+1)
    if d[key][0] >= 1:
        for i in range(-2,2,1):
            r_ = int(key)+i
            if r_ <= 0:
                r_ = len(uniq_trips) + i
            if r_ > len(uniq_trips):
                r_ = i
            if r_ != int(key):
                #print(r_,key,i)
                previous_trip=d[str(r_)][1][:]
                b = d[key][1][:]
                previous_trip, b = prev_path_opt(previous_trip, b)
                previous_trip, b = prev_path_opt_s1(previous_trip, b)
                #previous_trip, b = split_trips(b) - Need to increment trip Ids by one to keep sequence for further optimization
                previous_trip = bb_sort(previous_trip)
                b = bb_sort(b)
                d[str(r_)][1]=previous_trip[:]
                d[key][1]=b[:]
for key in d:
    b = d[key][1][:]
    for x_ in range(len(b)):
        ou_.write(str(key)+","+str(b[x_][0])+"\n")
ou_.close()

# Lets see those optimized paths again
Submission = pd.read_csv('submission_v1.csv') 
gifts = pd.read_csv("../input/gifts.csv").fillna(" ")[:2000]
print(Submission.head())
Submission = pd.merge(Submission, gifts, how='left', on=['GiftId'])
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = [16, 12]
fig = plt.figure()
plt.scatter(Submission['Longitude'], Submission['Latitude'], c=Submission['TripId'],  alpha=0.8, s=8, linewidths=0)
for t in Submission.TripId.unique():
    trip = Submission[Submission['TripId'] == t]
    plt.plot(trip['Longitude'], trip['Latitude'], 'k.-', alpha=0.1)
plt.colorbar()
plt.grid()
plt.title('Trips')
plt.tight_layout()
fig.savefig('Trips2.png', dpi=300)

fig = plt.figure()
plt.scatter(Submission['Longitude'].values, Submission['Latitude'].values, c='k', alpha=0.1, s=1, linewidths=0)
for t in Submission.TripId.unique():
    previous_location = north_pole
    trip = Submission[Submission['TripId'] == t]
    i = 0
    for _, gift in trip.iterrows():
        plt.plot([previous_location[1], gift['Longitude']], [previous_location[0], gift['Latitude']],
                 color=plt.cm.copper_r(i/90.), alpha=0.1)
        previous_location = tuple(gift[['Latitude', 'Longitude']])
        i += 1
    plt.scatter(gift['Longitude'], gift['Latitude'], c='k', alpha=0.5, s=20, linewidths=0)

plt.scatter(gift['Longitude'], gift['Latitude'], c='k', alpha=0.5, s=20, linewidths=0, label='TripEnds')
plt.legend(loc='upper right')
plt.grid()
plt.title('TripOrder')
plt.tight_layout()
fig.savefig('TripsinOrder2.png', dpi=300)

