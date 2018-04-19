from pandas.tseries.holiday import USFederalHolidayCalendar
import numpy, pandas, haversine, multiprocessing, math
from datetime import timedelta
from geopy import distance
from sklearn import *
import lightgbm as lgb
import xgboost as xgb

cal = USFederalHolidayCalendar()
holidays = [d.date() for d in cal.holidays(start='2016-01-01', end='2016-06-30').to_pydatetime()]
business = [d.date() for d in pandas.date_range('2016-01-01', '2016-06-30') if d not in pandas.bdate_range('2016-01-01', '2016-06-30')]
holidays_prev = [d + timedelta(days=-1) for d in holidays]
holidays_after = [d + timedelta(days=1) for d in holidays]

def fdist(params):
    i, ll = params
    l1, l2, l3, l4 = ll
    h = haversine.haversine((l1, l2),(l3, l4))
    v = distance.vincenty((l1, l2),(l3, l4)).miles
    g = distance.great_circle((l1, l2),(l3, l4)).miles
    d =  math.degrees(math.atan2(math.sin(l3 - l2) * math.cos(l3), math.cos(l1) * math.sin(l3) - math.sin(l1) * math.cos(l3) * math.cos(l3 - l2)))
    return [i, h, v, g, d]

def mixer(ingredients):
    p = multiprocessing.Pool(multiprocessing.cpu_count())
    ll = list(enumerate(ingredients[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']].values))
    dist = numpy.array(sorted(p.map(fdist, ll)))
    p.close(); p.join()
    ingredients['hdistance'] = dist[:,1]
    ingredients['vdistance'] = dist[:,2]
    ingredients['gdistance'] = dist[:,3]
    ingredients['direction'] = dist[:,4]
    ingredients['sdistance'] = numpy.sqrt(numpy.square(ingredients['pickup_longitude'] - ingredients['dropoff_longitude']) + numpy.square(ingredients['pickup_latitude'] - ingredients['dropoff_latitude']))
    for i in range(1,6):
        ingredients['pickup_latitude'+str(i)] = numpy.round(ingredients['pickup_latitude'], i)
        ingredients['pickup_longitude'+str(i)] = numpy.round(ingredients['pickup_longitude'], i)
        ingredients['dropoff_latitude'+str(i)] = numpy.round(ingredients['dropoff_latitude'], i)
        ingredients['dropoff_longitude'+str(i)] = numpy.round(ingredients['dropoff_longitude'], i)
    ingredients['quarter'] = ingredients.pickup_datetime.dt.quarter
    ingredients['month'] = ingredients.pickup_datetime.dt.month
    ingredients['day'] = ingredients.pickup_datetime.dt.day
    ingredients['dow'] = ingredients.pickup_datetime.dt.dayofweek
    ingredients['wd'] = ingredients.pickup_datetime.dt.weekday
    ingredients['hr'] = ingredients.pickup_datetime.dt.hour
    ingredients['m'] = ingredients.pickup_datetime.dt.minute
    ingredients['h'] = ingredients.pickup_datetime.dt.date.map(lambda x: 1 if x in holidays else 0)
    ingredients['hp'] = ingredients.pickup_datetime.dt.date.map(lambda x: 1 if x in holidays_prev else 0)
    ingredients['ha'] = ingredients.pickup_datetime.dt.date.map(lambda x: 1 if x in holidays_after else 0)
    ingredients['b'] = ingredients.pickup_datetime.dt.date.map(lambda x: 1 if x in business else 0)
    ingredients['store_and_fwd_flag'] = ingredients['store_and_fwd_flag'].map(lambda x: 0 if x =='N' else 1)
    ingredients = pandas.merge(ingredients, weather, on = ['month', 'day', 'hr'], how = 'left').fillna(0.0)
    layer = [l for l in ingredients.columns if l not in ['id','pickup_datetime', 'dropoff_datetime', 'trip_duration']]
    print('Ohhh smells good...')
    return ingredients[layer]

flour = pandas.read_csv('../input/nyc-taxi-trip-duration/train.csv', parse_dates=['pickup_datetime'])
flour = flour[flour.trip_duration <= 1800000].reset_index(drop=True)
candles = numpy.log(flour['trip_duration']+1)
frosting = pandas.read_csv('../input/nyc-taxi-trip-duration/test.csv', parse_dates=['pickup_datetime'])
icing = frosting['id']

weather = pandas.read_csv('../input/knycmetars2016/KNYC_Metars.csv', parse_dates=['Time'])
weather['year'] = weather['Time'].dt.year
weather['month'] = weather['Time'].dt.month
weather['day'] = weather['Time'].dt.day
weather['hr'] = weather['Time'].dt.hour
weather = weather[weather['year'] == 2016][['month','day','hr','Temp.','Humidity','Pressure']]
weather = pandas.DataFrame(weather.groupby(by=['month','day','hr'])['Temp.','Humidity','Pressure'].mean().reset_index()) 

flour = mixer(flour)
frosting = mixer(frosting)

col = ['pickup_latitude', 'pickup_longitude','dropoff_latitude', 'dropoff_longitude']
full = pandas.concat([flour[col], frosting[col]])
coords = numpy.vstack((full[['pickup_latitude', 'pickup_longitude']], full[['dropoff_latitude', 'dropoff_longitude']]))

pca = decomposition.PCA().fit(coords)
flour['pickup_pca0'] = pca.transform(flour[['pickup_latitude', 'pickup_longitude']])[:, 0]
flour['pickup_pca1'] = pca.transform(flour[['pickup_latitude', 'pickup_longitude']])[:, 1]
flour['dropoff_pca0'] = pca.transform(flour[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
flour['dropoff_pca1'] = pca.transform(flour[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
frosting['pickup_pca0'] = pca.transform(frosting[['pickup_latitude', 'pickup_longitude']])[:, 0]
frosting['pickup_pca1'] = pca.transform(frosting[['pickup_latitude', 'pickup_longitude']])[:, 1]
frosting['dropoff_pca0'] = pca.transform(frosting[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
frosting['dropoff_pca1'] = pca.transform(frosting[['dropoff_latitude', 'dropoff_longitude']])[:, 1]

x1, x2, y1, y2 = model_selection.train_test_split(flour, candles, test_size=0.3); flour = []

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'l2_root',
    'learning_rate': 0.2111,
    'max_depth': 15
}
oven2 = lgb.train(params, lgb.Dataset(x1, label=y1), 2111, lgb.Dataset(x2, label=y2), verbose_eval=50, early_stopping_rounds=50)
prep2 = oven2.predict(frosting, num_iteration=oven2.best_iteration)
cake = pandas.DataFrame({'id': icing, 'trip_duration': numpy.exp(prep2)-1})

import sqlite3, io #blending
c = sqlite3.connect('../input/dataset01/database.sqlite')
df1 = pandas.read_sql('Select * From stacks Where Id=2', c)
df1 = pandas.read_csv(io.StringIO(df1['file'][0]))
df1.columns = [x+'_' if x not in ['id'] else x for x in df1.columns]
cake = pandas.merge(cake, df1, how='left', on='id')
cake['trip_duration'] = (cake['trip_duration'] * 0.30) + (cake['trip_duration_'] * 0.70)

cake[['id','trip_duration']].to_csv('eat_me2.csv', index=False)