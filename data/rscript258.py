import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.basemap import Basemap  

#read in data
shipdata = pd.read_csv('../input/CLIWOC15.csv')
lat = shipdata.Lat3
lon = shipdata.Lon3
coord=np.column_stack((list(lon),list(lat)))
ship=shipdata.ShipName
utc=shipdata.UTC
year=shipdata.Year
month=shipdata.Month
day=shipdata.Day

#take out lon/lat nan
utc=utc[~np.isnan(coord).any(axis=1)]
ship=ship[~np.isnan(coord).any(axis=1)]
year=year[~np.isnan(coord).any(axis=1)]
month=month[~np.isnan(coord).any(axis=1)]
day=day[~np.isnan(coord).any(axis=1)]
coord=coord[~np.isnan(coord).any(axis=1)]
data=np.column_stack((coord,ship,year,month,day,utc))

#find Endeavour
np.count_nonzero(data[:,2]=='Endeavour')
cook=data[data[:,2]=='Endeavour']

#sort time
cook=cook[cook[:,6].argsort()]

#set up map
m = Basemap(projection='robin',lon_0=180,resolution='c', llcrnrlon=120, urcrnrlon=-30)
m.drawcoastlines()
m.drawcountries()
m.drawmeridians(np.arange(0,360,30))
m.drawparallels(np.arange(-90,90,30))
m.fillcontinents(color='grey')

#draw path on the background
x,y=m(cook[:,0],cook[:,1])
m.plot(x,y,'.',color='grey',alpha=0.2)

#animation (based on https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/)

x,y = m(0, 0)
point = m.plot(x, y, 'o', markersize=7, color='red')[0]
def init():
    point.set_data([], [])
    return point,

def animate(i):
    x,y=m(cook[i][0],cook[i][1])
    point.set_data(x,y)
    plt.title('%d %d %d' % (cook[i][3],cook[i][4],cook[i][5]))
    return point,

output = animation.FuncAnimation(plt.gcf(), animate, init_func=init, frames=355, interval=100, blit=True, repeat=False)


#have problems with saving
output.save('captaincook.gif', writer='imagemagick')
plt.show()
