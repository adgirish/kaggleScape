
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

df = pd.read_csv('../input/data.csv')
extinct = df[df["Degree of endangerment"] == "Extinct"]
usa = df[df.Countries == "United States of America"]
usa_extinct = df[(df.Countries == "United States of America") & (df["Degree of endangerment"] == "Extinct")]
brazil = df[df.Countries == "Brazil"]
brazil_extinct = df[(df.Countries == "Brazil") & (df["Degree of endangerment"] == "Extinct")]
aus = df[df.Countries == "Australia"]
aus_extinct = df[(df.Countries == "Australia") & (df["Degree of endangerment"] == "Extinct")]
indonesia = df[df.Countries == "Indonesia"]
indonesia_extinct = df[(df.Countries == "Indonesia") & (df["Degree of endangerment"] == "Extinct")]
mexico = df[df.Countries == "Mexico"]
mexico_extinct = df[(df.Countries == "Mexico") & (df["Degree of endangerment"] == "Extinct")]
papau = df[df.Countries == "Papua New Guinea"]
papau_extinct = df[(df.Countries == "Papua New Guinea") & (df["Degree of endangerment"] == "Extinct")]
china = df[df.Countries == "China"]
china_extinct = df[(df.Countries == "China") & (df["Degree of endangerment"] == "Extinct")]


plt.figure(figsize=(15,8))
m = Basemap(projection = 'mill', llcrnrlat = -15, urcrnrlat = 5, llcrnrlon = 90, urcrnrlon = 150, resolution = 'h')
m.drawcoastlines()
m.drawcountries()

x, y = m(list(indonesia["Longitude"].astype(float)), list(indonesia["Latitude"].astype(float)))
m.plot(x, y, 'go', markersize = 12, alpha = 0.8, color = "#990099")

plt.title('Extinct and Endangered Languages - Indonesia')
plt.show()


# In[ ]:


plt.figure(figsize=(15,8))
m = Basemap(projection = 'mill', llcrnrlat = -15, urcrnrlat = 5, llcrnrlon = 90, urcrnrlon = 150, resolution = 'h')
m.drawcoastlines()
m.drawcountries()

x, y = m(list(indonesia_extinct["Longitude"].astype(float)), list(indonesia_extinct["Latitude"].astype(float)))
m.plot(x, y, 'go', markersize = 12, alpha = 0.8, color = "#990099")

plt.title('Extinct Languages - Indonesia')
plt.show()


# In[ ]:



plt.figure(figsize=(15,8))
m = Basemap(projection='mill', llcrnrlat = 20, urcrnrlat = 50, llcrnrlon = -130, urcrnrlon = -60, resolution = 'h')
m.drawcoastlines()
m.drawcountries()
m.drawstates()

x, y = m(list(usa["Longitude"].astype(float)), list(usa["Latitude"].astype(float)))
m.plot(x, y, 'go', markersize = 12, alpha = 0.8, color = "blue")

plt.title('Extinct and Endangered Languages in USA')
plt.show()


# In[ ]:


plt.figure(figsize=(15,8))
m = Basemap(projection='mill', llcrnrlat = 20, urcrnrlat = 50, llcrnrlon = -130, urcrnrlon = -60, resolution = 'h')
m.drawcoastlines()
m.drawcountries()
m.drawstates()

x, y = m(list(usa_extinct["Longitude"].astype(float)), list(usa_extinct["Latitude"].astype(float)))
m.plot(x, y, 'go', markersize = 12, alpha = 0.8, color = "blue")

plt.title('Extinct Languages in USA')
plt.show()


# In[ ]:


plt.figure(figsize=(15,8))
m = Basemap(projection = 'mill', llcrnrlat = -80, urcrnrlat = 80, llcrnrlon = -180, urcrnrlon = 180, resolution = 'h')
m.drawcoastlines()
m.drawcountries()

x, y = m(list(df["Longitude"].astype(float)), list(df["Latitude"].astype(float)))
m.plot(x, y, 'go', markersize = 12, alpha = 0.7, color = "red")

plt.title('Extinct and Endangered Languages Across the World')
plt.show()


# In[ ]:



plt.figure(figsize=(15,8))
m = Basemap(projection = 'mill', llcrnrlat = -80, urcrnrlat = 80, llcrnrlon = -180, urcrnrlon = 180, resolution = 'h')
m.drawcoastlines()
m.drawcountries()

x, y = m(list(extinct["Longitude"].astype(float)), list(extinct["Latitude"].astype(float)))
m.plot(x, y, 'go', markersize = 12, alpha = 0.8, color = "maroon")

plt.title('Extinct Languages Across the World')
plt.show()


# In[ ]:


plt.figure(figsize=(15,8))
m = Basemap(projection = 'mill', llcrnrlat = -30, urcrnrlat = 10, llcrnrlon = -80, urcrnrlon = -20, resolution = 'h')
m.drawcoastlines()
m.drawcountries()

x, y = m(list(brazil["Longitude"].astype(float)), list(brazil["Latitude"].astype(float)))
m.plot(x, y, 'go', markersize = 12, alpha = 0.8, color = "yellow")

plt.title('Extinct and Endangered Languages - Brazil')
plt.show()


# In[ ]:


plt.figure(figsize=(15,8))
m = Basemap(projection = 'mill', llcrnrlat = -30, urcrnrlat = 10, llcrnrlon = -80, urcrnrlon = -20, resolution = 'h')
m.drawcoastlines()
m.drawcountries()

x, y = m(list(brazil_extinct["Longitude"].astype(float)), list(brazil_extinct["Latitude"].astype(float)))
m.plot(x, y, 'go', markersize = 12, alpha = 0.8, color = "yellow")

plt.title('Extinct Languages - Brazil')
plt.show()


# In[ ]:


plt.figure(figsize=(15,8))
m = Basemap(projection = 'mill', llcrnrlat = -50, urcrnrlat = -5, llcrnrlon = 100, urcrnrlon = 160, resolution = 'h')
m.drawcoastlines()
m.drawcountries()

x, y = m(list(aus["Longitude"].astype(float)), list(aus["Latitude"].astype(float)))
m.plot(x, y, 'go', markersize = 12, alpha = 0.8, color = "#0072B2")

plt.title('Extinct and Endangered Languages - Australia')
plt.show()


# In[ ]:


plt.figure(figsize=(15,8))
m = Basemap(projection = 'mill', llcrnrlat = -50, urcrnrlat = -5, llcrnrlon = 100, urcrnrlon = 160, resolution = 'h')
m.drawcoastlines()
m.drawcountries()

x, y = m(list(aus_extinct["Longitude"].astype(float)), list(aus_extinct["Latitude"].astype(float)))
m.plot(x, y, 'go', markersize = 15, alpha = 0.8, color = "#0072B2")

plt.title('Extinct Languages - Australia')
plt.show()


# In[ ]:


plt.figure(figsize=(15,8))
m = Basemap(projection = 'mill', llcrnrlat = 10, urcrnrlat = 35, llcrnrlon = -120, urcrnrlon = -80, resolution = 'h')
m.drawcoastlines()
m.drawcountries()

x, y = m(list(mexico["Longitude"].astype(float)), list(mexico["Latitude"].astype(float)))
m.plot(x, y, 'go', markersize = 15, alpha = 0.8, color = "darkgreen")

plt.title('Extinct and Endangered Languages - Mexico')
plt.show()


# In[ ]:


plt.figure(figsize=(15,8))
m = Basemap(projection = 'mill', llcrnrlat = -12, urcrnrlat = 0, llcrnrlon = 140, urcrnrlon = 160, resolution = 'h')
m.drawcoastlines()
m.drawcountries()

x, y = m(list(papau_extinct["Longitude"].astype(float)), list(papau_extinct["Latitude"].astype(float)))
m.plot(x, y, 'go', markersize = 15, alpha = 0.8, color = "darkorange")

plt.title('Extinct Languages - Papua New Guinea')
plt.show()


# In[ ]:


plt.figure(figsize=(15,8))
m = Basemap(projection = 'mill', llcrnrlat = -12, urcrnrlat = 0, llcrnrlon = 140, urcrnrlon = 160, resolution = 'h')
m.drawcoastlines()
m.drawcountries()

x, y = m(list(papau["Longitude"].astype(float)), list(papau["Latitude"].astype(float)))
m.plot(x, y, 'go', markersize = 15, alpha = 0.8, color = "darkorange")

plt.title('Extinct and Endangered Languages - Papua New Guinea')
plt.show()


# In[ ]:


plt.figure(figsize=(15,8))
m = Basemap(projection = 'mill', llcrnrlat = 20, urcrnrlat = 55, llcrnrlon = 75, urcrnrlon = 140, resolution = 'h')
m.drawcoastlines()
m.drawcountries()

x, y = m(list(china["Longitude"].astype(float)), list(china["Latitude"].astype(float)))
m.plot(x, y, 'go', markersize = 15, alpha = 0.8, color = "#006633")

plt.title('Extinct and Endangered Languages - China')
plt.show()

