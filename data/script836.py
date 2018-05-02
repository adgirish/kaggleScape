
# coding: utf-8

# *Update: Added analysis for the 2017 data below comparing logerror to kde. *
# <h1>The Parcel Density Hypothesis</h1>
# This Notebook analyses the relationship between parcel density and logerror. For a given parcel, a high parcel density  indicates a high amount of data points in its geographical region. This 'density' around a parcel could be related to the logerror we are predicting, since model performance (i.e. logerror) is generally dependent on the amount of representative data it was trained on. Therefore, I will test the following hypothesis:
# 
# <h3>Hypothesis: A higher parcel density is associated with a lower absolute logerror. </h3>
# <p></p>
# <p></p>
# This notebook approaches this in the following order:
# 1. Imports
# 2. Method
# 3. Results
# 4. Conclusion
# 5. Update on 2017 training data - the whole analysis again with logerror (not absolute logerror)
# 
# Let's start!
# 

# <h1>1. Imports </h1>
# First we import the packages we need.  Then we import data. Since we want to know the relationship between logerror and parcel density, we only need the rows from the training set that contain the x,y coordinates.

# In[ ]:


#  Packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
from matplotlib import pyplot as plt
from sklearn.neighbors import KDTree
from scipy.stats.stats import pearsonr
from scipy.stats import rankdata


# In[ ]:


#  Data
train = pd.read_csv('../input/train_2016_v2.csv')
props = pd.read_csv('../input/properties_2016.csv',low_memory=False)
train = train.merge(props,how='left',on='parcelid')
train = train[['parcelid','longitude','latitude','logerror']]
train.dropna(inplace=True)  #
del props  # delete redundant data
gc.collect()  # Free up memory
print("DataFrame sample:")
print("***************************************************")
train.head()
print("***************************************************")
print("shape = ",train.shape)


# <h1>2. Method: Kernel Density Estimation </h1>
# Now that the data and packages are loaded, we continue with estimating parcel densities. This technique  is more generally referred to as** "Kernel Density Estimation", or KDE**. It estimates the probability density function (PDF) on given data points. In this case, these data points are the geographical coordinates of parcels. Therefore, I refer to KDE as **Parcel Density Estimation, or Parcel Density Estimate (PDE).**
# 
# *More info: https://en.wikipedia.org/wiki/Kernel_density_estimation *

# <h3> 2.1 Parcel Density calculation</h3>
# 
# In this kernel, I use a** Gaussian kernel** because it seems the most intuitive method (other options are ‘gaussian’, ‘tophat’, ‘epanechnikov’, ‘exponential’, ‘linear’, ‘cosine’) . Here is 1D example from the sklearn documentation:
# ![](http://scikit-learn.org/stable/_images/sphx_glr_plot_kde_1d_001.png)
# 
# The KDE depends on the **bandwidth (bw)**. If the bandwidth is high, the KDE is relatively sensitive to more distant parcels. When the bandwidth is low, the KDE is relatively senstive to local parcels. For now, I calculate the bandwidth on 5 arbitrarily chosen bandwidths: 30,000, 10,000, 3,000, 1,000 and 300. Here is an illustration of how bandwidth affects KDE from wikipedia:
# ![](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2a/Kernel_density.svg/250px-Kernel_density.svg.png)
# 
# For optimizing the **performance** of the function, I chose leafsize 20 and relative tolerance of 0.00001. Leafsize is essentially a memory-cpu tradeoff and relative tolerance is a precision-cpu tradeoff. Both can greatly influence the time it takes to calculate the KDE. A great article on this topic can be found here: https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/  . 

# In[ ]:


def get_pde(train,bw):
    x = train['longitude'].values
    y = train['latitude'].values
    xy = np.vstack([x,y])
    X = np.transpose(xy)
    tree = KDTree(X,leaf_size = 20 )     
    parcelDensity = tree.kernel_density(X, h=bw,kernel='gaussian',rtol=0.00001)
    return parcelDensity


# In[ ]:


parcelDensity30000 = get_pde(train,30000)
parcelDensity10000 = get_pde(train,10000)
parcelDensity3000 = get_pde(train,3000)
parcelDensity1000 = get_pde(train,1000)
parcelDensity300 = get_pde(train,300)


# <h3>2.1 Parcel Density visualization at bandwith 30,000</h3>
# Let's visualize the results of our KDE in a scatter plot. We use the longitude and latitude of our parcels as x,y coordinates and color these points by their density. Below is the plot for a high bandwidth. It clearly shows the most dense area in bright yellow, the more semi-central locations in orange, the pheripheral locations in purple and the relatively isolated locations in black.
# 
# 

# In[ ]:


plt.figure(figsize=(14,14))
plt.axis("off")
plt.title("Gaussian Parcel Density Estimate at bandwidth 30,000")
plt.scatter(train['longitude'].values, train['latitude'].values, c=parcelDensity30000,cmap='inferno', s=1, edgecolor='')


# The bright yellow dense area on the map allignes almost perfectly with what google maps defines as the LA downtown. This means that the parcel density feature now contains informations about the centrality of a particular parcel. https://www.google.nl/maps/place/Downtown,+Los+Angeles,+CA,+USA/@33.9761623,-118.5305526,10z/data=!4m5!3m4!1s0x80c2c634253dfd01:0x26fe52df19a5a920!8m2!3d34.040713!4d-118.2467693 

# <h3>2.2 Parcel Density visualization at bandwith 10,000, 3,000, 1,000 and 300</h3>
# Now I plot the KDE's in four subplots for the remaining four bandwidths.  In smaller bandwidths, local fluctuations are relatively strong and that cann cause outliers that outscale the "normal" flucations in KDE in the plot image. To improve the visibility of the moderate fluctuations, I **rankscaled** the KDE's for these plots by their percentile. The script in the next box does this.

# In[ ]:


rankScaled30000 = 100*rankdata(parcelDensity30000)/len(parcelDensity30000)
rankScaled10000 = 100*rankdata(parcelDensity10000)/len(parcelDensity10000)
rankScaled3000 = 100*rankdata(parcelDensity3000)/len(parcelDensity3000)
rankScaled1000 = 100*rankdata(parcelDensity1000)/len(parcelDensity1000)
rankScaled300 = 100*rankdata(parcelDensity300)/len(parcelDensity300)


# In[ ]:


fig = plt.figure(figsize=(15,15))

ax1 = fig.add_subplot(221)
ax1.set_title('bandwidth = 10,000')
ax1.set_axis_off()
ax1.scatter(train['longitude'].values, train['latitude'].values, c=rankScaled10000,cmap='inferno', s=1, edgecolor='')

ax2 = fig.add_subplot(222)
ax2.set_title('bandwidth = 3,000')
ax2.set_axis_off()
ax2.scatter(train['longitude'].values, train['latitude'].values, c=rankScaled3000,cmap='inferno', s=1, edgecolor='')

ax3 = fig.add_subplot(223)
ax3.set_title('bandwidth = 1,000')
ax3.set_axis_off()
ax3.scatter(train['longitude'].values, train['latitude'].values, c=rankScaled1000,cmap='inferno', s=1, edgecolor='')

ax4 = fig.add_subplot(224)
ax4.set_title('bandwidth = 300')
ax4.set_axis_off()
ax4.scatter(train['longitude'].values, train['latitude'].values, c=rankScaled300,cmap='inferno', s=1, edgecolor='')


# <h1>3. Results </h1>
# We want to test whether higher KDE's are associated with lower absolute errors. Therefore, we want to compare the KDE to the absolute logerror. We draw scatter plots and measure the Pearson correlation for all five bandwidths

# <h3>3.1 Result for Bandwidth 30,000</h3>
# First, we look at the results for bandwidth 30,000 in detail

# In[ ]:


abs_logerrors = np.abs(train['logerror'].values)
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# In[ ]:


fig = plt.figure(figsize=(15,15))
fig.suptitle("Parcel Density Vs. Logerror")
ax1, ax2= fig.add_subplot(221),fig.add_subplot(222)
x1,x2 = parcelDensity30000, rankScaled30000
index1, index2 = x1.argsort(), x2.argsort()
x1 = x1[index1[::-1]]
x2 = x2[index2[::-1]]
y = abs_logerrors
y = y[index1[::-1]]
y_av = moving_average(y,n=250)
y_av = [0]*249 + list(y_av)

m, b = np.polyfit(x1,y , 1)
ax1.plot(x1, y, '.',alpha=0.5,color='skyblue')
ax1.plot(x1,y_av,linewidth=6,color="steelblue")
ax1.plot(x1, m*x1 + b, '--',linewidth=3,color='red')

ax1.set_xlim([-0.000000001,0.00000025])
ax1.set_ylim([-0.0,2])
ax1.set_ylabel("logerror",fontsize='large')
ax1.set_xlabel("PDE",fontsize='large')

m, b = np.polyfit(x2,y , 1)
ax2.plot(x2, y, '.',alpha=0.5,color='skyblue')
ax2.plot(x2,y_av,linewidth=6,color="steelblue")
ax2.plot(x2, m*x2 + b, '--',linewidth=3,color='red')
ax2.set_xlabel("PDE - ranked",fontsize='large')
ax2.set_ylabel("Logerror",fontsize='large')
ax2.set_xlim([0,100])
ax2.set_ylim([-0.0,2])



#     > light blue dots: individual data points
#     > dark blue line: moving average (window=250)
#     > red dashed line: linear fit
# **Analysis**
# 
# **Left **The red coefficient line indicates a weak positive relation between PDE and absolute logerror. Observering the scatter plot from the naked eye, there seems to be a stronger relationship between PDE and logerror than the coefficient line indicates. This is because some ranges for the PDE are relatively more common and therefore have more extreme values. I compensated for this on the** right** by putting the **ranked** PDE on the x-axis rather than the PDE itself.  Each percentile in the ranked PDE is (by definition) equally sized and therefore the **right  **provides  a  clearer view on the weak relationship between absolute logerror and PDE. 
# 
# **Now we'll check if the correlation is significant:**

# In[ ]:


corrCoef_30000_1, p_twoTailed_30000_1 = pearsonr(x1,y)
corrCoef_30000_2, p_twoTailed_30000_2 = pearsonr(x2,y)
p_oneTailed_30000_1 = 1 - (p_twoTailed_30000_1/2)

print("Result for bandwidth 30,000:")
print("*******************************************************************")
print("Correlation Coefficient: ",corrCoef_30000_1)
print("Two tailed_p: ",p_twoTailed_30000_1)
print("One tailed p for negative correlation: ",p_oneTailed_30000_1)
print("*******************************************************************")


# 
# 
# **Result: **The p-value indicates that **there is no significant negative correlation between PDE and absolute logerror at bandwidth 30,000**. 

# <h3> 3.2 Results for bandwidths 10,000, 3,000, 1,000 and 300 </h3>
# To avoid an overkill of plots and information, we'll ignore the ranked PDE and focus the normal PDE. 

# In[ ]:


fig = plt.figure(figsize=(15,15))
x1,x2,x3,x4 = parcelDensity10000, parcelDensity3000, parcelDensity1000,parcelDensity300
y = abs_logerrors

index1, index2,index3,index4 = x1.argsort(), x2.argsort(), x3.argsort(), x4.argsort()
x1, x2, x3, x4 = x1[index1[::-1]],  x2[index2[::-1]],  x3[index3[::-1]], x4[index4[::-1]]
x1 = x1 - min(x1)
x2 = x2 - min(x2)
x3 = x3 - min(x3)
x4 = x4 - min(x4)

y1, y2, y3, y4 = y[index1[::-1]], y[index2[::-1]],y[index3[::-1]],y[index4[::-1]]
y_av1 = moving_average(y1,n=100)
y_av1 = [0]*99 + list(y_av1)
y_av2 = moving_average(y2,n=100)
y_av2 = [0]*99 + list(y_av2)
y_av3 = moving_average(y3,n=100)
y_av3 = [0]*99 + list(y_av3)
y_av4 = moving_average(y4,n=100)
y_av4 = [0]*99 + list(y_av4)

ax1 = fig.add_subplot(221)
m, b = np.polyfit(x1,y1 , 1)
ax1.plot(x1, y1, '.',alpha=0.5,color='skyblue')
ax1.plot(x1,y_av1,linewidth=6,color="steelblue")
ax1.plot(x1, m*x1 + b, '--',linewidth=3,color='red')
ax1.set_xlim([0,0.0000005])
ax1.set_ylim([-0.0,2])
ax1.set_ylabel("logerror",fontsize='large')
ax1.set_xlabel("PDE",fontsize='large')

ax2 = fig.add_subplot(222)
m, b = np.polyfit(x2,y2 , 1)
ax2.plot(x2, y2, '.',alpha=0.5,color='skyblue')
ax2.plot(x2,y_av2,linewidth=4,color="steelblue")
ax2.plot(x2, m*x2 + b, '--',linewidth=3,color='red')
ax2.set_xlim([0,0.0000015])
ax2.set_ylim([-0.0,2])
ax2.set_ylabel("logerror",fontsize='large')
ax2.set_xlabel("PDE",fontsize='large')

ax3 = fig.add_subplot(223)
m, b = np.polyfit(x3,y3 , 1)
ax3.plot(x3, y3, '.',alpha=0.5,color='skyblue')
ax3.plot(x3,y_av3,linewidth=4,color="steelblue")
ax3.plot(x3, m*x3 + b, '--',linewidth=3,color='red')
ax3.set_xlim([-0,0.000005])
ax3.set_ylim([-0.0,2])
ax3.set_ylabel("logerror",fontsize='large')
ax3.set_xlabel("PDE",fontsize='large')

ax4 = fig.add_subplot(224)
m, b = np.polyfit(x4,y4 , 1)
ax4.plot(x4, y, '.',alpha=0.5,color='skyblue')
ax4.plot(x4,y_av4,linewidth=4,color="steelblue")
ax4.plot(x4, m*x4 + b, '--',linewidth=3,color='red')
ax4.set_xlim([-0.000000001,0.00004])
ax4.set_ylim([-0.0,2])
ax4.set_ylabel("logerror",fontsize='large')
ax4.set_xlabel("PDE",fontsize='large')



# **Analysis:** The plots vaguely suggest, for all four bandwidths,  that the mean absolute logerror declines as PDE increases. At the higher PDE's, the sparsity of datapoints causes some jumpy behavior in the moving average. These are probably result of the reduced precision I used when calculating the KDE (remains to be confirmed).
# 
# ** Next I calculate if this correlation is significant**
# 

# In[ ]:


corrCoef_10000, p_twoTailed_10000 = pearsonr(x1,y1)
p_oneTailed_10000 = p_twoTailed_10000/2
corrCoef_3000, p_twoTailed_3000 = pearsonr(x2,y2)
p_oneTailed_3000 = p_twoTailed_3000/2
corrCoef_1000, p_twoTailed_1000 = pearsonr(x3,y3)
p_oneTailed_1000 = p_twoTailed_1000/2
corrCoef_300, p_twoTailed_300 = pearsonr(x4,y4)
p_oneTailed_300 = p_twoTailed_300/2

print("For BW 10,000, Correlation Coefficient: ",corrCoef_10000)
print("For BW 10,000, One tailed_p: ",p_oneTailed_10000)
print("**********************************************************")
print("For BW 3,000, Correlation Coefficient: ",corrCoef_3000)
print("For BW 3,000, One tailed_p: ",p_oneTailed_3000)
print("**********************************************************")
print("For BW 1,000, Correlation Coefficient: ",corrCoef_1000)
print("For BW 1,000, One tailed_p: ",p_oneTailed_1000)
print("**********************************************************")
print("For BW 500, Correlation Coefficient: ",corrCoef_300)
print("For BW 500, One tailed_p: ",p_oneTailed_300)


# **Result**: For bandwidths 10,000, 3,000, 1,000 and 300 there is a significant negative correlation between KDE and absolute logerror.

# <h1> 4. Conclusion </h1>
# **The result indicate that higher parcel density is associated with lower absolute logerrors.** This hypothesis was confirmed in bandwidths 10,000, 3,000, 1,000 and 300. Suprisingly, the opposite relationship was observed for bandwidth 30,000. Overall, This means that parcel density could be a promising feature to use for the prediction of the Zestimate logerror.
# 
# **Thanks all for reading, hope you enjoyed!!**

# <h1> 5. Update: repeated with 2017 data, but now I use logerror instead of abs logerror</h1>

# **Load...**

# In[ ]:


#  Data
train = pd.read_csv('../input/train_2017.csv')
props = pd.read_csv('../input/properties_2017.csv',low_memory=False)
train = train.merge(props,how='left',on='parcelid')
train = train[['parcelid','longitude','latitude','logerror']]
train.dropna(inplace=True)  #
del props  # delete redundant data
gc.collect()  # Free up memory


# In[ ]:


parcelDensity30000 = get_pde(train,30000)
parcelDensity10000 = get_pde(train,10000)
parcelDensity3000 = get_pde(train,3000)
parcelDensity1000 = get_pde(train,1000)
parcelDensity300 = get_pde(train,300)


# In[ ]:


rankScaled30000 = 100*rankdata(parcelDensity30000)/len(parcelDensity30000)
rankScaled10000 = 100*rankdata(parcelDensity10000)/len(parcelDensity10000)
rankScaled3000 = 100*rankdata(parcelDensity3000)/len(parcelDensity3000)
rankScaled1000 = 100*rankdata(parcelDensity1000)/len(parcelDensity1000)
rankScaled300 = 100*rankdata(parcelDensity300)/len(parcelDensity300)


# In[ ]:


logerrors = train['logerror'].values
y = logerrors


# **Result @30.000 BW**

# In[ ]:


fig = plt.figure(figsize=(15,15))
fig.suptitle("Parcel Density Vs. Logerror")
ax1, ax2= fig.add_subplot(221),fig.add_subplot(222)
x1,x2 = parcelDensity30000, rankScaled30000
index1, index2 = x1.argsort(), x2.argsort()
x1 = x1[index1[::-1]]
x2 = x2[index2[::-1]]
y = y[index1[::-1]]
y_av = moving_average(y,n=250)
y_av = [0]*249 + list(y_av)

m, b = np.polyfit(x1,y , 1)
ax1.plot(x1, y, '.',alpha=0.5,color='skyblue')
ax1.plot(x1,y_av,linewidth=6,color="steelblue")
ax1.plot(x1, m*x1 + b, '--',linewidth=3,color='red')

ax1.set_xlim([-0.000000001,0.00000025])
ax1.set_ylim([-2.0,2])
ax1.set_ylabel("logerror",fontsize='large')
ax1.set_xlabel("PDE",fontsize='large')

m, b = np.polyfit(x2,y , 1)
ax2.plot(x2, y, '.',alpha=0.5,color='skyblue')
ax2.plot(x2,y_av,linewidth=6,color="steelblue")
ax2.plot(x2, m*x2 + b, '--',linewidth=3,color='red')
ax2.set_xlabel("PDE - ranked",fontsize='large')
ax2.set_ylabel("Logerror",fontsize='large')
ax2.set_xlim([0,100])
ax2.set_ylim([-2.0,2])



# **significance level @ 30k bw, 2-tailed**

# In[ ]:


corrCoef_30000_1, p_twoTailed_30000_1 = pearsonr(x1,y)

print("Result for bandwidth 30,000:")
print("*******************************************************************")
print("Correlation Coefficient: ",corrCoef_30000_1)
print("Two tailed_p: ",p_twoTailed_30000_1)


# **Results at @ 10k, 3k, 1k and 300 BW, 2-tailed**

# In[ ]:


fig = plt.figure(figsize=(15,15))
x1,x2,x3,x4 = parcelDensity10000, parcelDensity3000, parcelDensity1000,parcelDensity300
y = logerrors

index1, index2,index3,index4 = x1.argsort(), x2.argsort(), x3.argsort(), x4.argsort()
x1, x2, x3, x4 = x1[index1[::-1]],  x2[index2[::-1]],  x3[index3[::-1]], x4[index4[::-1]]
x1 = x1 - min(x1)
x2 = x2 - min(x2)
x3 = x3 - min(x3)
x4 = x4 - min(x4)

y1, y2, y3, y4 = y[index1[::-1]], y[index2[::-1]],y[index3[::-1]],y[index4[::-1]]
y_av1 = moving_average(y1,n=100)
y_av1 = [0]*99 + list(y_av1)
y_av2 = moving_average(y2,n=100)
y_av2 = [0]*99 + list(y_av2)
y_av3 = moving_average(y3,n=100)
y_av3 = [0]*99 + list(y_av3)
y_av4 = moving_average(y4,n=100)
y_av4 = [0]*99 + list(y_av4)

ax1 = fig.add_subplot(221)
m, b = np.polyfit(x1,y1 , 1)
ax1.plot(x1, y1, '.',alpha=0.5,color='skyblue')
ax1.plot(x1,y_av1,linewidth=6,color="steelblue")
ax1.plot(x1, m*x1 + b, '--',linewidth=3,color='red')
ax1.set_xlim([0,0.0000005])
ax1.set_ylim([-02.0,2])
ax1.set_ylabel("logerror",fontsize='large')
ax1.set_xlabel("PDE",fontsize='large')

ax2 = fig.add_subplot(222)
m, b = np.polyfit(x2,y2 , 1)
ax2.plot(x2, y2, '.',alpha=0.5,color='skyblue')
ax2.plot(x2,y_av2,linewidth=4,color="steelblue")
ax2.plot(x2, m*x2 + b, '--',linewidth=3,color='red')
ax2.set_xlim([0,0.0000015])
ax2.set_ylim([-02.0,2])
ax2.set_ylabel("logerror",fontsize='large')
ax2.set_xlabel("PDE",fontsize='large')

ax3 = fig.add_subplot(223)
m, b = np.polyfit(x3,y3 , 1)
ax3.plot(x3, y3, '.',alpha=0.5,color='skyblue')
ax3.plot(x3,y_av3,linewidth=4,color="steelblue")
ax3.plot(x3, m*x3 + b, '--',linewidth=3,color='red')
ax3.set_xlim([-0,0.000005])
ax3.set_ylim([-2.0,2])
ax3.set_ylabel("logerror",fontsize='large')
ax3.set_xlabel("PDE",fontsize='large')

ax4 = fig.add_subplot(224)
m, b = np.polyfit(x4,y4 , 1)
ax4.plot(x4, y, '.',alpha=0.5,color='skyblue')
ax4.plot(x4,y_av4,linewidth=4,color="steelblue")
ax4.plot(x4, m*x4 + b, '--',linewidth=3,color='red')
ax4.set_xlim([-0.000000001,0.00004])
ax4.set_ylim([-02.0,2])
ax4.set_ylabel("logerror",fontsize='large')
ax4.set_xlabel("PDE",fontsize='large')



# **significance level @ 30k bw, 2-tailed**

# In[ ]:


corrCoef_10000, p_twoTailed_10000 = pearsonr(x1,y1)
corrCoef_3000, p_twoTailed_3000 = pearsonr(x2,y2)
corrCoef_1000, p_twoTailed_1000 = pearsonr(x3,y3)
corrCoef_300, p_twoTailed_300 = pearsonr(x4,y4)
print("For BW 10,000, Correlation Coefficient: ",corrCoef_10000)
print("For BW 10,000,  2-tailed_p: ",p_twoTailed_10000)
print("**********************************************************")
print("For BW 3,000, Correlation Coefficient: ",corrCoef_3000)
print("For BW 3,000,  2-tailed_p: ",p_twoTailed_3000)
print("**********************************************************")
print("For BW 1,000, Correlation Coefficient: ",corrCoef_1000)
print("For BW 1,000, 2-tailed_p: ",p_twoTailed_1000)
print("**********************************************************")
print("For BW 500, Correlation Coefficient: ",corrCoef_300)
print("For BW 500, tailed_p: ",p_twoTailed_300)

