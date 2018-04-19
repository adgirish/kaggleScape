
# coding: utf-8

# Here, we'll take a look at mapping the latitude and longitude data provided to us by the Two Sigma Rental Listing competition.  Next, we'll try applying K-Nearest Neighbor to the location data to see if it will actually provide any insight.  Finally, out of my own interest, I plotted the cost of living across new york city to see what the distribution looks like.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
terrain = sns.color_palette(palette='terrain',n_colors=10)
plasma = sns.color_palette(palette='plasma',n_colors=10)
rainbow = sns.color_palette(palette='rainbow',n_colors=6)
get_ipython().run_line_magic('matplotlib', 'inline')

from bokeh.io import output_notebook
from bokeh.layouts import gridplot,row,column
from bokeh.plotting import figure,show
output_notebook()


# We'll load in the data and take a look at its shape to get an idea of how many features and how many samples we have.

# In[ ]:


trainDF=pd.read_json('../input/train.json')
testDF=pd.read_json('../input/test.json')
print('Training data dimensions:',trainDF.shape)
print('Testing data dimensions:',testDF.shape)


# ### Interest based on geographical location
# As they always say in real estate, "location, location, location" (Lord Harold Samuel).  We'll know that people will have preferences in terms of where they are looking.  Because of this, we should look at the interest level mapped across different areas.

# In[ ]:


p = figure(title="interest level based on geography",y_range=(40.65,40.85),x_range=(-74.05,-73.85))
p.xaxis.axis_label = 'longitude'
p.yaxis.axis_label = 'latitude'
lowLat=trainDF['latitude'][trainDF['interest_level']=='low']
lowLong=trainDF['longitude'][trainDF['interest_level']=='low']
medLat=trainDF['latitude'][trainDF['interest_level']=='medium']
medLong=trainDF['longitude'][trainDF['interest_level']=='medium']
highLat=trainDF['latitude'][trainDF['interest_level']=='high']
highLong=trainDF['longitude'][trainDF['interest_level']=='high']
p.circle(lowLong,lowLat,size=3,color=terrain.as_hex()[1],fill_alpha=0.1,line_alpha=0.1,legend='low')
p.circle(medLong,medLat,size=3,color=plasma.as_hex()[9],fill_alpha=0.1,line_alpha=0.1,legend='med')
p.circle(highLong,highLat,size=3,color=plasma.as_hex()[5],fill_alpha=0.1,line_alpha=0.1,legend='high')
show(p, notebook_handle=True)


# If you zoom in on manhattan, you'll quickly realize that people tend to have more interest near East Village, Chelsea, Hell's Kitchen, and Upper East Side.  But the reason why it probably looks brighter in those areas is probably due to the number of requests in those areas.  So I plotted each individual color separately below.  From there, you can really see that even though there's a lot of high interest in the areas I described above, there's still quite a lot of low interests.  We will need to dig deeper into this feature to determine whether it will be useful or not.

# In[ ]:


p1 = figure(width=500, height=500, title=None,y_range=(40.65,40.85),x_range=(-74.05,-73.85))
p1.circle(lowLong,lowLat,size=3,color=terrain.as_hex()[1],fill_alpha=0.1,line_alpha=0.1,legend='low')
p2 = figure(width=500, height=500, title=None,y_range=(40.65,40.85),x_range=(-74.05,-73.85))
p2.circle(medLong,medLat,size=3,color=plasma.as_hex()[9],fill_alpha=0.1,line_alpha=0.1,legend='med')
p3 = figure(width=500, height=500, title=None,y_range=(40.65,40.85),x_range=(-74.05,-73.85))
p3.circle(highLong,highLat,size=3,color=plasma.as_hex()[5],fill_alpha=0.1,line_alpha=0.1,legend='high')
show(column(p1,p2,p3), notebook_handle=True)


# ### KNN performance on longitude and latitude data
# Since I am quite interested in whether longitutde and latitude data will perform well in our prediction task, I've decided to further pursue this option.  First I'll need to create a dataframe from just the longitutde and latitude data and our dependent variable, interest level.  
# 
# I feel like KNN would perform the best with this kind of data because the assumption is that people interested in apartments in one building may be interested other apartments in that building.

# In[ ]:


X=pd.concat([trainDF['latitude'],trainDF['longitude']],axis=1)
y=trainDF['interest_level']


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import scipy as sp


# First we'll split the training data into testing and training sets

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=45)
neigh = KNeighborsClassifier(n_neighbors=9)
neigh.fit(X_train, y_train)


# I want to check what is the % accuracy in predicting the right interest level

# In[ ]:


predVal=neigh.predict(X_test)
mat=[predVal,y_test]
df=pd.DataFrame(mat).transpose()
df.columns=('h0','y')
df['diff']=np.where(df.h0==df.y,1,0)
print('% correct =',sum(df['diff'])/len(df['diff'])*100)


#  I looked into building out the log loss function to see how much error there is in the predictions

# In[ ]:


PredProb=neigh.predict_proba(X_test)
pred=np.asmatrix(PredProb)
pred.columns=('high','low','medium')
s=np.asmatrix(pd.get_dummies(y_test))
def f(x):
    return sp.log(sp.maximum(sp.minimum(x,1-10**-5),10**-5))
f=np.vectorize(f)
predf=f(pred)
mult=np.multiply(predf,s)
print('log loss =',np.sum(mult)/-len(y_test))


# This log loss is quite high so let's see if we can improve this by increasing our k value.  Since, it would be annoying to change the value and run it, I figure it'll be faster to run a for loop through values of k from odd numbers between 3 to 39 (represented by j).  I also wanted to have atleast 5 samples in each k to give us a good average (represented by i).

# In[ ]:


accbig=[]
loglossbig=[]

def f(x):
    return sp.log(sp.maximum(sp.minimum(x,1-10**-5),10**-5))
f=np.vectorize(f)

for j in range(3,40,2):
    logloss=[]
    acc=[]
    for i in range(5):
        #split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=i)
        neigh = KNeighborsClassifier(n_neighbors=j)
        #train classifier
        neigh.fit(X_train, y_train)
        
        #find % predicted correctly for this k
        predVal=neigh.predict(X_test)
        mat=[predVal,y_test]
        df=pd.DataFrame(mat).transpose()
        df.columns=('h0','y')
        df['diff']=np.where(df.h0==df.y,1,0)
        acc.append(sum(df['diff'])/len(df['diff']))
        
        #find the logloss for this k
        PredProb=neigh.predict_proba(X_test)
        pred=np.asmatrix(PredProb)
        pred.columns=('high','low','medium')
        s=np.asmatrix(pd.get_dummies(y_test))
        predf=f(pred)
        mult=np.multiply(predf,s)
        logloss.append(np.sum(mult)/-len(y_test))
    loglossbig.append(np.mean(logloss))
    accbig.append(np.mean(acc))
print(accbig)
print(loglossbig)


# #### Now let's plot this against every K to see the decrease

# In[ ]:


plt.plot(range(3,40,2),loglossbig)
plt.ylabel('logloss')
plt.xlabel('k value')
plt.title('KNN logloss on longitude and latitude')


# In[ ]:


plt.plot(range(3,40,2),accbig)
plt.ylabel('% predicted correctly')
plt.xlabel('k value')
plt.title('KNN prediction on longitude and latitude')


# #### Even though the performance for this isn't amazing, we can see that there are some predictive value that we can use from longtitude and latitude data

# ### (sidetrack) Rent cost in NYC
# Just for my own entertainment, I wanted to see what the pricing distribution is in this list of training samples, then I scatter plotted this for NYC and provided different price points that contained most of the data in the histogram

# In[ ]:


sns.distplot(trainDF['price'][trainDF['price']<6000])


# In[ ]:


Lat25=trainDF['latitude'][trainDF['price']<2500]
Long25=trainDF['longitude'][trainDF['price']<2500]
Lat30=trainDF['latitude'][(trainDF['price']<3000)&(trainDF['price']>=2500)]
Long30=trainDF['longitude'][(trainDF['price']<3000)&(trainDF['price']>=2500)]
Lat35=trainDF['latitude'][(trainDF['price']<3500)&(trainDF['price']>=3000)]
Long35=trainDF['longitude'][(trainDF['price']<3500)&(trainDF['price']>=3000)]
Lat40=trainDF['latitude'][(trainDF['price']<4000)&(trainDF['price']>=3500)]
Long40=trainDF['longitude'][(trainDF['price']<4000)&(trainDF['price']>=3500)]
Latup=trainDF['latitude'][(trainDF['price']>=4000)]
Longup=trainDF['longitude'][(trainDF['price']>=4000)]


# In[ ]:


p = figure(title="Cost",y_range=(40.65,40.85),x_range=(-74.05,-73.85))
p.xaxis.axis_label = 'latitude'
p.yaxis.axis_label = 'longitude'

p.circle(Long25,Lat25,size=3,color=rainbow.as_hex()[0],fill_alpha=0.6,line_alpha=0.6,legend='<$2500')
p.circle(Long30,Lat30,size=3,color=rainbow.as_hex()[2],fill_alpha=0.6,line_alpha=0.6,legend='$3000')
p.circle(Long35,Lat35,size=3,color=rainbow.as_hex()[4],fill_alpha=0.6,line_alpha=0.6,legend='$3500')
p.circle(Long40,Lat40,size=3,color=rainbow.as_hex()[5],fill_alpha=0.6,line_alpha=0.6,legend='$4000')
#p.circle(Latup,Longup,size=3,color=rainbow.as_hex()[5],fill_alpha=0.6,line_alpha=0.6,legend='up')
p.legend.location = 'bottom_right'
show(p, notebook_handle=True)


# If you enjoyed the contents of this Notebook, please upvote =)
