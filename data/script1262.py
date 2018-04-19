
# coding: utf-8

# ## Quick Visual ##
# 
# Trying to get a quick visual of this competition.

# In[ ]:


import pandas as pd
import numpy as np
import datetime


import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)



# Read data 

dateparse = lambda x: datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')

# Read/clean data

# This Python 3 environment comes with many helpful analytics libraries installed
FILE="../input/gifts.csv"
d=pd.read_csv(FILE, encoding = "ISO-8859-1")


# In[ ]:


d['type'] = d['GiftId'].apply(lambda x: x.split('_')[0])
d['id'] = d['GiftId'].apply(lambda x: x.split('_')[1])



# In[ ]:


d['type'].value_counts()


# In[ ]:


def Weight(mType):
    if mType == "horse":
        return max(0, np.random.normal(5,2,1)[0])
    if mType == "ball":
        return max(0, 1 + np.random.normal(1,0.3,1)[0])
    if mType == "bike":
        return max(0, np.random.normal(20,10,1)[0])
    if mType == "train":
        return max(0, np.random.normal(10,5,1)[0])
    if mType == "coal":
        return 47 * np.random.beta(0.5,0.5,1)[0]
    if mType == "book":
        return np.random.chisquare(2,1)[0]
    if mType == "doll":
        return np.random.gamma(5,1,1)[0]
    if mType == "blocks":
        return np.random.triangular(5,10,20,1)[0]
    if mType == "gloves":
        return 3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3 else np.random.rand(1)[0]


# In[ ]:


# Build Weights
d['weight'] = d['type'].apply(lambda x: Weight(x))


# In[ ]:


d.head()


# In[ ]:


sns.distplot(d[d['type']=='horse'] ['weight'],bins=100, label="horse" );
sns.distplot(d[d['type']=='ball'] ['weight'],bins=100,  label="ball"  );
sns.distplot(d[d['type']=='bike']['weight'],bins=100,   label="bike"  );
sns.distplot(d[d['type']=='train']['weight'],bins=100,  label="train" );
sns.distplot(d[d['type']=='coal']['weight'],bins=100,   label="coal"  );
sns.distplot(d[d['type']=='book']['weight'],bins=100,   label="book"  );
sns.distplot(d[d['type']=='doll']['weight'],bins=100,   label="doll"  );
sns.distplot(d[d['type']=='blocks']['weight'],bins=100,   label="blocks"  );
sns.distplot(d[d['type']=='gloves']['weight'],bins=100,   label="gloves"  );
plt.legend();


# ## Light Items ##
# 
# Find maximum weights for each item

# In[ ]:


g = d.groupby(['type']).agg({'weight':max})
g = g.reset_index()
g.sort_values(by=['weight'],ascending=True,inplace=True)
g


# ## Only One of This Item (Heavy)##
# 
# The maximum weight is 50.  So, which items go over 25 in weight?  That could be a potential problem with 2 or more.

# In[ ]:


# Bike and Coal only ones over 25
d[d['weight']> 25]['type'].unique()


# In[ ]:


#sns.distplot(d[d['type']=='horse'] ['weight'],bins=100, label="horse" );
#sns.distplot(d[d['type']=='ball'] ['weight'],bins=100,  label="ball"  );
sns.distplot(d[d['type']=='bike']['weight'],bins=100,   label="bike"  );
sns.distplot(d[d['type']=='train']['weight'],bins=100,  label="train" );
sns.distplot(d[d['type']=='coal']['weight'],bins=100,   label="coal"  );
#sns.distplot(d[d['type']=='book']['weight'],bins=100,   label="book"  );
#sns.distplot(d[d['type']=='doll']['weight'],bins=100,   label="doll"  );
#sns.distplot(d[d['type']=='blocks']['weight'],bins=100,   label="blocks"  );
#sns.distplot(d[d['type']=='gloves']['weight'],bins=100,   label="gloves"  );
plt.legend();


# In[ ]:


sns.distplot(d[d['type']=='coal']['weight'],bins=100,   label="coal"  );

plt.legend();


# In[ ]:


sns.distplot(d[d['type']=='bike']['weight'],bins=100,   label="bike"  );

plt.legend();


# In[ ]:


# So you could have 2 trains...maybe
sns.distplot(d[d['type']=='train']['weight'],bins=100,   label="train"  );
plt.legend();


# ## Submit Requirements ##
# 
#  - Must have at least 3 items in each bag
#  - 1000 bags
#  - Bag over 50 lbs gets removed
# 

# In[ ]:


# Total amount of weight
d['weight'].sum()


# ## Combinations - How does the distribution look? ##

# In[ ]:


# Use this to create  multiples
def mul(mType,number):
    a=[]
    for i in range(0,number):
        a.append(Weight(mType))
    return a


# In[ ]:


# Working with combinations
# Combine coal and books
# This is just to show the distribution..

a=[]
for i in range(0,10000):
    tmp = mul("coal",1)+mul("book",3) 
    a.append(sum(tmp))

t=pd.DataFrame(a,columns=['weights'])
sns.distplot(t,bins=100,   label="(1) coal and (3) books"  );
print("Greater than 50lb {:03.2f} %".format( t[t['weights'] > 50].sum()[0]/10000)  )
print('Greater than 40lb and less 50lb {:03.2f}%'.format(t[(t['weights'] > 40) & (t['weights'] < 50)   ].sum()[0]/10000.0))
plt.legend();



# In[ ]:


# 2 Coals are more interesting
# ..but probably not a good pick...

a=[]
for i in range(0,10000):
    tmp = mul("coal",2) 
    a.append(sum(tmp))

t=pd.DataFrame(a,columns=['weights'])
sns.distplot(t,bins=100,   label="(2) coal "  );
print("Greater than 50lb {:03.2f} %".format( t[t['weights'] > 50].sum()[0]/10000)  )
print('Greater than 40lb and less 50lb {:03.2f}%'.format(t[(t['weights'] > 40) & (t['weights'] < 50)   ].sum()[0]/10000.0))
plt.legend();


# ## Quantile Table with Counts ##
# 
# Build a table to see what combinations we can do by hand... just to get an idea if we're going in the right direction.

# In[ ]:


def BuildDef():
    b=[]
    for atype in  ["book","ball","horse","blocks","doll","train","bike","gloves","coal"]:
        b.append([atype, 
                  d[d['type']==atype].quantile(q=0.95, interpolation='linear')[0],
                  d[d['type']==atype].quantile(q=0.85, interpolation='linear')[0],
                  d[d['type']==atype].quantile(q=0.60, interpolation='linear')[0],
                  d[d['type']==atype].quantile(q=0.40, interpolation='linear')[0],
                  d[d['type']==atype].quantile(q=0.20, interpolation='linear')[0]]
                )
    dk=pd.DataFrame(b,columns=['type','q95','q85','q60','q40','q20'])
    return dk
        
        


# In[ ]:


dk = BuildDef()
# Take a look at what we have
dk['type'].head()


# In[ ]:


j=d['type'].value_counts().to_frame().reset_index()
def getCount(atype):
    val = j[j['index']==atype].iloc[0]['type']
    return val


dk['count'] = dk['type'].apply(lambda x: getCount(x))
dk.sort_values(by=['q85'],ascending=True,inplace=True)
dk


# ## Sample Submission ##
# 
# Score: 31212.68873

# In[ ]:


coal=0
book=0
bike=0
train=0
blocks=0
doll=0
horse=0
gloves=0
ball=0
with open("Santa_03.csv", 'w') as f:
        f.write("Gifts\n")
        for i in range(1000):
            if coal < 166:
                f.write('coal_'+str(coal)+' book_'+str(book))
                coal+=1
                book+=1
                f.write(' book_'+str(book)+'\n')
                book+=1
            elif blocks < 1000 and train < 1000:
                f.write('blocks_'+str(blocks)+' train_'+str(train))
                blocks+=1
                train+=1
                f.write(' blocks_'+str(blocks)+' train_'+str(train)+'\n')
                blocks+=1
                train+=1
            elif bike < 500 and blocks < 1000:
                f.write('bike_'+str(bike)+' train_'+str(train)+' blocks_'+str(blocks)+'\n')
                bike+=1
                train+=1
                blocks+=1
            elif book < 1000 and gloves < 200: 
                f.write('doll_'+str(doll))
                doll+=1
                f.write(' doll_'+str(doll))
                doll+=1
                f.write(' doll_'+str(doll))
                doll+=1
                f.write(' horse_'+str(horse))
                horse+=1
                f.write(' horse_'+str(horse))
                horse+=1
                f.write(' horse_'+str(horse))
                horse+=1
                f.write(' gloves_'+str(gloves))
                gloves+=1
                f.write(' ball_'+str(ball))
                ball+=1
                f.write(' ball_'+str(ball))
                ball+=1
                f.write(' ball_'+str(ball))
                ball+=1
                f.write(' ball_'+str(ball))
                ball+=1
                f.write(' ball_'+str(ball))
                ball+=1
                f.write(' book_'+str(book)+'\n')
                book+=1
            # (1) bike  (4) books  (2) horse -- See Graph below    
            elif bike < 500 and horse < 1000 and book < 1200: 
                f.write('bike_'+str(bike))
                bike+=1
                f.write(' book_'+str(book))
                book+=1
                f.write(' book_'+str(book))
                book+=1
                f.write(' book_'+str(book))
                book+=1
                f.write(' book_'+str(book))
                book+=1
                f.write(' horse_'+str(horse))
                horse+=1
                f.write(' horse_'+str(horse)+'\n')
                horse+=1
            
                


print("coal max(166)",coal)                
print("horse max(1000)",horse)
print("book max(1200)",book)
print("bike max(500)",bike)
print("gloves max(200)",gloves)
print("train max(1000)",train)
print("ball max(1100)",ball)
print("doll max(1000)",doll)
print("blocks max(1000)",blocks)


# In[ ]:


a=[]
for i in range(0,10000):
    tmp = mul("bike",1)+mul("book",4)+mul("horse",2)
    a.append(sum(tmp))

t=pd.DataFrame(a,columns=['weights'])
sns.distplot(t,bins=100,   label="(1) bike  (4) books  (2) horse"  );
print("Greater than 50lb {:03.2f} %".format( t[t['weights'] > 50].sum()[0]/10000)  )
print('Greater than 40lb and less 50lb {:03.2f}%'.format(t[(t['weights'] > 40) & (t['weights'] < 50)   ].sum()[0]/10000.0))
print('Greater than 20lb and less 50lb {:03.2f}%'.format(t[(t['weights'] > 20) & (t['weights'] < 50)   ].sum()[0]/10000.0))


plt.legend();

