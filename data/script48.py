
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import bson
import io
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import tables ##enables hdf tables
import cv2 #opencv helpful for storing image as array

from matplotlib.colors import ListedColormap
from wordcloud import WordCloud


# In[ ]:


path = '../input/'
get_ipython().system('ls "$path"')


# #### Read Files

# In[ ]:


categories = pd.read_csv('{}{}'.format(path,'category_names.csv'),
                         index_col=0)


# In[ ]:


categories.head()


# #### Read bson file and convert to pandas DataFrame

# In[ ]:


with open('{}{}'.format(path,'train_example.bson'),'rb') as b:
    df = pd.DataFrame(bson.decode_all(b.read()))


# ###### keep only binary image data in the imgs column

# In[ ]:


df['imgs'] = df['imgs'].apply(lambda rec: rec[0]['picture'])


# #### set category_id as index

# In[ ]:


df.set_index('category_id',inplace=True)
df.head()


# ##### Combine images and categries

# In[ ]:


df[categories.columns.tolist()] = categories.loc[df.index]


# In[ ]:


df.head()


# In[ ]:


df['imgs'] = df['imgs'].apply(lambda img: Image.open(io.BytesIO(img)))


# In[ ]:


fig,axs  = plt.subplots(7,5 ,figsize=(10,10))
title = df['category_level1'].str.split('-').str[0].str.strip()
# title += ','
# title += df['category_level2'].str.split('-').str[0].str.strip()
title = title.tolist()
axs = axs.flatten()
for i,ax in enumerate(axs):
    ax.imshow(df.iloc[i,1],
              interpolation='nearest', 
              aspect='auto')
    ax.set_title(title[i])
    #remove frame and ticks
    ax.axis('off')
plt.tight_layout()


# ### Overview of categories

# #### Read the train dataset

# In[ ]:


CHUNK_SIZE = 100000
with open('{}{}'.format(path,'train.bson'),'rb',buffering=True) as b:
    i=0
    lst = []
    for line in bson.decode_file_iter(b):
        if i%CHUNK_SIZE == 0 and i !=0:
            df = pd.DataFrame(lst)
            # df['imgs'] = df['imgs'].apply(
            #lambda reclst: ','.join(rec['picture'].hex() for rec in reclst))
            try:
                df.iloc[:,:-1].to_hdf('file.h5',
                                      key='train',
                                      format='table',
                                      append=True)
            except Exception as e:
                #catch disk full
                print('error: ',e)
                break
            lst=[]
        lst.append(line)
        i+=1


# In[ ]:


train = pd.read_hdf('file.h5',key='train')
#combine with categries
train = pd.merge(categories,train,right_on='category_id',left_index=True)


# In[ ]:


train.info()


# In[ ]:


train.head()


# In[ ]:


train.shape


# #### see distinct categories

# In[ ]:


cats = train['category_level1'].value_counts()
cats.head()


# In[ ]:


abbriv = cats.index.str.split('\W').str[0].str.strip()
abbriv


# In[ ]:


sns.set_style('white')
fig,ax = plt.subplots(1,figsize=(12,6))
pal = ListedColormap(sns.color_palette('Paired').as_hex())
colors = pal(np.interp(cats,[cats.min(),cats.max()],[0,1]))
bars = ax.bar(range(1,len(cats)+1),cats,color=colors);
ax.set_xticks([]);
ax.set_xlim(0,len(cats))
ax1 = plt.twiny(ax)
ax1.set_xlim(0,len(cats))
ax1.set_xticks(range(1,len(abbriv)+1,1));
ax1.set_xticklabels(abbriv.values,rotation=90);
# sns.despine();


# In[ ]:


categories.head()


# In[ ]:


wc = WordCloud(max_words=500,  margin=20,
               random_state=0).generate(' '.join(categories.iloc[:,0].values.flatten()))


# In[ ]:


def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(0, 0%%, %d%%)" % np.random.randint(60, 100)
default_colors = wc.to_array()
plt.title("Custom colors")
plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3),
           interpolation="bilinear")
wc.to_file("a_new_hope.png")
plt.axis("off")
plt.figure()
plt.title("Default colors")
plt.imshow(default_colors, interpolation="bilinear")
plt.axis("off")
plt.show()

