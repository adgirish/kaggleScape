
# coding: utf-8

# ## Notes about machine generated images:
# <p> I'm republishing this notebook because questions about machine generated images are getting asked alot.
#  Machine generated images DO NOT count in the scoring of the public or private leaderboard.  It says this in the competition description!  <br>
# TL:DR - You can identify machine generated images by the length of the decimal in the incidence angle (natural have <= 4, machine generated have > 4).   <br>
# This really has no effect on your predictions, but I guess it could make it easier for you to hand label the 3425 natural images if you really wanted to.  
# </p>

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from subprocess import check_output
from mpl_toolkits.axes_grid1 import ImageGrid
import random
random.seed(1)


# In[ ]:


train = pd.read_json("../input/train.json")
agg_df = train.groupby('inc_angle').agg({"is_iceberg": [len, np.sum]}).sort_values([('is_iceberg', 'len')], ascending=False)
agg_df[0:20]


# In[ ]:


def plot_bands(df, ia):
    df = df[df['inc_angle'] == ia]
    i = int(np.sqrt(len(df))//1 * 2)
    j = int(2*len(df) // i + 1)
    fig = plt.figure(1, figsize=(24,24))
    grid = ImageGrid(fig, 111, nrows_ncols=(i, j), axes_pad=0.05)
    for i, (band1, band2, id_num, inc_angle, iceberg) in enumerate(df.values):
        # plot band 1
        ax = grid[(i*2)]
        band1_sample = band1
        band1_sample = np.array(band1_sample).reshape(75, 75)
        ax.imshow(band1_sample / 75.)
        ax.text(10, 4, 'Id: %s %s' % (id_num, "Band_1"), color='k', backgroundcolor='m', alpha=0.8)
        ax.text(10, 10, 'Incidence Angle: (%.4f)' % inc_angle, color='w', backgroundcolor='k', alpha=0.8)
        ax.text(10, 16, 'Is Iceberg: %s' % iceberg, color='k', backgroundcolor='w', alpha=0.8)
        ax.axis('on')
        # plot band 2
        ax = grid[(i*2)+1]
        band2_sample = band2
        band2_sample = np.array(band2_sample).reshape(75, 75)
        ax.imshow(band2_sample / 75.)
        ax.text(10, 4, 'Id: %s %s' % (id_num, "Band_2"), color='k', backgroundcolor='m', alpha=0.8)
        ax.text(10, 10, 'Incidence Angle: (%.4f)' % inc_angle, color='w', backgroundcolor='k', alpha=0.8)
        ax.text(10, 16, 'Is Iceberg: %s' % iceberg, color='k', backgroundcolor='w', alpha=0.8)
        ax.axis('on')


# ## Plot some of the leaky image pairs

# In[ ]:


test = pd.read_json("../input/test.json")
test['is_iceberg'] = -999
combined = pd.concat([train, test])


# In[ ]:


plot_bands(combined, 42.5128)


# In[ ]:


def plot_bands_test(df):
    df = df.sample(8)
    i = 4 #int(np.sqrt(len(df))//1 * 2)
    j = 4 #int(2*len(df) // i + 1)
    fig = plt.figure(1, figsize=(16,16))
    grid = ImageGrid(fig, 111, nrows_ncols=(i, j), axes_pad=0.05)
    for i, (band1, band2, id_num, inc_angle, iceberg) in enumerate(df.values):
        # plot band 1
        ax = grid[(i*2)]
        band1_sample = band1
        band1_sample = np.array(band1_sample).reshape(75, 75)
        ax.imshow(band1_sample / 75.)
        ax.text(10, 4, 'Id: %s %s' % (id_num, "Band_1"), color='k', backgroundcolor='m', alpha=0.8)
        ax.text(10, 10, 'Incidence Angle: (%.8f)' % inc_angle, color='w', backgroundcolor='k', alpha=0.8)
        ax.text(10, 16, 'Is Iceberg: %s' % iceberg, color='k', backgroundcolor='w', alpha=0.8)
        ax.axis('on')
        # plot band 2
        ax = grid[(i*2)+1]
        band2_sample = band2
        band2_sample = np.array(band2_sample).reshape(75, 75)
        ax.imshow(band2_sample / 75.)
        ax.text(10, 4, 'Id: %s %s' % (id_num, "Band_2"), color='k', backgroundcolor='m', alpha=0.8)
        ax.text(10, 10, 'Incidence Angle: (%.8f)' % inc_angle, color='w', backgroundcolor='k', alpha=0.8)
        ax.text(10, 16, 'Is Iceberg: %s' % iceberg, color='k', backgroundcolor='w', alpha=0.8)
        ax.axis('on')


# ## Plot random sample of test images

# In[ ]:


plot_bands_test(test)


# In[ ]:


def plot_bands_test_juxt(df):
    df['precision_4'] = df['inc_angle'].apply(lambda x: len(str(x))) <= 7
    df = pd.concat([df[df['precision_4'] == True].sample(8), df[df['precision_4'] == False].sample(8)])
    fig = plt.figure(1, figsize=(16,16))
    grid = ImageGrid(fig, 111, nrows_ncols=(4, 4), axes_pad=0.05)
    for i, (band1, band2, id_num, inc_angle, iceberg, precision_4) in enumerate(df.values):
        # plot band 1
        ax = grid[(i)]
        band1_sample = band1
        band1_sample = np.array(band1_sample).reshape(75, 75)
        ax.imshow(band1_sample / 75.)
        ax.text(10, 4, 'Id: %s %s' % (id_num, "Band_1"), color='k', backgroundcolor='m', alpha=0.8)
        ax.text(10, 10, 'Incidence Angle: (%.8f)' % inc_angle, color='w', backgroundcolor='k', alpha=0.8)
        ax.text(10, 16, 'Is Iceberg: %s' % iceberg, color='k', backgroundcolor='w', alpha=0.8)
        if i < 8:
            ax.text(10, 22, 'Precision is <= 4: %s' % precision_4, color='k', backgroundcolor='g', alpha=0.8)
        else:
            ax.text(10, 22, 'Precision is <= 4: %s' % precision_4, color='k', backgroundcolor='r', alpha=0.8)
        ax.axis('on')


# ## Something seems suspicious about the precision of the incidence angle

# In[ ]:


plot_bands_test_juxt(test)


# **It looks like images with incidence angles having less than or equal to 4 decimal places (like all of those in the training set) are the naturally captured images, and those with greater precision are machine generated.**

# In[ ]:


print('~%.1f%% of the test data is machine generated' % (100 * (1 - test['precision_4'].sum() / len(test))))
print('There are %i naturally captured images in the test set' % (test['precision_4'].sum() + 13))
# My method misses 13 of the natural images.  Thanks for the fix! Sorry I didn't update this sooner


# > ## Trying a leakage submission

# In[ ]:


CUTOFF = 2
agg_df = agg_df[agg_df['is_iceberg']['len'] >= CUTOFF]
my_df = []
for i in range(0,len(agg_df.index.values)):
    my_df.append([agg_df.index.values[i], agg_df['is_iceberg'].values[i][0], agg_df['is_iceberg'].values[i][1]])
my_df = pd.DataFrame(my_df, columns = ['ia', 'count', 'sum_is_iceberg']).drop(0) # remove 1st row NA

test['is_iceberg'] = 0.5
for (ia, count, sum_is_iceberg) in my_df.values:
    if(count == sum_is_iceberg):
        leak = 1
        test.loc[test['inc_angle'] == ia, 'is_iceberg'] = leak
    elif(sum_is_iceberg == 0):
        leak = 0
        test.loc[test['inc_angle'] == ia, 'is_iceberg'] = leak

test[['id', 'is_iceberg']].to_csv('littleleak_cutoff2.csv', index=False)


# <p> * The R version of this submission scores .46560 on the public LB, with about 1300 of the 3425 natural test images labelled 0 or 1. (This suggests that exactly 1 prediction of either 1 or 0 is incorrect on the public LB)
