
# coding: utf-8

# # TalkingData AdTracking Fraud Detection Challenge
# 
# This notebook presents some plots that simply show click frequency over time, for one value, e.g. all records with app==3.
# 
# Each plot is 600 pixels wide, one pixel is one second, so the width represents 10 minutes. The pixel value represents the click count for that second (with log scaling to make it clearer), so one hour is 6 rows of pixels, and one day is 144 rows. (The [color map](http://matplotlib.org/users/colormaps.html) is 'afmhot', black means 0 clicks.)
# 
# The training set spans 264460 seconds (3.06 days), and the test set 39600 seconds (0.46 days), though nearly half of the test set is blank (no records).
# 
# Beyond those facts, you're on your own, these plots are good for a general view but not specific details :)
# 
# I've also added a [script to generate all the plots](http://www.kaggle.com/jtrotman/generate-temporal-click-count-data-plots) for the top N most common values in each column, this notebook is for selected highlights.
# 
# I generated these visualisations expecting to just see fairly uniform/noisy variation over time.
# 
# Instead, many of the plots show extreme regularity, mechanistic clicking which obviously cannot be natural human behaviour... (The plots are the sum of lots of overlapping behaviour, even so, regular patterns presumably from heavy click automation are highly visible.)
# 
# The target (*is_attributed*) is not used to generate these plots, but I'd guess that feature engineering that describes the (conditional) density of clicks at a point in time will be needed to get the best performance...
# 
# In other words, good, predictive features will encode "is the record part of one of these mechanistic patterns?"

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, sys, time

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16'
        }

# read a subset, runs out of memory otherwise ('os' seems least interesting)
fields = [ 'ip', 'app', 'device', 'channel' ]
to_read = fields + [ 'click_time' ]

train_df  = pd.read_csv('../input/train.csv', usecols=to_read, parse_dates=['click_time'], dtype=dtypes) #, nrows=72000000)
test_df   = pd.read_csv('../input/test.csv', usecols=to_read, parse_dates=['click_time'], dtype=dtypes)
print('Loaded', train_df.shape, test_df.shape)


# In[ ]:


def datetime_to_deltas(series, delta=np.timedelta64(1, 's')):
    t0 = series.min()
    return ((series-t0)/delta).astype(np.int32)

train_df['sec'] = datetime_to_deltas(train_df.click_time)
test_df['sec'] = datetime_to_deltas(test_df.click_time)
print('Added seconds')

train_df.drop('click_time', axis=1, inplace=True)
test_df.drop('click_time', axis=1, inplace=True)
print('Dropped click_time')


# In[ ]:


from matplotlib.colors import LogNorm

# e.g. train_df.loc[train_df.ip==234]
def generate_plot(df):
    w = 600
    n = df.sec.max()+1
    l = int(np.ceil(n/float(w))*w)
    c = np.zeros(l, dtype=np.float32)
    np.add.at(c, df.sec.values, 1)
    print(f'total clicks {c.sum():.0f} \t max clicks {c.max():.0f} \t mean click rate {c.mean():.02f} ')
    return c.reshape((-1,w))

def show(pix, title):
    pix += 1 # matplotlib restriction
    ysize = 5 if pix.shape[0] < 400 else 8
    fig, ax0 = plt.subplots(figsize=(18, ysize))
    ax0.invert_yaxis()
    ax0.set_yticks(np.arange(0, pix.shape[0], 144), False)
    ax0.set_yticks(np.arange(0, pix.shape[0], 6), True)
    ax0.set_xticks(np.arange(0, pix.shape[1], 60), False)
    ax0.set_xticks(np.arange(0, pix.shape[1], 10), True)
    c = ax0.pcolormesh(pix, norm=LogNorm(vmin=1, vmax=pix.max()), cmap='afmhot')
    ax0.set_title(title)
    return fig.colorbar(c, ax=ax0)

def gen_show(df, col, value):
    idx = df[col]==value
    if idx.sum()<1:
        print('Not found!')
    else:
        pix = generate_plot(df.loc[idx])
        show(pix, f'{col} {value}')


# # Training Set
# 
# ## Apps

# App 3, most common in train set. Not much striking here, it serves as a guide to the general form of the plots, e.g. the darker bands indicate night time. The black pixels at the top indicate 0 clicks, the data starts off slowly... The x-axis is marked with seconds 0..600 and the y-axis counts 10-minute rows, 0..144 for one day.
# 
# Also the plotting function lists some stats: total clicks for the plot, the max clicks (in one second), and the mean clicks per second.
# 

# In[ ]:


gen_show(train_df, 'app', 3)


# App 1, strong periodic behaviour every minute, and a comb-like pattern on the right (some hourly regularity?)
# 

# In[ ]:


gen_show(train_df, 'app', 1)


# App 7, heavy but short spikes.
# 

# In[ ]:


gen_show(train_df, 'app', 7)


# App 20, very striking mechanistic behaviour, busy one minute, quiet the next!
# 

# In[ ]:


gen_show(train_df, 'app', 20)


# App 46, too regular to be human (clicks every ~5 seconds), plus some streaks that last 20-30 seconds.
# 

# In[ ]:


gen_show(train_df, 'app', 46)


# App 151, this is the most clicked app in the training set that has no *is_attributed==1* records at all (a target rate of 0). Zooming in on the plot it's clearly a highly regular mechanistic pattern.
# 

# In[ ]:


gen_show(train_df, 'app', 151)


# App 56 is the second most clicked app with no positive targets (3rd is 183, 4th is 93), these ar similar to app 151 above but with a very heavy spike in day 2.
# 

# In[ ]:


gen_show(train_df, 'app', 56)


# ## Channels

# Channel 236 and 237, more periodic behaviour, phase shifting over time, in different ways. (This is a real whoa! moment... anyone have any ideas on the cause of this?)
# 

# In[ ]:


gen_show(train_df, 'channel', 236)


# Channel 105 - more minutely switching, the exact same point in every minute, no phase drift...
# 

# In[ ]:


gen_show(train_df, 'channel', 105)


# Channel 244 - some fine horizontal banding particularly visible in darker night-time periods
# .

# In[ ]:


gen_show(train_df, 'channel', 244)


# Channel 419 is an example of what I expected to see: occasional clicks, some smooth day/night variation, no odd patterns. There are 10,371 clicks and for 6,675 *is_attributed==1*, a download rate of 64%, very high: apparently channel 419 is not a lucrative target for click fraud.

# In[ ]:


gen_show(train_df, 'channel', 419)


# Channel 272 is low volume, but with no *is_attributed==1*, the regularity of the presumably bot-generated clicks is easier to see, with 5-6 second gaps. (Unfortunately the test set only has 79 records for channel 272... but are there other channels in the test set with click patterns with easy-to-detect regularity like this? ;)

# In[ ]:


gen_show(train_df, 'channel', 272)


# Channel 114 is also low volume: 677 clicks but 629 downloads, a ~93% *is_attributed* rate, kind of abnormally high, perhaps a special offer? Also very few nighttime clicks, could it be a *channel* targeting young children or pensioners? (Just my speculation.)

# In[ ]:


gen_show(train_df, 'channel', 114)


# ## Devices

# Device 2 (2nd most common) some faint traces of patterns.
# 

# In[ ]:


gen_show(train_df, 'device', 2)


# Devices 3032, 3543, 3866, total cut outs in time. (Seems to cut off bottom of image in the notebook...)
# 

# In[ ]:


gen_show(train_df, 'device', 3543)


# Device 154, normal looking sporadic clicks, but with one streak on day 1.
# 

# In[ ]:


gen_show(train_df, 'device', 154)


# ## IPs

# IP 5314 and 5348 - the most active IPs in train - featuring an hour or so of abnormally low activity on day 2.
# 

# In[ ]:


gen_show(train_df, 'ip', 5314)


# In[ ]:


gen_show(train_df, 'ip', 5348) 


# IP 86767 - the same hour? This takes over from the other two IPs? (This may well have nothing to do with click fraud and instead be a network architecture effect.)
# 

# In[ ]:


gen_show(train_df, 'ip', 86767)


# # Test Set
# 
# Now what is in the test set? Note that the [test set has some time periods missing](http://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/51877)...
# 
# > **Aaron Yin wrote**:
# 
# >  We choose three ranges of time, as we believed they would play an more significant role in Chinese daily life.
# The ranges of time are: 12pm-14pm, 17pm-19pm, 21pm-23pm
# Note: the data's timestamp is in UTC+0, and Chinese timezone is UTC+8
# 
# The [original test set](http://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/51506) has uniform coverage of the whole test set time period and so would present a much better picture than we have here...
# 
# ## Apps
# 
# Test App 24, same checkerboard pattern as seen in train set. Seems to be generally ramping up in intensity, especially in the interval between band 2 and 3...

# In[ ]:


gen_show(test_df, 'app', 24)


# App 183, if you zoom in, the pattern is very regular...

# In[ ]:


gen_show(test_df, 'app', 183)


# ## Channels
# 
# Test Channel 125, 145, same 1 minute schedule switching pattern as above.

# In[ ]:


gen_show(test_df, 'channel', 125)


# Channel 236, some smoother 1-minute peaks (10 per row), again drifting in phase a little...

# In[ ]:


gen_show(test_df, 'channel', 236)


# That's it for now, happy feature engineering!
