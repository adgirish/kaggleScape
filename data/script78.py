
# coding: utf-8

# For the normalized signal calculate FFT per band. Get some stats horizontaly/verically for each band. Save it in a file. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import os
import io
from datetime import datetime as dt
import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt
import json


# In[ ]:


pd.options.display.max_columns  = 999
pd.options.display.max_colwidth = 999
pd.options.display.max_rows = 999


# In[ ]:


train_norm = pd.read_json("../input/normalize-singal-per-inc-angle/normalized_train.json").fillna(-1.0).replace('na', -1.0)
test_norm  = pd.read_json("../input/normalize-singal-per-inc-angle/normalized_test.json").fillna(-1.0).replace('na', -1.0)


# In[ ]:


train = pd.read_json("../input/statoil-iceberg-classifier-challenge/train.json").fillna(-1.0).replace('na', -1.0)
test  = pd.read_json("../input/statoil-iceberg-classifier-challenge/test.json").fillna(-1.0).replace('na', -1.0)


# ### Generate

# In[ ]:


def gen_fft(dt, out_file, make_plot):
    with open(out_file, 'w+') as f:
        for j in range(0,dt.shape[0]):            
            if len(str(dt.iloc[j,:].inc_angle))<=7:
                for band in ['band_1', 'band_2']:
                    x = np.array(dt.iloc[j,:][band])
                    multiplier = 1.1
                    threshold_h = 45.0
                    threshold_v = 45.0
                    mean_value = {}


                    mean_value['h'] = []
                    xx = []
                    sph = None
                    for i in range(75):
                        th = np.reshape(x,(75,75))[i,:]
                        sph = np.fft.fft(th)
                        mnh = np.mean(abs(sph))     
                        sph[abs(sph)<mnh*multiplier] = 0.0
                        xx.append(abs(np.fft.ifft(sph)))
                        mean_value['h'].append(mnh)
                    mxh = np.max(mean_value['h'])
                    mih = np.min(mean_value['h'])
                    mnh = np.mean(mean_value['h'])


                    mean_value['v']=[]
                    yy = []
                    spv = None
                    for i in range(75):
                        tv = np.reshape(x,(75,75))[:,i]
                        spv = np.fft.fft(tv)
                        mnv = np.mean(abs(spv))
                        spv[abs(spv)<mnv*multiplier] = 0.0
                        yy.append(abs(np.fft.ifft(spv)))
                        mean_value['v'].append(mnv)

                    mxv = np.max(mean_value['v'])
                    miv = np.min(mean_value['v'])                
                    mnv = np.mean(mean_value['v'])

                    estimate_size = sum(mean_value['v'] > mnv*multiplier)*sum(mean_value['h'] > mnh*multiplier)

                    yy = np.transpose(yy)
                    fft_data = {'band':band, 'id':dt.iloc[j,:].id, 'inc_angle':dt.iloc[j,:].inc_angle, 'is_iceberg':dt.iloc[j,:].is_iceberg, 'mxv':mxv,'miv':miv,'mnv':mnv,'mean_value_v':mean_value['v'], 'size_v': sum(mean_value['v'] > mnv*multiplier)
                                ,'mxh':mxh,'mih':mih,'mnh':mnh,'mean_value_h':mean_value['h'], 'size_h': sum(mean_value['h'] > mnh*multiplier) }
                    f.write(str(fft_data)+'\n')

                    if make_plot:        
                        fig = plt.figure(1,figsize=(20,10))
                        # vertical FFT means
                        ax = fig.add_subplot(4,1,1)
                        plt.plot((0, 75), (mnh*multiplier, mnh*multiplier), '--')
                        ax.plot(mean_value['v'])

                        # horizontal FFT means
                        axx = fig.add_subplot(4,1,2)
                        plt.plot((0, 75), (mnv*multiplier, mnv*multiplier), '--')
                        axx.plot(mean_value['h'])


                        axz = fig.add_subplot(4,1,3)
                        axz.plot(np.convolve(mean_value['h'], mean_value['v'], 'same'))

                        # original
                        axxx = fig.add_subplot(4,4,9)
                        axxx.imshow(np.reshape(x,(75,75)))

                        # vertial ifft 
                        axxx = fig.add_subplot(4,4,10)
                        axxx.imshow(yy)

                        # horizontal ifft 
                        axxx = fig.add_subplot(4,4,11)
                        axxx.imshow(xx)

                        # vertical * horizontal ifft
                        axxx = fig.add_subplot(4,4,12)
                        axxx.imshow(np.array(yy)*np.array(xx))

                        txx = ''
                        fc = 'blue'
                        if 'is_iceberg' in dt.columns:
                            if dt.iloc[j,:].is_iceberg==0:
                                txx = 'ice'
                                fc = 'red'
                            else:
                                txx = 'ship'
                                fc = 'green'

                        if band == 'band_1':
                            fcc = 'yellow'
                        else:
                            fcc = 'purple'
                        axxx.text(200,0,  '%s' % (band), color='k', bbox=dict(facecolor=fcc, alpha=0.3), horizontalalignment='center', verticalalignment='center')
                        axxx.text(200,25,  'Id: %s' % (dt.iloc[j,:].id), color='k', bbox=dict(facecolor=fc, alpha=0.3), horizontalalignment='center', verticalalignment='center')
                        axxx.text(200,50, 'Type: %s' % (txx), color='k', bbox=dict(facecolor=fc, alpha=0.3), horizontalalignment='center', verticalalignment='center')
                        axxx.text(200,75, 'Inc angle: %s' % (dt.iloc[j,:].inc_angle), color='k', bbox=dict(facecolor=fc, alpha=0.3), horizontalalignment='center', verticalalignment='center')

                        axxx.text(200,100, 'Size: %s' % (estimate_size), color='k', bbox=dict(facecolor=fc, alpha=0.3), horizontalalignment='center', verticalalignment='center')
                        #axxx.text(200,70,'avg H: %s avg V: %s' % (np.mean(abs(sp))), color='k', bbox=dict(facecolor=fc, alpha=0.3), horizontalalignment='center', verticalalignment='center')

                        plt.show()


# In[ ]:


gen_fft(train, 'fft.txt', False)


# In[ ]:


gen_fft(train_norm, 'normalized_fft.txt', False)


# ### Read back FFT

# In[ ]:


def read_back_fft(out_file):
    xxh = dict()
    xxv = dict()
    row = 0 
    df = pd.DataFrame()
    with open(out_file,'r') as f:
        for l in f:
            dc = eval(l)
            #'mean_value_h', 'mean_value_v'
            for key in ['band','size_v','size_h','inc_angle','is_iceberg','mxh','mxv','mnv','mnh','mih','miv','id']:
                df.loc[row, key] = dc[key]
            if not dc['id'] in xxv:
                xxv[dc['id']] = {}
            if not dc['id'] in xxh:
                xxh[dc['id']] = {}            
            xxv[dc['id']].update({dc['band']:dc['mean_value_v']})
            xxh[dc['id']].update({dc['band']:dc['mean_value_h']})
            row +=1
    df['size'] = df['size_v']*df['size_h']  
    return df, xxh, xxv


# In[ ]:


df_norm, xxh_norm, xxv_norm = read_back_fft('normalized_fft.txt')
df, xxh, xxv      = read_back_fft('fft.txt')


# ### Helper functionsaaa

# In[ ]:


def get_color(x):
    if x == 0.0:
        return 'red'
    return 'green'

s = lambda x : (((x-x.min())/float(x.max()-x.min())+1)*8)**2


# In[ ]:


import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


# # Graphs

# ### size x size

# In[ ]:


def size_size(df,title):
    cond1 = df.band=='band_1'
    cond2 = df.band=='band_2'

    x_var = 'size_v'
    y_var = 'size_v'

    # calculate counts for size
    u, c = np.unique(np.c_[df.loc[cond1,:][x_var], df.loc[cond2,:][y_var]], return_counts=True, axis=0)


    fig = plt.figure(1,figsize=(20,10))
    fig.suptitle(title, fontsize=14, fontweight='bold')



    ax = fig.add_subplot(111)
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)

    plt.scatter(x=df.loc[cond1,:][x_var], y=df.loc[cond2,:][y_var], c=df['is_iceberg'].apply(lambda x:get_color(x)), s=s(c), alpha=.5)

    red_patch = mpatches.Patch(color='red', label='Iceberg')
    green_patch = mpatches.Patch(color='green', label='Ship')
    plt.legend(handles=[red_patch, green_patch])

    slope = 1.8
    intercept = -29
    x = df.loc[cond1,:][x_var].unique()
    abline_values = [slope * i + intercept for i in x]
    plt.plot(x, abline_values, 'b')

    ax.set_ylim((0, 30))
    plt.grid(True)
    plt.show()


# In[ ]:


size_size(df, 'size_v*size_v for both bands')


# In[ ]:


size_size(df_norm, 'size_v*size_v for both bands (NORMALIZED)')


# #### band_1 mean_value_v

# In[ ]:


def band_mean(df, xxv, band, title):
    fig = plt.figure(3,figsize=(20,10))

    band = 'band_1'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    ax = fig.add_subplot(3,1,1)
    axx = fig.add_subplot(3,1,2)
    axxx = fig.add_subplot(3,1,3)

    ax.grid(True)
    axx.grid(True) 
    axxx.grid(True)

    red_patch = mpatches.Patch(color='red', label='Iceberg')
    green_patch = mpatches.Patch(color='green', label='Ship')
    ax.legend(handles=[red_patch, green_patch])
    axx.legend(handles=[red_patch])
    axxx.legend(handles=[green_patch])


    for k in xxv.keys():
        cond = (df['id']==k) & (df['band']==band)
        vl = xxv[k][band]
        ax.plot(vl,c=df.loc[cond, 'is_iceberg'].apply(lambda x:get_color(x)).values[0], alpha=.2)
        if (df.loc[cond, 'is_iceberg']==0.0).bool():
            axx.plot(vl,c='red', alpha=.2 )
        else:
            axxx.plot(vl,c='green', alpha=.2 )


    plt.show()


# In[ ]:


band_mean(df, xxv, 'band_1', 'mean_value_v for band_1')


# In[ ]:


band_mean(df, xxv, 'band_2', 'mean_value_v for band_2')


# In[ ]:


band_mean(df_norm, xxv_norm, 'band_1', 'mean_value_v for band_1 (NORMALIZED)')


# In[ ]:


band_mean(df_norm, xxv_norm, 'band_2', 'mean_value_v for band_2 (NORMALIZED)')


# In[ ]:


band_mean(df, xxh, 'band_1', 'mean_value_h for band_1')


# In[ ]:


band_mean(df, xxh, 'band_2', 'mean_value_h for band_2')


# In[ ]:


band_mean(df_norm, xxh_norm, 'band_1', 'mean_value_h for band_1 (NORMALIZED)')


# In[ ]:


band_mean(df_norm, xxh_norm, 'band_2', 'mean_value_h for band_2 (NORMALIZED)')


# # Tests

# In[ ]:


def threshold_fft_gt(df, xxv,  dataset, band, title, threshold):
    cnt = 0
    iceberg_count = 0
    for k in xxv.keys():
        if np.sum([1.0 for x in xxv[k][band] if x > threshold ])>0:
            #print(k + ' ' + str(df.loc[(df['id']==k) & (df['band']=='band_1')]['is_iceberg'].values[0]))
            iceberg_count += df.loc[(df['id']==k) & (df['band']==band)]['is_iceberg'].values[0]
            cnt += 1

    print('{:{width}} | {:{width}} | {:{width}} | th: {:{width}} | icebergs: {:{width}} | all:{:{width}} | ship hit rate:{:{width}}'.format(title, dataset, band, threshold, iceberg_count,cnt,round(1-iceberg_count/cnt,4), width=10))


# In[ ]:


threshold_fft_gt(df, xxv, 'xxv', 'band_1', 'standard', 70.0)
threshold_fft_gt(df, xxv, 'xxv', 'band_2', 'standard', 60.0)
threshold_fft_gt(df_norm, xxv_norm, 'xxv_norm', 'band_1', 'normalized', 17.0)
threshold_fft_gt(df_norm, xxv_norm, 'xxv_norm', 'band_2', 'normalized', 17.0)
print('---------------------')
threshold_fft_gt(df, xxh, 'xxh', 'band_1', 'standard', 70.0)
threshold_fft_gt(df, xxh, 'xxh', 'band_2', 'standard', 60.0)
threshold_fft_gt(df_norm, xxh_norm, 'xxh_norm', 'band_1', 'normalized', 17.0)
threshold_fft_gt(df_norm, xxh_norm, 'xxh_norm', 'band_1', 'normalized', 19.0)
threshold_fft_gt(df_norm, xxh_norm, 'xxh_norm', 'band_2', 'normalized', 17.0)


# In[ ]:


def threshold_fft_lt(df, xxv,  dataset, band, title, threshold):
    cnt = 0
    iceberg_count = 0
    for k in xxv.keys():
        if np.sum([1.0 for x in xxv[k][band] if x < threshold ])>0:
            #print(k + ' ' + str(df.loc[(df['id']==k) & (df['band']=='band_1')]['is_iceberg'].values[0]))
            iceberg_count += df.loc[(df['id']==k) & (df['band']==band)]['is_iceberg'].values[0]
            cnt += 1

    print('{:{width}} | {:{width}} | {:{width}} | th: {:{width}} | icebergs: {:{width}} | all:{:{width}} | ship hit rate:{:{width}}'.format(title, dataset, band, threshold, iceberg_count,cnt,round(1-iceberg_count/cnt,4), width=10))


# In[ ]:


threshold_fft_lt(df, xxv, 'xxv', 'band_1', 'standard', 30.0)
threshold_fft_lt(df, xxv, 'xxv', 'band_2', 'standard', 35.0)
threshold_fft_lt(df_norm, xxv_norm, 'xxv_norm', 'band_1', 'normalized', 3.0)
threshold_fft_lt(df_norm, xxv_norm, 'xxv_norm', 'band_2', 'normalized', 4.0)
print('---------------------')
threshold_fft_lt(df, xxh, 'xxh', 'band_1', 'standard', 30.0)
threshold_fft_lt(df, xxh, 'xxh', 'band_2', 'standard', 35.0)
threshold_fft_lt(df_norm, xxh_norm, 'xxh_norm', 'band_1', 'normalized', 2.0)
threshold_fft_lt(df_norm, xxh_norm, 'xxh_norm', 'band_1', 'normalized', 2.0)
threshold_fft_lt(df_norm, xxh_norm, 'xxh_norm', 'band_2', 'normalized', 4.0)


# ## Combined & normalized

# In[ ]:


cnt = 0
iceberg_count = 0
band = 'band_2'
for k in xxv_norm.keys():
    if (
           np.sum([1.0 for x in xxv[k]['band_2']      if x >60.0 ])>0 
        or np.sum([1.0 for x in xxv_norm[k]['band_2'] if x >17.0 ])>0 
        or np.sum([1.0 for x in xxh_norm[k]['band_1'] if x >19.0 ])>0 
        or np.sum([1.0 for x in xxh_norm[k]['band_2'] if x >17.0 ])>0     
        )    :
        iceberg_count += df_norm.loc[(df_norm['id']==k) & (df_norm['band']==band)]['is_iceberg'].values[0]
        cnt += 1

print('norm and standard | band 1 & 2 | icebergs: {} | all:{} | ship hit rate:{}'.format(iceberg_count,cnt,round(1-iceberg_count/cnt,4)))


# In[ ]:


cnt = 0
iceberg_count = 0
band = 'band_2'
for k in xxv_norm.keys():
    if np.sum([1.0 for x in xxh_norm[k][band] if x >19.0 ])>0 or np.sum([1.0 for x in xxv_norm[k][band] if x >17.0 ])>0:
        iceberg_count += df_norm.loc[(df['id']==k) & (df_norm['band']==band)]['is_iceberg'].values[0]
        cnt += 1

print('xxh & xxv | {} | icebergs: {} | all:{} | ship hit rate:{}'.format(band, iceberg_count,cnt,round(1-iceberg_count/cnt,4)))


# In[ ]:


cnt = 0
iceberg_count = 0
for k in xxv_norm.keys():
    if     np.sum([1.0 for x in xxh_norm[k]['band_2'] if x >19.0 ])>0         or np.sum([1.0 for x in xxv_norm[k]['band_2'] if x >17.0 ])>0         or np.sum([1.0 for x in xxh_norm[k]['band_1'] if x >19.0 ])>0:
        #print(k + ' ' + str(df.loc[(df['id']==k) & (df['band']=='band_1')]['is_iceberg'].values[0]))
        iceberg_count += df_norm.loc[(df_norm['id']==k) & (df_norm['band']=='band_1')]['is_iceberg'].values[0]
        cnt += 1

print('normalized | xxh & xxv | {} | xxh | band_1 | icebergs: {} | all:{} | ship hit rate:{}'.format(band, iceberg_count,cnt,round(1-iceberg_count/cnt,4)))


# ## Combined & non-normalized

# In[ ]:


cnt = 0
iceberg_count = 0
band = 'band_2'
for k in xxv.keys():
    if np.sum([1.0 for x in xxh[k][band] if x >60.0 ])>0 or np.sum([1.0 for x in xxv[k][band] if x >60.0 ])>0:
        #print(k + ' ' + str(df.loc[(df['id']==k) & (df['band']=='band_1')]['is_iceberg'].values[0]))
        iceberg_count += df.loc[(df['id']==k) & (df['band']==band)]['is_iceberg'].values[0]
        cnt += 1

print('xxh & xxv | {} | icebergs: {} | all:{} | ship hit rate:{}'.format(band, iceberg_count,cnt,round(1-iceberg_count/cnt,4)))

