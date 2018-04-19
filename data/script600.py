
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import seaborn as sns
import matplotlib.pyplot as plt
color = sns.color_palette()
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Load training data and see what's inside.**

# In[ ]:


train_df = pd.read_csv('../input/train.csv',nrows=10000)
train_df.info(memory_usage='deep')


# In[ ]:


test_df = pd.read_csv('../input/test.csv',nrows=10000)
test_df.info(memory_usage='deep')


# In[ ]:


test_df['click_time'] = pd.to_datetime(test_df['click_time'])
test_df['day'] = test_df['click_time'].dt.day
test_df['day'].value_counts()


# * We have 6 parameters types as int64 and 2  parameters types as object in training data, and 6 parameters types as int64 and 1  parameters types as object in testing data.
# * The parameter, attributed_time, is not in the testing data, so let's drop it to save memory.
# * ** Because we have only one day of test data, I using chunk to get one day of training data.**

# In[ ]:


del train_df
del test_df
df = pd.read_csv('../input/train.csv', iterator=True, chunksize=10000,nrows= 3700000, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
train_df = pd.concat([chunk[chunk['click_time'].str.contains("2017-11-06")] for chunk in df])
train_df.info(memory_usage='deep')


# In[ ]:


train_df.head()


# * Now, let's take a look at the distribution of target (is_attributed).

# In[ ]:


group_df = train_df.is_attributed.value_counts().reset_index()
k = group_df['is_attributed'].sum()
plt.figure(figsize = (12,8))
sns.barplot(group_df['index'], (group_df.is_attributed/k), alpha=0.8, color=color[0])
print((group_df.is_attributed/k))
plt.ylabel('Frequency', fontsize = 12)
plt.xlabel('Attributed', fontsize = 12)
plt.title('Frequency of Attributed', fontsize = 16)
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:


print(train_df.ip.describe())
plt.figure(figsize=(12, 8))
sns.kdeplot(train_df.ip, shade=True)
plt.title('IP distribution', fontsize = 15)
plt.xlabel('IP', fontsize = 12)
plt.ylabel('Percent', fontsize = 12)
plt.show()


# * **The distribution of IP data is average.**
# * **After the int64 type parameters have been processed, we need to process the object type parameters.**

# In[ ]:


train_df['click_time'] = pd.to_datetime(train_df['click_time'])
train_df['hour'] = train_df['click_time'].dt.hour
train_df['minute'] = train_df['click_time'].dt.minute
train_df['second'] = train_df['click_time'].dt.second
train_df=train_df.drop(['click_time'], axis =1)


# In[ ]:


train_df.info(memory_usage='deep')


# In[ ]:


import gc
del df
gc.collect()


# In[ ]:


colormap = plt.cm.viridis
plt.figure(figsize=(16,16))
plt.title(' The Absolute Correlation Coefficient of Features', y=1.05, size=15)
sns.heatmap(abs(train_df.astype(float).corr()),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True, )
plt.show()


# * **From the picture.**
# *  **We can know more important influence on is_attributed parameter is weekday, hour, ip, and app parameter.**
# * **Thanks for Bryan Arnold, perhaps using the correlation coefficient is not the best way to express the method of targeting non-continuous data.**

# **Let's group by some parameters**

# In[ ]:


#The frequency of each parameter (hours)
def group_by(lis_p, select_p, data, Agg=''):
    print('group by...')
    newname = '{}'.format('_'.join(lis_p))
    all_p = lis_p[:]
    all_p.append(select_p)
    if Agg=='':
        gp = data[all_p].groupby(by=lis_p)
        gp = gp[select_p].count().reset_index().rename(index=str, columns={select_p: newname})
    else:
        gp = data[all_p].groupby(by=lis_p).agg(Agg)
        gp = gp[select_p].reset_index().rename(index=str, columns={select_p: newname})
    print('merge...')
    data = data.merge(gp, on=lis_p, how='left')
    return data, newname
#The frequency of each parameter with IP (hours)
train_df, tmp1 = group_by(['ip', 'hour'], 'channel', train_df, 'count')
train_df, tmp2 = group_by(['ip', 'hour', 'device'], 'channel', train_df, 'count')
train_df, tmp3 = group_by(['ip', 'hour', 'app'], 'channel', train_df, 'count')                  
train_df, tmp4 = group_by(['ip', 'hour', 'channel'], 'os', train_df, 'count')
train_df, tmp5 = group_by(['ip', 'hour', 'os'], 'channel', train_df, 'count')
parameter_with_IP = [tmp1,tmp2,tmp3,tmp4,tmp5]
del tmp1
del tmp2
del tmp3
del tmp4
del tmp5
#train_df = train_df.drop( ['ip'], axis=1)
gc.collect()


# In[ ]:


def calc_iv(df, feature, target, pr=False):
    """
    Set pr=True to enable printing of output.
    
    Output: 
      * iv: float,
      * data: pandas.DataFrame|
    """

    lst = []

    df[feature] = df[feature].fillna("NULL")

    for i in range(df[feature].nunique()):
        val = list(df[feature].unique())[i]
        lst.append([feature,                                                        # Variable
                    val,                                                            # Value
                    df[df[feature] == val].count()[feature],                        # All
                    df[(df[feature] == val) & (df[target] == 0)].count()[feature],  # Good (think: Fraud == 0)
                    df[(df[feature] == val) & (df[target] == 1)].count()[feature]]) # Bad (think: Fraud == 1)

    data = pd.DataFrame(lst, columns=['Variable', 'Value', 'All', 'Good', 'Bad'])

    data['Share'] = data['All'] / data['All'].sum()
    data['Bad Rate'] = data['Bad'] / data['All']
    data['Distribution Good'] = (data['All'] - data['Bad']) / (data['All'].sum() - data['Bad'].sum())
    data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()
    data['WoE'] = np.log(data['Distribution Good'] / data['Distribution Bad'])

    data = data.replace({'WoE': {np.inf: 0, -np.inf: 0}})

    data['IV'] = data['WoE'] * (data['Distribution Good'] - data['Distribution Bad'])
    data['abs_WoE'] = abs(data['WoE'])
    data = data.sort_values(by=['Variable', 'Value'], ascending=[True, True])
    data.index = range(len(data.index))

    if pr:
        print(data)
        print('IV = ', data['IV'].sum())
    return data
def loop_iv(lis_val, train_df):
    dic_val = {}
    for i in lis_val:
        data = calc_iv(train_df, i, 'is_attributed')
        dic_val[i] = data
        print("Done {0}".format(i))
    return dic_val


# In[ ]:


lis=list(train_df.columns)
lis.remove('is_attributed')
lis.remove('ip')
lis.remove('hour')
lis.remove('minute')
lis.remove('second')
dic=loop_iv(lis, train_df)


# In[ ]:


dic_num = {}
for i in lis:
    nr_woe = dic[i]['WoE'].argmax()
    nr_iv = dic[i]['IV'].argmax()
    dic_num[i] = [nr_woe, nr_iv]


# In[ ]:


sum_iV={}
for i in lis:
    sum_num = dic[i]['IV'].sum()
    sum_iV[i] = sum_num
    print("The sum of IV parameter in {0} : {1}".format(i, sum_num))
plt.figure(figsize = (16,8))
m_colors=[]
k_num=0
for num in range(len(lis)):
    if (num//len(color))>k_num:
        k_num+=1
    t = num - k_num*len(color)
    m_colors.append(color[t])
sns.barplot(list(sum_iV.keys()), list(sum_iV.values()), alpha=0.8, palette = m_colors)
plt.ylabel('IV value', fontsize = 12)
plt.xlabel('Parameters', fontsize = 12)
plt.title("Each parameter's IV", fontsize = 16)
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:


train_df, tmp1 = group_by(['ip','app'], 'channel', train_df, 'count')
train_df, tmp2 = group_by(['ip', 'app', 'os'], 'channel', train_df, 'count')
train_df, tmp3 = group_by(['ip', 'device'], 'channel', train_df, 'count')                  
train_df, tmp4 = group_by(['app', 'channel'], 'os', train_df, 'count')
parameter_IA_IAO_ID_AC = [tmp1,tmp2,tmp3,tmp4]
dic.update(loop_iv(parameter_IA_IAO_ID_AC, train_df))


# In[ ]:


for i in parameter_IA_IAO_ID_AC:
    sum_num = dic[i]['IV'].sum()
    sum_iV[i] = sum_num
    print("The sum of IV parameter in {0} : {1}".format(i, sum_num))


# In[ ]:


#train_df = train_df.drop( parameter_with_IP, axis=1)
#The frequency of each parameter with IP (minute)
train_df, tmp1 = group_by(['ip', 'hour','minute'], 'channel', train_df, 'count')
train_df, tmp2 = group_by(['ip', 'hour', 'device','minute'], 'channel', train_df, 'count')
#train_df, tmp3 = group_by(['ip', 'hour', 'app','minute'], 'channel', train_df, 'count')                  
#train_df, tmp4 = group_by(['ip', 'hour', 'channel','minute'], 'os', train_df, 'count')
#train_df, tmp5 = group_by(['ip', 'hour', 'os','minute'], 'channel', train_df, 'count')
train_df, tmp3 = group_by(['ip', 'hour', 'os','minute'], 'channel', train_df, 'count')
#parameter_with_IP_minute = [tmp1,tmp2,tmp3,tmp4,tmp5]
parameter_with_IP_minute = [tmp1,tmp2,tmp3]
dic.update(loop_iv(parameter_with_IP_minute, train_df))


# In[ ]:


for i in parameter_with_IP_minute:
    sum_num = dic[i]['IV'].sum()
    sum_iV[i] = sum_num
    print("The sum of IV parameter in {0} : {1}".format(i, sum_num))


# In[ ]:


##train_df = train_df.drop( parameter_with_IP_minute, axis=1)
##The frequency of each parameter with IP (second)
#train_df, tmp1 = group_by(['ip', 'hour','minute','second'], 'channel', train_df, 'count')
#train_df, tmp2 = group_by(['ip', 'hour', 'device','minute','second'], 'channel', train_df, 'count')
#train_df, tmp3 = group_by(['ip', 'hour', 'app','minute','second'], 'channel', train_df, 'count')                  
#train_df, tmp4 = group_by(['ip', 'hour', 'channel','minute','second'], 'os', train_df, 'count')
#train_df, tmp5 = group_by(['ip', 'hour', 'os','minute','second'], 'channel', train_df, 'count')
#parameter_with_IP_second = [tmp1,tmp2,tmp3,tmp4,tmp5]
#dic.update(loop_iv(parameter_with_IP_second, train_df))


# In[ ]:


#for i in parameter_with_IP_second:
#    sum_num = dic[i]['IV'].sum()
#    sum_iV[i] = sum_num
#    print("The sum of IV parameter in {0} : {1}".format(i, sum_num))


# In[ ]:


#train_df = train_df.drop( parameter_with_IP_second, axis=1)
#The frequency of each parameter with IP and channel (hours)
train_df, tmp1 = group_by(['ip', 'app', 'hour', 'channel'], 'os', train_df, 'count')
#train_df, tmp2 = group_by(['ip', 'app','minute','second', 'hour', 'channel'], 'os', train_df, 'count')
#train_df, tmp3 = group_by(['ip', 'device', 'hour', 'channel'], 'os', train_df, 'count')
#train_df, tmp4 = group_by(['ip', 'os', 'hour', 'channel'], 'app', train_df, 'count')
train_df, tmp2 = group_by(['ip', 'device', 'hour', 'channel'], 'os', train_df, 'count')
parameter_with_IP_channel = [tmp1,tmp2,tmp3,tmp4]
parameter_with_IP_channel = [tmp1,tmp2]
dic.update(loop_iv(parameter_with_IP_channel, train_df))


# In[ ]:


for i in parameter_with_IP_channel:
    sum_num = dic[i]['IV'].sum()
    sum_iV[i] = sum_num
    print("The sum of IV parameter in {0} : {1}".format(i, sum_num))


# In[ ]:


#train_df = train_df.drop( parameter_with_IP_channel, axis=1)
#The frequency of each parameter with IP and app (hours)
train_df, tmp1 = group_by(['ip', 'app', 'hour', 'device'], 'os', train_df, 'count')
train_df, tmp2 = group_by(['ip', 'os', 'hour', 'app'], 'device', train_df, 'count')
#The frequency of each parameter with IP and device (hours)
train_df, tmp3 = group_by(['ip', 'os', 'hour', 'device'], 'app', train_df, 'count')
parameter_with_IP_app = [tmp1,tmp2,tmp3]
dic.update(loop_iv(parameter_with_IP_app, train_df))


# In[ ]:


for i in parameter_with_IP_app:
    sum_num = dic[i]['IV'].sum()
    sum_iV[i] = sum_num
    print("The sum of IV parameter in {0} : {1}".format(i, sum_num))


# In[ ]:


#train_df = train_df.drop( parameter_with_IP_app, axis=1)
#The frequency of each parameter with app (IP)
train_df, tmp1 = group_by(['ip', 'app', 'channel'], 'os', train_df)
train_df, tmp2 = group_by(['ip', 'device', 'app'], 'os', train_df)
train_df, tmp3 = group_by(['ip', 'os', 'app'], 'channel', train_df)
parameter_with_app = [tmp1,tmp2,tmp3]
dic.update(loop_iv(parameter_with_app, train_df))


# In[ ]:


for i in parameter_with_app:
    sum_num = dic[i]['IV'].sum()
    sum_iV[i] = sum_num
    print("The sum of IV parameter in {0} : {1}".format(i, sum_num))


# In[ ]:


#train_df = train_df.drop( parameter_with_app, axis=1)
#The frequency of each parameter with app and device(IP)
train_df, tmp1 = group_by(['ip', 'app', 'device','channel'], 'os', train_df)
train_df, tmp2 = group_by(['ip', 'device', 'app', 'os'], 'channel', train_df)
#The frequency of each parameter with app and channel(IP)
#train_df, tmp3 = group_by(['ip', 'app', 'os','channel'], 'device', train_df)
#train_df, tmp4 = group_by(['app','channel', 'hour','minute','second' ], 'device', train_df, 'count')
train_df, tmp3 = group_by(['app','channel', 'hour','minute','second' ], 'device', train_df, 'count')
#parameter_with_app_device = [tmp1,tmp2,tmp3,tmp4]
parameter_with_app_device = [tmp1,tmp2,tmp3]
dic.update(loop_iv(parameter_with_app_device, train_df))


# In[ ]:


for i in parameter_with_app_device:
    sum_num = dic[i]['IV'].sum()
    sum_iV[i] = sum_num
    print("The sum of IV parameter in {0} : {1}".format(i, sum_num))


# In[ ]:


#train_df = train_df.drop( parameter_with_app_device, axis=1)
#The frequency of each parameter with device (IP)
train_df, tmp1 = group_by(['ip', 'device', 'channel'], 'os', train_df)
train_df, tmp2 = group_by(['ip', 'os', 'device'], 'channel', train_df)
#The frequency of each parameter with device and channel (IP)
#train_df, tmp3 = group_by(['ip', 'device', 'channel', 'os'], 'app', train_df)
##The frequency of each parameter with channel (IP)
#train_df, tmp4 = group_by(['ip', 'os', 'channel'], 'app', train_df)
#parameter_with_device = [tmp1,tmp2,tmp3,tmp4]
parameter_with_device = [tmp1,tmp2]
dic.update(loop_iv(parameter_with_device, train_df))


# In[ ]:


for i in parameter_with_device:
    sum_num = dic[i]['IV'].sum()
    sum_iV[i] = sum_num
    print("The sum of IV parameter in {0} : {1}".format(i, sum_num))
#train_df = train_df.drop( parameter_with_device, axis=1)


# In[ ]:


v=list(sum_iV.values())
v_m = list(sum_iV.values())
v_n = list(sum_iV.keys())
v.sort()
kv = []
for i in range(len(v)):
    vv = v_m.index(v[i])
    kv.append(v_n[vv])
print('The Top ten IV parameters-----------------------')
for i in range(1, 11):
    print('The {0} big IV paramter is {1}.....'.format(i, kv[-i]))


# **Now, let's add the time interval parameters. **

# In[ ]:


def group_by_click(lis_p, data):
    print('group by...')
    newname = '{}_click_time_gap'.format('_'.join(lis_p))
    all_p = lis_p[:]
    all_p.append('click_time')
    data[newname] = data[all_p].groupby(by=lis_p).click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds
    data[newname] = data[newname].fillna(-1)
    print('merge...')
    return data, newname


# In[ ]:


df = pd.read_csv('../input/train.csv', iterator=True, chunksize=10000,nrows= 3500000, usecols=['click_time'])
tmp_train_df = pd.concat([chunk[chunk['click_time'].str.contains("2017-11-06")] for chunk in df])


# In[ ]:


train_df['index'] = train_df.index
tmp_train_df['index'] = tmp_train_df.index
train_df = train_df.merge(tmp_train_df, on=['index'], how='left')
train_df = train_df.drop( ['index'], axis=1)
train_df['click_time'] = pd.to_datetime(train_df['click_time'])
del df
del tmp_train_df
gc.collect()


# In[ ]:


train_df, tmp1 = group_by_click(['ip'], train_df)
train_df, tmp2 = group_by_click(['ip', 'app'],train_df)
train_df, tmp3 = group_by_click(['ip', 'channel'],train_df)
train_df, tmp4 = group_by_click(['ip', 'device'],train_df)
parameter_with_ip_next = [tmp1,tmp2,tmp3, tmp4]
dic.update(loop_iv(parameter_with_ip_next, train_df))


# In[ ]:


for i in parameter_with_ip_next:
    sum_num = dic[i]['IV'].sum()
    sum_iV[i] = sum_num
    print("The sum of IV parameter in {0} : {1}".format(i, sum_num))
#train_df = train_df.drop( parameter_with_ip_next, axis=1)


# In[ ]:


train_df, tmp1 = group_by_click(['ip', 'device', 'channel'],train_df)
train_df, tmp2 = group_by_click(['ip', 'device', 'app'],train_df)
train_df, tmp3 = group_by_click(['ip', 'channel', 'app'],train_df)
parameter_with_ip2_next = [tmp1,tmp2,tmp3]
dic.update(loop_iv(parameter_with_ip2_next, train_df))


# In[ ]:


for i in parameter_with_ip2_next:
    sum_num = dic[i]['IV'].sum()
    sum_iV[i] = sum_num
    print("The sum of IV parameter in {0} : {1}".format(i, sum_num))
#train_df = train_df.drop( parameter_with_ip2_next, axis=1)


# In[ ]:


v=list(sum_iV.values())
v_m = list(sum_iV.values())
v_n = list(sum_iV.keys())
v.sort()
kv = []
for i in range(len(v)):
    vv = v_m.index(v[i])
    kv.append(v_n[vv])
print('The Top 25 IV parameters-----------------------')
top_25 = {}
for i in range(1, 26):
    print('The {0} big IV paramter is {1}.....'.format(i, kv[-i]))
    top_25[kv[-i]] = sum_iV[kv[-i]]

plt.figure(figsize = (22,8))
m_colors=[]
k_num=0
for num in range(len(top_25)):
    if (num//len(color))>k_num:
        k_num+=1
    t = num - k_num*len(color)
    m_colors.append(color[t])
sns.barplot(list(top_25.keys()),list(top_25.values()), alpha=0.8, palette = m_colors)
plt.ylabel('IV value', fontsize = 12)
plt.xlabel('Parameters', fontsize = 12)
plt.title("TOP 20 parameter's IV", fontsize = 16)
plt.xticks(rotation='vertical')
plt.show()


# * **From the above result, we can know  the is_attributed parameter is closely related to time interval.**

# In[ ]:


tmp = pd.DataFrame()
sns.set(font_scale=1.2)
for i in top_25:
    tmp[i] = train_df[i]
del train_df
gc.collect()
colormap = plt.cm.viridis
plt.figure(figsize=(26,26))
plt.title(' The Absolute Correlation Coefficient of Features', y=1.05, size=15)
sns.heatmap(abs(tmp.astype(float).corr()),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True, )
plt.show()
plt.savefig('feature_Correlation_Coefficient.png')


# *  **From the above result, we can choose to avoid parameters that have similar information.**
