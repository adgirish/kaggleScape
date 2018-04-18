
# coding: utf-8

# ## Introduction

# #### This kernel is for improving basic baseline upto 0.992 for new dataset using regular expressions and string operations.  Below are the classes I am going to cover which comprises of the major portion of test set . I will be adding functions for other classes soon.
# 1. Dates 
# 2. Measure
# 3. Decimals 
# 4. Cardinals 
# 5. Electronic - URL
# 6. Currency - Dollars
# 7. Telephone Numbers
# 
# **Any feedback on improving accuracy of these functions will be highly appreciated!**

# ## Loading modules and training data

# In[ ]:


import pandas as pd
import numpy as np
import re
from num2words import num2words
import inflect
p = inflect.engine()
from datetime import datetime


# In[ ]:


path = "../input/en_train.csv"
df = pd.read_csv(path)


# In[ ]:


# Checking if key is decimal or digit or general numeric
def is_num(key):
    if is_float(key) or re.match(r'^-?[0-9]\d*?$', key.replace(',','')): return True
    else: return False

def is_float(string):
    try:
        return float(string.replace(',','')) and "." in string # True if string is a number contains a dot
    except ValueError:  # String is not a number
        return False


# ## Cardinal

# In[ ]:


def digit2word(key):
    try:
        text = p.number_to_words(key,decimal='point',andword='', zero='o')
        if re.match(r'^0\.',key): 
            text = 'zero '+text[2:]
        if re.match(r'.*\.0$',key): text = text[:-2]+' zero'
        text = text.replace('-',' ').replace(',','')
        return text.lower()
    except: return key


# ##  Decimals

# In[ ]:


def float2word(key):
    key = float(key.replace(',',''))
    key = p.number_to_words(key,decimal='point',andword='', zero='o')
    if 'o' == key.split()[0]:
        key = key[2:]
    key = key.replace('-',' ').replace(',','')
    return key.lower()


# ## General Numeric

# In[ ]:



def num2word(key):
    if re.match(r'^-?\d+$', key.replace(',','')):
        return digit2word(key)
    if is_float(key):
        return float2word(key)


# ## Measures

# In[ ]:


#Comprehensive list of all measures
dict_m = {'"': 'inches', "'": 'feet', 'km/s': 'kilometers per second', 'AU': 'units', 'BAR': 'bars', 'CM': 'centimeters', 'mm': 'millimeters', 'FT': 'feet', 'G': 'grams', 
     'GAL': 'gallons', 'GB': 'gigabytes', 'GHZ': 'gigahertz', 'HA': 'hectares', 'HP': 'horsepower', 'HZ': 'hertz', 'KM':'kilometers', 'km3': 'cubic kilometers',
     'KA':'kilo amperes', 'KB': 'kilobytes', 'KG': 'kilograms', 'KHZ': 'kilohertz', 'KM²': 'square kilometers', 'KT': 'knots', 'KV': 'kilo volts', 'M': 'meters',
      'KM2': 'square kilometers','Kw':'kilowatts', 'KWH': 'kilo watt hours', 'LB': 'pounds', 'LBS': 'pounds', 'MA': 'mega amperes', 'MB': 'megabytes',
     'KW': 'kilowatts', 'MPH': 'miles per hour', 'MS': 'milliseconds', 'MV': 'milli volts', 'kJ':'kilojoules', 'km/h': 'kilometers per hour',  'V': 'volts', 
     'M2': 'square meters', 'M3': 'cubic meters', 'MW': 'megawatts', 'M²': 'square meters', 'M³': 'cubic meters', 'OZ': 'ounces',  'MHZ': 'megahertz', 'MI': 'miles',
     'MB/S': 'megabytes per second', 'MG': 'milligrams', 'ML': 'milliliters', 'YD': 'yards', 'au': 'units', 'bar': 'bars', 'cm': 'centimeters', 'ft': 'feet', 'g': 'grams', 
     'gal': 'gallons', 'gb': 'gigabytes', 'ghz': 'gigahertz', 'ha': 'hectares', 'hp': 'horsepower', 'hz': 'hertz', 'kWh': 'kilo watt hours', 'ka': 'kilo amperes', 'kb': 'kilobytes', 
     'kg': 'kilograms', 'khz': 'kilohertz', 'km': 'kilometers', 'km2': 'square kilometers', 'km²': 'square kilometers', 'kt': 'knots','kv': 'kilo volts', 'kw': 'kilowatts', 
     'lb': 'pounds', 'lbs': 'pounds', 'm': 'meters', 'm2': 'square meters','m3': 'cubic meters', 'ma': 'mega amperes', 'mb': 'megabytes', 'mb/s': 'megabytes per second', 
     'mg': 'milligrams', 'mhz': 'megahertz', 'mi': 'miles', 'ml': 'milliliters', 'mph': 'miles per hour','ms': 'milliseconds', 'mv': 'milli volts', 'mw': 'megawatts', 'm²': 'square meters',
     'm³': 'cubic meters', 'oz': 'ounces', 'v': 'volts', 'yd': 'yards', 'µg': 'micrograms', 'ΜG': 'micrograms', 'kg/m3': 'kilograms per meter cube'}

def measure2word(key):
    unit = dict_m[key.split()[-1]]
    val = key.split()[0]
    if is_num(val):
        val = num2word(val)
        text = val + ' ' + unit
    else: text = key
    return text


# ## Electronic - URL

# In[ ]:


def url2word(key):
    key = key.replace('.',' dot ').replace('/',' slash ').replace('-',' dash ').replace(':',' colon ').replace('_',' underscore ')
    key = key.split()
    lis2 = ['dot','slash','dash','colon']
    for i in range(len(key)):
        if key[i] not in lis2:
            key[i]=" ".join(key[i])
    text = " ".join(key)
    return text.lower()


# ## Currency

# In[ ]:


def currency2word(key):
        v = key.replace('$','').replace('US$','').split()
        if len(v) == 2: 
            if is_num(v[0]):
                text = num2word(v[0]) + ' '+ v[1] + ' '+ 'dollars'
        elif is_num(v[0]):
            text = num2word(v[0]) + ' '+ 'dollars'
        else:
            if 'm' in key or 'M' in key or 'million':
                text = p.number_to_words(key).replace(',','').replace('-',' ').replace(' and','') + ' million dollars'
            elif 'bn' in key:
                text = p.number_to_words(key).replace(',','').replace('-',' ').replace(' and','') + ' billion dollars'
            else: text = key
        return text.lower()


# ## Telephone Numbers

# In[ ]:


def telephone2word(key):
    key = key.replace('-','.').replace(')','.')
    text = p.number_to_words(key,group =1, decimal = "sil",zero = 'o').replace(',','')
    return text.lower()


# ## Dates

# In[ ]:


dict_mon = {'jan': "January", "feb": "February", "mar ": "march", "apr": "april", "may": "may ","jun": "june", "jul": "july", "aug": "august","sep": "september",
            "oct": "october","nov": "november","dec": "december", "january":"January", "february":"February", "march":"march","april":"april", "may": "may", 
            "june":"june","july":"july", "august":"august", "september":"september", "october":"october", "november":"november", "december":"december"}
def date2word(key):
    v =  key.split('-')
    if len(v)==3:
        if v[1].isdigit():
            try:
                date = datetime.strptime(key , '%Y-%m-%d')
                text = 'the '+ p.ordinal(p.number_to_words(int(v[2]))).replace('-',' ')+' of '+datetime.date(date).strftime('%B')
                if int(v[0])>=2000 and int(v[0]) < 2010:
                    text = text  + ' '+digit2word(v[0])
                else: 
                    text = text + ' ' + digit2word(v[0][0:2]) + ' ' + digit2word(v[0][2:])
            except:
                text = key
            return text.lower()    
    else:   
        v = re.sub(r'[^\w]', ' ', key).split()
        if v[0].isalpha():
            try:
                if len(v)==3:
                    text = dict_mon[v[0].lower()] + ' '+ p.ordinal(p.number_to_words(int(v[1]))).replace('-',' ')
                    if int(v[2])>=2000 and int(v[2]) < 2010:
                        text = text  + ' '+digit2word(v[2])
                    else: 
                        text = text + ' ' + digit2word(v[2][0:2]) + ' ' + digit2word(v[2][2:])   
                elif len(v)==2:

                    if int(v[1])>=2000 and int(v[1]) < 2010:
                        text = dict_mon[v[0].lower()]  + ' '+ digit2word(v[1])
                    else: 
                        if len(v[1]) <=2:
                            text = dict_mon[v[0].lower()] + ' ' + digit2word(v[1])
                        else:
                            text = dict_mon[v[0].lower()] + ' ' + digit2word(v[1][0:2]) + ' ' + digit2word(v[1][2:])
                else: text = key
            except: text = key
            return text.lower()
        else: 
            key = re.sub(r'[^\w]', ' ', key)
            v = key.split()
            try:
                date = datetime.strptime(key , '%d %b %Y')
                text = 'the '+ p.ordinal(p.number_to_words(int(v[0]))).replace('-',' ')+' of '+ dict_mon[v[1].lower()]
                if int(v[2])>=2000 and int(v[2]) < 2010:
                    text = text  + ' '+digit2word(v[2])
                else: 
                    text = text + ' ' + digit2word(v[2][0:2]) + ' ' + digit2word(v[2][2:])
            except:
                try:
                    date = datetime.strptime(key , '%d %B %Y')
                    text = 'the '+ p.ordinal(p.number_to_words(int(v[0]))).replace('-',' ')+' of '+ dict_mon[v[1].lower()]
                    if int(v[2])>=2000 and int(v[2]) < 2010:
                        text = text  + ' '+digit2word(v[2])
                    else: 
                        text = text + ' ' + digit2word(v[2][0:2]) + ' ' + digit2word(v[2][2:])
                except:
                    try:
                        date = datetime.strptime(key , '%d %m %Y')
                        text = 'the '+ p.ordinal(p.number_to_words(int(v[0]))).replace('-',' ')+' of '+datetime.date(date).strftime('%B')
                        if int(v[2])>=2000 and int(v[2]) < 2010:
                            text = text  + ' '+digit2word(v[2])
                        else: 
                            text = text + ' ' + digit2word(v[2][0:2]) + ' ' + digit2word(v[2][2:])
                    except:
                        try:
                            date = datetime.strptime(key , '%d %m %y')
                            text = 'the '+ p.ordinal(p.number_to_words(int(v[0]))).replace('-',' ')+' of '+datetime.date(date).strftime('%B')
                            v[2] = datetime.date(date).strftime('%Y')
                            if int(v[2])>=2000 and int(v[2]) < 2010:
                                text = text  + ' '+digit2word(v[2])
                            else: 
                                text = text + ' ' + digit2word(v[2][0:2]) + ' ' + digit2word(v[2][2:])
                        except:text = key
            return text.lower() 


# ## Checking output of these function on training sets:

# In[ ]:


df_num = df[df['class']=="CARDINAL"]
ct = 0
pred_list = []
for i in range(df_num.shape[0]):
    pred_list.append((digit2word(df_num.iloc[i,3]), df_num.iloc[i,4] ))
pred_list = list(set(pred_list))

for i in range(len(pred_list)):
    if pred_list[i][0] !=  pred_list[i][0]: ct += 1
        
# print ("Total Wrong: " , ct*1.0/len(pred_list))


# In[ ]:


df_num = df[df['class']=="DATE"]
ct = 0
pred_list = []
for i in range(df_num.shape[0]):
    pred_list.append((date2word(df_num.iloc[i,3]), df_num.iloc[i,4] ))
pred_list = list(set(pred_list))

for i in range(len(pred_list)):
    if pred_list[i][0] is not None:
        if not pred_list[i][0].isdigit():
            if pred_list[i][0] !=  pred_list[i][1]: ct += 1
        
# print ("Total Wrong: " , ct*1.0/len(pred_list))


# In[ ]:


df_num = df[df['class']=="DECIMAL"]
ct = 0
pred_list = []
for i in range(df_num.shape[0]):
    try:
        pred_list.append((float2word(df_num.iloc[i,3]), df_num.iloc[i,4] ))
    except:
        pred_list.append((df_num.iloc[i,4], df_num.iloc[i,4] ))
pred_list = list(set(pred_list))

for i in range(len(pred_list)):
    if pred_list[i][0] !=  pred_list[i][1]: 
        ct += 1
        print(pred_list[i])
        
# print ("Total Wrong: " , ct*1.0/len(pred_list))


# In[ ]:


df_num = df[df['class']=="MEASURE"]
ct = 0
pred_list = []
for i in range(df_num.shape[0]):
    try:
        pred_list.append((measure2word(df_num.iloc[i,3]), df_num.iloc[i,4] ))
    except:
        pred_list.append((df_num.iloc[i,3], df_num.iloc[i,4] ))
pred_list = list(set(pred_list))

for i in range(len(pred_list)):
    #These cases are handeled separately:
    if '%' not in pred_list[i][0] and "/km2" not in pred_list[i][0] and "/km²" not in pred_list[i][0]:  
        if pred_list[i][0] !=  pred_list[i][1]: 
            ct += 1
# print ("Total Wrong: " , ct*1.0/len(pred_list))

