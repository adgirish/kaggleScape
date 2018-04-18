
# coding: utf-8

# # Introduction
# 
# Solution outlined in the notebook can be arrived at by:
# 
# 1. Training XGboost model with context
# 1. Generating class predictions from the trained model
# 1. Create bag-of-words from training set to lookup normalized text of words available in test
# 1. Create class-wise regex function for new words, after looking up from bag-of-words created above
# 1. Normalize text and generate output**
# 
# This solution is an improvement to existing kernels and I would like to thank BingQing Wei, Neerja Doshi and Alvira for sharing starter codes on XGboost model and class-wise processing functions. 
# 
# 
# ### I've worked on python 2.7 and the code doesn't work here (partially have to do with package dependency). You can leverage this to build your own notebook in 2.7
# ### I've also uploaded the final results (output/sub2.csv) if anyone is interested in comparing the results.

# In[ ]:


# Import necessary packages
import pandas as pd
import numpy as np
import re
from collections import defaultdict, Counter
from datetime import datetime
import string
import roman
import num2words
import inflect
p = inflect.engine()


# # Step 1: XGboost Model Creation
# 
# Refer to https://www.kaggle.com/alphasis/xgboost-with-context-label-data-acc-99-637 for training XGboost model

# # Step 2: Class Perdictions
# 
# I've uploaded data set with predictions from trained model that can be used for further processing.

# In[ ]:


get_ipython().system('ls ../input')


# In[ ]:


test_df = pd.read_csv("../input/classpredictions/test_rf.csv")
test_df.drop("Unnamed: 0", axis=1,inplace=True)
train = pd.read_csv("../input/text-normalization-challenge-english-language/en_train.csv")


# Create id column that is required for final submission

# In[ ]:


test_df['id'] = test_df[['sentence_id','token_id']].apply(lambda x: str(x[0])+"_"+str(x[1]),axis=1)


# In[ ]:


test_df.head()


# # Step 3: Create bag-of-words

# In[ ]:


def bagOfWords(cls):
    d = defaultdict(list)
    train_cls = train[train["class"]==cls]
    train_list = [(train_cls.iloc[i,3],train_cls.iloc[i,4]) for i in range(train_cls.shape[0])]
    for k,v in train_list:
        d[k].append(v)
    counter_dict = {}
    for key in d:
        c = Counter(d[key]).most_common(1)[0][0]
        counter_dict[key] = c
    return counter_dict


# Create a bag-of-words dictionary for each class 

# In[ ]:


plain_trained = bagOfWords("PLAIN")
punt_trained = bagOfWords("PUNCT")
time_trained = bagOfWords("TIME")
frac_trained = bagOfWords('FRACTION')
add_trained = bagOfWords('ADDRESS')
tel_trained = bagOfWords('TELEPHONE')
dec_trained = bagOfWords('DECIMAL')
mon_trained = bagOfWords('MONEY')
digit_trained = bagOfWords('DIGIT')
mes_trained = bagOfWords('MEASURE')
ord_trained = bagOfWords('ORDINAL')
elec_trained = bagOfWords('ELECTRONIC')
verb_trained = bagOfWords('VERBATIM')
card_trained = bagOfWords('CARDINAL')
let_trained = bagOfWords('LETTERS')
date_trained = bagOfWords('DATE')


# # Step 4: Create Class-wise regex Functions

# Generic function definition

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

def bag2word(key,bag_dict):
    try:
        return bag_dict[key]
    except:
        return key


# ## Generic Numeric

# In[ ]:


def num2word(key):
    bag_res = bag2word(key, digit_trained)
    if bag_res != key: return bag_res
    if re.match(r'^-?\d+$', key.replace(',','')):
        return digit2word(key)
    if is_float(key):
        return float2word(key)


# ## Ordinal

# In[ ]:


def ordinal2word(key):
    bag_res = bag2word(key, ord_trained)
    if bag_res != key: return bag_res
    num = re.sub(r'[^0-9]',"",key).strip()
    if len(num)!=0: return num2words(int(num),ordinal=True)
    try:
        num = roman.fromRoman(key)
        return "the "+ num2words(int(num),ordinal=True)
    except:
        return key


# ## Measures

# In[ ]:


#Comprehensive list of all measures
dict_m = {'"': 'inches', "'": 'feet', 'km/s': 'kilometers per second', 'AU': 'units', 'BAR': 'bars',
          'CM': 'centimeters', 'mm': 'millimeters', 'FT': 'feet', 'G': 'grams', 'GAL': 'gallons', 'GB': 'gigabytes',
          'GHZ': 'gigahertz', 'HA': 'hectares', 'HP': 'horsepower', 'HZ': 'hertz', 'KM':'kilometers',
          'km3': 'cubic kilometers','KA':'kilo amperes', 'KB': 'kilobytes', 'KG': 'kilograms', 'KHZ': 'kilohertz',
          'KM²': 'square kilometers', 'KT': 'knots', 'KV': 'kilo volts', 'M': 'meters','KM2': 'square kilometers',
          'Kw':'kilowatts', 'KWH': 'kilo watt hours', 'LB': 'pounds', 'LBS': 'pounds', 'MA': 'mega amperes',
          'MB': 'megabytes','KW': 'kilowatts', 'MPH': 'miles per hour', 'MS': 'milliseconds', 'MV': 'milli volts',
          'kJ':'kilojoules', 'km/h': 'kilometers per hour',  'V': 'volts', 'M2': 'square meters', 'M3': 'cubic meters',
          'MW': 'megawatts', 'M²': 'square meters', 'M³': 'cubic meters', 'OZ': 'ounces',  'MHZ': 'megahertz',
          'MI': 'miles','MB/S': 'megabytes per second', 'MG': 'milligrams', 'ML': 'milliliters', 'YD': 'yards',
          'au': 'units', 'bar': 'bars', 'cm': 'centimeters', 'ft': 'feet', 'g': 'grams', 'gal': 'gallons',
          'gb': 'gigabytes', 'ghz': 'gigahertz', 'ha': 'hectares', 'hp': 'horsepower', 'hz': 'hertz',
          'kWh': 'kilo watt hours', 'ka': 'kilo amperes', 'kb': 'kilobytes', 'kg': 'kilograms', 'khz': 'kilohertz',
          'km': 'kilometers', 'km2': 'square kilometers', 'km²': 'square kilometers', 'kt': 'knots',
          'kv': 'kilo volts','kw': 'kilowatts', 'lb': 'pounds', 'lbs': 'pounds', 'm': 'meters', 'm2': 'square meters',
          'm3': 'cubic meters', 'ma': 'mega amperes', 'mb': 'megabytes', 'mb/s': 'megabytes per second', 
          'mg': 'milligrams', 'mhz': 'megahertz', 'mi': 'miles', 'ml': 'milliliters', 'mph': 'miles per hour',
          'ms': 'milliseconds', 'mv': 'milli volts', 'mw': 'megawatts', 'm²': 'square meters','m³': 'cubic meters',
          'oz': 'ounces', 'v': 'volts', 'yd': 'yards', 'µg': 'micrograms', 'ΜG': 'micrograms',"sq mi":"square miles",
          'kg/m3': 'kilograms per cubic meter', "mg/kg":"milli grams per kilogram"}

def measure2word(key):
    bag_res = bag2word(key, mes_trained)
    if bag_res != key: return bag_res
    if "%" in key: unit = "percent"; val = key[:len(key)-1]
    elif "/" in key and key.split("/")[0].replace(".","").isdigit():
        try:
            unit = "per " + dict_m[key.split("/")[-1]]
        except KeyError:
            unit = "per " + key.split("/")[-1].lower()
        
        val = key.split("/")[0]
    else:
        v = key.split()
        if len(v)>2:
            try:
                unit = " ".join(v[1:-1])+" "+dict_m[v[-1]]
            except KeyError:
                unit = " ".join(v[1:-1])+" "+v[-1].lower()
        else:
            try:
                unit = dict_m[v[-1]]
            except KeyError:
                unit = v[-1].lower()
        val = v[0]
    if is_num(val):
        val = p.number_to_words(val,andword='').replace("-"," ").replace(',','')
        text = val + ' ' + unit
    else: text = key
    return text


# ## Electronic

# In[ ]:


def url2word(key):
    bag_res = bag2word(key, elec_trained)
    if bag_res != key: return bag_res
    key = key.replace('.',' dot ').replace('/',' slash ').replace('-',' dash ').replace(':',' colon ').replace('_',' underscore ').replace('#',' hashtag ')
    key = key.split()
    if "hashtag" in key: return "hash tag " + " ".join(key[1:]).lower()
    lis2 = ['dot','slash','dash','colon']
    for i in range(len(key)):
        if key[i] not in lis2:
            key[i]=" ".join(key[i])
    text = " ".join(key)
    return text.lower()


# ## Verbatim

# In[ ]:


def verb2word(key):
    bag_res = bag2word(key, verb_trained)
    if bag_res != key: return bag_res
    dict_verb = {"#":"number","&":"and","α":"alpha","Α":"alpha","β":"beta","Β":"beta","γ":"gamma","Γ":"gamma",
                 "δ":"delta","Δ":"delta","ε":"epsilon","Ε":"epsilon","Ζ":"zeta","ζ":"zeta","η":"eta","Η":"eta",
                 "θ":"theta","Θ":"theta","ι":"iota","Ι":"iota","κ":"kappa","Κ":"kappa","λ":"lambda","Λ":"lambda",
                 "Μ":"mu","μ":"mu","ν":"nu","Ν":"nu","Ξ":"xi","ξ":"xi","Ο":"omicron","ο":"omicron","π":"pi","Π":"pi",
                 "ρ":"rho","Ρ":"rho","σ":"sigma","Σ":"sigma","ς":"sigma","Φ":"phi","φ":"phi","τ":"tau","Τ":"tau",
                 "υ":"upsilon","Υ":"upsilon","Χ":"chi","χ":"chi","Ψ":"psi","ψ":"psi","ω":"omega","Ω":"omega",
                 "$":"dollar","€":"euro","~":"tilde","_":"underscore","ₐ":"sil","%":"percent","³":"cubed"}
    if key in dict_verb: return dict_verb[key]
    if len(key)==1 or not(key.isalpha()): return key
    return letter2word(key)


# ## Telephone

# In[ ]:


def telephone2word(key):
    bag_res = bag2word(key, tel_trained)
    if bag_res != key: return bag_res
    key = key.replace('-','.').replace(')','.')
    text = p.number_to_words(key,group =1, decimal = "sil",zero = 'o').replace(',','')
    return text.lower()


# ## Dates

# In[ ]:


dict_mon = {'jan': "January", "feb": "February", "mar ": "march", "apr": "april", "may": "may ","jun": "june", "jul": "july", "aug": "august","sep": "september",
            "oct": "october","nov": "november","dec": "december", "january":"January", "february":"February", "march":"march","april":"april", "may": "may", 
            "june":"june","july":"july", "august":"august", "september":"september", "october":"october", "november":"november", "december":"december",
           "bc":"b c", "bc.":"b c","bce":"b c e","ad":"a d"}
def date2word(key):
    bag_res = bag2word(key, date_trained)
    if bag_res != key: return bag_res
    key = key.strip(string.punctuation)
    if key.isdigit(): return digit2word(key)
        #if (int(key)>=2000 and int(key) < 2010) or int(key)<1000:
        #    text = digit2word(key)
        #else:
        #    if int(key)%100 < 10 and int(key)/100<20:
        #        text = digit2word(key[0:2]) + ' o ' + digit2word(key[2:])
        #    else:
        #        text = digit2word(key[0:2]) + ' ' + digit2word(key[2:])
        #return text
    
    if key.lower().replace("s","").isdigit():
        v = key.lower().replace("s","")
        if (int(v)>=2000 and int(v) < 2010) or int(v)<1000:
            text = digit2word(v)
        else:
            if int(v)%100 < 10 and int(v)/100<20:
                text = digit2word(v[0:2]) + ' o ' + digit2word(v[2:])
            else:
                text = digit2word(v[0:2]) + ' ' + digit2word(v[2:])
        return re.sub('ys$','ies',text+ 's')

    key = key.replace("/","-")    
    v =  key.split('-')
    if len(v)==3:
        if v[1].isdigit():
            try:
                date = datetime.strptime(key , '%Y-%m-%d')
                text = 'the '+ p.ordinal(p.number_to_words(int(v[2]))).replace('-',' ')+' of '+datetime.date(date).strftime('%B')
                if int(v[0])>=2000 and int(v[0]) < 2010:
                    text = text  + ' '+digit2word(v[0])
                else:
                    if int(v[0])%100 < 10 and int(v[0])/100<20:
                        text = text + ' ' + digit2word(v[0][0:2]) + ' o ' + digit2word(v[0][2:])
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
                        if int(v[2])%100 < 10 and int(v[2])/100<20:
                            text = text + ' ' + digit2word(v[2][0:2]) + ' o ' + digit2word(v[2][2:])
                        else:
                            text = text + ' ' + digit2word(v[2][0:2]) + ' ' + digit2word(v[2][2:])   
                elif len(v)==2:
                    if int(v[1])>=2000 and int(v[1]) < 2010:
                        text = dict_mon[v[0].lower()]  + ' '+ digit2word(v[1])
                    else: 
                        if len(v[1]) <=2:
                            text = dict_mon[v[0].lower()] + ' ' + digit2word(v[1])
                        else:
                            if int(v[1])%100 < 10 and int(v[1])/100<20:
                                text = dict_mon[v[0].lower()]+ ' ' +digit2word(v[1][0:2])+ ' o ' +digit2word(v[1][2:])
                            else:
                                text = dict_mon[v[0].lower()]+ ' ' +digit2word(v[1][0:2])+ ' ' +digit2word(v[1][2:])
                
                else: text = key
            except: text = key
            return text.lower()
        else:       
            key = re.sub(r'[^\w]', ' ', key)
            v = key.split()
            if len(v)==2 and v[0].isdigit():
                if v[1]=='s': return re.sub('ys$','ies',digit2word(v[0])+ 's')
                if v[1] not in dict_mon: return digit2word(v[0])+ ' ' + v[1].lower()
                return digit2word(v[0])+ ' ' + dict_mon[v[1].lower()]
            
            try:
                date = datetime.strptime(key , '%d %b %Y')
                val =1
                text = 'the '+ p.ordinal(p.number_to_words(int(v[0]))).replace('-',' ')+' of '+ dict_mon[v[1].lower()]
            except:
                try:
                    date = datetime.strptime(key , '%d %b %y')
                    val = 2
                    text = 'the '+ p.ordinal(p.number_to_words(int(v[0]))).replace('-',' ')+' of '+ dict_mon[v[1].lower()]
                    v[2] = datetime.date(date).strftime('%Y')
                except:
                    try:
                        date = datetime.strptime(key , '%d %B %Y')
                        val = 3
                        text = 'the '+ p.ordinal(p.number_to_words(int(v[0]))).replace('-',' ')+' of '+ dict_mon[v[1].lower()]
                    except:
                        try:
                            date = datetime.strptime(key , '%d %B %y')
                            val = 4
                            text = 'the '+ p.ordinal(p.number_to_words(int(v[0]))).replace('-',' ')+' of '+ dict_mon[v[1].lower()]
                            v[2] = datetime.date(date).strftime('%Y')
                        except:
                            try:
                                date = datetime.strptime(key , '%d %m %Y')
                                val = 5
                                text = 'the '+ p.ordinal(p.number_to_words(int(v[0]))).replace('-',' ')+' of '+datetime.date(date).strftime('%B')
                            except:
                                try:
                                    date = datetime.strptime(key , '%d %m %y')
                                    val = 6
                                    text = 'the '+ p.ordinal(p.number_to_words(int(v[0]))).replace('-',' ')+' of '+datetime.date(date).strftime('%B')
                                    v[2] = datetime.date(date).strftime('%Y')
                                except:
                                    text = key
                                    val = -1
            if val != -1:
                if (int(v[2])>=2000 and int(v[2]) < 2010) or int(v[2])<1000:
                    text = text  + ' '+digit2word(v[2])
                else: 
                    if int(v[2])%100 < 10 and int(v[2])/100<20:
                        text = text + ' ' + digit2word(v[2][0:2]) + ' o ' + digit2word(v[2][2:])
                    else:
                        text = text + ' ' + digit2word(v[2][0:2]) + ' ' + digit2word(v[2][2:])

            return text.lower()


# ## Money

# In[ ]:


def currency2word(key):
    bag_res = bag2word(key, mon_trained)
    if bag_res != key: return bag_res
    dict_mon = {"$":"dollars","€":"euros","£":"pounds","¥":"yen"}
    dict_unit = {"m":"million","b":"billion","k":"thousand","h":"hundered","usd":"united states dollars",
                 "aud":"australian dollars","gbp":"british pounds","usdm":"million united states dollars",
                 "gbpm":"million british pounds","audm":"million australian dollars",
                 "usdb":"billion united states dollars","usdk":"thousand united states dollars",
                 "gbpb":"billion british pounds","audb":"billion australian dollars",
                 "gbpk":"thousand british pounds","audk":"thousand australian dollars",
                 "nok":"norwegian kroner"}
    final = []
    num = p.number_to_words(re.sub(r'[^\d]','',key).strip(),andword='').replace("-"," ").replace(',','')
    final.append(num)
    v = re.sub(r'[!\"#$€££%&\'()*+,-./:;<=>?@\[\\\]\^_`{\|}~]',"",key).split()
    flag = [a.isalpha() for a in v]
    if any(flag):
        if len(v)==2:
            try:
                text = dict_unit[v[1].strip().lower()]
            except KeyError:
                text = v[1].strip().lower()
        else:
            text = " ".join([x.lower() for i,x in zip(flag,v) if i])
        final.append(text)
    elif len(v)==1:
        try:
            text = dict_unit[re.sub(r'[\d]','',v[0]).strip().lower()]
        except KeyError:
            text = re.sub(r'[\d]','',v[0]).strip().lower()
        final.append(text)
        
    try:
        unit = dict_mon[re.sub(r'[^$€£₦￥]','',key).strip()]
    except KeyError:
        unit = re.sub(r'[^$€£₦￥]','',key).strip()
    final.append(unit)
    
    return " ".join(final)


# ## Fraction

# In[ ]:


def fraction2word(x):
    bag_res = bag2word(x, frac_trained)
    if bag_res != x: return bag_res
    if x.find("½") != -1:
        x = x.replace("½","").strip()
        if len(x) != 0: return p.number_to_words(x,andword='').replace("-"," ").replace(',','')+" and a half"
        else: return "one half"
    elif x.find("¼") != -1:
        x = x.replace("¼","").strip()
        if len(x) != 0: return p.number_to_words(x,andword='').replace("-"," ").replace(',','')+" and a quarter"
        else: return "one quarter"
    elif x.find("⅓") != -1:
        x = x.replace("⅓","").strip()
        if len(x) != 0: return p.number_to_words(x,andword='').replace("-"," ").replace(',','')+" and a third"
        else: return "one third"
    elif x.find("⅔") != -1:
        x = x.replace("⅔","").strip()
        if len(x) != 0: return p.number_to_words(x,andword='').replace("-"," ").replace(',','')+" and two thirds"
        else: return "two third"
    elif x.find("⅞") != -1:
        x = x.replace("⅞","").strip()
        if len(x) != 0: return p.number_to_words(x,andword='').replace("-"," ").replace(',','')+" and seven eighths"
        else: return "seven eighth"
    elif x.find(" ") != -1:
        v = x.split(" ")
        res = " and ".join([fraction2word(val) for val in v])
        return res.replace("and one","and a")
    
    try:
        y = x.split('/')
        result_string = ''
        if len(y)==1 and y[0].isdigit(): return p.number_to_words(y[0],andword='').replace("-"," ").replace(',','')
        y[0] = p.number_to_words(y[0],andword='').replace("-"," ").replace(',','')
        y[1] = ordinal2word(y[1]).replace("-"," ").replace(" and "," ").replace(',','')
        if y[1] == "first":
            return y[0]+" over one"
        if y[1] == 'fourth':
            if y[0]=='one': result_string = y[0] + ' quarter'
            else: result_string = y[0] + ' quarters'
        elif y[1] == 'second':
            if y[0]=='one': result_string = y[0] + ' half'
            else: result_string = y[0] + ' halves'
        else:
            if y[0]=='one': result_string = y[0] + " "+ y[1]
            else: result_string = y[0] + ' ' + y[1] + 's'
        return(result_string)
    except:    
        return(x)


# ## Plain

# In[ ]:


def plain2word(key):
    bag_res = bag2word(key, plain_trained)
    if bag_res != key: return bag_res
    if key.find(".") != -1: return url2word(key)
    return key


# ## Address

# In[ ]:


def address(key):
    bag_res = bag2word(key, add_trained)
    if bag_res != key: return bag_res
    if len(key)==1: return key
    try:
        text = re.sub('[^a-zA-Z]+', '', key)
        num = re.sub('[^0-9]+', '', key)
        result_string = ''
        if len(text)>0: result_string = ' '.join(list(text.lower()))
        if num.isdigit():
            if int(num)<1000:
                result_string = result_string + " " + digit2word(num)
            else:
                result_string = result_string + " " + telephone2word(num)
        return(result_string.strip())        
    except:    
        return(key)


# ## Letters

# In[ ]:


def letter2word(key):
    bag_res = bag2word(key, let_trained)
    if bag_res != key: return bag_res
    if len(key)==1: return key
    key = re.sub(r'[!\"#$€££%&\()*+,-./:;<=>?@\[\\\]\^_`{\|}~]',"",key)
    if key.replace("'","").isalpha():
        result = ' '.join(list(key.strip(string.punctuation).lower()))
        return result.replace(" ' s","'s")
    return key


# ## Cardinal

# In[ ]:


def digit2word(key):
    bag_res = bag2word(key,card_trained)
    if bag_res != key: return bag_res
    #if key.isalpha(): return ordinal2word(key)
    try:
        text = p.number_to_words(key,decimal='point',andword='', zero='o')
        if re.match(r'^0\.',key): 
            text = 'zero '+text[2:]
        if re.match(r'.*\.0$',key): text = text[:-2]+' zero'
        text = text.replace('-',' ').replace(',','')
        return text.lower()
    except: return key


# ## Decimal

# In[ ]:


def float2word(key):
    bag_res = bag2word(key, dec_trained)
    if bag_res != key: return bag_res
    if key.find(" ") != (-1): return measure2word(key)
    try:
        key = float(key.replace(',',''))
    except ValueError:
        return measure2word(key)
    key = p.number_to_words(key,decimal='point',andword='', zero='o')
    if 'o' == key.split()[0]:
        key = key[2:]
    key = key.replace('-',' ').replace(',','')

    return key.lower()


# # Step 5: Normalize text and generate output

# In[ ]:


df_num = test_df[test_df['class']=="CARDINAL"]
out = []
for i in range(df_num.shape[0]):
    out.append(digit2word(df_num.iloc[i,2]))

df_num['after'] = out

df_date = test_df[test_df['class']=="DATE"]
out = []
for i in range(df_date.shape[0]):
    out.append(date2word(df_date.iloc[i,2]))

df_date['after'] = out

df_let = test_df[test_df['class']=="LETTERS"]
out = []
for i in range(df_let.shape[0]):
    out.append(letter2word(df_let.iloc[i,2]))

df_let['after'] = out

df_verb = test_df[test_df['class']=="VERBATIM"]
out = []
for i in range(df_verb.shape[0]):
    out.append(verb2word(df_verb.iloc[i,2]))

df_verb['after'] = out

df_elec = test_df[test_df['class']=="ELECTRONIC"]
out = []
for i in range(df_elec.shape[0]):
    out.append(url2word(df_elec.iloc[i,2]))

df_elec['after'] = out

df_ord = test_df[test_df['class']=="ORDINAL"]
out = []
for i in range(df_ord.shape[0]):
    out.append(ordinal2word(df_ord.iloc[i,2]))

df_ord['after'] = out

df_mes = test_df[test_df['class']=="MEASURE"]
out = []
for i in range(df_mes.shape[0]):
    out.append(measure2word(df_mes.iloc[i,2]))

df_mes['after'] = out

df_dig = test_df[test_df['class']=="DIGIT"]
out = []
for i in range(df_dig.shape[0]):
    out.append(digit2word(df_dig.iloc[i,2]))

df_dig['after'] = out

df_mon = test_df[test_df['class']=="MONEY"]
out = []
for i in range(df_mon.shape[0]):
    out.append(currency2word(df_mon.iloc[i,2]))

df_mon['after'] = out

df_dec = test_df[test_df['class']=="DECIMAL"]
out = []
for i in range(df_dec.shape[0]):
    out.append(float2word(df_dec.iloc[i,2]))

df_dec['after'] = out

df_tel = test_df[test_df['class']=="TELEPHONE"]
out = []
for i in range(df_tel.shape[0]):
    out.append(telephone2word(df_tel.iloc[i,2]))

df_tel['after'] = out

df_frac = test_df[test_df['class']=="FRACTION"]
out = []
for i in range(df_frac.shape[0]):
    out.append(fraction2word(df_frac.iloc[i,2]))

df_frac['after'] = out

df_add = test_df[test_df['class']=="ADDRESS"]
out = []
for i in range(df_add.shape[0]):
    out.append(address(df_add.iloc[i,2]))

df_add['after'] = out

df_plain = test_df[test_df['class'] == "PLAIN"]
df_plain['after'] = df_plain['before'].apply(lambda x: plain2word(x))

df_punct = test_df[test_df['class'] == "PUNCT"]
df_punct['after'] = df_punct['before'].apply(lambda x: bag2word(x,punt_trained))

df_time = test_df[test_df['class'] == "TIME"]
df_time['after'] = df_time['before'].apply(lambda x: bag2word(x,time_trained))


# In[ ]:


# Append output dataframes
result_df = df_plain.append(df_date).append(df_punct).append(df_time)
result_df = result_df.append(df_let).append(df_num).append(df_verb)
result_df = result_df.append(df_elec).append(df_ord).append(df_mes)
result_df = result_df.append(df_dig).append(df_mon).append(df_dec)
result_df = result_df.append(df_tel).append(df_frac).append(df_add)


# In[ ]:


result_df[['id','after']].to_csv("submission.csv",index=False)


# # Possible Improvements:
# 
# As can be seen above, the function are not capable of handling all possible inputs. If you're patient enough, you can look at updating Date/Measure/Money/Time.
