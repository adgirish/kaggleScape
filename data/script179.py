
# coding: utf-8

# The methods used to generate the filter terms involved looking at the most frequent words/client names for clues about what types of establishments were mentioned in this data set. 
# 
# Developing the filters took a fair amount of "human" sleuthing while looking at TF-IDF scores, frequency counts of Client names, as well as just general knowledge of common Spanish words used to refer to certain establishment types. Especially with all the noise provided by the Clients referred to only by a proper name, the filtered data is a proportionally small figure- but, significant.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# Load in the Client Name data
# Make sure all names uppercase (there are some mixed instances)
pd.set_option('display.max_rows', 50)
vf = pd.read_csv('../input/cliente_tabla.csv',header=0)
vf['NombreCliente'] = vf['NombreCliente'].str.upper()


# In[ ]:


# Begin with a scan of the Client Name Data based on Top Frequency Client Names
# Notice there are a lot of Proper Names
vf['NombreCliente'].value_counts()[0:200]


# In[ ]:


# Let's also generate a list of individual word frequency across all names
def tfidf_score_list(vf2, list_len):
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer()

    vf2['New'] = 'na'
    a = " ".join(vf2['NombreCliente'])
    vf2['New'][0] = a

    tfidf = v.fit_transform(vf2['New'])

    feature_names = v.get_feature_names()

    freq = []
    doc = 0
    feature_index = tfidf[doc,:].nonzero()[1]
    tfidf_scores = zip(feature_index, [tfidf[doc, x] for x in feature_index])
    for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
            freq.append((w.encode('utf-8'),s))
    
    del vf2['New']
    
    import numpy as np
    names = ['word','score']
    formats = ['S50','f8']
    dtype = dict(names = names, formats=formats)
    array = np.array(freq, dtype=dtype)

    b = np.sort(array, order='score')
    
    if list_len > len(b)+1:
        list_len = len(b)+1
    for i in range(1,list_len):
        print(b[-i])


# In[ ]:


tfidf_score_list(vf, 200)


# Again, notice the prevalence of personal names. By looking at a long enough list, however, we can start to see some other useful terms appear such as particles of speech (i.e. 'ag', 'los', 'san') or more useful words like "super", "oxxo", "mini", "comodin". If I found a word that I thought represent a type of establishment, I'd double check it by doing a single search. If that search was fruitful and had a good amount of relevant results, I'd add it to my filter. An example of a single term search is below:

# In[ ]:


print(vf[vf['NombreCliente'].str.contains('.*CAFE.*')])


# The result is a filter derived from hand-picking the best, most common, most interesting terms I could determine.

# In[ ]:


# --- Begin Filtering for specific terms

# Note that the order of filtering is significant.
# For example: 
# The regex of .*ERIA.* will assign "FRUITERIA" to 'Eatery' rather than 'Fresh Market'.
# In other words, the first filters to occur have a bigger priority.

def filter_specific(vf2):
    
    # Known Large Company / Special Group Types
    vf2['NombreCliente'] = vf2['NombreCliente'].str.replace('.*REMISION.*','Consignment')
    vf2['NombreCliente'] = vf2['NombreCliente'].replace(['.*WAL MART.*','.*SAMS CLUB.*'],'Walmart', regex=True)
    vf2['NombreCliente'] = vf2['NombreCliente'].str.replace('.*OXXO.*','Oxxo Store')
    vf2['NombreCliente'] = vf2['NombreCliente'].str.replace('.*CONASUPO.*','Govt Store')
    vf2['NombreCliente'] = vf2['NombreCliente'].str.replace('.*BIMBO.*','Bimbo Store')

    

    # General term search for a random assortment of words I picked from looking at
    # their frequency of appearance in the data and common spanish words for these categories
    vf2['NombreCliente'] = vf2['NombreCliente'].replace(['.*COLEG.*','.*UNIV.*','.*ESCU.*','.*INSTI.*',                                                        '.*PREPAR.*'],'School', regex=True)
    vf2['NombreCliente'] = vf2['NombreCliente'].str.replace('.*PUESTO.*','Post')
    vf2['NombreCliente'] = vf2['NombreCliente'].replace(['.*FARMA.*','.*HOSPITAL.*','.*CLINI.*'],'Hospital/Pharmacy', regex=True)
    vf2['NombreCliente'] = vf2['NombreCliente'].replace(['.*CAFE.*','.*CREMERIA.*','.*DULCERIA.*',                                                        '.*REST.*','.*BURGER.*','.*TACO.*', '.*TORTA.*',                                                        '.*TAQUER.*','.*HOT DOG.*',                                                        '.*COMEDOR.*', '.*ERIA.*','.*BURGU.*'],'Eatery', regex=True)
    vf2['NombreCliente'] = vf2['NombreCliente'].str.replace('.*SUPER.*','Supermarket')
    vf2['NombreCliente'] = vf2['NombreCliente'].replace(['.*COMERCIAL.*','.*BODEGA.*','.*DEPOSITO.*',                                                            '.*ABARROTES.*','.*MERCADO.*','.*CAMBIO.*',                                                        '.*MARKET.*','.*MART .*','.*MINI .*',                                                        '.*PLAZA.*','.*MISC.*','.*ELEVEN.*','.*EXP.*',                                                         '.*SNACK.*', '.*PAPELERIA.*', '.*CARNICERIA.*',                                                         '.*LOCAL.*','.*COMODIN.*','.*PROVIDENCIA.*'
                                                        ],'General Market/Mart'\
                                                       , regex=True)

    vf2['NombreCliente'] = vf2['NombreCliente'].replace(['.*VERDU.*','.*FRUT.*'],'Fresh Market', regex=True)
    vf2['NombreCliente'] = vf2['NombreCliente'].replace(['.*HOTEL.*','.*MOTEL.*'],'Hotel', regex=True)


# In[ ]:


filter_specific(vf)


# In[ ]:


# --- Begin filtering for more general terms
# The idea here is to look for names with particles of speech that would
# not appear in a person's name.
# i.e. "Individuals" should not contain any participles or numbers in their names.
def filter_participle(vf2):
    vf2['NombreCliente'] = vf2['NombreCliente'].replace([
            '.*LA .*','.*EL .*','.*DE .*','.*LOS .*','.*DEL .*','.*Y .*', '.*SAN .*', '.*SANTA .*',\
            '.*AG .*','.*LAS .*','.*MI .*','.*MA .*', '.*II.*', '.*[0-9]+.*'\
    ],'Small Franchise', regex=True)


# In[ ]:


filter_participle(vf)


# In[ ]:


# Any remaining entries should be "Individual" Named Clients, there are some outliers.
# More specific filters could be used in order to reduce the percentage of outliers in this final set.
def filter_remaining(vf2):
    def function_word(data):
        # Avoid the single-words created so far by checking for upper-case
        if (data.isupper()) and (data != "NO IDENTIFICADO"): 
            return 'Individual'
        else:
            return data
    vf2['NombreCliente'] = vf2['NombreCliente'].map(function_word)


# In[ ]:


filter_remaining(vf)


# With the filtering complete, let's look at the breakdown of how the data is classified now:

# In[ ]:


vf['NombreCliente'].value_counts()


# Finally, we can apply these new tags on the actual Training and Test data sets that have been provided!

# In[ ]:


#trdf = pd.read_csv('../input/train.csv',header=0)
#trdf_stores = trdf.merge(vf.drop_duplicates(subset=['Cliente_ID']), how="left")


# In[ ]:


#tsdf = pd.read_csv('../input/test.csv',header=0)
#tsdf_stores = tsdf.merge(vf.drop_duplicates(subset=['Cliente_ID']), how="left")


# Write the data to file to save it for a new session....

# In[ ]:


#trdf.to_csv('../output/train_with_cnames.csv')
#tsdf.to_csv('../output/test_with_cnames.csv')

