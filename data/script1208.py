
# coding: utf-8

# To visualize used the data that have the exact date and place of the attack. Next, all the data are splited by hemispheres and months. The graphics clearly show that the attacks in the northern hemisphere have a peak in the summer (July), and in the southern the peak in winter (January).
# 
# Для визуализации использованы те данные, что имеют точную дату и место нападения. Далее все данные разбиваются по полушариям и по месяцам. По графикам четко видно, что нападения в северном полушарии имеют пик летом (июль), а в южном этот пик приходится на зиму (январь).

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm


# In[ ]:


data = pd.read_csv("../input/attacks.csv", encoding = "ISO-8859-1")


# In[ ]:


# N - 0, S - 1
countrys = {'CROATIA': 0, 'NORWAY': 0, 'FRANCE': 0, 'MARTINIQUE': 0, 'ICELAND': 0, 
            'JAVA': 1, 'Sierra Leone': 0, 'CYPRUS': 0, 'LIBERIA': 0, 'NEW BRITAIN': 1, 
            'URUGUAY': 1, 'NORTH ATLANTIC OCEAN ': 0, 'ADMIRALTY ISLANDS': 1, 
            'PAPUA NEW GUINEA': 1, 'DJIBOUTI': 0, 'TAIWAN': 1, 'EL SALVADOR': 0, 
            'ST. MAARTIN': 0, 'ASIA?': 0, 'NAMIBIA': 1, 'OCEAN': 1, 'CAPE VERDE': 0, 
            'MID ATLANTIC OCEAN': 0, 'MAURITIUS': 1, 'ANTIGUA': 0, 'FRENCH POLYNESIA': 1, 
            'JOHNSTON ISLAND': 0, 'SUDAN': 0, 'SOUTH KOREA': 0, 'TUVALU': 1, 
            'SOUTH ATLANTIC OCEAN': 1, 'UNITED ARAB EMIRATES (UAE)': 0, 'DOMINICAN REPUBLIC': 0, 
            ' PHILIPPINES': 0, 'MALAYSIA': 0, 'BRITISH VIRGIN ISLANDS': 0, 'CHINA': 0, 
            'ATLANTIC OCEAN': 0, 'ITALY': 0, 'VENEZUELA': 0, 'SOLOMON ISLANDS / VANUATU': 1, 
            'SOUTH CHINA SEA': 0, 'Between PORTUGAL & INDIA': 2, 'DIEGO GARCIA': 1, 
            'MEDITERRANEAN SEA?': 0, 'INDIAN OCEAN?': 1, 'INDIA': 0, 'SOUTH AFRICA': 1, 
            'St Helena': 1, 'WESTERN SAMOA': 1, 'TASMAN SEA': 1, 'HONG KONG': 0, 'TONGA': 1, 
            'YEMEN': 0, 'COLUMBIA': 0, 'NORTHERN MARIANA ISLANDS': 0, 'GUAM': 0, 'GUINEA': 0, 
            'CENTRAL PACIFIC': 2, 'GUATEMALA': 0, 'FIJI': 1, 'GULF OF ADEN': 0, 'JAPAN': 0, 
            'MID-PACIFC OCEAN': 0, 'ST. MARTIN': 1, 'USA': 0, 'CRETE': 0, 'BRAZIL': 1, 
            'TURKS & CAICOS': 0, 'SOUTHWEST PACIFIC OCEAN': 1, 'GREENLAND': 0, 
            'BAY OF BENGAL': 1, 'PACIFIC OCEAN': 0, 'LEBANON': 0, 'MALTA': 0, 'NIGERIA': 0, 
            'GREECE': 0, 'MEXICO': 0, 'BERMUDA': 0, 'UNITED KINGDOM': 0, 'SINGAPORE': 0, 
            'BRITISH ISLES': 0, 'TURKEY': 0, 'NEVIS': 1, 'AUSTRALIA': 1, 'ENGLAND': 0, 
            'SIERRA LEONE': 0, 'VANUATU': 1, 'NORTH SEA': 0, 'RUSSIA': 0, 'MICRONESIA': 0, 
            'PORTUGAL': 0, 'RED SEA': 0, 'MONTENEGRO': 0, 'IRAQ': 0, 'SWEDEN': 0, 
            'PERSIAN GULF': 0, 'NORTH ATLANTIC OCEAN': 0, 'Fiji': 1, 'SLOVENIA': 0, 
            'PHILIPPINES': 0, 'IRAN / IRAQ': 0, 'TUNISIA': 0, 'SAN DOMINGO': 1, 'AZORES': 0, 
            'GEORGIA': 0, 'BURMA': 0, 'NEW GUINEA': 1, 'SUDAN?': 0, 'NETHERLANDS ANTILLES': 0, 
            'ALGERIA': 0, 'NICARAGUA': 0, 'SEYCHELLES': 1, 'RED SEA?': 0, 'BRITISH NEW GUINEA': 1, 
            'THAILAND': 0, 'PALESTINIAN TERRITORIES': 0, 'FALKLAND ISLANDS': 1, 'IRELAND': 0, 
            'MONACO': 0, 'PARAGUAY': 1, 'SYRIA': 0, 'EGYPT ': 0, 'MADAGASCAR': 1, 
            'NORTH PACIFIC OCEAN': 0, 'EGYPT / ISRAEL': 0, 'COOK ISLANDS': 1, 
            'TRINIDAD & TOBAGO': 0, 'PACIFIC OCEAN ': 0, 'EQUATORIAL GUINEA / CAMEROON': 0, 
            'ISRAEL': 0, 'SAMOA': 1, 'ECUADOR': 1, 'CARIBBEAN SEA': 0, 'NEW CALEDONIA': 1, 
            'MARSHALL ISLANDS': 0, 'PANAMA': 0, 'UNITED ARAB EMIRATES': 0, 'ITALY / CROATIA': 0, 
            'NEW ZEALAND': 1, 'MALDIVE ISLANDS': 0, 'GHANA': 0, 'MOZAMBIQUE': 0, 'SRI LANKA': 0, 
            'SOLOMON ISLANDS': 1, 'Coast of AFRICA': 1, 'BARBADOS': 0, 'BANGLADESH': 0, 
            'CHILE': 1, 'CANADA': 0, 'HONDURAS': 0, 'PALAU': 0, 'AMERICAN SAMOA': 1, 
            'SAUDI ARABIA': 0, ' TONGA': 1, 'SPAIN': 0, 'ARGENTINA': 1, 'CURACAO': 0, 
            'ANDAMAN / NICOBAR ISLANDAS': 0, 'KENYA': 1, 'EGYPT': 0, 'THE BALKANS': 0, 
            'PUERTO RICO': 0, 'KIRIBATI': 0, 'OKINAWA': 0, 'REUNION': 1, 
            'BRITISH WEST INDIES': 0, 'NICARAGUA ': 0, 'FEDERATED STATES OF MICRONESIA': 0, 
            'IRAN': 0, 'CAYMAN ISLANDS': 0, 'SOMALIA': 0, 'INDONESIA': 1, 'KUWAIT': 0, 
            'Seychelles': 1, 'COSTA RICA': 0, 'INDIAN OCEAN': 1, 'CEYLON (SRI LANKA)': 0, 
            'YEMEN ': 0, 'HAITI': 0, 'SCOTLAND': 0, 'CUBA': 0, 'GUYANA': 0, 'LIBYA': 0, 
            'MEXICO ': 0, 'SENEGAL': 0, 'GRAND CAYMAN': 0, 'GABON': 1, 'GRENADA': 0, 
            'RED SEA / INDIAN OCEAN': 0, 'VIETNAM': 0, 'BAHAMAS': 0, 'BAHREIN': 0, 
            'NORTHERN ARABIAN SEA': 0, 'BELIZE': 0, 'MEDITERRANEAN SEA': 0, 'ANGOLA': 1, 
            'SOUTH PACIFIC OCEAN': 1, 'TANZANIA': 1, 'KOREA': 0, 'JAMAICA': 0, 'ARUBA': 0, 
            'MAYOTTE':1}
            


# In[ ]:


start = 1940
by_year = np.zeros((2017-start, 2))
by_year_fatal = np.zeros((2017-start, 4))
year_mon = np.zeros((12, 4))
activity_south = np.zeros((12, 7))
activity_north = np.zeros((12, 7))
type_south = np.zeros((12, 6))
type_north = np.zeros((12, 6))

months_numb = {'Jan': 0, 'Feb': 1, 'Mar': 2, 
               'Apr': 3, 'Ap-': 3, 'May': 4, 
               'Jun': 5, 'Jul': 6, 'Aug': 7, 
               'Sep': 8, 'Oct': 9, 'Nov': 10, 
               'Dec': 11}

type_atc = {'Invalid': 0, 'Unprovoked': 1, 'Boat': 2, 
            'Provoked': 3, 'Sea Disaster': 4, 'Boating': 5}

for i in data.values:
    date = i[1].replace(' ', '')
    date = date.replace('July', 'Jul')
    date = date.replace('Sept', 'Sep')
    date = date.replace('--', '-')
    date = date.replace('y2', 'y-2')
    date = date.replace('v2', 'v-2')
    type = i[3]
    activity = i[7]
    fatal = i[12]
    day, month, year = 0, 0, 'not'
    if len(date) >= 11 and len(date) <= 12 and date[2] == '-':
        year = date[-4:]
        if len(date) == 12:
            year = date[-5:-1]
        day = int(date[:2])
        month = date[3:6]
    elif len(date) == 10 and date[1] == '-':
        year = date[-4:]
        day = int(date[:1])
        month = date[2:5]
    elif len(date) == 19 and date[10] == '-':
        year = date[-4:]
        day = int(date[8:10])
        month = date[11:14]
    
    t = -1
    if fatal == 'Y':    
        t = 1
    elif fatal == 'N' or fatal == ' N' or fatal == 'N ' or fatal == 'n ':
        t = 0
    
    act = 6
    if isinstance(activity, str):
        if 'Surfing' in activity or 'surfing' in activity:
            act = 0
        elif 'Swimming' in activity or 'swimming' in activity:
            act = 1
        elif 'Fishing' in activity or 'fishing' in activity:
            act = 2
        elif 'Bathing' in activity or 'bathing' in activity:
            act = 3
        elif 'Wading' in activity or 'wading' in activity:
            act = 4
        elif 'Diving' in activity or 'diving' in activity:
            act = 5
    tp = type_atc[type]
        
    if countrys.get(i[4]) != 2 and countrys.get(i[4]) != None and months_numb.get(month) != None and t != -1:
        year_mon[months_numb[month], 2*countrys[i[4]] + t] += 1
        
        if countrys[i[4]] == 0:
            activity_north[months_numb[month], act] += 1
            type_north[months_numb[month], tp] += 1
        else:
            activity_south[months_numb[month], act] += 1
            type_south[months_numb[month], tp] += 1
            
        if year.isnumeric() and int(year) >= start:
            by_year[int(year) - start, countrys[i[4]]] += 1
            by_year_fatal[int(year) - start, 2*countrys[i[4]] + t] += 1


# In[ ]:


months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
x = np.arange(year_mon[:,0].shape[0])
plt.figure(figsize=(15, 8))
plt.plot([], [], 'b', label='Northern Hemisphere', linewidth=10)
plt.plot([], [], 'r', label='Southern Hemisphere', linewidth=10)
plt.stackplot(x, year_mon[:,:2].sum(axis=1), year_mon[:,-2:].sum(axis=1), 
              colors=['b', 'r'])
plt.title('Attacks by hemispheres')
plt.legend(loc='best')
plt.ylabel('Attacks')
plt.axis([0, 11, 0, 600])
plt.xticks(x, months)
plt.show()


# In[ ]:


x = np.arange(year_mon[:,0].shape[0])
plt.figure(figsize=(15, 8))
plt.plot([], [], 'b', label='Not Fatal', linewidth=10)
plt.plot([], [], 'r', label='Fatal', linewidth=10)
plt.stackplot(x, year_mon[:,1], year_mon[:,0], colors=['r', 'b'])
plt.title('Attacks by Northern Hemisphere')
plt.legend(loc='best')
plt.ylabel('Attacks')
plt.axis([0, 11, 0, 500])
plt.xticks(x, months)
plt.show()

plt.figure(figsize=(15, 8))
plt.plot([], [], 'b', label='Not Fatal', linewidth=10)
plt.plot([], [], 'r', label='Fatal', linewidth=10)
plt.stackplot(x, year_mon[:,3], year_mon[:,2], colors=['r', 'b'])
plt.title('Attacks by Southern Hemisphere')
plt.legend(loc='best')
plt.ylabel('Attacks')
plt.axis([0, 11, 0, 400])
plt.xticks(x, months)
plt.show()


# In[ ]:


dim = activity_north.shape[1]
w = 0.8
dimw = w / dim
x = np.arange(activity_north.shape[0])
c = ['red', 'blue', 'green', 'yellow', 'magenta', 'cyan', 'lightcoral']
l = ['Surfing', 'Swimming', 'Fishing', 'Bathing', 'Wading', 'Diving', 'Other']
plt.figure(figsize=(15, 8))
for i in range(activity_north.shape[1]):
    plt.bar(x + i * dimw, activity_north[:,i], dimw, bottom=0.001, color=c[i], label=l[i])
plt.title('Attacks by Northern Hemisphere')
plt.legend(loc='best')
plt.ylabel('Attacks')
plt.xticks(x+0.15, months)
plt.show()


# In[ ]:


dim = activity_north.shape[1]
w = 0.8
dimw = w / dim
x = np.arange(activity_north.shape[0])
c = ['red', 'blue', 'green', 'yellow', 'magenta', 'cyan', 'lightcoral']
l = ['Surfing', 'Swimming', 'Fishing', 'Bathing', 'Wading', 'Diving', 'Other']
plt.figure(figsize=(15, 8))
for i in range(activity_south.shape[1]):
    plt.bar(x + i * dimw, activity_south[:,i], dimw, bottom=0.001, color=c[i], label=l[i])
plt.title('Attacks by Southern Hemisphere')
plt.legend(loc='best')
plt.ylabel('Attacks')
plt.xticks(x+0.15, months)
plt.show()


# In[ ]:


dim = activity_north.shape[1]
w = 0.8
dimw = w / dim
x = np.arange(activity_north.shape[0])
c = ['red', 'blue', 'green', 'yellow', 'magenta', 'cyan']
l = ['Invalid', 'Unprovoked', 'Boat', 'Provoked', 'Sea Disaster', 'Boating']
plt.figure(figsize=(15, 8))
for i in range(type_north.shape[1]):
    plt.bar(x + i * dimw, type_north[:,i], dimw, bottom=0.001, color=c[i], label=l[i])
plt.title('Attacks by Northern Hemisphere')
plt.legend(loc='best')
plt.ylabel('Attacks')
plt.xticks(x+0.15, months)
plt.show()


# In[ ]:


dim = activity_north.shape[1]
w = 0.8
dimw = w / dim
x = np.arange(activity_north.shape[0])
c = ['red', 'blue', 'green', 'yellow', 'magenta', 'cyan']
l = ['Invalid', 'Unprovoked', 'Boat', 'Provoked', 'Sea Disaster', 'Boating']
plt.figure(figsize=(15, 8))
for i in range(type_south.shape[1]):
    plt.bar(x + i * dimw, type_south[:,i], dimw, bottom=0.001, color=c[i], label=l[i])
plt.title('Attacks by Southern Hemisphere')
plt.legend(loc='best')
plt.ylabel('Attacks')
plt.xticks(x+0.15, months)
plt.show()


# In[ ]:


plt.figure(figsize=(15, 8))
x = np.arange(by_year.shape[0])
w = 1
h = by_year[:,0]/by_year[:,0].max()
cl = cm.OrRd(h)
plt.bar(x, by_year[:,0], w, color=cl)
plt.title('Attacks by Northern Hemisphere')
plt.ylabel('Attacks')
plt.xticks(x+0.5, np.arange(start, 2017), rotation='vertical')
plt.axis([0, 2017-start, 0, by_year[:,0].max()+5])
plt.show()

plt.figure(figsize=(15, 8))
nf = by_year_fatal[:,0]/by_year_fatal[:,0].max()
f = by_year_fatal[:,1]/by_year_fatal[:,1].max()
cl1 = cm.OrRd(nf)
cl2 = cm.cool(f)
plt.bar(x, by_year_fatal[:,1], w, color=cl2, label='Fatal')
plt.bar(x, by_year_fatal[:,0], w, bottom=by_year_fatal[:,1], color=cl1, label='Not Fatal')
plt.title('Attacks by Northern Hemisphere')
plt.ylabel('Attacks')
plt.xticks(x+0.5, np.arange(start, 2017), rotation='vertical')
plt.axis([0, 2017-start, 0, by_year[:,0].max()+5])
plt.legend(loc='best')
plt.show()


# In[ ]:


plt.figure(figsize=(15, 8))
x = np.arange(by_year.shape[0])
w = 1
cl = cm.OrRd(by_year[:,1]/by_year[:,1].max())
plt.bar(x, by_year[:,1], w, color=cl)
plt.title('Attacks by Southern Hemisphere')
plt.ylabel('Attacks')
plt.xticks(x+0.5, np.arange(start, 2017), rotation='vertical')
plt.axis([0, 2017-start, 0, by_year[:,1].max()+5])
plt.show()

plt.figure(figsize=(15, 8))
nf = by_year_fatal[:,2]/by_year_fatal[:,2].max()
f = by_year_fatal[:,3]/by_year_fatal[:,3].max()
cl1 = cm.OrRd(nf)
cl2 = cm.cool(f)
plt.bar(x, by_year_fatal[:,3], w, color=cl2, label='Fatal')
plt.bar(x, by_year_fatal[:,2], w, bottom=by_year_fatal[:,3], color=cl1, label='Not Fatal')
plt.title('Attacks by Southern Hemisphere')
plt.ylabel('Attacks')
plt.xticks(x+0.5, np.arange(start, 2017), rotation='vertical')
plt.axis([0, 2017-start, 0, by_year[:,1].max()+5])
plt.legend(loc='best')
plt.show()


# In[ ]:


year = by_year.sum(axis=1)
plt.figure(figsize=(15, 8))
x = np.arange(year.shape[0])
w = 1
cl = cm.OrRd(year/year.max())
plt.bar(x, year, w, color=cl)
plt.title('Attacks by both Hemispheres')
plt.ylabel('Attacks')
plt.xticks(x+0.5, np.arange(start, 2017), rotation='vertical')
plt.axis([0, 2017-start, 0, year.max()+5])
plt.show()

plt.figure(figsize=(15, 8))
nf = by_year_fatal[:,0] + by_year_fatal[:,2]
f = by_year_fatal[:,1] + by_year_fatal[:,3]
nf1 = nf / nf.max()
f1 = f /f .max()
cl1 = cm.OrRd(nf1)
cl2 = cm.cool(f1)
plt.bar(x, f, w, color=cl2, label='Fatal')
plt.bar(x, nf, w, bottom=f, color=cl1, label='Not Fatal')
plt.title('Attacks by Southern Hemisphere')
plt.ylabel('Attacks')
plt.xticks(x+0.5, np.arange(start, 2017), rotation='vertical')
plt.axis([0, 2017-start, 0, (f+nf).max()+5])
plt.legend(loc='best')
plt.show()

