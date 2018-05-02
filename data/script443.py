
# coding: utf-8

# ### In this Notebook I  will Explore the UP elections 
# 
#  1. Thank Aaniket  for the dataset
#  2. The Data will be explored in the following ways:-
#     3. UP Analysis
#     3. District Analysis
#     3. Phase wise Analysis
#     3. Party Analysis
#     3.  Winner Analysis
#     3. Region wise Analysis
# 
#  
# 
# 
# **If u like it please up vote it **

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Import  input file
Data = pd.read_csv("../input/up_res.csv")


# In[ ]:


# Lets see the Data 
Data.head() # this will give us the top 5(by default ) rows of dataset


# In[ ]:


Data.info() # it will give us the information about the data columns


# **Insights**
# 
#  1. First of all there is no missing values
#  2. I do not understand what do mean by column seat allotment 
#  3. Other column are as described in Data over view
# 
# **Let's Start the exploring the UP : - A Indian state which is bigger than many countries in the world**
# 
# **UP Analysis**

# In[ ]:


# Now lets count total  number of votes
print("The numbers of votes voted in UP election result are:",Data["votes"].sum() )


# **Insights**
# 
#  1. The number of Votes voted in UP is 86 Million, If we look at the world's Population Data  [link](https://www.infoplease.com/world/population-statistics/worlds-50-most-populous-countries-2016)
# then the number of voters in UP are more than the population of Iran which is placed at the 16th in the list.
# 
# 2. FYI : In India the age eligibility for the vote is 18 years. 

# In[ ]:


# Now lets count the number of districts in UP
print(" The number of districts in UP is :" ,len(Data['district'].unique()))
# in this the unique will give only unique values in district
# len will count the number of unique values


# In[ ]:


# Now lets count the number of constituencies in UP 
# same as we did in the the district
print(" The number of constituencies in UP is :" ,len(Data['ac'].unique()))


# **2. District Analysis**

# In[ ]:


# there is 403 assembly consistuencies in UP within 75 districts
# Lets Count How much Constituency in a disrict
Data_districts= Data.groupby(["district"])["ac"].nunique()
# In this we grouped the data by districs wise and then we counted the uniques values of constitutncy
Data_districts=Data_districts.reset_index().sort_values(by="ac",ascending=False).reset_index(drop=True)
# Now we sorted values on the basis of count of assembly seats


# In[ ]:


# let see which districts are at the top
Data_districts.head()


# In[ ]:


# Allabhad is with the most number of Assembly seats 
# Now look at the bottom who are down in the list
Data_districts.tail(7)
# here tail will give the bottom of Table


# In[ ]:


# from this we can see that the there are district more than 12 Assembly seats and also with only 2 seats 
# not fair distibution
# lets see the mode of this 
Data_districts.ac.value_counts() # this is to count the number of values


#  1. Here the 50 % district have the 5,4 and 3 number of Assembly seats

# In[ ]:


# Let's Count the number of votes per district
Data_Districts_votes = Data.groupby("district")['votes'].sum().reset_index().sort_values(by="votes",ascending= False).reset_index(drop=True)
# group by district and sum the number of votes


# In[ ]:


Data_Districts_votes.head(10)


# In[ ]:


# Allahabad is at the top in both case so it may be due to a large number of assembly seats 
# lest calculate the avearage number of votes per seat District wise


# In[ ]:


Data_districts= pd.merge(Data_districts,Data_Districts_votes,on="district")
# in this we are merging two data set one contain the number of votes and one contain the number of Assembly seats


# In[ ]:


Data_districts.head()


# In[ ]:


Data_districts["Average Votes Per assembly"]= (Data_districts['votes']/Data_districts["ac"]).astype(int)
# Making a new coloumn which contains the average number of votes per assembely Seats


# In[ ]:


Data_districts.sort_values(by="Average Votes Per assembly",ascending=False).reset_index(drop=True)


#  1. Lalitpur is at the top 
# 
# 2. Allahabad moves to 61st position

# In[ ]:


# Lets Count the number of Candidate per assembly seat
Data_Candidate= Data.groupby("ac")['candidate'].count().reset_index().sort_values(by="candidate",ascending=False).reset_index(drop=True)


# In[ ]:


Data_Candidate.head(15)


# **3. Analysis of Phases**

# In[ ]:


# Let's explore the number of votes per phase 
Votes_Phase= Data.groupby('phase')['votes'].sum().reset_index().sort_values(by="votes",ascending=False).reset_index(drop=True)
Votes_Phase


# In[ ]:


sns.barplot(x='phase',y='votes',data=Votes_Phase) # to plot the votes in each phase
plt.title("No. Of Votes In each Phase")


#  1. No of votes are decreasing with the next phase it may be due to no of Assembly so lets count the number of assembly phase wise

# In[ ]:


Assembly_phase =  Data.groupby("phase")["ac"].count().reset_index().sort_values(by="ac",ascending=False).reset_index(drop=True)
sns.barplot(x='phase',y='ac',data=Assembly_phase)


#  1. oh ! The pattern is not same as the number of votes, means some phase have a better percentage of votes

# **4.Party Analysis**

# In[ ]:


# Let's see which parties are there 
Data.party.unique() # the parties which are participating in the elections


# In[ ]:


# Here none of the above means nothing we can convert them into also others 
Data.party.replace("None of the Above","others",inplace=True)


# In[ ]:


Data.party.unique()


# In[ ]:


# Vote Distribution of Parties
plt.figure(figsize=(10,8))
sns.pointplot(x='party',y='votes',data=Data)


# In[ ]:


plt.figure(figsize=(10,8))
sns.boxplot(x='party',y='votes',data=Data)


#  1. Here is some insights from above two graphs
#  2. First graph is known as Point plot [Pointplot](http://seaborn.pydata.org/generated/seaborn.pointplot.html). From this graph we can interpret the BJP is having the Highest mean number of votes (As expected) but surprise will be INC have more mean than BSP while BSP won more seat. It is due to INC and SP was in alliance so the INC had less number of seat.
# 
#  3. The Second graph is known as boxplot [Boxplot](http://seaborn.pydata.org/generated/seaborn.boxplot.html). By this graph same thing are here but here we can get the quratiles and outliers too . From this we observed that the independent and others having very less number of votes some of them are ouliers may be independent camdidate who won the election.
# 
#  4. There is one outlier for BJP that is nearby 250 thousand votes

# In[ ]:


# Let see the patteren of votes get by parties
Votes_party=Data.groupby("party")['votes'].sum().reset_index().sort_values(by='votes',ascending=False).reset_index(drop=True)
Votes_party


# In[ ]:


# lest plot the barplot of it 
sns.barplot(x='party',y='votes',data=Votes_party)
plt.title("No oF votes got by parties")
plt.xticks(rotation=90)


#  1. As known BJP is leading in the table but surprise to see that the BSP is ahead of SP 
#  2.  BJP got nearby double of the vote got by BSP
#  3. This shows how much  they dominated in  Election of UP
#  4. The INC is in very bad condition

# In[ ]:


# Let's see the number of votes get by parties phase wise
No_of_phase= len(Data.phase.unique())
fig=plt.subplots(figsize=(8,10*(No_of_phase+1)))
for i in range(No_of_phase):
    index_values= Data[Data["phase"]==i+1].index.values
    phase_votes= Data.ix[index_values,:] # getting all the value by phase wise
    votes_party_phase= phase_votes.groupby('party')['votes'].sum().reset_index().sort_values(by='votes',ascending=False).reset_index(drop=True)
    plt.subplot(No_of_phase+1,1,i+1) # No of Phase +1 is for total no of plots 
    sns.barplot(x='party',y='votes',data=votes_party_phase)
    plt.subplots_adjust(hspace=.3)
    plt.xticks(rotation=90)
    plt.title("Phase {}".format(i+1))


#  1. The pattern is same for parties other than SP and BSP in all phases.
#  2. BJP is leading in all phase
#  3. But for SP and BSP in 2nd and 3rd phase the SP is dominating BSP but BSP is dominating in all other phases
#  4. In 1st Phase RLD perform better than the INC and acquired  4th position it may be due to the home districts (like as meerut, muzzafurnagar, shamli etc) of RLD are in 1st phase.

# **5. Winner Analysis**

# In[ ]:


#Lets find the number of Assembly seats won by parties
# This thing I am doing in my way other suggestion will be helpful
# Please comment if you know any other way for it 
Winner= Data.groupby(["ac"])['votes'].max().reset_index()
# This will give us the maximum number of votes for every assembly seat
Winner2=pd.merge(Winner,Data,on=['ac','votes'],how="left",copy=False)

# Now I am merging the this data with the our original data to get all values

winner_party=Winner2.groupby(['party'])['candidate'].count().reset_index() # Now counting the seats won by a party
print(winner_party)
sns.barplot(x='party',y='candidate',data=winner_party)


#  1. Graph is telling the whole story about dominance of the BJP in the UP election

# In[ ]:


# We can see these for phase wise
Winner2.groupby(['phase','party'])['candidate'].count()


# In[ ]:


# Let's do it for who are at the last Postion 
Last_position= Data.groupby(["ac"])['votes'].min().reset_index().sort_values(by='votes').reset_index(drop=True)
Last_position2=pd.merge(Last_position,Data,on="votes",how="left",copy=False)
Last_position2=Last_position2.drop_duplicates('ac_x').reset_index(drop=True) # drop any duplicate if it is there


# In[ ]:


Last_position2[["ac_x",'candidate','party','votes']]


#  1. This list shows the candidate getting votes less than 50 too.
#  2. There is no main party in the list 

# In[ ]:


# Now lets Find who are at the second positions
Second_place=Data.groupby("ac")['votes'].nlargest(2).reset_index() # nlargest(2) will give us the two largest value for each category
Second_place1 = Second_place.groupby('ac')['votes'].min().reset_index().sort_values(by='votes',ascending=False).reset_index(drop=True) # from this we will get the miinimum of those two
#print(second_place1) you can do it for your confirmation
# Now we will merge it with our oringinal data so to get all the fields here

Second_place2=pd.merge(Second_place1,Data,on=['ac','votes'],how="left",copy=False)
#print(Second_place2) you can do it for your confirmation
Second_party=Second_place2.groupby(['party'])['candidate'].count().reset_index() # Now counting the seats won by a party
print(Second_party)
sns.barplot(x='party',y='candidate',data=Second_party)
 


#  1. BSP and SP are the Party who finished second most time so from this we conclude that the main parties of this election were BJP,SP and BSP 

# In[ ]:


# now I want to see the difference b/w candidate who won the election and the one who finished second
winner= Winner2[['ac','votes']] # here we got the data of winners
second_place= Second_place2[['ac','votes']] # here we got the data of second_place
Winner_comparison= pd.merge(winner,second_place,on='ac')
# Now get the difference b/w the these two position
Winner_comparison["Difference"]=Winner_comparison['votes_x']-Winner_comparison['votes_y']
Winner_comparison.sort_values(by="Difference",ascending=False).reset_index(drop=True)


#  1. From this list we can see that the in some constituencies the difference is more than 100 thousands 
# while in some it is only 100 or 200

# In[ ]:


#lets plot a graph for more information
x=Winner_comparison["Difference"]
sns.distplot(x)


# In[ ]:


# reduce the xlimit to clear view

plt.figure(figsize=(12,10))
plt.xlim(0,100000)
sns.distplot(x)


#  1. From here we get maximum time  the difference is nearby 20,000

# **6. Region wise Analysis** 
# 
#  1. In this we will divide the UP in four regions which are Harit Pardesh ( Western UP), Purvanchal (Eastern UP), Bundelkhand ( Central UP), Avadh Pardesh (Central UP) 
#  2.  These divisions were suggested by the Mayawati the BSP president to divide UP in four states when she was chief minister of UP . fortunately it did not work out

# In[ ]:


# Let's divide UP In four regions
# these list's name repersent the name the region and element repersent distric in them 
# I know it all because I am from neighbouring state of UP
Harit_Pardesh=['Saharanpur',
'Shamli',
'Muzaffarnagar',
'Bijnor',
'Moradabad',
'Sambhal',
'Rampur',
'Amroha',
'Meerut',
'Baghpat',
'Ghaziabad',
'Hapur',
'Gautam Buddha Nagar',
'Bulandshahr',
'Aligarh',
'Hathras',
'Mathura',
'Agra',
'Firozabad',
'Kasganj',
'Etah',
'Mainpuri',
'Budaun',
'Bareilly',
'Pilibhit',
'Shahjahanpur'
]

Avadh_Pardesh=['Lakhimpur Kheri',
'Sitapur',
'Hardoi',
'Unnao',
'Lucknow',
'Raebareli',
'Farrukhabad',
'Kannauj',
'Etawah',
'Auraiya',
'Kanpur Dehat',
'Kanpur Nagar',
'Barabanki'
]

BundelKhand = ['Jalaun',
'Jhansi',
'Lalitpur',
'Hamirpur',
'Mahoba',
'Banda',
'Chitrakoot'
]

Purvanchal= ['Amethi',
'Sultanpur',
'Fatehpur',
'Pratapgarh',
'Kaushambi',
'Allahabad',
'Faizabad',
'Ambedkar Nagar',
'Bahraich',
'Shravasti',
'Balarampur',
'Gonda',
'Siddharthnagar',
'Basti',
'Sant Kabir Nagar',
'Maharajganj',
'Gorakhpur',
'Kushinagar',
'Deoria',
'Azamgarh',
'Mau',
'Ballia',
'Jaunpur',
'Ghazipur',
'Chandauli',
'Varanasi',
'Sant Ravidas Nagar',
'Mirzapur',
'Sonbhadra'
]
print("No of District in Harit Pardesh:",len(Harit_Pardesh))
print("No of District in Purvanchal:",len(Purvanchal))
print("No of District in Avadh Pardesh:",len(Avadh_Pardesh))
print("No of District in BundelKhand:",len(BundelKhand))


# In[ ]:


mapper={} # now taking a empty dictonary
for i in Harit_Pardesh: # Now iterating through list and adding districts as key and assigning them value Region
    mapper[i]="Harit Pardesh"
for i in Purvanchal: # Same as above
    mapper[i]="Purvanchal"
for i in Avadh_Pardesh:
    mapper[i]="Avadh Pardesh"
for i in BundelKhand:
    mapper[i]="BundelKhand"
    


# In[ ]:


Data['Region']=Data["district"].map(mapper)  # Now mapping districts to region using mapper dictonary


# In[ ]:


# Just rechecking the mapping is it correct or not so again counting the number of districts per region
District_Region=Data.groupby("Region")["district"].nunique().reset_index()
District_Region


# In[ ]:


# Let's Now see vote Per Region
Region_Votes = Data.groupby("Region")["votes"].sum().reset_index().sort_values(by=['votes']).reset_index(drop=True)
Region_Votes


# In[ ]:


# Lets plot a pie plot of it
plt.figure(figsize=(8,8))
plt.pie(Region_Votes["votes"],labels=Region_Votes["Region"] ,autopct='%1.1f%%',shadow=True,explode=[0.10,0.10,0.10,0.10])


#  1. Here we can see that the Purvanchal and Harit Pardesh is having 75 % of votes that are voted in UP. 
#  2. It may be due to they have more number of districts. so we will find their % of votes per districs 

# In[ ]:


Votes_Region_per_district = pd.merge(Region_Votes,District_Region,on="Region")
Votes_Region_per_district["Votes_Per_District"]=(Votes_Region_per_district["votes"]/Votes_Region_per_district["district"])*100
Votes_Region_per_district.Votes_Per_District=Votes_Region_per_district.Votes_Per_District.astype(int)
Votes_Region_per_district.sort_values(by="Votes_Per_District",ascending=False).reset_index(drop=True)


#  1. oh ! Avadh is at the top but Bundelkhand is still at low it is because BundelKhand Includes district which have only two or three ac like as lalitpur

# In[ ]:


# Now let's see the number of seat won by party resion wise
Winner= Data.groupby(["ac"])['votes'].max().reset_index()
Winner2=pd.merge(Winner,Data,on=['ac','votes'],how="left",copy=False)
Winner_Region = Winner2.groupby(['Region','party'])['candidate'].count()
Winner_Region


#  1. BJP is leading in all Region but in BundelKhand BJP  did clean sweep it may be due to BundelKhand is near to state Madhya Pardesh and BJP has been ruling MP since last 15 years
#  2. The Independent and others won only in Purvanchal 
#  3.  BSP Won 14 seats in Purvanchal which is a large number compare to their seat number in other Regions. 
#  4. INC is winning only  2 or 3 seats other than BundelKhand (FYI: INC is India's Most older Party which belong to Nehru and Gandhi )
#  5.  SP who was the Ruling party before election is  better than other party. but in MODI STROM they are flown away too.
#  6. RLD won only one seat and that is also in Harit_ Pardesh their own hometown

#  1. Thank you for reading this 
#  2. vote it if you liked it 
# 
# 
# ----------
# 
