
# coding: utf-8

# **Scroll to the bottom of the page if you just want to see the rankings**

# This notebook uses the Trueskill rating system to compute the world kitefoil rankings based on the major 2016 races. Trueskill was developed by Microsoft Research for ranking XBox players. It works similarly to the well-known Elo rating system used to rank chess players. The principle behind Trueskill is easy to understand: if racer A beats racer B and in a different regatta, racer B beats racer C, Trueskill deduces that A is probably better than C. 
# 
# For those familiar with Elo, Trueskill has two main that make it more suitable for kitefoil rankings:
# 1. it's more flexible and can handles races (not just 1 on 1 matches)
# 2. it accounts for uncertainty. If a racer has not raced very much, then each race will have a bigger impact on ranking than a racer that has raced a lot for example.
# 
# *Regatta's included*
#    - February 2016: NZ National Championships in Lake Taupo
#    - March 2016: Mexican Leg of 2016 Hydrofoil Pro Tour in La Ventana
#    - May 2016: France Leg of the IFKO Silver Cup in Montpellier 
#    - May 2016: European Championship in Cagliari 
#    - May 2016: Florida Kite League Spring Regatta in St Petersburg
#    - May 2016: Rippin the Rio in Sherman Island California
#    - July 2016: Sail Brisbane in Brisbane 
#    - July 2016: Italian Leg of the IKA Gold Cup in Gizzeria
#    - August 2016: US Leg of 2016 Hydrofoil Pro Tour in San Francisco
#    - August 2016: Mauritius Leg of the 2016 Hydrofoil Pro Tour in Mauritius 
#    - August 2016: Pringles Foil World Cup in Fehnmarn  
#    - September 2016: Danish National Championship
#    - September 2016: IKA World Championship in Weifang 
#    - October 2016: Vineyard Cup in Martha's Vineyard
#    - November 2016: Florida Kite League Fall Regatta in St Petersburg
#    - November 2016: Qatar Leg of the IKA Gold Cup 
#    - November 2016: Australian National Championships in Rockingham
#    - November 2016: Australian Leg of the 2016 Hydrofoil Pro Tour in Rockingham
#    - December 2016: Sail Melbourne in Melbourne
#    - January 2017: Lord of the Wind in La Ventana
#    - February 2017: NZ National Championships
#    - February 2017: WA State Titles  
#    - March 2017:  Hydrofoil Pro Tour in La Ventana, Mexico 
#    - April 2017: New South Wales State Championships in George's River 
#    - April 2017: Sailing World Cup in Hyeres, France 
#    - May 2017: Elvstrom Zellerbach Regatta in San Francisco
#    - May 2017: Florida Kite League Spring Regatta in St Petersburg
#    - June 2017: Hydrofoil Pro Tour in Montpellier
#    - June 2017: Sailing World Cup in Santander
#    - August 2017: Hydrofoil Pro Tour in San Francisco
#    - August 2017: Hydrofoil Pro Tour in Germany
#    - September 2017: Hydrofoil Pro Tour in Mauritius
#    - October 2017: Kitefoil Gold Cup in Sardinia
# 
# *To add*
#    - 2016 and 2017 SF Club Races
#    - 2016 Perth Kite Reviews
#    - 2016 Zellebach Regatta
#    - 2017 SF Club Races
#    
# *Missing Results*
#    - Turkish National Championship
#    - French National Championship
# 
# *Assumptions*
# - Dropped races don't count towards ranking
# - DNF, DSQ, RET, RCT count as last place
# - DNC, DNS, OCS, UFD don't count towards ranking even if they're not dropped races
# - No scoring penalty is applied for an SCP
# - I'm only ranking those who have a result for at least three races
# - Silver Fleet rankings are added to the back of the Gold Fleet (so 1st in the Silver Fleet counts as 25th in the regatta if the Gold Fleet had 24 racers). The Bronze Fleet is added to the back of the Silver Fleet and Gold Fleet. 
#  - Removes anybody who hasn't competed in at least 2 regattas, 5 races
#  - Removes anybody who hasn't raced in the last year
# 
# *Technical Assumption*
#  - I'm using Mu - 3 x sigma rather than Mu, which has penalizes those who Trueskill is less sure about (either because they have competed in fewer races or have inconsistent results). Microsoft use this by default. It makes the ratings a little conservative. 
#  - I doubled the default value of tau, to make the impact of new races much higher. The reason is that races can be few and far between and people can improve a lot between races. 
# 

# In[ ]:


import pandas as pd 
import numpy as np
import trueskill as ts
from IPython.display import display
import datetime


resultDir = '../input/RaceResults/'

pd.set_option('display.max_rows', 400)
pd.set_option('display.max_columns', 300)


# *Clean names and handle things like DNC, DNF etc*
# 
# This section loads the libraries and cleans up names. For example Rikki Leccese sometimes races as Riccardo Andrea Leccese. Please use consistent names when you register for races in future. 
# 
# This section also has the logic that handles the assumptions around how DNF, DNC, DNS etc are handled. 

# In[ ]:


def cleanResults(raceColumns,dfResultsTemp,appendScore, nameAllCaps):
    for raceCol in raceColumns:

        #Clean up Names
        
        if (nameAllCaps):
            dfResultsTemp.index = dfResultsTemp.index.str.lower()
        
        dfResultsTemp.index = dfResultsTemp.index.str.replace(r"(\w)([A-Z])", r"\1 \2")
        
        dfResultsTemp.index = dfResultsTemp.index.str.title()
        dfResultsTemp.index = dfResultsTemp.index.str.replace('\([A-Z\ 0-9]*\)','')
        dfResultsTemp.index = dfResultsTemp.index.str.strip()
        
        
        dfResultsTemp.index = dfResultsTemp.index.str.replace('Riccardo Andrea Leccese','Rikki Leccese')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('Riccardo Leccese','Rikki Leccese')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('Nicolas Parlier','Nico Parlier')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('Alejandro Climent Hernã¥_ Ndez', 'Alejandro Climent Hernandez')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('Alexandre Caizergues','Alex Caizergues')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('Florian Trit.*','Florian Trittel Paul')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('Jean Guillaume Rivaud','Jean-Guillaume Rivaud')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('^Kieran Le$','Kieran Le Borgne')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('Marvin Baumeister.*','Marvin Baumeister Schoenian')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('Mavin Baumeister Schoenian','Marvin Baumeister Schoenian')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('Theo De Ramecourt','Theo De-Ramecourt')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('James Johnson','James Johnsen')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('^Enrico$','Enrico Tonon')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('Roman Liubimtsev','Roman Lyubimtsev')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('Tomek Glazik','Tomasz Glazik')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('Andrew Hansen','Andy Hansen')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('Andrew Mc Manus','Andrew McManus')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('Matthew Taggart','Matt Taggart')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('Joey Pasqauli','Joey Pasquali')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('Stefanus Viljoen','Stefaans Viljoen')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('^Alejandro Climent$','Alejandro Climent Hernandez')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('?','')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('Jan Blaesiide','Jan Blaesino')  	
        dfResultsTemp.index = dfResultsTemp.index.str.replace('Zachary Marks','Zack Marks')  	
        dfResultsTemp.index = dfResultsTemp.index.str.replace('Matthew Reinhardt','Matt Reinhardt')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('Georgina Hewson','Gina Hewson')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('Normand Mc Guire','Normand Mcguire')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('Henrik Baerentzen','Henrik Baerentsen')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('Jade Oconnor','Jade O\'Connor')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('James Mc Grath','James Mcgrath')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('.*Vieujot.*','Mateo Vieujot')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('Steph Bridge','Stephanie Bridge')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('Oliver Bridge','Olly Bridge')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('Natalie Flintrop-Clarke','Natalie Clarke')              
        dfResultsTemp.index = dfResultsTemp.index.str.replace('Timothy Mossholder','Tim Mossholder')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('Pete Mc Kewen','Pete Mckewen')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('Charlie Morano','Charles Morano')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('Katia Rose','Katja Roose')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('Johnvon Tesmar','John Von Tesmar')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('William Morris','Will Morris')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('Yang Fung','Yang Fung')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('Nicholas Leason','Nick Leason')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('Phillip Rowlands','Philip Rowlands')
        dfResultsTemp.index = dfResultsTemp.index.str.replace('Sylvain Hocieni','Sylvain Hoceini')  
    
    
        
             
        #Handle DNF, DNS etc
        dfResultsTemp[raceCol] = dfResultsTemp[raceCol].astype(str)
        #dfResultsTemp[raceCol] = dfResultsTemp[raceCol].str.replace('','')
        droppedRaces = ['\(.*\)','\(.*','-.*','\[.*\]','\*']
        countAsMissing = ['D\+D','DCT','DNS','DNC','D\+0','OCS','\/','UFD','DNE','^[0-9\.]*C']
        countAsNoPenalty = ['SCP-*','RDG-*','RDG','^[0-9\.]*G']
        
        removeCharacter = ['\\n','nan']
        

        missingStr =  ("%s%s%s%s%s%s%s") % ('|'. join(droppedRaces),'|.*','.*|.*'.join(countAsMissing),'|','|' . join(countAsNoPenalty),'|','|' . join(removeCharacter)) 

        dfResultsTemp[raceCol] = dfResultsTemp[raceCol].str.replace(missingStr,'')              
        
        countASLast = ['RET','DNF','DSQ','RCT']
        lastStr = ("%s%s%s") % ('.*','.*|.*'.join(countASLast),'.*')
        dfResultsTemp[raceCol] = dfResultsTemp[raceCol].str.replace(lastStr,str(len(dfResultsTemp)+1))
        
        #remove any remaining white space
        dfResultsTemp[raceCol] = dfResultsTemp[raceCol].str.strip()
        
        #convert result to int or float
        
        #for debugging the regex expressions above
        #print(dfResultsTemp[raceCol])
        
        dfResultsTemp[raceCol] = pd.to_numeric(dfResultsTemp[raceCol])  
        #append score adjusts positions for silver fleet and bronze fleets
        dfResultsTemp[raceCol] = dfResultsTemp[raceCol] + appendScore
        
    return dfResultsTemp


# In[ ]:


#This function adds results to the Results dataframe, which stores all races in one dataframe
def mergeResults(raceColumns,raceName,dfResultsTemp,dfResults):
    for raceCol in raceColumns:
        raceIndex = ("%s%s%s") % (raceName,'-',raceCol)     
        dfResultsTemp[raceIndex] = dfResultsTemp[raceCol]
        del(dfResultsTemp[raceCol])
        dfResults = pd.merge(dfResults,dfResultsTemp[[raceIndex]],left_index=True,right_index=True,how='outer')
        
        
    lastRegatta = pd.to_datetime([raceName[0:8]])[0]
    dfResults.loc[dfResults.index.isin(dfResultsTemp.index),'lastRegatta'] = lastRegatta
    dfResults['numRegattas'] = dfResults['numRegattas'].fillna(0)
    dfResults.loc[dfResults.index.isin(dfResultsTemp.index),'numRegattas'] += 1

    return dfResults


# In[ ]:


#call this if the regatta doesn't explicitly allow for droped races
def dropRaces(numDrops,dfResultsTemp,raceColumns):
    for i in (dfResultsTemp[raceColumns].isnull().sum(axis=1) < numDrops).index:
        toDelete = numDrops-dfResultsTemp[raceColumns][dfResultsTemp.index == i].isnull().sum(axis=1).values[0]
        if toDelete > 0:
            for j in range(1,toDelete+1):
                maxToDelete = dfResultsTemp[raceColumns][dfResultsTemp.index == i].idxmax(axis=1).values[0]            
                dfResultsTemp.loc[dfResultsTemp.index == i,maxToDelete] = np.nan 
    return dfResultsTemp


# In[ ]:


#initialize the results table
dfResults = pd.DataFrame(columns=['numRegattas','lastRegatta'])
#dfResults['numRegattas'] = dfResults['numRegattas'].astype(int)


# **Load Results: February 2016 - New Zealand National Championships **
# 
# [Raw results][1]
# 
#   [1]: https://kaggle2.blob.core.windows.net/inbox-files/1924/Sailwave%20results%20for%20New%20Zealand%20Open%20Kite%20Foil%20National%20Championships%20at%20Lake%20Taupo%202016.pdf?sv=2012-02-12&se=2016-09-19T15%3A31%3A18Z&sr=b&sp=r&sig=w%2BgVqy0ZG6koKaBd3Du%2FBQe7l%2BGrSL6O2pvfZooTzsg%3D

# In[ ]:


raceName = '20160229-LakeTaupo-NZNationals'
dfResultsTemp = pd.read_csv(resultDir + raceName + '.csv')
dfResultsTemp = dfResultsTemp.set_index(dfResultsTemp['HelmName'])
raceColumns = ['R1','R2','R3','R4','R5','R6']

dfResultsTemp = cleanResults(raceColumns,dfResultsTemp,0,False)
dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults) 


# **Load Results: March 2016 - Mexico Leg of 2106 Hydrofoil Pro Tour **
# 
# [Raw results][1]
# 
# 
#   [1]: https://kaggle2.blob.core.windows.net/inbox-files/1925/Results%20Gold%20&%20Silver%20Fleet%20_%20HydroFoil%20Pro%20Tour.pdf?sv=2012-02-12&se=2016-09-19T15%3A31%3A18Z&sr=b&sp=r&sig=O%2FlT9KdABg60%2BaHRlVjotF8SG4AWgtXZbOqJ9RX1fYg%3D

# In[ ]:


##Load LaVentana  Results
raceName = '20160323-LaVentana-HydrofoilProTour'
raceColumns = ['Q1','Q2','R1','R2','R3','R4','R5','R6']

dfResultsTempGold = pd.read_csv(resultDir+raceName+ '-Gold.csv')
dfResultsTempGold = dfResultsTempGold.set_index(dfResultsTempGold['Name'] + ' ' + dfResultsTempGold['LastName'])
dfResultsTempGold = cleanResults(raceColumns,dfResultsTempGold,0,False)

dfResultsTempSilver = pd.read_csv(resultDir+raceName+ '-Silver.csv')
dfResultsTempSilver = dfResultsTempSilver.set_index(dfResultsTempSilver['Name'] + ' ' + dfResultsTempSilver['LastName'])
dfResultsTempSilver = cleanResults(raceColumns,dfResultsTempSilver,len(dfResultsTempGold),False)

dfResultsTemp = dfResultsTempGold.append(dfResultsTempSilver)
dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)


# **Load Result: May 2016 - Rippin the Rio**

# In[ ]:


raceName = '20160515-ShermanIsland-RippinTheRio'
raceColumns = ['R1', 'R2','R3','R4','R5','R6','R7']
dfResultsTemp = pd.read_csv(resultDir+raceName+ '.csv')
dfResultsTemp = dfResultsTemp.set_index(dfResultsTemp['Name'])
dfResultsTemp = cleanResults(raceColumns,dfResultsTemp,0,True)

#drop worst race
dfResultsTemp = dropRaces(1,dfResultsTemp,raceColumns)
            
dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)


# **Load Result: May 2016 - Florida Kite League Fall Regatta**
# 
# [Raw results][1]
# 
#   [1]: http://www.regattanetwork.com/event/12210#_newsroom

# In[ ]:


raceName = '20160515-StPete-FKLSpringRegatta'
raceColumns = ['R1', 'R2','R3','R4','R5','R6','R7','R8','R9','R10']

dfResultsTemp = pd.read_csv(resultDir+raceName+ '.csv')
dfResultsTemp = dfResultsTemp.set_index(dfResultsTemp['Skipper'])
dfResultsTemp = cleanResults(raceColumns,dfResultsTemp,0,True)
dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)


# **Load Result: May 2016 - France Leg of IFKO Silver Cup**
# 
# [Raw results][1]
# 
#   [1]: https://kaggle2.blob.core.windows.net/inbox-files/1888/Montpellier.png?sv=2012-02-12&se=2016-09-14T17%3A39%3A22Z&sr=b&sp=r&sig=uz1gSwGb7Wwl3bio8ZK5K2xrBZr4EmBJcOGaEEdsAmc%3D

# In[ ]:


raceName = '20160516-MontPellier-IFKOSilverCup'
raceColumns = ['CO 1', 'CO 2','CO 3','CO 4','CO 5','CO 6','CO 7','CO 8','CO 9','CO 10','CO 11','CO 12']
dfResultsTemp = pd.read_csv(resultDir+raceName+ '.csv')

for index, row in dfResultsTemp.iterrows():
    numNames = len(row['Name'].split(' '))
    dfResultsTemp.loc[dfResultsTemp.index == index,'Name'] = row['Name'].split(' ')[numNames-1] + ' ' + row['Name'].split(' ')[0]
    
   
dfResultsTemp = dfResultsTemp.set_index(dfResultsTemp['Name'])

dfResultsTemp = cleanResults(raceColumns,dfResultsTemp,0,True)

#drop worst 3 races
dfResultsTemp = dropRaces(3,dfResultsTemp,raceColumns)
            
dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)


# **Load Result: May 2016 - European Championship**
# 
# [Raw results][1]
# 
# 
#   [1]: https://kaggle2.blob.core.windows.net/inbox-files/1894/Cagliari.png?sv=2012-02-12&se=2016-09-16T05%3A01%3A10Z&sr=b&sp=r&sig=eSZRz4Zi6IPv8Sj9NIOhTffFSe3H%2FYwP%2BRqPlTxhi3Q%3D

# In[ ]:


##Load Cagliari Results
raceName = '20160522-Cagliari-EuropeanChampionships'
raceColumns = ['CF1','CF2','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','M1','M2','M3','M4']


dfResultsTempPlatinum = pd.read_csv(resultDir+raceName+ '-Platinum.csv',encoding = "ISO-8859-1")
dfResultsTempPlatinum = dfResultsTempPlatinum.set_index(dfResultsTempPlatinum['Name'])
dfResultsTempPlatinum = cleanResults(raceColumns,dfResultsTempPlatinum,0,False)

dfResultsTempGold = pd.read_csv(resultDir+raceName+ '-Gold.csv',encoding = "ISO-8859-1")
dfResultsTempGold = dfResultsTempGold.set_index(dfResultsTempGold['Name'])
dfResultsTempGold = cleanResults(raceColumns,dfResultsTempGold,len(dfResultsTempPlatinum),False)

dfResultsTempSilver = pd.read_csv(resultDir+raceName+ '-Silver.csv',encoding = "ISO-8859-1")
dfResultsTempSilver = dfResultsTempSilver.set_index(dfResultsTempSilver['Name'])
dfResultsTempSilver = cleanResults(raceColumns,dfResultsTempSilver,len(dfResultsTempGold),False)

dfResultsTemp = dfResultsTempPlatinum.append(dfResultsTempGold)
dfResultsTemp = dfResultsTemp.append(dfResultsTempSilver)

dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)


# **Load Result: July 2016 - Sail Brisbane**

# In[ ]:


##Load Brisbane Results
raceName = '20160710-Brisbane-SailBrisbane'
raceColumns = ['Race 11','Race 10','Race 9','Race 8','Race 7','Race 6','Race 5','Race 4','Race 3','Race 2','Race 1']
dfResultsTemp = pd.read_csv(resultDir+raceName+ '.csv')
dfResultsTemp = dfResultsTemp.set_index(dfResultsTemp['Skipper'])
dfResultsTemp = cleanResults(raceColumns,dfResultsTemp,0,False)

dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)


# **Load Result: July 2016 - Italian Leg of the IKA Gold Cup**
# 
# [Raw results][1]
# 
# 
#   [1]: https://kaggle2.blob.core.windows.net/inbox-files/1891/Gizerria.png?sv=2012-02-12&se=2016-09-14T17%3A39%3A22Z&sr=b&sp=r&sig=C0doGeMU63l02XiRXk%2BRAoWlRLx8v2ZrxTCxXBcGsQ8%3D`

# In[ ]:


###Load Italy results
raceName = '20160717-Gizzeria-IKAGoldCup'



raceColumns = ['CF 1','CF 2','F 1','F 2','F 3','F 4','F 5','F 6','F 7','F 8',	'F 9','F 10']
dfResultsTempGold = pd.read_csv(resultDir+raceName+ '-Gold.csv')
dfResultsTempGold = dfResultsTempGold.set_index(dfResultsTempGold['Name'])
dfResultsTempGold = cleanResults(raceColumns,dfResultsTempGold,0,False)

raceColumns = ['CF 1','CF 2','F 1','F 2','F 3','F 4','F 5','F 6','F 8']
dfResultsTempSilver = pd.read_csv(resultDir+raceName+ '-Silver.csv')
dfResultsTempSilver = dfResultsTempSilver.set_index(dfResultsTempSilver['Name'])
dfResultsTempSilver = cleanResults(raceColumns,dfResultsTempSilver,len(dfResultsTempGold),False)

raceColumns = ['CF 1','CF 2','F 1','F 2','F 3','F 4','F 5','F 6']
dfResultsTemp = dfResultsTempGold.append(dfResultsTempSilver)

dfResultsTempBronze = pd.read_csv(resultDir+raceName+ '-Bronze.csv',encoding = "ISO-8859-1")
dfResultsTempBronze = dfResultsTempBronze.set_index(dfResultsTempBronze['Name'])
dfResultsTempBronze = cleanResults(raceColumns,dfResultsTempBronze,len(dfResultsTemp),False)

dfResultsTemp = dfResultsTemp.append(dfResultsTempBronze)

raceColumns = ['CF 1','CF 2','F 1','F 2','F 3','F 4','F 5','F 6','F 7','F 8',	'F 9','F 10']


dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)


# **Load Result: August 2016 - US leg of the Hydrofoil Pro Tour**
# 
# [Raw results][1]
# 
# 
#   [1]: https://kaggle2.blob.core.windows.net/inbox-files/1892/SanFrancisco.png?sv=2012-02-12&se=2016-09-14T17%3A39%3A22Z&sr=b&sp=r&sig=vaq39h3n3S0EVE4gR%2BWpij%2BKY4UjwkbC%2BCmfi0vV9RQ%3D

# In[ ]:


##Load SF results
raceName = '20160807-SanFrancisco-HydrofoilProTour'
dfResultsTemp = pd.read_csv(resultDir + raceName + '.csv')
dfResultsTemp = dfResultsTemp.set_index(dfResultsTemp['Name'])
raceColumns = ['R1','R2','R3','R4','R5','R6','R7','R8','R9','R10','R11','R12','R13','R14','R15','R16']

dfResultsTemp = cleanResults(raceColumns,dfResultsTemp,0,False)
dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)


# **Load Result: August 2016 - Mauritius leg of the Hydrofoil Pro Tour**
# 
# [Raw results][1]
# 
# 
#   [1]: https://kaggle2.blob.core.windows.net/inbox-files/1926/Mauritius%20Results%20_%20HydroFoil%20Pro%20Tour.pdf?sv=2012-02-12&se=2016-09-19T15%3A31%3A18Z&sr=b&sp=r&sig=48Y8tXleSDLFa0bHpTRgDAfXM7hcKMQEQnHhKA%2Fwb2o%3D

# In[ ]:


##Load Mauritius results
raceName = '20160820-Mauritius-HydrofoilProTour'
dfResultsTemp = pd.read_csv(resultDir + raceName + '.csv')
dfResultsTemp = dfResultsTemp.set_index(dfResultsTemp['HelmName'])
raceColumns = ['R1','R2','R3','R4','R5','R6','R7','R8','R9','R10','R11','R12','R13','R14','R15','R16','R17']

dfResultsTemp = cleanResults(raceColumns,dfResultsTemp,0,True)
dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)


# **Load Result: August 2016 - Pringles Foil World Cup in Fehnmarn **

# In[ ]:


##Load Fehnmanr Results - Men
raceName = '20160830-Fehnmarn-PringlesFoilWorldCup'
raceColumns = ['R1','R2','R3','R4','R5','R6','R7','R8']

dfResultsTemp = pd.read_csv(resultDir+raceName+ '.csv')
dfResultsTemp = dfResultsTemp.set_index(dfResultsTemp['Name'])
dfResultsTemp = cleanResults(raceColumns,dfResultsTemp,0,False)

#drop worst two 
dfResultsTemp = dropRaces(2,dfResultsTemp,raceColumns)

dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)


# **Load Result: September 2016 - Danish National Championship**

# In[ ]:


raceName = '20160911-Denmark-NationalChampionship'
raceColumns = ['R1', 'R2','R3','R4','R5','R6','R7','R8','R9','R10']
dfResultsTemp = pd.read_csv(resultDir+raceName+ '.csv',encoding = "ISO-8859-1")
dfResultsTemp = dfResultsTemp.set_index(dfResultsTemp['Name'])
dfResultsTemp = cleanResults(raceColumns,dfResultsTemp,0,True)

#drop worst 2 races
dfResultsTemp = dropRaces(2,dfResultsTemp,raceColumns)
            
dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)


# **Load Result: September 2016 - IKA World Championship in Weifang**

# In[ ]:


##Load Weifang Results - Men
raceName = '20160915-Weifang-WorldChampionship'
raceColumns = ['CF1','CF2','F1','F2','F3','F4','F5','M1','M2','M3','M4']

dfResultsTempPlatinumAndGold = pd.read_csv(resultDir+raceName+ '-PlatinumAndGold.csv',encoding = "ISO-8859-1")
dfResultsTempPlatinumAndGold = dfResultsTempPlatinumAndGold.set_index(dfResultsTempPlatinumAndGold['Name'])
dfResultsTempPlatinumAndGold = cleanResults(raceColumns,dfResultsTempPlatinumAndGold,0,False)

raceColumns = ['CF1','CF2','F1','F2','F3','F4']
dfResultsTempSilver = pd.read_csv(resultDir+raceName+ '-Silver.csv',encoding = "ISO-8859-1")
dfResultsTempSilver = dfResultsTempSilver.set_index(dfResultsTempSilver['Name'])
dfResultsTempSilver = cleanResults(raceColumns,dfResultsTempSilver,len(dfResultsTempPlatinumAndGold),False)

dfResultsTemp = dfResultsTempPlatinumAndGold.append(dfResultsTempSilver)

#put all race columns back before merging results
raceColumns = ['CF1','CF2','F1','F2','F3','F4','F5','M1','M2','M3','M4']
dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)


# In[ ]:


##Load Weifang Results - Women
raceName = '20160915-Weifang-WorldChampionship-Women'
raceColumns = ['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','M1','M2','M3','M4']

dfResultsTemp = pd.read_csv(resultDir+raceName+ '.csv',encoding = "ISO-8859-1")
dfResultsTemp = dfResultsTemp.set_index(dfResultsTemp['Name'])
dfResultsTemp = cleanResults(raceColumns,dfResultsTemp,0,False)

dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)


# **Load: September 2016 Battle of the Bay in San Francisco**

# In[ ]:


raceName = '20160918-SanFrancisco-BattleOfTheBay'
raceColumns = ['Race 1','Race 2','Race 3','Race 4','Race 5','Race 6']

dfResultsTemp = pd.read_csv(resultDir+raceName+ '.csv')
dfResultsTemp = dfResultsTemp.set_index(dfResultsTemp['Name'])
dfResultsTemp = cleanResults(raceColumns,dfResultsTemp,0,False)

dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)


# **Load: October 2016: Vineyard Cup**

# In[ ]:


##Load Vineyard Cup Results
raceName = '20161002-MarthasVineyard-VineyardCup'
raceColumns = ['R1','R2','R3','R4','R5','R6','R7','R8','R9','R10']

dfResultsTemp = pd.read_csv(resultDir+raceName+ '.csv',encoding = "ISO-8859-1")
dfResultsTemp = dfResultsTemp.set_index(dfResultsTemp['Name'])
dfResultsTemp = cleanResults(raceColumns,dfResultsTemp,0,False)

dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)


# **Load Result: November 2016 - Florida Kite League Spring Regatta**
# 
# [Raw results][1]
# 
#   [1]: http://www.regattanetwork.com/event/13310#_newsroom

# In[ ]:


raceName = '20161106-StPete-FKLFallRegatta'
raceColumns = ['R1', 'R2','R3','R4','R5','R6','R7','R8','R9']

dfResultsTemp = pd.read_csv(resultDir + raceName+ '.csv',encoding = "ISO-8859-1")
dfResultsTemp = dfResultsTemp.set_index(dfResultsTemp['Skipper'])
dfResultsTemp = cleanResults(raceColumns,dfResultsTemp,0,True)
dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)


# **Load Result: November 2016 - Qatar Leg of the IKA Gold Cup**

# In[ ]:


raceName = '20161119-Doha-IKAGoldCup'
raceColumns = ['R1', 'R2','R3','R4','R5','R6','R7','R8','R9','R10','R11','R12','R13','R14','R15']

dfResultsTemp = pd.read_csv(resultDir+raceName+ '.csv',encoding = "ISO-8859-1")
dfResultsTemp = dfResultsTemp.set_index(dfResultsTemp['HelmName'])
dfResultsTemp = cleanResults(raceColumns,dfResultsTemp,0,True)
dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)


# **Load Result: November 2016 - Australian National Formula Championships in Rockingham**

# In[ ]:


raceName = '20161128-Rockingham-Australian-Formula-Nationals'
raceColumns = ['R1', 'R2','R3','R4','R5','R6','R7']

dfResultsTemp = pd.read_csv(resultDir+raceName+ '.csv',encoding = "ISO-8859-1")
dfResultsTemp = dfResultsTemp.set_index(dfResultsTemp['Name'])
dfResultsTemp = cleanResults(raceColumns,dfResultsTemp,0,True)
dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)


# **Load Result: November 2016 - Australian Leg of the 2016 Hydrofoil Pro Tour in Rockingham**

# In[ ]:


raceName = '20161129-Rockingham-HydrofoilProTour-Australia'
raceColumns = ['1', '2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17']

dfResultsTemp = pd.read_csv(resultDir + raceName+ '.csv',encoding = "ISO-8859-1")
dfResultsTemp = dfResultsTemp.set_index(dfResultsTemp['Name'])
dfResultsTemp = cleanResults(raceColumns,dfResultsTemp,0,True)
dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)


# **Load Result: November 2016 - 2016 Australian National Kitefoil Championship in Rockingham**

# In[ ]:


raceName = '20161129-Rockingham-Australian-Kitefoil-Nationals'
raceColumns = ['R1', 'R2','R3','R4','R5','R6','R7','R8', 'R9','R10','R11','R12','R13','R14','R15','R16','R17']

dfResultsTemp = pd.read_csv(resultDir+raceName+ '.csv',encoding = "ISO-8859-1")
dfResultsTemp = dfResultsTemp.set_index(dfResultsTemp['Name'])
dfResultsTemp = cleanResults(raceColumns,dfResultsTemp,0,False)
dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)


# **Load Result: December 2016 - 2016 Sail Melbourne**

# In[ ]:


raceName = '20161210-Melbourne-SailMelbourne'

raceColumns = ['R13','R14','R15','R16','R17','R18','R19','R20','R21','R22','R23','R24']

dfResultsTempGold = pd.read_csv(resultDir+raceName+ '-Gold.csv')
dfResultsTempGold = dfResultsTempGold.set_index(dfResultsTempGold['Name'])
dfResultsTempGold = cleanResults(raceColumns,dfResultsTempGold,0,True)

raceColumns = ['R13','R14','R15','R16','R17','R18','R19','R20']
dfResultsTempSilver = pd.read_csv(resultDir+raceName+ '-Silver.csv')
dfResultsTempSilver = dfResultsTempSilver.set_index(dfResultsTempSilver['Name'])
dfResultsTempSilver = cleanResults(raceColumns,dfResultsTempSilver,len(dfResultsTempGold),True)

dfResultsTemp = dfResultsTempGold.append(dfResultsTempSilver)

#put all race columns back before merging results
raceColumns = ['R13','R14','R15','R16','R17','R18','R19','R20','R21','R22','R23','R24']
dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)


# **Load Result: January 2017 - Lord of the Wind in La Ventana**

# In[ ]:


raceName = '20170122-LosBarriles-LordOfTheWind'
raceColumns = ['R1','R2','R3','R4','R5','R6','R7','R8']

dfResultsTemp = pd.read_csv(resultDir+raceName+ '.csv')
dfResultsTemp = dfResultsTemp.set_index(dfResultsTemp['Name'])
dfResultsTemp = cleanResults(raceColumns,dfResultsTemp,0,True)

#drop worst two 
dfResultsTemp = dropRaces(2,dfResultsTemp,raceColumns)

dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)


# **Load Result: February 2017 - New Zealand Nationals**

# In[ ]:


raceName = '20170207-Takapuna-NZNationalChampionships'
raceColumns = ['R1','R2','R3','R4','R5','R6','R7','R8','R9','R10','R11']

dfResultsTemp = pd.read_csv(resultDir+raceName+ '.csv')
dfResultsTemp = dfResultsTemp.set_index(dfResultsTemp['HelmName'])
dfResultsTemp = cleanResults(raceColumns,dfResultsTemp,0,True)

dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)


# **Load Result: February 2017 - WA State Titles**

# In[ ]:


raceName = '20170219-JervoiseBay-WAStateTitles'
raceColumns = ['R1','R2','R3','R4']

dfResultsTemp = pd.read_csv(resultDir+raceName+ '.csv')
dfResultsTemp = dfResultsTemp.set_index(dfResultsTemp['Sailor(s)'])
dfResultsTemp = cleanResults(raceColumns,dfResultsTemp,0,True)

dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)


# **Load Result: March 2017 - Hydrofoil Pro Tour in Mexico**

# In[ ]:


raceName = '20170329-LaVentana-HydrofoilProTour'
raceColumns = ['R1', 'R2','R3','R4','R5','R6','R7','R8', 'R9','R10','R11','R12','R13','R14','R15','R16','R17','R18','R19']

dfResultsTemp = pd.read_csv(resultDir+raceName+ '.csv')
dfResultsTemp = dfResultsTemp.set_index(dfResultsTemp['NAME'])
dfResultsTemp = cleanResults(raceColumns,dfResultsTemp,0,True)

dfResultsTemp = dropRaces(3,dfResultsTemp,raceColumns)

dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)


# **Load Result April 2017 - NSW State Championship**

# In[ ]:


raceName = '20170409-NSWChampionship-GeorgeRiverSailingClub'
raceColumns = ['R1', 'R2','R3','R4','R5','R6','R7','R8']

dfResultsTemp = pd.read_csv(resultDir+raceName+ '.csv')
dfResultsTemp = dfResultsTemp.set_index(dfResultsTemp['Name'])
dfResultsTemp = cleanResults(raceColumns,dfResultsTemp,0,True)

dfResultsTemp = dropRaces(1,dfResultsTemp,raceColumns)

dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)


# **Load Result April 2017 - Sailing World Cup in Hyeres**

# In[ ]:


raceName = '20170429-Hyeres-SailingWorldCup-QualificationSplit1Blue'
raceColumns = ['R1', 'R2','R3','R4','R5','R6']

dfResultsTemp = pd.read_csv(resultDir+raceName+ '.csv')
dfResultsTemp = dfResultsTemp.set_index(dfResultsTemp['Name'])
dfResultsTemp = cleanResults(raceColumns,dfResultsTemp,0,True)
dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)

raceName = '20170429-Hyeres-SailingWorldCup-QualificationSplit1Yellow'
raceColumns = ['R1', 'R2','R3','R4','R5','R6']

dfResultsTemp = pd.read_csv(resultDir+raceName+ '.csv')
dfResultsTemp = dfResultsTemp.set_index(dfResultsTemp['Name'])
dfResultsTemp = cleanResults(raceColumns,dfResultsTemp,0,True)
dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)

raceName = '20170429-Hyeres-SailingWorldCup-QualificationSplit2Blue'
raceColumns = ['R1', 'R2','R3','R4','R5','R6']

dfResultsTemp = pd.read_csv(resultDir+raceName+ '.csv')
dfResultsTemp = dfResultsTemp.set_index(dfResultsTemp['Name'])
dfResultsTemp = cleanResults(raceColumns,dfResultsTemp,0,True)
dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)


raceName = '20170429-Hyeres-SailingWorldCup-QualificationSplit2Yellow'
raceColumns = ['R1', 'R2','R3','R4','R5','R6']

dfResultsTemp = pd.read_csv(resultDir+raceName+ '.csv')
dfResultsTemp = dfResultsTemp.set_index(dfResultsTemp['Name'])
dfResultsTemp = cleanResults(raceColumns,dfResultsTemp,0,True)
dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)


raceName = '20170429-Hyeres-SailingWorldCup-Gold'
raceColumns = ['R13','R14','R15','R16','R17','R18','M1','M2','M3']
dfResultsTempGold = pd.read_csv(resultDir+raceName+ '.csv')
dfResultsTempGold = dfResultsTempGold.set_index(dfResultsTempGold['Name'])
dfResultsTempGold = cleanResults(raceColumns,dfResultsTempGold,0,True)

raceName = '20170429-Hyeres-SailingWorldCup-Silver'
raceColumns = ['R13','R14','R15','R16','R17','R18']
dfResultsTempSilver = pd.read_csv(resultDir+raceName+ '.csv')
dfResultsTempSilver = dfResultsTempSilver.set_index(dfResultsTempSilver['Name'])
dfResultsTempSilver = cleanResults(raceColumns,dfResultsTempSilver,len(dfResultsTempGold),True)

dfResultsTemp = dfResultsTempGold.append(dfResultsTempSilver)
dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)




# **Load Result May 2017 - Elvstrom Zellerbach Regatta in SF**

# In[ ]:


raceName = '20170507-SanFrancisco-Elvstrom-Zellerbach'
raceColumns = ['R1', 'R2','R3','R4','R5','R6']

dfResultsTemp = pd.read_csv(resultDir+raceName+ '.csv')
dfResultsTemp = dfResultsTemp.set_index(dfResultsTemp['Sailor(s)'])
dfResultsTemp = cleanResults(raceColumns,dfResultsTemp,0,True)

dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)


# **Load Result May 2017 - Florida Kite League Spring Regatta**

# In[ ]:


raceName = '20170507-StPete-SpringRegatta2017'
raceColumns = ['1', '2','3','4','5','6','7','8']
dfResultsTemp = pd.read_csv(resultDir+raceName+ '.csv')
dfResultsTemp = dfResultsTemp.set_index(dfResultsTemp['Skipper'])
dfResultsTemp = cleanResults(raceColumns,dfResultsTemp,0,True)

dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)


# **Load Result June 2017 - Hydrofoil Pro Tour Montpellier**

# In[ ]:


raceName = '20170605-Montpellier-HydrofoilProTour'

raceColumns = ['R1', 'R2','R3','R4','R5','R6','R7','R8', 'R9','R10','R11','R12','R13']

dfResultsTempGold = pd.read_csv(resultDir+raceName+ '-Gold.csv')
dfResultsTempGold = dfResultsTempGold.set_index(dfResultsTempGold['Name'])
dfResultsTempGold = cleanResults(raceColumns,dfResultsTempGold,0,True)

raceColumns = ['R1', 'R2','R3','R4','R5','R6','R7','R8', 'R9','R10','R11','R12']
dfResultsTempSilver = pd.read_csv(resultDir+raceName+ '-Silver.csv')
dfResultsTempSilver = dfResultsTempSilver.set_index(dfResultsTempSilver['Name'])
dfResultsTempSilver = cleanResults(raceColumns,dfResultsTempSilver,len(dfResultsTempGold),True)

dfResultsTemp = dfResultsTempGold.append(dfResultsTempSilver)


#put all race columns back before merging results
raceColumns = ['R1', 'R2','R3','R4','R5','R6','R7','R8', 'R9','R10','R11','R12','R13']
dfResultsTemp = dropRaces(2,dfResultsTemp,raceColumns)

dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)


# **Load Result June 2016: Sailing World Cup in Santander**

# In[ ]:


raceName = '20170610-Santander-SailingWorldCupSeriesFinal-QualificationSplit1Blue'
raceColumns = ['R1', 'R2','R3','R4','R5','R6']

dfResultsTemp = pd.read_csv(resultDir+raceName+ '.csv')
dfResultsTemp = dfResultsTemp.set_index(dfResultsTemp['Name'])
dfResultsTemp = cleanResults(raceColumns,dfResultsTemp,0,True)
dfResultsTemp = dropRaces(1,dfResultsTemp,raceColumns)

dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)

raceName = '20170610-Santander-SailingWorldCupSeriesFinal-QualificationSplit1Yellow'
raceColumns = ['R1', 'R2','R3','R4','R5','R6']
dfResultsTemp = pd.read_csv(resultDir+raceName+ '.csv')
dfResultsTemp = dfResultsTemp.set_index(dfResultsTemp['Name'])
dfResultsTemp = cleanResults(raceColumns,dfResultsTemp,0,True)
dfResultsTemp = dropRaces(1,dfResultsTemp,raceColumns)
dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)

raceName = '20170610-Santander-SailingWorldCupSeriesFinal-QualificationSplit2Blue'
raceColumns = ['R7', 'R8','R9','R10','R11','R12']

dfResultsTemp = pd.read_csv(resultDir+raceName+ '.csv')
dfResultsTemp = dfResultsTemp.set_index(dfResultsTemp['Name'])
dfResultsTemp = cleanResults(raceColumns,dfResultsTemp,0,True)
dfResultsTemp = dropRaces(1,dfResultsTemp,raceColumns)
dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)


raceName = '20170610-Santander-SailingWorldCupSeriesFinal-QualificationSplit2Yellow'
raceColumns = ['R7', 'R8','R9','R10','R11','R12']
dfResultsTemp = pd.read_csv(resultDir+raceName+ '.csv')
dfResultsTemp = dfResultsTemp.set_index(dfResultsTemp['Name'])
dfResultsTemp = cleanResults(raceColumns,dfResultsTemp,0,True)
dfResultsTemp = dropRaces(1,dfResultsTemp,raceColumns)
dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)


raceName = '20170610-Santander-SailingWorldCupSeriesFinal-Gold'
raceColumns = ['R13','R14','R15','R16','R17','R18','R19','R20','R21','R22','R23','R24','M1','M2','M3']
dfResultsTempGold = pd.read_csv(resultDir+raceName+ '.csv')
dfResultsTempGold = dfResultsTempGold.set_index(dfResultsTempGold['Name'])
dfResultsTempGold = cleanResults(raceColumns,dfResultsTempGold,0,True)

raceName = '20170610-Santander-SailingWorldCupSeriesFinal-Silver'
raceColumns = ['R13','R14','R15','R16','R17','R18','R19','R20','R21','R22','R23','R24']
dfResultsTempSilver = pd.read_csv(resultDir+raceName+ '.csv')
dfResultsTempSilver = dfResultsTempSilver.set_index(dfResultsTempSilver['Name'])
dfResultsTempSilver = cleanResults(raceColumns,dfResultsTempSilver,len(dfResultsTempGold),True)


dfResultsTemp = dfResultsTempGold.append(dfResultsTempSilver)
dfResultsTemp = dropRaces(3,dfResultsTemp,raceColumns)
dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)




# **Load Result August 2018: Hydrofoil Pro Tour in San Francisco**

# In[ ]:


raceName = '20170803-SanFrancisco-HydrofoilProTour'
raceColumns = ['R1', 'R2','R3','R4','R5','R6','R7','R8', 'R9','R10','R11','R12','R13']

dfResultsTemp = pd.read_csv(resultDir+raceName+ '.csv',encoding = "ISO-8859-1")
dfResultsTemp = dfResultsTemp.set_index(dfResultsTemp['Sailor(s)'])
dfResultsTemp = cleanResults(raceColumns,dfResultsTemp,0,True)

dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults)


# In[ ]:


raceName = '20170823-Fehman-HydrofoilProTour'
dfResultsTemp = pd.read_csv(resultDir + raceName + '.csv',encoding = "ISO-8859-1")
raceColumns = ['R1', 'R2','R3','R4','R5','R6','R7','R8', 'R9','R10','R11','R12']
dfResultsTemp = dfResultsTemp.set_index(dfResultsTemp['Sailor(s)'])
dfResultsTemp = cleanResults(raceColumns,dfResultsTemp,0,True)
dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults) 


# In[ ]:


raceName = '20170922-Mauritius-HydrofoilProTour'
dfResultsTemp = pd.read_csv(resultDir + raceName + '.csv',encoding = "ISO-8859-1")
raceColumns = ['R1', 'R2','R3','R4','R5','R6','R7','R8', 'R9','R10','R11','R12','R13','R14','R15','R16','R17','R18']
dfResultsTemp = dfResultsTemp.set_index(dfResultsTemp['Sailor(s)'])
dfResultsTemp = cleanResults(raceColumns,dfResultsTemp,0,True)
dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults) 


# In[ ]:


raceName = '20171008-Sardinia-KitefoilGoldCup'
dfResultsTemp = pd.read_csv(resultDir + raceName + '.csv',encoding = "ISO-8859-1")
raceColumns = ['R1', 'R2','R3','R4','R5','R6','R7','R8', 'R9','R10','R11','R12']
dfResultsTemp = dfResultsTemp.set_index(dfResultsTemp['Name'])
dfResultsTemp = cleanResults(raceColumns,dfResultsTemp,0,True)
dfResults = mergeResults(raceColumns,raceName,dfResultsTemp,dfResults) 


# **Do Rating**

# In[ ]:


def doRating(dfResults):
    
    env = ts.TrueSkill()
    ts.setup(tau=0.2)
    
    columns = ['Name','mu_minus_3sigma','numRaces','Rating']
        
    dfRatings = pd.DataFrame(columns=columns,index=dfResults.index)
    dfRatings = pd.merge(dfRatings,dfResults[['numRegattas','lastRegatta']],left_index=True,right_index=True,how='outer')
    
    
    dfRatings['numRaces'] = dfResults.count(axis=1)
    dfRatings['Rating'] = pd.Series(np.repeat(env.Rating(),len(dfRatings))).T.values.tolist()
    
    
    
    for raceCol in dfResults:
        if (raceCol != 'numRegattas') or (raceCol != 'lastRegatta'):  
            competed = dfRatings.index.isin(dfResults.index[dfResults[raceCol].notnull()])
            rating_group = list(zip(dfRatings['Rating'][competed].T.values.tolist()))
            ranking_for_rating_group = dfResults[raceCol][competed].T.values.tolist()
            dfRatings.loc[competed, 'Rating'] = ts.rate(rating_group, ranks=ranking_for_rating_group)

    
    dfRatings = pd.DataFrame(dfRatings) #convert to dataframe

    dfRatings['mu_minus_3sigma'] = pd.Series(np.repeat(0.0,len(dfRatings))) #calculate mu - 3 x sigma: MSFT convention

    for index, row in dfRatings.iterrows():
        dfRatings.loc[dfRatings.index == index,'mu_minus_3sigma'] = float(row['Rating'].mu) - 3 * float(row['Rating'].sigma)

    #competed in at least 5 races and 1 regatta and has competed in the last 12 months
    dfRatings = dfRatings[dfRatings['numRaces'] > 4]
    dfRatings = dfRatings[dfRatings['numRegattas'] > 1]
    dfRatings = dfRatings[(datetime.datetime.now() - dfRatings['lastRegatta'] ) / np.timedelta64(1, 'D') < 365] 

    dfRatings['Name'] = dfRatings.index
    dfRatings.index = dfRatings['mu_minus_3sigma'].rank(ascending=False).astype(int) #set index to ranking
    dfRatings.index.names = ['Rank']

    
    return dfRatings.sort_values('mu_minus_3sigma',ascending=False) 


# **Kitefoil World Ranking**

# In[ ]:


dfRatings = doRating(dfResults)
display(dfRatings)


# **Check duplicates**

# In[ ]:


dfRatings['Name'].sort_values()

