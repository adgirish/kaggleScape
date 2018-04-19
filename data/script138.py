
# coding: utf-8

# In[ ]:


import hashlib
import os
''' I have made this simple class to save the data i am getting from query, So that If I
Ever Run this program again I dont have to use my quota but get it directly from the
place where I have saved it... :##'''
class DataSaver:
    def __init__(self, bq_assistant):
        self.bq_assistant=bq_assistant
        
    def Run_Query(self, query, max_gb_scanned=1):
        hashed_query=''.join(query.split()).encode("ascii","ignore")
        query_hash=hashlib.md5(hashed_query).hexdigest()
        query_hash+=".csv"
        if query_hash in os.listdir(os.getcwd()):
            print ("Data Already present getting it from file")
            return pd.read_csv(query_hash)
        else:
            data=self.bq_assistant.query_to_pandas_safe(query, max_gb_scanned=max_gb_scanned)
            data.to_csv(query_hash, index=False,encoding='utf-8')
            return data


# In[ ]:


import time
start_time=time.time()
import numpy as np
import operator
from collections import Counter
import pandas as pd 
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go
init_notebook_mode(connected=True)
plt.rcParams['figure.figsize']=(12,5)
import numpy as np
import operator
import pandas as pd 
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go
init_notebook_mode(connected=True)
plt.rcParams['figure.figsize']=(12,5)
from google.cloud import bigquery
from bq_helper import BigQueryHelper
client = bigquery.Client()
bq_assistant = BigQueryHelper("bigquery-public-data", "bitcoin_blockchain")
def satoshi_to_bitcoin(satoshi):
    return float(float(satoshi)/ float(100000000))
test_data=bq_assistant.head("transactions")


bq=DataSaver(bq_assistant)


def Create_Bar_plotly(list_of_tuples, items_to_show=40, title=""):
    #list_of_tuples=list_of_tuples[:items_to_show]
    data = [go.Bar(
            x=[val[0] for val in list_of_tuples],
            y=[val[1] for val in list_of_tuples]
    )]
    layout = go.Layout(
    title=title,xaxis=dict(
        autotick=False,
        tickangle=290 ),)
    fig = go.Figure(data=data, layout=layout)
    py.offline.iplot(fig)


# 
# ## So Bitcoins were pretty difficult to understand at first but while digging deep into it it seems quite simple. 
# <br>
# ### So in this notebook we will first learn about bitcoins and their workings, then try to find out some interesting insights from it.
# ### In case you already know about bitcoin transactions you can click through hyperlink and go directly to visualizations.
# ## CONTENTS
# <b><a href='#1'>1:  Wallets</a></b>
# <br>
# <b><a href='#2'>2:  Transactions</a></b><br>
# <b> &nbsp;&nbsp;  <a href='#2.2'>2.2: Transaction In a nutshell</a></b><br>
# <b> &nbsp;&nbsp;  <a href='#2.3'>2.3: Sample Transaction</a></b><br>
# <b> &nbsp;&nbsp;  <a href='#2.4'>2.4: Security in Transaction</a></b><br>
# <b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#2.4.1'>2.4.1: Short Description</a></b><br>
# <b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#2.4.2'>2.4.2: Long Description</a></b>
# <br>
# <b><a href='#3'>3: Visualizations</a></b><br>
# <b>&nbsp;&nbsp;<a href='#3.1'>3.1: Who have the most number of Bitcoins.</a></b><br>
# <b>&nbsp;&nbsp;<a href='#3.2'>3.2: TimeSeries For Highest Valued Transactions</a></b><br>
# <b>&nbsp;&nbsp;<a href='#3.3'>3.3: Average Highest value transaction per year</a></b><br>
# <b>&nbsp;&nbsp;<a href='#3.4'>3.4: Bar Chart for top transaction counts per year</a></b><br>
# <b>&nbsp;&nbsp;<a href='#3.5'>3.5: Guess who got rich in 2018??</a></b><br>
# <b>&nbsp;&nbsp;<a href='#3.6'>3.6: Average number BITCOINS transacted per day</a></b><br>
# <b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#3.6.1'>3.6.1: Time Series for avg no of bitcoins transacted per day</a></b><br>
# <b>&nbsp;&nbsp;<a href='#3.7'>3.7: Total number of Transactions Per day</a></b><br>
# <b>&nbsp;&nbsp;<a href='#3.8'>3.8: Self Transactions per day (literally) (what a burden on miners)</a></b><br>
# <b>&nbsp;&nbsp;<a href='#3.9'>3.9: Who did most of the transactions</a></b><br>
# <b>&nbsp;&nbsp;<a href='#3.10'>3.10: Inactive Addresses(someone who received but never spent)</a></b><br>
# <b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#3.10.1'>3.10.1: Timeseries for Value of unspent outputs</a></b><br>
# <b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#3.10.2'>3.10.2: Unspent output value yearly</a></b><br>
# <b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#3.10.3'>3.10.3: 2010 Analysis</a></b>
# <br>
# <b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#3.10.4'>3.10.3: Total Number of bitcoins unspent (or losssssssssssssssssst)</a></b>
# <br>
# 
# <b>&nbsp;&nbsp;<a href='#3.11'>3.11: Comparison of Bitcoin Transactions Prices and count with Exchange Rates</a></b><br>
# <b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#3.11.1'>3.11.1: Timeseries plot signifying the relation between the spike in bitcoin value, the number of transactions, the total bitcoins transacted per day.</a></b><br>
# <b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#3.11.2'>3.11.2: Average bitcoin price in usd to transaction counts</a></b><br>
# <b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#3.11.2.1'>3.11.2.1: 2012</a></b><br>
# <b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#3.11.2.2'>3.11.2.2: 2013</a></b><br>
# <b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#3.11.2.3'>3.11.2.3: 2014</a></b><br>
# <b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#3.11.2.4'>3.11.2.4: 2015</a></b><br>
# <b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#3.11.2.5'>3.11.2.5: 2016</a></b><br>
# <b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#3.11.2.6'>3.11.2.6: 2017</a></b><br>
# 
# 
# 
# <b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#3.11.3'>3.11.3: Number of bitcoins transacted with the average bitcoin value in USD</a></b><br>
# <b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#3.11.3.1'>3.11.3.1: 2012</a></b><br>
# <b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#3.11.3.2'>3.11.3.2: 2013</a></b><br>
# <b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#3.11.3.3'>3.11.3.3: 2014</a></b><br>
# <b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#3.11.3.4'>3.11.3.4: 2015</a></b><br>
# <b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#3.11.3.5'>3.11.3.5: 2016</a></b><br>
# <b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#3.11.3.6'>3.11.3.6: 2017</a></b><br>
# <br>
# 
# 
# 
# 
# 
# <br>
# <br>
# <a id='1'></a>
# ## Wallets:
# <br>
# Concept of Wallets are quite simple, Basically lets say you have 100 dollars note. Where you will put it?, in a wallet right, just like that in Bitcoin wallets you put bitcoins but difference is that bitcoin wallet is digital.
# <br>
# 
# But there is a problem, How can you ensure that ONLY you can access 100 dollar note in your wallet and no one else. 
# 
# <b> To solve the above mentioned problem lets start from the beginning and see  how bitcoin wallet is created. </b>
# <br>
# 
# 
# <b> How you got to the address, your public bitcoin address.</b>
# <br>
# <b> How a wallet is created?</b>
# <br>
# Lets understand how a wallet is created and how it ensures that only you have access to the bitcoins you have
# <br>
# Here are three keywords you need to remember Private_key, Public_key, and Bitcoin_address.
# 
# 
# <b>Step 1:</b>
# <br>
# First of all you get a fixed length of random characters its called PRIVATE_KEY. You need to keep it hidden and dont have to give it to anyone else.  
# <br>
# <br>
# <b> Step 2:</b>
# <br>
# Now that you have a private key what you need to do is pass this private_key to a function. This is a special function which will give you a new key, lets name it Public Key.  
# <br>
# <br>
# public_key=FUNCTION(private_key)
# 
# We will skip how a function is implemented. To know how it is implemented we need to understand cry.
# 
# <br>
# But the point to notice is that it is a <b>one way function</b> and there is no practical way to get  private_key from public key.
# <br>
# 
# <b> Step 3</b>
# <br>
# Now that you have two things <b>Private_key</b> and <b>public_key</b>.
# What we will do now is pass the public_key to another function which gives us another key. Lets name it <b>address</b>.
# <br>
# <b> ADDRESS= Some_other_function(public_key)</b>
# 
# Now Address is something that you can give to anyone.
# If you want anyone to transfer some money intor your wallet  you will provide him this ADDRESS.
# 
# What wallet stores is your private_address and unspent outputs, we will undertand the unspent outputs in the next section.
# 
# Now the question which arises over here how can we use private_key, public_key and address to transfer money and ensure safety of your wallet.
# 
# <b>Summary</b><br>
# So here are few points if you missed anything, you had a private key: a random characters of fixed length, you passed it to a function to get a public key which was passed again to a function to get address. These functions are one way function which means if
# a=function(b)
# then there is no practical way to find <b>b</b> if you know <b>a</b>.
# <br>
# <br>
# <br>
# <br>
# <a id='2'></a>
# ## Transactions:
# <br>
# In this section we will understand how is bitcoins put in a wallet and how can only the wallet holder spends it and how it is transferred.
# <a id='2.1'></a>
# ### How transaction works in a NUTSHELL?
# In transactions there is a concept of input and outputs. 
# 
# In bitcoins each transaction have a number of inputs and outputs.
# In this scenario there are three people: IBAD, READER(you) and amazon
# 
# Now lets say you want to buy a nice watch. 
# So you go to the ecommerce website and it says to transfer 1 bitcoin to their address to get a watch.
# 
# So you will do is open your wallet and transfer 1 btc to Amazon (ecommerce website) to get a watch.
# 
# 
# Unfortunately you have no bitcoins.
# 
# So you came to me and asked me to transfer 100btc to your address and you provide me your "address" you made in the previous section.
# RECALL:    address=Function(public_key)
#                   public_keye=Function(private key)
#                   private key= "Random Characters of Fixed Length" Must be kept hidden
# 
# What is going to happen is this that I will write a transaction and it will look something like this. 
# 
# <b>TRANSACTION#1 by Ibad:</b><br>
# <b>INPUT:</b> Hey miner, I have 1000 bitcoins, Want a proof?. look at OUTPUT#2762
# 
# <b><a id='output123'>OUTPUT#123:</a></b>
# Hey Miner,  This is for the reader, he can spent 100 Bitcoins
# 
# <b><a id='output124'>OUTPUT#124:</a></b>
# Hey Miner, This is for ME, I must be able to spent 900 Bitcoins
# 
# I had 1000 bitcoins, I transferred 100 to you, and got 900 change back to myself.
# .
# Now after writing this transaction I send it to the bitcoin network, After some times a MINER sees this transactions, Validates it(checks it) just like the banker checks your cheque for forgery and approves it. Once approved this transaction will be marked as Confirmed and you can now spend your 100 bitcoins. 
# 
# 
# Now that you have 100 bitcoins now So you want to buy that watch from amazon so what you are going to do is write a transaction.
# 
# <b>TRANSACTION#2 by READER:</b>
# <br>
# <b>INPUT:</b> Hey miner, I have 100 bitcoins, Want a proof?.  look at <a href='#output123'>OUTPUT#123</a>
# 
# 
# <b>OUTPUT#125: </b>
# Hey Miner, This is for the amazon, they can spend 1 Bitcoin.
# 
# <b><a id='output126'>OUTPUT#126:</a></b>
# Hey Miner, This is for me(reader)(you)(as it is written by you hence i am using me), I must be able to spend 99 Bitcoins.
# 
# You had 100 bitcoins you sent 1 bitcoin to amazon and got 99 back to you.
# 
# #### Now  <a href='#output123'>OUTPUT#123</a>  is  SPENT OUTPUT
# 
# ####  <a href='#output126'>OUTPUT#126</a> is unspent output
# 
# #### <a href='output124'>Output#124</a> is unspent output.
# 
# 
# For the next time if you want to do another transaction you will reference <a href='#output126'>Output#126</a> in your INPUT.
# 
# 
# 
# 
# <a id='2.3'></a>
# ## SAMPLE TRANSACTION:
# Lets look at a sample input and sample output and lets see how they help in transactions.
# 
# To make it simple assume there are two people you and me, Now you have some bitcoins with you and want to transfer some of it to me. 
# 
# <b>Scenario:</b> You want to transfer me <b>0.04</b> bitcoins to me.
# <br>
# You currently have: <b>0.06</b> bitcoins.
# 
# 
# Let take a look at sample transaction input First and understand what does it implies.
# 

# In[ ]:


x=test_data.iloc[2].inputs[0]
x["input_pubkey_base58"]="1KEH32noJFb3tiBbWzLZo9nie6C4VhNP7Y"
print (x)
#MODIFIED THE INPUT BECAUSE HEAD VALUE KEEPS ON CHANGING


# There are total of 6 properties over here, we will consider only those that are important to get the basic idea.
# The first property "input_pubkey_base58" is your ADDRESS which in this case is: <b>1KEH32noJFb3tiBbWzLZo9nie6C4VhNP7Y</b>
# 
# The property "input_script_string" contains the reference to previous unspent output from where you are spending.
# 
# 
# inputs are just there for referencing the previous unspent outputs.
# 
# Now lets look at the output for this transaction.

# In[ ]:


x=test_data.iloc[3].outputs[0]
x["output_satoshis"]=4000000
x["output_pubkey_base58"]="1bonesF1NYidcd5veLqy1RZgF4mpYJWXZ"
print (x)
print ("-"*0)
x["output_satoshis"]=1159000
x["output_pubkey_base58"]="1KEH32noJFb3tiBbWzLZo9nie6C4VhNP7Y"
print (x)
#MODIFIED THE INPUT BECAUSE HEAD VALUE KEEPS ON CHANGING


# Recall from above that your address is: 1KEH32noJFb3tiBbWzLZo9nie6C4VhNP7Y
# 
# Here we can see there are two outputs.
# 
# output_satoshis is the amount of bitcoins you are sending. Sathoshi is the smallest unit of bitcoin. just like cents in dollars.
# 
# 1 Bitcoin=100,000,000 satoshis. 
# 
# So the first output says to transfer 4000000 satoshis (or 0.004 bitcoins) to  address output_pubkey_base58': '1bonesF1NYidcd5veLqy1RZgF4mpYJWXZ' which is my address. 
# 
# 
# Now the second output:
# So the second output says to transfer 1159000 satoshis (or 0.01159 bitcoins) to  address output_pubkey_base58': '1KEH32noJFb3tiBbWzLZo9nie6C4VhNP7Y' which is YOUR address. 
# 
# 
# Heres a summary of what happened: I had 0.006 bitcoins with me, I transferred 0.004 to your account,
# Got a change of 0.001159 back to my address.
# 
# But wait I should get 0.002 bitcoins back but I just got 0.001159 back, That's because the the rest of it will be counted as the transaction fees paid to the miner, In bitcoins you can set the transaction fees yourself. There is no limit on what transaction fees you set, you can set it from 0 to (total_amount - sent_amount)
# 
# 
# <a id='2.4'></a>
# ## SECURITY IN TRANSACTION
# 
# Now the most important part is: How can we ensure safety, Since all outputs and inputs are public, so anyone can write a transaction and transfer money from someone unspent output to his address. 
# 
# To ensure security bitcoins utilizes the concept of Digital Signatures.
# Whenever you write a transaction: the input is digitally signed:  To understand what it means take a look at this picture:
# [](https://www.cryptocompare.com/media/1284/digital_signature.png)
# 
# 
# If you are interested in just getting the idea of security in bitcoin you can just read the short description, if you want to get indepth detailed idea then read the long description of security in bitcoin.
# <br><br><br><br>
# <a id='2.4.1'></a>
# <br>
# <b>Short Description:</b>
# <br>
# Lets say you create a transaction to send 100 bitcoins to me, What you are going to do before sending it to miner is this that you will sign it with your private key, YOU will generate a signature with your private key, so
# Signature= Some_Function(private_key)
# 
# But there one property of this signature which will help us ensure security
# Now the miner can verify the signature with your PUBLIC_KEY, 
# 
# isSignatureValid= Verification_function(signature, public_key)
# 
# if isSignatureValid==True then the miner will validate transaction else miner will not accept the transaction.
# <br>
# <br>
# <a id='2.4.2'></a>
# <b>Long Description: </b>
# <br>
# Recall that the proof of having a bitcoin is the unspent output on your address. 
# The output looks something like this:
# 
# 
# 
# 
# 

# In[ ]:


print (test_data.iloc[2].outputs[0])


# Look closely there is a variable named output_script_string, this is a script, an output script which puts on the condition that only the private_key holder of the specified public_key have the rights to spend this output.
# Look at the last command <b>EQUALVERIFY CHECKSIG</b>, This checks the signature whether it matches the address or not.
# 

# Recall that inputs gives reference to the previous output,  In the input you will satisfy the condition put on by the output, to use it. 
# 
# In this way every bitcoin transaction is available to everyone to see but still its secure and immutable. 

# #### There are many more factors in transactions. 
#  https://unglueit-files.s3.amazonaws.com/ebf/05db7df4f31840f0a873d6ea14dcc28d.pdf
#  #### I read few chapters of this book, It contains detailed information about how Bitcoins works.
#  ## I tried to summarize the transaction and wallet chapter of this book. Do give suggestions on where I might be wrong or how can I improve my explanation. 

# <a id='3'></a>
# ## Now lets find something interesting
# 

# 
# ### Lets find the addresses who have the most number of bitcoins

# In[ ]:


q = """
SELECT  o.output_pubkey_base58, sum(o.output_satoshis) as output_sum from 
    `bigquery-public-data.bitcoin_blockchain.transactions`JOIN
    UNNEST(outputs) as o 
    where o.output_pubkey_base58 not in (select i.input_pubkey_base58
    from UNNEST(inputs) as i)
    group by o.output_pubkey_base58 order by output_sum desc limit 1000
"""
print (str(round((bq_assistant.estimate_query_size(q)),2))+str(" GB"))

results2=bq.Run_Query(q, max_gb_scanned=70)
results2["output_sum"]=results2["output_sum"].apply(lambda x: float(x/100000000))


# In[ ]:


import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
objects = results2["output_pubkey_base58"][:10]
y_pos = np.arange(len(objects))
performance = results2["output_sum"][:10]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects, rotation=90)
plt.ylabel('Bitcoins')
plt.title('Bitcoins Addresses Who received Most number of bitcoins')
plt.show()


# <a id='3.1'></a>
# ## The most number of bitcoins someone got in just ONE transactions.
# ### To do this what we are going to do is we will find out top1000 outputs from the data sorted by the amount of bitcoins transacted in a single output.

# In[ ]:


q = """
SELECT  TIMESTAMP_MILLIS((timestamp - MOD(timestamp,
          86400000))) AS day,o.output_pubkey_base58, o.output_satoshis as output_max from 
    `bigquery-public-data.bitcoin_blockchain.transactions`JOIN
    UNNEST(outputs) as o order by output_max desc limit 1000
"""
print (str(round((bq_assistant.estimate_query_size(q)),2))+str(" GB"))
results3=bq.Run_Query(q, max_gb_scanned=56)

#CONVERT SATOSHIS TO BITCOINS
results3["output_max"]=results3["output_max"].apply(lambda x: float(x/100000000))
results3.head()


# ### On 16th December 2011  "1M8s2S5bgAzSSzVTeL7zruvMPLvzSkEAuv".        This address received 500000 bitcoins IN A SINGLE TRANSACTION !!! What a lucky person...

# <a id='3.2'></a>
# ## Lets find out the timeseries plot for highest valued transactions.

# In[ ]:


results4=results3.sort_values(by="day")
layout = go.Layout(title="Time Series of Highest single output transaction")
data = [go.Scatter(x=results4.day, y=results4.output_max)]
fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig)


# ### It is quite clear that transactions of more than 100K bitcoins stopped in 2016.
# ### Maybe because people were transferring using multiple transactions. (using multiple addresses to evade the eyes of hacker on their addresses).

# In[ ]:


results4["day"]=results4["day"].apply(lambda x: x.year)
years_output={}
years_max_output_count={}
for i,x in results4.iterrows():
    if x["day"] not in years_output:
        years_output[x["day"]]=[]
    if x["day"] not in years_max_output_count:
        years_max_output_count[x["day"]]=[]
    years_output[x["day"]].append(x["output_max"])
    years_max_output_count[x["day"]].append(x["output_pubkey_base58"])
years_output_final={}
for x in years_output.keys():
    years_output_final[str(x)]=np.mean(years_output[x])
years_max_output_count_final={}
for x in years_max_output_count.keys():
    years_max_output_count_final[str(x)]=len(years_max_output_count[x])


# <a id='3.3'></a>
# ### Lets find out the average AMOUNT of transaction per year among top 1000 transactions from the data
# 
# ### Note that among the top1000 amount transactions the transaction with lowest amount was done on 19th September 2012!

# In[ ]:


print (results3.iloc[len(results3)-1]["output_max"])
print (results3.iloc[len(results3)-1]["day"])


# In[ ]:


d=Counter(years_output_final)
d.most_common(1)
Create_Bar_plotly(d.most_common(), title="Single Highest Valued Transaction Average Per Year")


# <a id='3.4'></a>
# ## Lets find out which year have most number of top1000 transactions.

# In[ ]:


d=Counter(years_max_output_count_final)
Create_Bar_plotly(d.most_common(), title="Most number of high transaction yearwise")


# ### This means that in 2016 there were just three transactions above 50343 Bitcoins !!!
# ### Moreover in 2018 there was just one transaction above 50343, 
# 
# ## Suprisingly in 2017, due to the high bitcoins price and its popularity there were 199 transactions that were above 50343 btc.
# 
# 
# ### 2012 seems to have majority of top1000 transactions, followed by 2011.
# ## One thing to notice that high amount transaction got quite low in 2013,2014,2015 and very low in 2016. 
# 
# ## It seems that in 2017 when bitcoins price were topped at 20000 dollars the investors sold out their bitcoins resulting in a huge number of transactions.
# 
# 
# ### One thing important to notice is this that among the top1000 amount transactions in bitcoin the LOWEST is 50343.455 Bitcoins that is done on 19th September 2012....

# <a id='3.5'></a>
# ### In 2018 there was just one transaction which made up to the top 1000 transactions list 

# In[ ]:


results4[results4.day==2018]
results3[results3.index==970]


# ### On 7th January 2018, someone just earned 51044 BITCOINS in just a single transaction !!!!.
# 
# ### Lets look at the transactions in 2010 which comes under the top1000 amount transactions.

# In[ ]:


results3.iloc[list(results4[results4.day==2010].index)]


# ### Lets look at the transactions in 2016 which comes under the top1000 amount transactions.

# In[ ]:


results3.iloc[list(results4[results4.day==2016].index)]


# <a id='3.6'></a>
# ## Lets look at the average number of bitcoins transacted per day.
# Note: This also includes the self transactions and changes etc.

# In[ ]:


q = """
SELECT  TIMESTAMP_MILLIS((timestamp - MOD(timestamp,
          86400000))) AS day, avg(o.output_satoshis) as output_avg from 
    `bigquery-public-data.bitcoin_blockchain.transactions`JOIN
    UNNEST(outputs) as o group by day order by output_avg desc
"""
print (str(round((bq_assistant.estimate_query_size(q)),2))+str(" GB"))

results5=bq.Run_Query(q, max_gb_scanned=89)

#CONVERT SATOSHIS TO BITCOINS
results5["output_avg"]=results5["output_avg"].apply(lambda x: float(x/100000000))
results5.head()


# <a id='3.6.1'></a>
# ## Time Series showing the average number of bitcoins transacted per day

# In[ ]:


results6=results5.sort_values(by="day")
layout = go.Layout(title="Time Series of AVERAGE in no of bitcoins transacted per day")
data = [go.Scatter(x=results6.day, y=results6.output_avg)]
fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig)


# ### it is quite clear that with even with the increas in popularity of bitcoins in 2016 and 2017,  the average number of bitcoins transacted per day still remains less than average no of bitcoins transacted in 2010, 2011, and 2012..
# 
# #### The possible reason behind could be the spike in bitcoins price.
# 
# <a id='3.7'></a>
# ### Now lets find out the total number of transactions per day. 
# 
# Note: this also includes the self transactions.

# In[ ]:


q = """
SELECT  TIMESTAMP_MILLIS((timestamp - MOD(timestamp,
          86400000))) AS day, count(o.output_satoshis) as counts from 
    `bigquery-public-data.bitcoin_blockchain.transactions`JOIN
    UNNEST(outputs) as o group by day order by counts
"""
print (str(round((bq_assistant.estimate_query_size(q)),2))+str(" GB"))
results7=bq.Run_Query(q,max_gb_scanned=180)
results7.tail()


# #### On 2nd September 2015 there were total of 2035035 transactions in total that's highest  number of transactions per day !!. Too much load on the miners.
# #### Have a look at the timeseries  for transaction count for each day

# In[ ]:


results7=results7.sort_values(by="day")
layout = go.Layout(title="Time Series of transaction COUNT in Bitcoins")
data = [go.Scatter(x=results7.day, y=results7.counts)]
fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig)


# <a id='3.8'> </a>
# ## SELF Transactions
# ### What if you have a 1000 dollars in your account and you transferred it to your own account :D
# ### That's exactly what is meant by self transactions in bitcoins. You will be surprised to know that how many self transactions were done by bitcoin holders.

# In[ ]:


q = """
SELECT  count(o.output_pubkey_base58) from
    `bigquery-public-data.bitcoin_blockchain.transactions`JOIN
    UNNEST(outputs)as o, UNNEST(inputs) as i where ARRAY_LENGTH(outputs)=1 and
    ARRAY_LENGTH(inputs)=1 and i.input_pubkey_base58=o.output_pubkey_base58
"""
print (str(round((bq_assistant.estimate_query_size(q)),2))+str(" GB"))
results9=bq.Run_Query(q, max_gb_scanned=580)
print (results9)


# ## There were atleast 170352 Self transactions done by bitcoin holders !!!.
# 
# ### Lets find out the top1000 valued self transactions done by bitcoin holders.

# In[ ]:


q = """
SELECT  TIMESTAMP_MILLIS((timestamp - MOD(timestamp,
          86400000))) AS day,o.output_pubkey_base58 as key, o.output_satoshis as price from 
    `bigquery-public-data.bitcoin_blockchain.transactions`JOIN
    UNNEST(outputs)as o, UNNEST(inputs) as i where ARRAY_LENGTH(outputs)=1 and
    ARRAY_LENGTH(inputs)=1 and i.input_pubkey_base58=o.output_pubkey_base58
    order by price desc limit 1000
"""
print (str(round((bq_assistant.estimate_query_size(q)),2))+str(" GB"))
results9=bq.Run_Query(q, max_gb_scanned=588)
#CONVERT SATOSHIS TO BITCOINS
results9["price"]=results9["price"].apply(lambda x: float(x/100000000))
results9.head()


# ### On 29th June 2010 someone just transacted 4000 bitcoins from his own address to his own address.
# 
# 
# ### Lets look at the time series graph for the top1000 priced self transactions (sending your money to your own address) (sending your money to your own account) (sending your money to your own SAME account) 

# In[ ]:


results9=results9.sort_values(by="day")
layout = go.Layout(title="Self Transactions price by bitcoins holders TimeSeries")
data = [go.Scatter(x=results9.day, y=results9.price)]
fig = go.Figure(data=data, layout=layout)

py.offline.iplot(fig, image="png")


# <a id='3.9'></a>
# ### Now lets find out who did most of the transactions

# In[ ]:


QUERY = """
SELECT
    inputs.input_pubkey_base58 AS input_key, count(*)
FROM `bigquery-public-data.bitcoin_blockchain.transactions`
    JOIN UNNEST (inputs) AS inputs
WHERE inputs.input_pubkey_base58 IS NOT NULL
GROUP BY inputs.input_pubkey_base58 order by count(*) desc limit 1000
"""
bq_assistant.estimate_query_size(QUERY)
ndf=bq.Run_Query(QUERY, max_gb_scanned=238)


# In[ ]:


import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
objects = ndf["input_key"][:10]
y_pos = np.arange(len(objects))
performance = ndf["f0_"][:10] 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects, rotation=90)
plt.ylabel('Number of transactions')
plt.title('BITCOIN ADDRESSES WITH MOST NUMBER OF TRANSACTIONS')
plt.show()


# In[ ]:


ndf.iloc[0]


# <b> The highest number of transactions was 1893290 done by the above mentioned address.</b>

# <a id='3.10'></a>
# ## Inactive Addresses: As suggested in comments:
# #### There are some wallets which received some money but never did any of the transactions, 
# #### Here is a concept of bitcoins lost wallets:
# #### lets say you hold a bitcoin wallet, which means you have a private key and a public key. Now assume that you somehow lost your private key. Now all of the bitcoins you have will be gone foreover and NO-ONE can recover them in any case.
# #### In this visualization we will analyze the addresses of bitcoin which received some bitcoin but never used them.
# ## OR more precisely the outputs which were never referenced by another input.
# 
# Note: something given over here is just the analysis from the data given, In bitcoins if there is no condition in the output_script_string then anyone can validate it. Moreover You can look at the query and please do tell me if I am making any mistake in query.

# In[ ]:


q = """
SELECT  TIMESTAMP_MILLIS((timestamp - MOD(timestamp,
          86400000))) AS day,o.output_pubkey_base58, o.output_satoshis as output_max from 
    `bigquery-public-data.bitcoin_blockchain.transactions`JOIN
    UNNEST(outputs) as o
    where o.output_pubkey_base58 not in 
    (
    select i.input_pubkey_base58 from 
    `bigquery-public-data.bitcoin_blockchain.transactions`,
    UNNEST(inputs) as i where i.input_pubkey_base58 is not null
    )
    order by output_max desc limit 10000
"""
print (str(round((bq_assistant.estimate_query_size(q)),2))+str(" GB"))
inactive_wallets=bq.Run_Query(q, max_gb_scanned=69)
inactive_wallets["output_max"]=inactive_wallets["output_max"].apply(lambda x: float(x/100000000))


# <a id=3.10.1></a>
# ### Lets find out the value of unspent outputs with the passage of time

# In[ ]:


inactive_wallets_2=inactive_wallets.sort_values(by="day")
layout = go.Layout(title="Value of unspent outputs with respect to time")
data = [go.Scatter(x=inactive_wallets_2.day, y=inactive_wallets.output_max)]
fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig)


# <a id=3.10.2></a>
# ### How many outputs were there which are not spent yearly??
# ### Lets find out.

# In[ ]:


inactive_wallets_2["day"]=inactive_wallets_2["day"].apply(lambda x: x.year)
years_outputs={}
for i,x in inactive_wallets_2.iterrows():
    if x["day"] not in years_outputs:
        years_outputs[x["day"]]=0
    years_outputs[x["day"]]+=1
years=Counter(years_outputs)
Create_Bar_plotly(years.most_common(),title="Count of outputs never spent with respect to year")


# In[ ]:


yearly_unspent={}
for i,x in inactive_wallets_2.iterrows():
    if x["day"] not in yearly_unspent:
        yearly_unspent[x["day"]]={}
    if x["output_pubkey_base58"] not in yearly_unspent[x["day"]]:
        yearly_unspent[x["day"]][x["output_pubkey_base58"]]=0    
    yearly_unspent[x["day"]][x["output_pubkey_base58"]]+=x["output_max"]


# <a id=3.10.3></a>
# ### These 5 addresses received more than 1k bitcoins but NEVER spent it 

# In[ ]:


yearly=Counter(yearly_unspent[2010])
yearly.most_common()


# <a id=3.10.4></a>
# ### Lets find out the number of bitocins which are NEVER spent, the amount of bitcoins received by the addresses but never spent on anything.

# In[ ]:


np.sum(inactive_wallets["output_max"])


# <a id='3.11'></a>
# ## Let us look at the data of bitcoin prices from 2012 to 2018

# In[3]:


data_f=pd.read_csv("../input/bitstampUSD_1-min_data_2012-01-01_to_2018-03-27.csv")
data_f["Timestamp"]=pd.to_datetime(data_f["Timestamp"], unit='s')
data_f.Timestamp=data_f.Timestamp.apply(lambda x: x.replace(hour=0, minute=0, second=0))
data_f=data_f[["Timestamp", "Weighted_Price"]]
data_f=data_f.drop_duplicates(keep="first")
data_f=data_f.drop_duplicates(subset='Timestamp', keep="last")
data_f.head()


# In[ ]:


import time

import numpy as np
import operator
from collections import Counter
import pandas as pd 
from wordcloud import WordCloud, STOPWORDS
import numpy as np
import operator
import pandas as pd 
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
plt.style.use('fivethirtyeight')


# #### Lets make a query to get a total sum of bitcoins  transacted for every day.

# In[ ]:


q = """
SELECT  TIMESTAMP_MILLIS((timestamp - MOD(timestamp,
          86400000))) as Timestamp, sum(o.output_satoshis) as output_price from 
    `bigquery-public-data.bitcoin_blockchain.transactions`JOIN
    UNNEST(outputs) as o group by timestamp
"""
print (str(round((bq_assistant.estimate_query_size(q)),2))+str(" GB"))
results3=bq_assistant.query_to_pandas(q)
results3["output_price"]=results3["output_price"].apply(lambda x: float(x/100000000))
results3=results3.sort_values(by="Timestamp")
results3.head()


# ### In this query we will find the total count of transactions per day

# In[ ]:


q = """
SELECT  TIMESTAMP_MILLIS((timestamp - MOD(timestamp,
          86400000))) as Timestamp , count(Timestamp) as output_count from 
    `bigquery-public-data.bitcoin_blockchain.transactions` group by timestamp
"""
print (str(round((bq_assistant.estimate_query_size(q)),2))+str(" GB"))
transaction_count=bq_assistant.query_to_pandas(q)
transaction_count=transaction_count.sort_values(by="Timestamp")
transaction_count.head()


# <a id='3.11.1'></a>
# ### Here comes our first plot, Timeseries plot signifying the relation between the spike in bitcoin value, the number of transactions, the total bitcoins transacted per day.
# ### But there are some problems: 
# ### The number of bitcoins transacted per day and the total number of transactions are very different, so we have to somehow make the graph in such a manner that we can see the difference in these attributes with the passage of time.
# 
# ### For scaling purposes we divide the transaction counts by 6
# ### Multiply the bitcoin prize by 10
# ### and finally divide the number of bitcoins transacted per day by 500.
# I need your suggestions over here: Please do tell me if there is more elegant way to scale the values and what more techniques can I use to compare the change in trend.

# In[ ]:


import datetime
def to_unix_time(dt):
    epoch =  datetime.datetime.utcfromtimestamp(0)
    return (dt - epoch).total_seconds() * 1000
import plotly
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.tools as tls
import plotly.graph_objs as go
import plotly.tools as tls
data = [go.Scatter(x=transaction_count.Timestamp, y=transaction_count.output_count/6, name="transaction_count/6"), go.Scatter(x=data_f.Timestamp,y=data_f.Weighted_Price*10, name="BITCOIN_PRICE*10"),
       go.Scatter(x=results3.Timestamp, y=results3.output_price/500, name="transactions price/500")
       
       ]
layout = go.Layout(
    xaxis=dict(
        range=[
        to_unix_time(datetime.datetime(2012, 1, 1)),
            to_unix_time(datetime.datetime(2018, 5, 1))]
    )
)

fig=go.Figure(data=data, layout=layout)
py.iplot(fig)


# ### We can clearly see that with increase in bitcoins prize in the end of 2017, the transaction count also increased and the suprisingly transactions prize(total number of bitcoins transacted per day) remained significantly low.
# ### This implies that although with the recent spike in bitcoins price the number of bitcoin transactions increased but the (number of bitcoins) transacted remained constant.
# ### One more insights we can find from this plot is this that in 2016 the NUMBER OF BITCOINS transacted increased.
# ### One conclusion I can make from this is that in the end of 2017, the bitcoin value spiked upto 15000 dollars, hence there were small transactions instead of bigger transactions as in 2016 and 2012

# In[ ]:


all_months=["","January","February","March","April","May","June","July","August","September","October","November","December"]
def Bitcoin_Price_avg_monthly(year):
    new_data_f=data_f[(data_f['Timestamp']>datetime.date(year,1,1)) & (data_f['Timestamp']<datetime.date(year+1,1,1))]
    new_data_f["Timestamp"]=new_data_f.Timestamp=data_f.Timestamp.apply(lambda x: x.month)
    
    month_dictionary={}
    for i,x in new_data_f.iterrows():
        if x["Timestamp"] not in month_dictionary:
            month_dictionary[int(x["Timestamp"])]=[]
            
        month_dictionary[int(x["Timestamp"])].append(x["Weighted_Price"])
        
    for i in month_dictionary.keys():
        all_sum=month_dictionary[i]
        
        all_sum=float(sum(all_sum))/float(len(all_sum))
        month_dictionary[i]=all_sum
        
    return month_dictionary
    
def Average_transaction_count_monthly(year, average=True,  mode="Price"):
    if mode=="Price":
        new_data_ff=results3[(results3['Timestamp']>datetime.date(year,1,1)) & (transaction_count['Timestamp']<datetime.date(year+1,1,1))]
    else:
        new_data_ff=transaction_count[(transaction_count['Timestamp']>datetime.date(year,1,1)) & (transaction_count['Timestamp']<datetime.date(year+1,1,1))]
    new_data_ff["Timestamp"]=new_data_ff.Timestamp=transaction_count.Timestamp.apply(lambda x: x.month)
    
    month_dictionary={}
    key="output_price"
    if mode!="Price":
        key="output_count"
    for i,x in new_data_ff.iterrows():
        if x["Timestamp"] not in month_dictionary:
            month_dictionary[int(x["Timestamp"])]=[]
        month_dictionary[int(x["Timestamp"])].append(x[key])
    
    for i in month_dictionary.keys():
        all_sum=month_dictionary[i]
        if not average:
            all_sum=int(sum(all_sum))
        else:
            all_sum=float(sum(all_sum))/float(len(all_sum))
        month_dictionary[i]=int(all_sum)  
    return month_dictionary


# In[ ]:


all_months=["","January","February","March","April","May","June","July","August","September","October","November","December"]
from operator import itemgetter
def Compare_Transaction_Price_Yearly(year, average=True, mode="Price",title=""):
    title=title+" "+str(year)
    new_x=Bitcoin_Price_avg_monthly(year)
    new_x2=Average_transaction_count_monthly(year, average=average, mode=mode)
    new_x=Counter(new_x).most_common()
    new_x2=Counter(new_x2).most_common()
    new_x=sorted(new_x, key=itemgetter(0))
    new_x2=sorted(new_x2, key=itemgetter(0))
    for i in range(0,len(new_x)):
        x=list(new_x[i])
        x[0]=all_months[i+1]
        new_x[i]=tuple(x)
        x=list(new_x2[i])
        x[0]=all_months[i+1]
        new_x2[i]=tuple(x)
    
    x0=[x[0] for x in new_x]
    y0=[x[1] for x in new_x]
    x1=[x[0] for x in new_x2]
    y1=[x[1] for x in new_x2]
    plt.figure(figsize=(12,8))
    plt.subplot(1, 2, 1)
    g = sns.barplot( x=x0, y=y0, palette="winter")
    plt.xticks(rotation=90)
    plt.title('Bitcoin average Price monthly '+str(year))
    plt.xlabel("Month")
    plt.ylabel("Price in USD")

    plt.subplot(1, 2, 2)
    g = sns.barplot( x=x1, y=y1, palette="winter")
    plt.xticks(rotation=90)
    plt.title(title)
    plt.ylabel("Transaction_"+mode)
    plt.xlabel("Month")
    plt.tight_layout()


# <a id='3.11.2'></a>
# ### Lets get some more indepth analysis with respect to year
# ### Lets find out the average bitcoin price in usd for 2012 and the number of transactions in 2012
# <a id='3.11.2.1'></a>

# In[ ]:


import warnings
warnings.filterwarnings("ignore")

Compare_Transaction_Price_Yearly(2012,average=False, mode="Count", title="Transaction count of ")


# <a id='3.11.2.2'></a>
# ## Similarly for 2013

# In[ ]:


Compare_Transaction_Price_Yearly(2013,average=False, mode="Count", title="Transaction count of ")


# <a id='3.11.2.3'></a>

# In[ ]:


Compare_Transaction_Price_Yearly(2014,average=False, mode="Count", title="Transaction count of ")


# <a id='3.11.2.4'></a>

# In[ ]:


Compare_Transaction_Price_Yearly(2015,average=False, mode="Count", title="Transaction count of ")


# <a id='3.11.2.5'></a>

# In[ ]:


Compare_Transaction_Price_Yearly(2016,average=False, mode="Count", title="Transaction count of ")


# <a id='3.11.2.6'></a>

# In[ ]:


Compare_Transaction_Price_Yearly(2017,average=False, mode="Count", title="Transaction count of ")


# <a id='3.11.3'></a>
# 
# ### Now lets find out the average no of bitcoins transacted with the average bitcoin price.
# <a id='3.11.3.1'></a>

# In[ ]:


Compare_Transaction_Price_Yearly(2012, mode="Price", title="Average transaction price of ")


# <a id='3.11.3.2'></a>

# In[ ]:


Compare_Transaction_Price_Yearly(2013, mode="Price", title="Average transaction price of ")


# <a id='3.11.3.3'></a>

# In[ ]:


Compare_Transaction_Price_Yearly(2014, mode="Price", title="Average transaction price of ")


# <a id='3.11.3.4'></a>

# In[ ]:


Compare_Transaction_Price_Yearly(2015, mode="Price", title="Average transaction price of ")


# <a id='3.11.3.5'></a>

# In[ ]:


Compare_Transaction_Price_Yearly(2016, mode="Price", title="Average transaction price of ")


# <a id='3.11.3.6'></a>

# In[ ]:



Compare_Transaction_Price_Yearly(2017, mode="Price", title="Average transaction price of ")


# ## Please give your feedback and do tell me I am an wrong anywhere conceptually/ logically
# ## Upvote if you like it.

# In[ ]:


end_time=time.time()
print ("TIME TAKEN for the kernel")
print (end_time-start_time)

