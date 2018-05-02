
# coding: utf-8

# I will do some explorations through the Loan Club Data. 
# 
# English is not my native language, so sorry about if you see any error

# Do you wanna see anothers interesting dataset analysis? <a href="https://www.kaggle.com/kabure/kernels">Click here</a> <br>
# Please, give your feedback and if you like this Kernel  <b>votes up </b> =)
# 

# <b>About the dataset</b> <br>
# These files contain complete loan data for all loans issued through the 2007-2015, including the current loan status (Current, Late, Fully Paid, etc.) and latest payment information. The file containing loan data through the "present" contains complete loan data for all loans issued through the previous completed calendar quarter. Additional features include credit scores, number of finance inquiries, address including zip codes, and state, and collections among others. The file is a matrix of about 890 thousand observations and 75 variables. A data dictionary is provided in a separate file.

# <h2> Importing the Librarys </h2> 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

#To plot figs on jupyter
get_ipython().run_line_magic('matplotlib', 'inline')
# figure size in inches
rcParams['figure.figsize'] = 8,6


# <h2> Importing our dataset</h2> 

# In[ ]:


df_loan = pd.read_csv("../input/loan.csv",low_memory=False)


# Looking the infos of our dataset

# In[ ]:


df_loan.info()


# <h2> Knowing our data </h2> 

# In[ ]:


df_loan.head()


# <h2> Column names </h2> 

# In[ ]:


print(df_loan.columns)


# <h2> Unique values </h2> 

# In[ ]:


#Let's see the data shape and NaN values
print(df_loan.shape)
print(df_loan.isnull().sum().value_counts())


# I will hand with Na's later

# <h2>Let's start by the distribuition of the LOAN AMOUNT  </h2>

# In[ ]:


#I will start looking the loan_amnt column
plt.figure(figsize=(12,6))

plt.subplot(121)
g = sns.distplot(df_loan["loan_amnt"])
g.set_xlabel("", fontsize=12)
g.set_ylabel("Frequency Dist", fontsize=12)
g.set_title("Frequency Distribuition", fontsize=20)

plt.subplot(122)
g1 = sns.violinplot(y="loan_amnt", data=df_loan, 
               inner="quartile", palette="hls")
g1.set_xlabel("", fontsize=12)
g1.set_ylabel("Amount Dist", fontsize=12)
g1.set_title("Amount Distribuition", fontsize=20)

plt.show()


# <h2>Another interesting value to a Loan is the interest rate. Let's look this colum: </h2>

# In[ ]:


df_loan['int_round'] = df_loan['int_rate'].round(0).astype(int)

plt.figure(figsize = (10,8))

#Exploring the Int_rate
plt.subplot(211)
g = sns.distplot(np.log(df_loan["int_rate"]))
g.set_xlabel("", fontsize=12)
g.set_ylabel("Distribuition", fontsize=12)
g.set_title("Int Rate Log distribuition", fontsize=20)

plt.subplot(212)
g1 = sns.countplot(x="int_round",data=df_loan, 
                   palette="Set2")
g1.set_xlabel("Int Rate", fontsize=12)
g1.set_ylabel("Count", fontsize=12)
g1.set_title("Int Rate Normal Distribuition", fontsize=20)

plt.subplots_adjust(wspace = 0.2, hspace = 0.6,top = 0.9)

plt.show()


# We can clearly see that the Int rate have a interesting distribuition

# <h1>Some exploration of loan_status</h1><br>
# Understanding the default
# 

# In[ ]:


df_loan.loc[df_loan.loan_status ==             'Does not meet the credit policy. Status:Fully Paid', 'loan_status'] = 'NMCP Fully Paid'
df_loan.loc[df_loan.loan_status ==             'Does not meet the credit policy. Status:Charged Off', 'loan_status'] = 'NMCP Charged Off'


# <h2>Loan Status Distribuition</h2>

# In[ ]:


print(df_loan.loan_status.value_counts())

plt.figure(figsize = (12,14))

plt.subplot(311)
g = sns.countplot(x="loan_status", data=df_loan)
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_xlabel("", fontsize=12)
g.set_ylabel("Count", fontsize=15)
g.set_title("Loan Status Count", fontsize=20)

plt.subplot(312)
g1 = sns.boxplot(x="loan_status", y="total_acc", data=df_loan)
g1.set_xticklabels(g1.get_xticklabels(),rotation=45)
g1.set_xlabel("", fontsize=12)
g1.set_ylabel("Total Acc", fontsize=15)
g1.set_title("Duration Count", fontsize=20)

plt.subplot(313)
g2 = sns.violinplot(x="loan_status", y="loan_amnt", data=df_loan)
g2.set_xticklabels(g2.get_xticklabels(),rotation=45)
g2.set_xlabel("Duration Distribuition", fontsize=15)
g2.set_ylabel("Count", fontsize=15)
g2.set_title("Loan Amount", fontsize=20)

plt.subplots_adjust(wspace = 0.2, hspace = 0.7,top = 0.9)

plt.show()


# <h2>ISSUE_D</h2>
# 
# Going depth in the default exploration to see the amount and counting though the <b>ISSUE_D </b>,<br>
# that is: <i><b> The month which the loan was funded</b></i>

# In[ ]:


df_loan['issue_month'], df_loan['issue_year'] = df_loan['issue_d'].str.split('-', 1).str


# In[ ]:


months_order = ["Jan", "Feb", "Mar", "Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
df_loan['issue_month'] = pd.Categorical(df_loan['issue_month'],categories=months_order, ordered=True)

#Issue_d x loan_amount
plt.figure(figsize = (14,6))

g = sns.pointplot(x='issue_month', y='loan_amnt', 
                  data=df_loan, 
                  hue='loan_status')
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_xlabel("Duration Distribuition", fontsize=15)
g.set_ylabel("Mean amount", fontsize=15)
g.legend(loc='best')
g.set_title("Loan Amount by Months", fontsize=20)
plt.show()


# Looking the years

# In[ ]:


plt.figure(figsize = (14,6))
#Looking the count of defaults though the issue_d that is The month which the loan was funded
g = sns.countplot(x='issue_year', data=df_loan,
                  hue='loan_status')
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_xlabel("Dates", fontsize=15)
g.set_ylabel("Count", fontsize=15)
g.legend(loc='upper left')
g.set_title("Analysing Loan Status by Years", fontsize=20)
plt.show()


# <h2> Taking a look of Default counting  through the Years column </h2>

# In[ ]:


plt.figure(figsize = (14,6))
#Looking the count of defaults though the issue_d that is The month which the loan was funded
g = sns.countplot(x='issue_year', data=df_loan[df_loan['loan_status'] =='Default'])
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_xlabel("Dates", fontsize=15)
g.set_ylabel("Count", fontsize=15)
g.legend(loc='upper left')
g.set_title("Analysing Defaults Count by Time", fontsize=20)
plt.show()


# Interesting, maybe anything happens in Dec'14. It's the most frequent Defaults in our data.

# <h2> Now let's take a look on the crosstab using a heatmap to better see this</h2>

# In[ ]:


#Exploring the loan_status x purpose
purp_loan= ['purpose', 'loan_status']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_loan[purp_loan[0]], df_loan[purp_loan[1]]).style.background_gradient(cmap = cm)


# In[ ]:


loan_grade = ['loan_status', 'grade']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_loan[loan_grade[0]], df_loan[loan_grade[1]]).style.background_gradient(cmap = cm)


# In[ ]:


loan_home = ['loan_status', 'home_ownership']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_loan[loan_home[0]], df_loan[loan_home[1]]).style.background_gradient(cmap = cm)


# In[ ]:


#Looking the 'verification_status' column that is the Indicates 
#if the co-borrowers' joint income was verified by LC, not verified, or if the income source was verified
loan_verification = ['loan_status', 'verification_status']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_loan[loan_verification[0]], df_loan[loan_verification[1]]).style.background_gradient(cmap = cm)


# <h2>Looking verification status using plotly library</h2>

# In[ ]:


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
from collections import Counter

#First plot
trace0 = go.Bar(
    x = df_loan['verification_status'].value_counts().index.values,
    y = df_loan['verification_status'].value_counts().values,
    marker=dict(
        color=df_loan['verification_status'].value_counts().values,
        colorscale = 'Viridis'
    ),
)

data = [trace0]

layout = go.Layout(
    yaxis=dict(
        title='Count'
    ),
    xaxis=dict(
        title='Status'
    ),
    title='Verification Status Count'
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='verification-bar')


# <h2>Looking the column INSTALLMENT that is: </h2> <br>
# <i>The monthly payment owed by the borrower if the loan originates.</i>

# In[ ]:


sns.distplot(df_loan['installment'])
plt.show()


# <h2>Crossing the Loan Status and Installment</h2>

# In[ ]:


plt.figure(figsize = (12,6))

g = sns.violinplot(x='loan_status', y="installment",
                   data=df_loan)
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_xlabel("", fontsize=12)
g.set_ylabel("Installment Dist", fontsize=15)
g.set_title("Loan Status by Installment", fontsize=20)

plt.show()


# In[ ]:


#Exploring the loan_status x Application_type
loan_application = ['loan_status', 'application_type']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_loan[loan_application[0]], df_loan[loan_application[1]]).style.background_gradient(cmap = cm)


# <h2>Distribuition of Application_tye thought the Loan Amount and Interest Rate</h2>

# In[ ]:


plt.figure(figsize = (12,14))
#The amount and int rate x application_type 
plt.subplot(211)
g = sns.violinplot(x="application_type", y="loan_amnt",data=df_loan, 
            palette="hls")
g.set_title("Application Type - Loan Amount", fontsize=20)
g.set_xlabel("", fontsize=15)
g.set_ylabel("Loan Amount", fontsize=15)

plt.subplot(212)
g1 = sns.violinplot(x="application_type", y="int_rate",data=df_loan,
               palette="hls")
g1.set_title("Application Type - Interest Rate", fontsize=20)
g1.set_xlabel("", fontsize=15)
g1.set_ylabel("Int Rate", fontsize=15)

plt.subplots_adjust(wspace = 0.4, hspace = 0.4,top = 0.9)

plt.show()


# <h2>Looking the Home Ownership by Loan_Amount</h2>

# In[ ]:


plt.figure(figsize = (10,6))

g = sns.violinplot(x="home_ownership",y="loan_amnt",data=df_loan,
               kind="violin",
               split=True,palette="hls",
               hue="application_type")
g.set_title("Homer Ownership - Loan Distribuition", fontsize=20)
g.set_xlabel("", fontsize=15)
g.set_ylabel("Loan Amount", fontsize=15)

plt.show()


# <h2> Looking the Purpose distribuition  </h2>

# In[ ]:


# Now will start exploring the Purpose variable
print("Purposes count description: ")
print(pd.crosstab(df_loan.purpose, df_loan.application_type))

plt.figure(figsize = (12,8))

plt.subplot(211)
g = sns.countplot(x="purpose",data=df_loan,
                  palette='hls')
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_title("Application Type - Loan Amount", fontsize=20)
g.set_xlabel("", fontsize=15)
g.set_ylabel("Loan Amount", fontsize=15)

plt.subplot(212)
g1 = sns.violinplot(x="purpose",y="loan_amnt",data=df_loan,
               hue="application_type", split=True)
g1.set_xticklabels(g1.get_xticklabels(),rotation=45)
g1.set_title("Application Type - Loan Amount", fontsize=20)
g1.set_xlabel("", fontsize=15)
g1.set_ylabel("Loan Amount", fontsize=15)

plt.subplots_adjust(wspace = 0.2, hspace = 0.8,top = 0.9)
plt.show()


# <h2>Looking the Grade<h2>

# I will explore some variables.<br>
# the first variable I will explore is GRADE.<br>
# description of grade: <b>LC assigned loan grade</b>

# In[ ]:


fig, ax = plt.subplots(3,1, figsize=(14,10))
sns.boxplot(x="grade", y="loan_amnt", data=df_loan,
            palette="hls",ax=ax[0], hue="application_type", 
            order=["A",'B','C','D','E','F', 'G'])
sns.violinplot(x='grade', y="int_rate",data=df_loan, 
              hue="application_type", palette = "hls", ax=ax[1], 
            order=["A",'B','C','D','E','F', 'G'])
sns.lvplot(x="sub_grade", y="loan_amnt",data=df_loan, 
               palette="hls", ax=ax[2])

plt.show()


# Very very inteWe can clearly see difference patterns between Individual and Joint applications

# <h1>Let's look the Employment title Distribuition </h1>

# In[ ]:


#First plot
trace0 = go.Bar(
    x = df_loan.emp_title.value_counts()[:40].index.values,
    y = df_loan.emp_title.value_counts()[:40].values,
    marker=dict(
        color=df_loan.emp_title.value_counts()[:40].values
    ),
)

data = [trace0]

layout = go.Layout(
    yaxis=dict(
        title='Count'
    ),
    xaxis=dict(
        title='Employment name'
    ),
    title='TOP 40 Employment Title'
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='emp-title-bar')


# <h2>And, whats the most frequent Title in our Dataset? </h2>

# In[ ]:


#First plot
trace0 = go.Bar(
    x = df_loan.title.value_counts()[:40].index.values,
    y = df_loan.title.value_counts()[:40].values,
    marker=dict(
        color=df_loan.title.value_counts()[:40].values
    ),
)

data = [trace0]

layout = go.Layout(
    yaxis=dict(
        title='Count'
    ),
    xaxis=dict(
        title='Employment name'
    ),
    title='TOP 40 Employment Title'
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='emp-title-bar')


# <h2>Emp lenght crossed by some columns</h2>

# In[ ]:


# emp_lenght description: 
# Employment length in years. Possible values are between 0 and 10 where 0 means 
# less than one year and 10 means ten or more years. 

print(pd.crosstab(df_loan["emp_length"], df_loan["application_type"]))

fig, ax = plt.subplots(2,1, figsize=(12,10))
g = sns.boxplot(x="emp_length", y="int_rate", data=df_loan,
              palette="hls",ax=ax[0],
               order=["n/a",'< 1 year','1 year','2 years','3 years','4 years', '5 years',
                      '6 years', '7 years', '8 years','9 years','10+ years'])

z = sns.violinplot(x="emp_length", y="loan_amnt",data=df_loan, 
               palette="hls", ax=ax[1],
               order=["n/a",'< 1 year','1 year','2 years','3 years','4 years', '5 years',
                      '6 years', '7 years', '8 years','9 years','10+ years'])
               
plt.legend(loc='upper left')
plt.show()


# Interesting! We can see that the years do not influence the interest rate but it have a difference considering the loan_amount

# <h2>Terms column</h2>

# In[ ]:


print('Term x application type Description')
print(pd.crosstab(df_loan.term, df_loan.application_type))

#First plot
trace0 = go.Bar(
    x = df_loan.term.value_counts().index.values,
    y = df_loan.term.value_counts().values,
    marker=dict(
        color=df_loan.term.value_counts().values
    ),
)

data = [trace0]

layout = go.Layout(
    yaxis=dict(
        title='Count'
    ),
    xaxis=dict(
        title='Term name'
    ),
    title='Term Distribuition'
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='Term-bar')


# <h2>Looking the heatmap cross tab of Adress State x Loan Status<h2>

# In[ ]:


#Exploring the State Adress x Loan Status
adress_loan = ['addr_state', 'loan_status']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_loan[adress_loan[0]], df_loan[adress_loan[1]]).style.background_gradient(cmap = cm)


# <h1> Someone can tell me about plot the map using the initials? 
