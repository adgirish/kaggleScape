
# coding: utf-8

# # Kiva Data Exploration and Modeling
# ---
# 
# In this kernel I will focus on analysis of the data found in the `kiva_loans.csv` file. This analysis does not particularly address the problem statement for the competition but rather will seek to understand the data though Exploratory Data Analysis and Feature Engineering.
# 
# ** This is work in progress feel free to check back in later to see updates **. This version takes a deep dive into the loan information to see what we can learn about funding of these [loans](#loan_amounts) and then proceeds to explore borrower [gender](#gender). This is followed by analysis of the [lenders](#lender) (kiva.org users) and then analysis of the [loan structure](#structure) and [loan use](#loan_use). At the end of this kernel I finally start diving into the [location data](#location).
# 
# ## TL;DR
# 
# This kernel is getting long :). I will collect some of the interesting observations up here. 
# 
# * The majority of loans are asking for below \$10k with around \$1,000 being the mean loan but the most frequently applied for loan is \$225. So the data is heavily skewed.
# * Around 7% of loans are not funded.
# * The strongest predictor of full funding is the loan amount - bigger loans have lower chance of funding.
# * There are gender differences in loan funding - but these could be due to gender differences in loan amounts and types of loans.
# * Males have higher chance of having their loan not being funded, as do teams that contain lower ratio of females to males.
# * The majority of loans are funded within the first week, with another bump of funding at the one month mark.
# * The time from posting to funding is slower for males.
# * On average the majority of loans are funded by between 1-30 lenders.
# * The average amount of funding committed per lender is around \$10 to \$50. 
# * Male teams on average have longer repayment intervals (16 months) than females (13 months). 
# * For loans with terms longer than 5 years there appear to be a linear relationship between loan term and loan amount.
# * The overall majority of loans are repaid either montly or irregularly. However for male loans monthly and bullet payments are the preferred terms. 
# * From my exploration of the MPI region location file at least the GPS data in that file is mixed up. However, I have been assured that MPI and location names are correct.  
# 

# In[1]:


PUBLISH = True

#import graphing and data frame libs
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from wordcloud import WordCloud
from mpl_toolkits.basemap import Basemap

# statistics :)
from scipy.stats import ttest_ind, probplot

# we will do a little machine learning here - it is kaggle after all
from sklearn.ensemble import RandomForestClassifier

#ignore warnings
import warnings
warnings.filterwarnings('ignore')
#pd.options.display.max_rows = 999

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
plt.rcParams['figure.figsize'] = [14.0, 6.0]
plt.rcParams.update({'font.size': 14})


# In[2]:


# load the data
kiva_loans = pd.read_csv("../input/kiva_loans.csv")

# dont trust the column metrics that they give you - there are null values hiding in here!
kiva_loans.isnull().sum()


# # What is in the data?
# 
# We have:
# * [Loan amounts](#loan_amounts): Funded and loan amount - these are dollar values of how much was loaned and how much was contributed by kiva.org
# * [Loan use](#loan_use): Activity, sector, use, tags - these deal with what the loan will be used for
# * [Location data](#location): Country, region, and GPS data - geographical information 
# * [Loan structure](#structure): Term and repayment interval 
# * [Gender](#gender): Demographic about the borrower
# * [Lender information](#lender): Lender count and information about when loan was posted and funded. Then there is also information about the local loan partner and the time when the money was disbursed to the borrower 
# 

# # Loan amounts <a id="loan_amounts"></a>
# ---
# I will start out by exploring the information that is in the loan amount part of the dataset.
# 
# First a couple histograms that showing the loan amounts. There are a couple of extreme outliers that do not seem to fit the bill of microloans in my opinion, so I am truncating the dataset first at \$50,000 and then at \$10,000 for ease of visualization. 
# 
# The loan amounts are in dollars and it is interesting to see that people are asking for even amounts like \$1,000 - \$1,500 - \$2,000 and so on even though that would probably translate to an odd amount in their local currency. This brings up an interesting question about repayment - I wonder if people have to repay in their local currency or in dollars? This could severely affect people in countries with large inflation.
# 
# Also, it seems like the max amount that you can ask for easily is \$10,000 since there are very few loans that go over that limit.

# In[3]:


plt.figure(figsize=[14,14])

plt.subplot(211)
plt.hist(x = kiva_loans.loc[kiva_loans['loan_amount'] < 50000, 'loan_amount'], 
         color = ['#3CB371'], bins = 499)
plt.title('Loan amount histogram for loans less than $50,000 (log y scale)')
plt.xlabel('Loan amount ($)')
plt.ylabel('Number of people applying (log scale)')
plt.yscale('log', nonposy='clip')

plt.subplot(212)
plt.hist(x = kiva_loans.loc[kiva_loans['loan_amount'] < 10000, 'loan_amount'], 
         color = ['#3CB371'], bins = 99)
plt.title('Loan amount histogram for loans less than $5,000')
plt.xlabel('Loan amount ($)')
plt.ylabel('Number of people applying')

sns.despine()
print ("Mean loan amount is ${}, median ${}, mode ${}".format(kiva_loans['loan_amount'].mean(), kiva_loans['loan_amount'].median(), kiva_loans['loan_amount'].mode()[0]))


# In[9]:


fig = plt.figure()
ax1 = fig.add_subplot(121)
probplot(kiva_loans['loan_amount'], plot=ax1)

ax2 = fig.add_subplot(122)
probplot(kiva_loans['loan_amount'], plot=ax2)
ax2.set_yscale('log', nonposy='clip')
ax2.set_ylim([10, 100000])



# ### Partially funded loans
# 
# Now lets see what is going on with unfunded or partially funded loans. First I calculate the delta between loan amount and the funded amount. Then I check that delta against the loan amount to see if it was partially unfunded or fully unfunded. 
# 
# Since only a very small percentage of loans are fully unfunded and to make the text more concise I will refer to partially and unfunded loans collectively as **partially funded loans**. 
# 
# For this histogram I am again using log on the y-axis to better visualize the low occurrences and I am plotting the full ask behind (not stacked) for comparison. 

# In[ ]:


kiva_loans['ask_fund_delta'] = kiva_loans['loan_amount'] - kiva_loans['funded_amount']
#kiva_loans.loc[kiva_loans['ask_fund_delta'] > 0, 'ask_fund_delta']
kiva_loans['fully_unfunded'] = (kiva_loans['loan_amount'] == kiva_loans['ask_fund_delta'])
kiva_loans['partially_unfunded'] = (kiva_loans['ask_fund_delta'] > 0)

no_of_partially_funded = len(kiva_loans.loc[kiva_loans['ask_fund_delta'] > 0, 'ask_fund_delta'])
no_of_fully_unfunded   = kiva_loans['fully_unfunded'].sum()

print ("{:,} people have non- or partially funded loans ({:.1f}% of total loans). Of these {:,} are fully unfunded ({:.1f}% of partial, {:.1f}% of total)".format(no_of_partially_funded, 
         no_of_partially_funded/len(kiva_loans)*100.0,
         no_of_fully_unfunded,
         no_of_fully_unfunded/no_of_partially_funded*100.0,
         no_of_fully_unfunded/len(kiva_loans)*100.0,
        )
      )

plt.hist(x = kiva_loans.loc[kiva_loans['loan_amount'] < 50000, 'loan_amount'], 
         color = ['#FF6347'], bins = 500)

plt.hist(x = kiva_loans.loc[kiva_loans['ask_fund_delta'] > 0, 'ask_fund_delta'], 
         color = ['#3CB371'], bins = 500)

plt.title('Difference in ask and funded amount')
plt.xlabel('Difference ($)')
plt.ylabel('Number of people affected (log scale)')
plt.yscale('log', nonposy='clip')

sns.despine()


# Now it could be that these partially funded loan are recently posted loans that have not had a chance to be funded yet.
# 
# Lets plot the difference in ask and funded amount over time. To do this I set the index of the dataframe as the date when the loan was posted on kiva.org and then we group by week and plot both the difference in ask and funded amount and the total ask. 
# 
# From this it is clear that while there is a small uptick in unfunded amounts at the very end - there is a general trend for non-funding that is pretty consistent over the 3+ year period. So any analysis of partially funded loans may want to exclude loans made after May 2017.

# In[ ]:


kiva_loans.index = pd.to_datetime(kiva_loans['posted_time'])
ax = kiva_loans['ask_fund_delta'].resample('w').sum().plot()
ax = kiva_loans['loan_amount'].resample('w').sum().plot()
ax.set_ylabel('Amount ($)')
ax.set_xlabel('')
ax.set_xlim((pd.to_datetime(kiva_loans['posted_time'].min()), 
             pd.to_datetime(kiva_loans['posted_time'].max())))
ax.legend(["loan unfunded", "loan ask"])
plt.title('Loan ask and loan unfunded over time')

sns.despine()
plt.show()


# In[ ]:


# sanity check - above shows $3 mio per week is average ask. Is that reasonable? 
# use timeperiod of 3 years and 7 months - could probably figure out exactly how many
# weeks are covered in this dataset - but this is just a back of the envelope check.
print(" *"*30)
print("Sanity check: total loan amount asked for over time period was ${:,.0f} - that is approximately ${:,.0f} per week.".format(kiva_loans['loan_amount'].sum(),
         kiva_loans['loan_amount'].sum()/((3+7.0/12.0) * 52)
        )
     )
print(" *"*30)


# ### Funding biases
# 
# Now they did not ask us to do any machine learning on this data set but this is a nice big dataset with lots of information so there is a good chance here to look at what factors affect if a grant is funded or not. For this I dont want to look at the ful dataset - just at a subset of the data:
# * Loan amount < \$10k - exclude the outliers
# * Date posted < May 2017 - avoid grants posted towards the end of the dataset
# * Borrower gender is not null - gender is important it seems so I am excluding nulls here
# Also, I will look at just the following features:
# * Sector
# * Loan amount
# * Country
# * Borrower gender
# 
# Will use random forrest classifier since it allows interrogation of feature importances.

# In[ ]:


# Calculate percent female on loan - works because counting word male also counts female :)
kiva_loans['percent_female'] = kiva_loans['borrower_genders'].str.count('female') /                                kiva_loans['borrower_genders'].str.count('male')

kiva_loans['team_gender'] = 'mixed'
kiva_loans.loc[kiva_loans['percent_female'] == 1, 'team_gender'] = 'female'
kiva_loans.loc[kiva_loans['percent_female'] == 0, 'team_gender'] = 'male'
#kiva_loans['team_gender'].value_counts()


# now create training sub set

# drop all the nans
kiva_train = kiva_loans[['partially_unfunded', 'loan_amount', 'date', 'percent_female',
                        'sector', 'country', 'ask_fund_delta', 'repayment_interval']].dropna(axis=0, how='any')
print ("After dropna we still have {:,} of the {:,} partially unfunded".format(kiva_train['partially_unfunded'].sum(), kiva_loans['partially_unfunded'].sum()))

# limit the loan amount
kiva_train = kiva_train.drop(kiva_train[kiva_train.loan_amount > 10000].index)
print ("After loan limitation we still have {:,} of the {:,} partially unfunded".format(kiva_train['partially_unfunded'].sum(), kiva_loans['partially_unfunded'].sum()))

# limit by date to avoid loading up on partially unfunded loans that just 
# did not have enough time to get funded
kiva_train = kiva_train.drop(kiva_train[kiva_train.date >= '2017-05-01'].index)
print ("After date limitation we still have {:,} of the {:,} partially unfunded".format(kiva_train['partially_unfunded'].sum(), kiva_loans['partially_unfunded'].sum()))


# Limiting the dataset has reduced the number of partially funded loans - but we still have almost 40k cases in this class - so we should be able to learn something.
# 
# Below I am looking at the effect of loan amount and percent female (calculated from the `borrower_gender` variable).

# In[ ]:


# first explore the training set to see if we can see obvious differences 
# between funded and non funded
fig, (maxis1, maxis2) = plt.subplots(1, 2)

maxis1.set_title("Loan amount")
maxis2.set_title("Percent female")

sns.boxplot(x="partially_unfunded", y="loan_amount", data=kiva_train, 
            ax = maxis1, showmeans = True, meanline = True)
sns.boxplot(x="partially_unfunded", y="percent_female", data=kiva_train, 
            ax = maxis2, showmeans = True, meanline = True)

sns.despine()
plt.show()


# Now lets look at the top 10 countries that were funded and the top 10 that were not funded to see if there is a difference there.

# In[ ]:


# Definitely looks like there are some significant differences there. 
# Now lets look at countries.
fig, (maxis1, maxis2) = plt.subplots(2, 1, figsize=[14,12])

maxis1.set_title("Funded loans - top countries")
maxis2.set_title("Partially funded loans - top countries")

sns.barplot(x=kiva_train[kiva_train['partially_unfunded'] == False].country.value_counts().head(10).index, 
            y=kiva_train[kiva_train['partially_unfunded'] == False].country.value_counts().head(10), ax = maxis1)

sns.barplot(x=kiva_train[kiva_train['partially_unfunded'] == True].country.value_counts().head(10).index, 
            y=kiva_train[kiva_train['partially_unfunded'] == True].country.value_counts().head(10), ax = maxis2)

maxis1.set_ylabel('Number of funded loans')
maxis2.set_ylabel('Number of partially funded loans')

sns.despine()
plt.show()


# And the same for the top 10 sectors

# In[ ]:


# Same for sector
fig, (maxis1, maxis2) = plt.subplots(2, 1, figsize=[14,14])

maxis1.set_title("Funded loans - top sectors")
maxis2.set_title("Partially funded loans - top sectors")
sns.barplot(x=kiva_train[kiva_train['partially_unfunded'] == False].sector.value_counts().head(10).index, 
            y=kiva_train[kiva_train['partially_unfunded'] == False].sector.value_counts().head(10), ax = maxis1)

sns.barplot(x=kiva_train[kiva_train['partially_unfunded'] == True].sector.value_counts().head(10).index, 
            y=kiva_train[kiva_train['partially_unfunded'] == True].sector.value_counts().head(10), ax = maxis2)

maxis1.set_ylabel('Number of funded loans')
maxis2.set_ylabel('Number of partially funded loans')

for tick in maxis1.get_xticklabels():
    tick.set_rotation(10)

for tick in maxis2.get_xticklabels():
    tick.set_rotation(10)

sns.despine()
plt.show()


# There are too many countries to create meaningful dummy variables for each country. So I will just take the top 20 most loaned to countries and name the rest 'non-top20-country'. This is just a ranking based on how many loans a country has in the training set - not which country is good or bad :)

# In[ ]:


mostfrequentcountries = kiva_train['country'].value_counts().nlargest(20).keys()
kiva_train.loc[(kiva_train['country'].isin(mostfrequentcountries)==False), 'country'] = "non-top20-country"

# here is the value counts for the full dataset
print ("Country frequencies\n", kiva_train['country'].value_counts())


# Now create dummies for country and sector and put them in a NumPy array that can be sent to a random forest.

# In[ ]:


# now create dummies from sector and country
kiva_train_final = pd.concat([pd.get_dummies(kiva_train['country']), 
                              pd.get_dummies(kiva_train['sector']),
                              pd.get_dummies(kiva_train['repayment_interval']),
                             ], axis = 1)

kiva_train_final['loan_amount'] = kiva_train['loan_amount']
kiva_train_final['percent_female'] = kiva_train['percent_female']

kiva_train_final.sample(5)


# In[ ]:


np_train_features = kiva_train_final.as_matrix()
print ("training features shape", np_train_features.shape)

np_train_labels = kiva_train['partially_unfunded'].astype(int)
print ("training labels shape", np_train_labels.shape)

features = kiva_train_final.columns


# Classifier time. This one may take a while to run and will likely not tell us much that we dont already know from the plots above - but I do like running compute intensive models and looking at spinners.  

# In[ ]:


# this one is slow - only run it when we are creating the final kernel
if PUBLISH:
    rfc = RandomForestClassifier(n_estimators=50, min_samples_split=4)

    rfc.fit(np_train_features, np_train_labels)
    score = rfc.score(np_train_features, np_train_labels)

    print("Accuracy on full set: {:0.2f}".format(score*100))
    print(" *"*25)
    print("Top 20 feature importances in this model:")

    importances = rfc.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(20):
        print("%0.2f%% %s" % (importances[indices[f]]*100, features[indices[f]]))


# ## Conclusions from funding data and modeling
# ---
# 
# The majority of loans are asking for below \$10k with around \$1,000 being the mean loan but the most frequently applied for loan is \$225. So the data is heavily skewed. Around 7% of loans are not funded. The strongest predictor of full funding is the loan amount - bigger loans have lower chance of funding. In addition people from El Salvador and Colombia are less likely to be funded while people from the Philipines are more likely to be funded. As for sectors education is more likely to be funded while retail is less likely to be funded. Finally, males are less likely to get funded than females - but they are also asking for larger loans.
# 
# Please note that these patterns are effected by the people that log into kiva.org and decide who to fund in combination with the people who decide to ask for funding on kiva.org. I do not suggest that these patterns are part of a conspiracy :). 

# # Gender <a id="gender"></a>
#  ---
# Above, I created a new variable that calculates the percent of the borrower for a loan that is female. Lets take a closer look at this and see what we can learn. The first thing of interest is that only 20% of the loans on kiva.org are requested by teams that consist of only males. This is likely due to the much easier access to capital by males in these locations - i.e. males have other opportunities for obtaining money.

# In[ ]:


print("Percent loans requested by only female teams {:.2f}%".format(kiva_loans['team_gender'].value_counts()['female']/len(kiva_loans)*100  ))

print("Percent loans requested by only male teams {:.2f}%".format(kiva_loans['team_gender'].value_counts()['male']/len(kiva_loans)*100  ))


# A KDE can be useful to visualize the percent female data. When looking at all the data (in green below) there is such a preponderance of all male and all female teams or individuals that it is hard to see the data on the mixed gender teams. By excluding the 0 and 1 it is possible to just show the mixed gender teams (in red below). As expected there is a peak at 50% - likely husband and wife applying together. There are also small bumps at 33% and 67% as would be expected from 3 people teams. 
# 
# Of interest, the mixed gender teams still seem to be heavily balanced towards more females. 

# In[ ]:


ax=sns.kdeplot(kiva_loans['percent_female'], color='#3CB371',shade=True, label='all borrowers', bw=0.02)

ax=sns.kdeplot(kiva_loans.loc[(kiva_loans['percent_female'] > 0) & 
                              (kiva_loans['percent_female'] < 1), 'percent_female'], 
               color='#FF6347', shade=True, label='mixed gender teams', bw=0.02)

ax.annotate("0.33",
            xy=(0.33, 1), xycoords='data',
            xytext=(0.33, 6), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"),
            )

ax.annotate("0.67",
            xy=(0.67, 2.5), xycoords='data',
            xytext=(0.67, 6), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"),
            )

ax.annotate("0.5",
            xy=(0.5, 3), xycoords='data',
            xytext=(0.5, 6), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"),
            )

sns.despine()
plt.show()


# Lets look at the percent female distribution across the various sectors:
# 
# Only in `Construction` are there more men than women applying for loans
# 
# Other sectors where men are better represented than average are: `Transportation`, `Manufacturing`, `Wholesale`, `Education`, and `Entertainment`
# 
# Interestingly the `Personal Use` sector is where you see most teams of various sizes applying for loans
# 
# `Arts`, `Food`, `Clothing`, and `Health` have large female collectives applying for loans (many women get together to apply)
# 

# In[ ]:


# plot these as a facet grid...
g = sns.FacetGrid(kiva_loans, col = 'sector', col_wrap=4)
g.map(sns.kdeplot, 'percent_female', color='#3CB371', shade=True, label='mixed gender teams', bw=0.02)
plt.show()


# Turns out that collectives of people are applying for loans for water filters, stoves, solar panels, waste management, power generators, and other collective infrastructure under the `Personal Use` sector . The following interrogates the `use` variable for teams of more than one person who have applied for loans in the `Personal Use`.

# In[ ]:


kiva_loans[(kiva_loans['sector'] == 'Personal Use') & 
           (kiva_loans['percent_female'] > 0) &
           (kiva_loans['percent_female'] < 1)
          ].sample(20).use


# Turns out - in Peru - people like nice (\$600+) soundsystems in their houses:

# In[ ]:


kiva_loans[(kiva_loans['use'] == 'to buy a sound system for her house.')]


# Finally - lets bring this back to the partial funding data and look at chances of being funded if you are on an all male team or an all female team. Remember that for the full dataset the chance of not getting funded is around 7%. So all female teams have an increased chance of getting funded while all male teams have a decreased chance of getting funded. For this analysis I am going back to the kiva_train data set since it has been trimmed for outliers and the last 3 months of data. 

# In[ ]:


labels1 = 'Funded', 'Partially funded'
sizes1  = [len(kiva_train[(kiva_train['partially_unfunded'] == False) &
           (kiva_train['percent_female'] == 0)
          ]), len(kiva_train[(kiva_train['partially_unfunded'] == True) &
           (kiva_train['percent_female'] == 0)
          ])]

labels2 = 'Funded', 'Partially funded'
sizes2  = [len(kiva_train[(kiva_train['partially_unfunded'] == False) &
           (kiva_train['percent_female'] == 1)
          ]), len(kiva_train[(kiva_train['partially_unfunded'] == True) &
           (kiva_train['percent_female'] == 1)
          ])]

fig, (maxis1, maxis2) = plt.subplots(1, 2, figsize=[15, 8])

maxis1.pie(sizes1, labels=labels1, autopct='%1.1f%%',
        shadow=False, startangle=-45, colors = ['#3CB371', '#FF6347'], explode = (0, 0.1))
maxis1.axis('equal')

maxis2.pie(sizes2, labels=labels2, autopct='%1.1f%%',
        shadow=False, startangle=45, colors = ['#3CB371', '#FF6347'], explode = (0, 0.1))
maxis2.axis('equal')

maxis1.set_title("Male only")
maxis2.set_title("Female only")

maxis1.add_patch(
    patches.Arrow(
        0.1, 0.1,
        1.0, 1.0,
        color = '#3CB371'
    )
)


maxis2.add_patch(
    patches.Rectangle(
        (-0.15, -1.5),   # (x,y)
        0.3,          # width
        0.8,          # height
        facecolor='#3CB371'
    )
)
maxis2.add_patch(
    patches.Rectangle(
        (-0.4, -1.2),   # (x,y)
        0.8,          # width
        0.15,          # height
        facecolor='#3CB371'
    )
)

plt.show()


# As we saw from the classifier this may not be a true gender bias - since a lot of this difference could come from differences in the kinds of loans that males and females apply for. The classifier found that loan amount is the most influential feature when assessing if a loan is funded or not. So lets see if there is a difference in the loan amount between males and females. Also, construction has an even gender distribution - so lets see if there is a difference in the funded construction projects between males and females.

# In[ ]:


male = kiva_train[kiva_train['percent_female']==0]
female = kiva_train[kiva_train['percent_female']==1]
ttest = ttest_ind(male['loan_amount'], female['loan_amount'])

print("Mean loan amount for males is ${:,.2f} and for females ${:,.2f}. A t-test comparison find these are different with a p-value of {:,.4}.\n".format(male['loan_amount'].mean(),
         female['loan_amount'].mean(),
         ttest.pvalue
        ))


male = kiva_train[(kiva_train['percent_female']==0) & (kiva_train['sector'] == 'Construction')]
female = kiva_train[(kiva_train['percent_female']==1) & (kiva_train['sector'] == 'Construction')]
ttest = ttest_ind(male['loan_amount'], female['loan_amount'])

print("Mean loan amount for male construction projects is ${:,.2f} and for female construction projects ${:,.2f}. A t-test comparison find these are different with a p-value of {:,.4}.\n".format(male['loan_amount'].mean(),
         female['loan_amount'].mean(),
         ttest.pvalue
        ))


# In[ ]:


labels1 = 'Funded', 'Partially funded'
sizes1  = [len(kiva_train[(kiva_train['partially_unfunded'] == False) &
           (kiva_train['percent_female'] == 0) & (kiva_train['sector'] == 'Construction')
          ]), len(kiva_train[(kiva_train['partially_unfunded'] == True) &
           (kiva_train['percent_female'] == 0) & (kiva_train['sector'] == 'Construction')
          ])]

labels2 = 'Funded', 'Partially funded'
sizes2  = [len(kiva_train[(kiva_train['partially_unfunded'] == False) &
           (kiva_train['percent_female'] == 1) & (kiva_train['sector'] == 'Construction')
          ]), len(kiva_train[(kiva_train['partially_unfunded'] == True) &
           (kiva_train['percent_female'] == 1) & (kiva_train['sector'] == 'Construction')
          ])]

fig, (maxis1, maxis2) = plt.subplots(1, 2, figsize=[15, 8])

maxis1.pie(sizes1, labels=labels1, autopct='%1.1f%%',
        shadow=False, startangle=-45, colors = ['#3CB371', '#FF6347'], explode = (0, 0.1))
maxis1.axis('equal')

maxis2.pie(sizes2, labels=labels2, autopct='%1.1f%%',
        shadow=False, startangle=45, colors = ['#3CB371', '#FF6347'], explode = (0, 0.1))
maxis2.axis('equal')

maxis1.set_title("Male Construction Projects")
maxis2.set_title("Female Construction Projects")

maxis1.add_patch(
    patches.Arrow(
        0.1, 0.1,
        1.0, 1.0,
        color = '#3CB371'
    )
)

maxis2.add_patch(
    patches.Rectangle(
        (-0.15, -1.5),   # (x,y)
        0.3,          # width
        0.8,          # height
        facecolor='#3CB371'
    )
)
maxis2.add_patch(
    patches.Rectangle(
        (-0.4, -1.2),   # (x,y)
        0.8,          # width
        0.15,          # height
        facecolor='#3CB371'
    )
)

plt.show()


# ## Conclusions from gender analysis
# ---
# 
# It is clear that more women are applying for loans through kiva.org than men, and that when teams of people apply there is a larger percentage of women on these teams. This likely relects the societal gender bias and easier local access to capital for men.
# 
# Of interest, it does look like there is a significant difference in funding of loans that are being requested from male vs. females. Part of this could be due to the larger amounts asked for by males vs. females. But even in sectors such as construction where there is even distribution of males and females - the chance of a male loan being partially funded is much higher than the chance of a female loan bein partially funded. 

# # Lenders <a id="lenders"></a>
#  ---
# This is going to be some real interesing analysis. We are given information about how many people get together to fund each loan and we have times of posting on kiva.org and time of funding. In addition we have information about who was the local partner for the loan (this is the people on the ground responsible for giving the money to the lenders) and when the money was acutally given to the borrower. A couple of interesting tidbits from kiva.org API page:
# 
# About the `disbursed_time`
# >The date at which the funds from the loan were given to the borrowers. Note that it is possible for the money to be disbursed to borrowers before the loan is posted on Kiva.
# 
# About the `partner_id`
# >The partner works with Kiva to get funding for that loan from lenders. The association of a loan to a partner is very important since the risk associated with a loan correlates closely to the reputation of a partner. This is why every partner has a risk rating.
# 
# Lets start with some feature engineering and date manipulations.

# In[ ]:


# convert the dates to datetime so we can easily manipulate them
kiva_loans['posted_time'] = pd.to_datetime(kiva_loans['posted_time'])
kiva_loans['funded_time'] = pd.to_datetime(kiva_loans['funded_time'])
kiva_loans['disbursed_time'] = pd.to_datetime(kiva_loans['disbursed_time'])

posttofund = kiva_loans['funded_time'] - kiva_loans['posted_time']
posttodisburse = kiva_loans['disbursed_time'] - kiva_loans['posted_time']
fundtodisburse = kiva_loans['disbursed_time'] - kiva_loans['funded_time']

kiva_loans['posted_to_funded_time_in_hours'] = posttofund.dt.components.hours + (posttofund.dt.days*24)
if PUBLISH:
    kiva_loans['posted_to_disbursed_time_in_hours'] = posttodisburse.dt.components.hours + (posttodisburse.dt.days*24)
    kiva_loans['funded_to_disbursed_time_in_hours'] = fundtodisburse.dt.components.hours + (fundtodisburse.dt.days*24)


# Now that the times are calculated in hours it is easy to plot the data. The follwing KDE plots show that:
# 1. Some of the grants were funded before they were posted on kiva.org (negative time from posting to funding)
# 2. A large proportion of loans are funded within the first week and most are funded within the first month since posting 
# 3. Around the 1 month mark there is an uptick in funding - likely there is some feature on kiva.org that will highlight loans that are around a month old and still have not been funded.

# In[ ]:


plt.figure(figsize=[14,13])
maxis1 = plt.subplot(211)
maxis2 = plt.subplot(223)
maxis3 = plt.subplot(224)

# Full time
sns.kdeplot(kiva_loans['posted_to_funded_time_in_hours'], 
               color='#3CB371',shade=True, label='', bw=12, ax = maxis1)

# First two months
sns.kdeplot(kiva_loans[kiva_loans['posted_to_funded_time_in_hours'] < 1488]['posted_to_funded_time_in_hours'], 
               color='#3CB371',shade=True, label='', bw=12, ax = maxis2)
sns.kdeplot(kiva_loans[(kiva_loans['posted_to_funded_time_in_hours'] < 1488)& 
                                  (kiva_loans['percent_female'] == 0)]['posted_to_funded_time_in_hours'], 
               color='#5DADE2',shade=False, label='male', bw=12, ax = maxis2)
sns.kdeplot(kiva_loans[(kiva_loans['posted_to_funded_time_in_hours'] < 1488)& 
                                  (kiva_loans['percent_female'] == 1)]['posted_to_funded_time_in_hours'], 
               color='#FF6347',shade=False, label='female', bw=12, ax = maxis2)
maxis2.set_xlim([0,1500])

# First week
sns.kdeplot(kiva_loans[(kiva_loans['posted_to_funded_time_in_hours'] < 168)]['posted_to_funded_time_in_hours'], 
               color='#3CB371',shade=True, label='', bw=12, ax = maxis3)
maxis3.set_xlim([0,168])

maxis1.set_title("All time")
maxis2.set_title("First two months")
maxis3.set_title("First week")

maxis1.set_xlabel('Time from posting to funding (hours)')
maxis2.set_xlabel('Time from posting to funding (hours)')
maxis3.set_xlabel('Time from posting to funding (hours)')

maxis1.set_yticks([])
maxis1.set_yticklabels([])
maxis2.set_yticks([])
maxis2.set_yticklabels([])
maxis3.set_yticks([])
maxis3.set_yticklabels([])

maxis2.axvline(x=168, color = 'black', lw = 1) 
maxis2.annotate("One week",
            xy=(168, 0), xycoords='data',
            xytext=(174, 0.00025), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"),
            )

maxis2.axvline(x=774, color = 'black', lw = 1) 
maxis2.annotate("One month",
            xy=(774, 0), xycoords='data',
            xytext=(780, 0.00025), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"),
            )

sns.despine()
plt.show()


# Create variable that checks if a loan is funded within the first two week. Then we can look at what the difference is between male and female teams. From this we see that twice as many of the male loans are unfunded at the two week mark. Even if the time period is extended to one month or 1,000 hours the two fold difference persist. 

# In[ ]:


kiva_loans['funded_in_first_two_weeks'] = 0
kiva_loans.loc[kiva_loans['posted_to_funded_time_in_hours'] < 24*14, 'funded_in_first_two_weeks'] = 1


# In[ ]:


labels = 'Funded in two weeks', 'Not funded in two weeks'
sizes1  = [len(kiva_loans[(kiva_loans['funded_in_first_two_weeks'] == True) &
           (kiva_loans['percent_female'] == 0)
          ]), len(kiva_loans[(kiva_loans['funded_in_first_two_weeks'] == False) &
           (kiva_loans['percent_female'] == 0)
          ])]

sizes2  = [len(kiva_loans[(kiva_loans['funded_in_first_two_weeks'] == True) &
           (kiva_loans['percent_female'] == 1)
          ]), len(kiva_loans[(kiva_loans['funded_in_first_two_weeks'] == False) &
           (kiva_loans['percent_female'] == 1)
          ])]

fig, (maxis1, maxis2) = plt.subplots(1, 2, figsize=[15, 8])

maxis1.pie(sizes1, labels=labels, autopct='%1.1f%%', explode = (0, 0.02), 
        shadow=False, startangle=-30, colors = ['#3CB371', '#FF6347'])
maxis1.axis('equal')

maxis2.pie(sizes2, labels=labels, autopct='%1.1f%%', explode = (0, 0.02), 
        shadow=False, startangle=155, colors = ['#3CB371', '#FF6347'])
maxis2.axis('equal')

maxis1.set_title("Male only")
maxis2.set_title("Female only")

maxis1.add_patch(
    patches.Arrow(
        0.1, 0.1,
        1.0, 1.0,
        color = '#3CB371'
    )
)


maxis2.add_patch(
    patches.Rectangle(
        (-0.15, -1.5),   # (x,y)
        0.3,          # width
        0.8,          # height
        facecolor='#3CB371'
    )
)
maxis2.add_patch(
    patches.Rectangle(
        (-0.4, -1.2),   # (x,y)
        0.8,          # width
        0.15,          # height
        facecolor='#3CB371'
    )
)

plt.show()


# Again, there are many factors that can be affecting this gender difference in time to funding (such as the sector of the loan and the size of the loan).
# 
# Now lets look at time from funding to disbursal of the money (when the money is given to the borrower). I was hoping to get all kind of juicy information about how slow/fast various partners are about disbursing funding to the borrower once the loan was funded on kiva.org - but as the KDE plots show - most loans were actually disbursed before the loans were even poseted on kiva.org. I had to check my math many times to make sure that I was manipulating the dates correctly - but as my sanity check below shows the average of all the disbursal times is prior to the average of all the posting and funding dates. While there could be some skew to this since I am calculating thes times on the full 3 year data set - not the train set where I chopped off the last 2 months - the skew from these last 2 months over the full 3 year set is likely minimal. 

# In[ ]:


if PUBLISH:
    sns.kdeplot(kiva_loans['funded_to_disbursed_time_in_hours'], 
                color='#5DADE2',shade=True, label='funded to disbursed', bw=12)

    sns.kdeplot(kiva_loans['posted_to_disbursed_time_in_hours'], 
               color='#FF6347',shade=True, label='posted to disbursed', bw=12)

sns.kdeplot(kiva_loans['posted_to_funded_time_in_hours'], 
               color='#3CB371',shade=True, label='posted to funded', bw=12)

plt.xlabel('Time (hours)')

# Sanity check
if PUBLISH:
    avr_post = (kiva_loans.posted_time - kiva_loans.posted_time.min()).mean() + kiva_loans.posted_time.min()
    avr_fund = (kiva_loans.funded_time - kiva_loans.funded_time.min()).mean() + kiva_loans.funded_time.min()
    avr_disb = (kiva_loans.disbursed_time - kiva_loans.disbursed_time.min()).mean() + kiva_loans.disbursed_time.min()
    print ("average posting time {}, funding time {}, and disbural time {}.".format(avr_post, avr_fund, avr_disb))

sns.despine()


# So it looks like there is not much information to be gained from `disbursed_time`. So instead lets look quickly at the information about `lender_count` - the following KDEs show the distribution of lender counts for loans where there are more than 0 lenders. 

# In[ ]:


fig, (maxis1, maxis2) = plt.subplots(1, 2, figsize=[15, 8])

maxis1.set_title("All lender counts")
maxis2.set_title("Focus on lender counts < 100")

sns.kdeplot(kiva_loans[(kiva_loans['lender_count'] > 0)].lender_count, 
               color='#3CB371',shade=True, label='number of lenders', bw=4, ax=maxis1)

sns.kdeplot(kiva_loans[(kiva_loans['lender_count'] <100) & (kiva_loans['lender_count'] > 0)].lender_count, 
               color='#3CB371',shade=True, label='number of lenders', bw=2, ax=maxis2)

maxis1.set_xlabel('Number of lenders')
maxis2.set_xlabel('Number of lenders')

sns.despine()
plt.show()


# Lets take a further look at the lender count in relation to the time of loan funding as well as the loan amount. I am looking at a subset of the total data where lender count is > 40 to see if there is correlation between loan amount, number of lenders and the time to fund a loan. Interestingly, while more expensive loans have more lenders engaged - the time to fully fund those loans seem to be shorter. Mainly I just wnated a nice looking rainbow plot in this kernel. 

# In[ ]:


df = kiva_loans[(kiva_loans['lender_count'] > 40) & (kiva_loans['loan_amount'] < 10000) & (kiva_loans['posted_to_funded_time_in_hours'] < 6000)]#.sample(1000)
xval = df.lender_count
yval = df.posted_to_funded_time_in_hours
cval = df.loan_amount

if PUBLISH:
    plt.scatter(x=xval.values, y=yval.values, c=cval.values, cmap=plt.get_cmap('jet'), alpha = 0.2)
    plt.title('Lender count vs. time from posting to funding, color = loan amount')
    plt.xlabel("Lender count")
    plt.ylabel("Time from posting to funding (hours)")
    cbar = plt.colorbar()
    cbar.set_label('Loan amount ($)', rotation=90)

    # save this one so it shows up in feed with this image as the output
    plt.savefig("fname2.png")
    plt.show()


# Final piece of information that we can get from this is part of the dataset is the average amount of funding per lender.  I calculate it and then remove all funding per lender values where the grant is not fully funded. From the plots below you can see that the average funding is around \$25 per lender. 

# In[ ]:


kiva_loans['funding_per_lender'] = kiva_loans['loan_amount']/kiva_loans['lender_count']
kiva_loans.loc[kiva_loans['ask_fund_delta'] > 0, 'funding_per_lender'] = np.nan

fig, (maxis1, maxis2) = plt.subplots(1, 2, figsize=[15, 8])

maxis1.set_title("All lender counts")
maxis2.set_title("Focus on lender counts < 100")

sns.kdeplot(kiva_loans.funding_per_lender, 
               color='#3CB371',shade=True, label='', bw=100, ax=maxis1)

sns.kdeplot(kiva_loans[(kiva_loans['funding_per_lender'] < 200)].funding_per_lender, 
               color='#3CB371',shade=True, label='', bw=5, ax=maxis2)

maxis1.set_xlabel('Funding per lender ($)')
maxis2.set_xlabel('Funding per lender ($)')

sns.despine()
plt.show()


# ## Conclusions from lenders and lending time
# 
# Most loans are funded within the first week, with an additional hump of funding coming around 1 month after posting. There is again a gender difference in the speed of funding between male and female loans with the funding time distribution of male loans pushed towards slower funding. Again, this difference could be due to other factors such as loan amount and sector of the loan, although in general it seems like larger loans are funded quicker than medium sized loans. On average the majority of loans are funded by between 1-30 lenders and the amount of funding committed per lender is around \$10 to \$50.
# 
# Unfortunately, the majority of loans are disbursed before they are even posted on kiva.org - so it is not possible to assess time from funding to disbursement. In this regard it looks like kiva is acting mostly as a buyer of loans that have already been made by the local partners - much like US mortgages are bought and sold and transferred between lenders.

# # Loan structure <a id="structure"></a>
# ---
# There are two variables that we can use to learn about the loan structure `term_in_months` and `repayment_interval`. Lets take a look and see what sort of values we have in the loan terms. It looks like we have some spikes at non-conventional times: 8, 11, 14 months. I wonder if this reflects the fact that some of these loans were actually made around a month before they were posted on kiva.org. Also, as expected, loan terms are trending for longer for males vs females - most likely due to the higher amounts borrowed by males. 

# In[ ]:


fig, (maxis1, maxis2) = plt.subplots(1, 2, figsize=[15, 8])

maxis1.set_title("All loan terms")
maxis2.set_title("Loan terms up to 3 years")

sns.kdeplot(kiva_loans.term_in_months, 
               color='#3CB371',shade=True, label='', bw=0.5, ax=maxis1)

sns.kdeplot(kiva_loans[(kiva_loans['term_in_months'] <= 36)].term_in_months, 
               color='#3CB371',shade=True, label='', bw=0.5, ax=maxis2)

sns.kdeplot(kiva_loans[(kiva_loans['term_in_months'] <= 36) & (kiva_loans['percent_female'] == 1)].term_in_months, 
               color='#FF6347',shade=False, label='female', bw=0.5, ax=maxis2)
sns.kdeplot(kiva_loans[(kiva_loans['term_in_months'] <= 36) & (kiva_loans['percent_female'] == 0)].term_in_months, 
               color='#5DADE2',shade=False, label='male', bw=0.5, ax=maxis2)

maxis1.set_xlabel('Loan term (months)')
maxis2.set_xlabel('Loan term (months)')

maxis2.axvline(x=8, color = 'black', lw = 1) 
maxis2.axvline(x=11, color = 'black', lw = 1) 
maxis2.axvline(x=14, color = 'black', lw = 1) 

female = kiva_loans[(kiva_loans['percent_female'] == 1)].term_in_months
male   = kiva_loans[(kiva_loans['percent_female'] == 0)].term_in_months

print("Mean repayment for females {:0.2f}, median {:0.2f}, mode {:0.2f}". format(female.mean(), female.median(), female.mode()[0]))
print("Mean repayment for males {:0.2f}, median {:0.2f}, mode {:0.2f}". format(male.mean(), male.median(), male.mode()[0]))

sns.despine()


# Now lets look at how loan term interacts with loan amount for males and females. The following scatter is heavily dotted - but does show that there is a collection of mixed teams (green color) who have a loan term of about 122 months, and it shows that while there seem to be no correlation between loan amount and term for loans with terms less than 5 years (60 months) - for loans of longer term there is a nice correlation between the loan amount and the loan term. 

# In[ ]:


df = kiva_loans[(kiva_loans['loan_amount'] < 10000)]
xval = df.term_in_months
yval = df.loan_amount
cval = df.percent_female*100

if PUBLISH:
    plt.scatter(x=xval.values, y=yval.values, c=cval.values, cmap=plt.get_cmap('jet'), alpha = 0.2)
    plt.title('Loan amount vs loan term, color = percent female')
    plt.xlabel("Loan term (months)")
    plt.ylabel("Loan amount ($)")
    cbar = plt.colorbar()
    cbar.set_label('Percent female (%)', rotation=90)


# In[ ]:


fig, (maxis1, maxis2) = plt.subplots(1, 2, figsize=[15, 8])

maxis1.set_title("Repayment intervals")
maxis2.set_title("Repayment intervals (log scale)")

intervalcounts = kiva_loans.repayment_interval.value_counts()

sns.barplot(x=intervalcounts.index, 
            y=intervalcounts, ax = maxis1, palette="Greens_d", edgecolor=['black', 'black', 'black', 'black'])

sns.barplot(x=intervalcounts.index, 
            y=intervalcounts, ax = maxis2, palette="Greens_d", edgecolor=['black', 'black', 'black', 'black'])

maxis1.set_ylabel("Frequency")
maxis2.set_ylabel("Frequency (log scale)")

maxis2.set_yscale('log', nonposy='clip')

sns.despine()


# In[ ]:


sns.violinplot(x="repayment_interval", y="loan_amount", hue="team_gender", data=kiva_loans[(kiva_loans['team_gender'] != 'mixed')&(kiva_loans['loan_amount'] <= 4000)], split=True,
               inner="quart", palette={"male": "#5DADE2", "female": "#FF6347"})
plt.xlabel('Repayment interval')
plt.xlabel('Loan amount')
plt.title('Repayment interval vs loan amount for gender')
plt.legend(title = 'Gender')

sns.despine()
plt.show()


# In[ ]:


fig, (maxis1, maxis2) = plt.subplots(1, 2, figsize=[15, 8])

maxis1.set_title("Female")
maxis2.set_title("Male")

intervalcounts_male = kiva_loans[kiva_loans['percent_female'] == 0].repayment_interval.value_counts()
intervalcounts_female = kiva_loans[kiva_loans['percent_female'] == 1].repayment_interval.value_counts()

sns.barplot(x=intervalcounts_female.index, 
            y=intervalcounts_female, ax = maxis1, palette="Reds_d", edgecolor=['black', 'black', 'black', 'black'])

sns.barplot(x=intervalcounts_male.index, 
            y=intervalcounts_male, ax = maxis2, palette="Blues_d", edgecolor=['black', 'black', 'black', 'black'])

sns.despine()
plt.show()


# ## Conclusion from loan structure
# 
# Male teams on average have longer repayment intervals (16 months) than females (13 months). For loans with terms longer than 5 years there appear to be a linear relationship between loan term and loan amount. But for shorter loan periods the data is more messy. The majority of loans are repaid either montly or irregularly. However for male loans monthly and bullet payments are the preferred terms. Bullet is defined as below:
# 
# >In banking and finance, a bullet loan is a loan where a payment of the entire principal of the loan, and sometimes the principal and interest, is due at the end of the loan term.
# 
# This is an interesting type of loan and it certainly seems like the fact that it is so popular in the male category could have a big impact on lending. So I added this to the modeling above to see how it affects full/partial funding of loans. Adding this feature slightly increased the accuracy, but the repayment interval enters into the list of features that has more than 1% influence on the model (feature importances).
# 
# Before modeling with repayment interval (accuracy 94.36%)
# 
# <pre>56.82% loan_amount
# 14.05% percent_female
# 3.52% El Salvador
# 2.45% Colombia
# 1.66% Retail
# 1.41% Philippines
# 1.36% Education
# 1.27% Housing
# 1.25% non-top20-country
# 1.03% Peru</pre>
# 
# After including the repayment interval (accuracy 94.71%)
# 
# <pre>56.20% loan_amount
# 11.82% percent_female
# 2.78% El Salvador
# 2.09% bullet
# 2.05% irregular
# 1.81% Retail
# 1.69% Agriculture
# 1.55% non-top20-country
# 1.50% Colombia
# 1.31% monthly
# 1.16% Housing
# 1.09% Education
# 0.97% Philippines</pre>
# 
# Bullet type repayment enters in the top 5 most influential factors on the likelihood of full funding and the influence of percent female drops more than 3%.

# # Loan use <a id="loan_use"></a>
# ---
# What the loan will be used for is captured in the variables `activity`, `sector`, `use`, and `tags`.  All of these are text based and therefore are a little more tricky to visualize.
# 
# First lets take a closer look at the 4 features and see how many unique values are in each feature.

# In[ ]:


kiva_loans[['sector', 'activity', 'use', 'tags']].describe()


# Will use word clouds to visualize the various words that are in these features.

# In[ ]:


wordcloud = WordCloud(background_color='white', width = 1400, height = 600, max_words = 15, random_state=42, colormap='summer')
plt.imshow(wordcloud.fit_words(kiva_loans['sector'].value_counts()))
plt.axis('off')
plt.title('Word cloud of sector')
plt.savefig("fname1.png")
plt.show()


# In[ ]:


wordcloud = WordCloud(background_color='white', width = 1400, height = 600, max_words = 163, random_state=42, colormap='summer')
plt.imshow(wordcloud.fit_words(kiva_loans['activity'].value_counts()))
plt.axis('off')
plt.title('Word cloud of loan activity')
plt.show()


# In[ ]:


kiva_loans['use_simplified'] = kiva_loans['use'].copy()

kiva_loans.loc[kiva_loans['use_simplified'].str.contains('clean water').fillna(False), 'use_simplified'] = 'clean water'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('water filter').fillna(False), 'use_simplified'] = 'clean water'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('water filtration').fillna(False), 'use_simplified'] = 'clean water'

kiva_loans.loc[kiva_loans['use_simplified'].str.contains('toilet').fillna(False), 'use_simplified'] = 'sanitary toilet'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('latrine').fillna(False), 'use_simplified'] = 'sanitary toilet'

kiva_loans.loc[kiva_loans['use_simplified'].str.contains('university').fillna(False), 'use_simplified'] = 'school'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('school').fillna(False), 'use_simplified'] = 'school'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('tuition').fillna(False), 'use_simplified'] = 'school'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('studies').fillna(False), 'use_simplified'] = 'school'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('study').fillna(False), 'use_simplified'] = 'school'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('college').fillna(False), 'use_simplified'] = 'school'

kiva_loans.loc[kiva_loans['use_simplified'].str.contains('supplies to raise').fillna(False), 'use_simplified'] = 'farm supplies'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('feed and vitamins').fillna(False), 'use_simplified'] = 'farm supplies'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('feeds and vitamins').fillna(False), 'use_simplified'] = 'farm supplies'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('fertilizer').fillna(False), 'use_simplified'] = 'farm supplies'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('farm').fillna(False), 'use_simplified'] = 'farm supplies'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('maize').fillna(False), 'use_simplified'] = 'farm supplies'

kiva_loans.loc[kiva_loans['use_simplified'].str.contains(' to sell').fillna(False), 'use_simplified'] = 'merchandice to sell'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains(' for resale').fillna(False), 'use_simplified'] = 'merchandice to sell'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains(' more stock').fillna(False), 'use_simplified'] = 'merchandice to sell'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains(' restock').fillna(False), 'use_simplified'] = 'merchandice to sell'

kiva_loans.loc[kiva_loans['use_simplified'].str.contains('food production business').fillna(False), 'use_simplified'] = 'food production business'

kiva_loans.loc[kiva_loans['use_simplified'].str.contains('solar lamp').fillna(False), 'use_simplified'] = 'solar lamp'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('solar lantern').fillna(False), 'use_simplified'] = 'solar lamp'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('solar light').fillna(False), 'use_simplified'] = 'solar lamp'

kiva_loans.loc[kiva_loans['use_simplified'].str.contains('building materials').fillna(False), 'use_simplified'] = 'building materials'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('construction materials').fillna(False), 'use_simplified'] = 'building materials'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('cement').fillna(False), 'use_simplified'] = 'building materials'

kiva_loans.loc[kiva_loans['use_simplified'].str.contains(' fish ').fillna(False), 'use_simplified'] = 'fish'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains(' seafood ').fillna(False), 'use_simplified'] = 'fish'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains(' fishing ').fillna(False), 'use_simplified'] = 'fish'

kiva_loans.loc[kiva_loans['use_simplified'].str.contains('stove').fillna(False), 'use_simplified'] = 'stove'

kiva_loans.loc[kiva_loans['use_simplified'].str.contains('cattle').fillna(False), 'use_simplified'] = 'livestock'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('calves').fillna(False), 'use_simplified'] = 'livestock'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('cow').fillna(False), 'use_simplified'] = 'livestock'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('poultry').fillna(False), 'use_simplified'] = 'livestock'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('pig').fillna(False), 'use_simplified'] = 'livestock'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('goat').fillna(False), 'use_simplified'] = 'livestock'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('chicken').fillna(False), 'use_simplified'] = 'livestock'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('livestock').fillna(False), 'use_simplified'] = 'livestock'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('buffalo').fillna(False), 'use_simplified'] = 'livestock'

kiva_loans.loc[kiva_loans['use_simplified'].str.contains('items to sell').fillna(False), 'use_simplified'] = 'items to sell'

kiva_loans.loc[kiva_loans['use_simplified'].str.contains('clothes').fillna(False), 'use_simplified'] = 'clothes'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('clothing').fillna(False), 'use_simplified'] = 'clothes'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('tailor').fillna(False), 'use_simplified'] = 'clothes'
kiva_loans.loc[kiva_loans['use_simplified'].str.contains('sewing').fillna(False), 'use_simplified'] = 'clothes'

print("Using simple regular expression to simplify the use variable. After simplification the number of unique values is", 
      kiva_loans['use_simplified'].nunique(), "which is down from", kiva_loans['use'].nunique(),"in the original dataset")


# In[ ]:


wordcloud = WordCloud(background_color='white', width = 1400, height = 600, random_state=42, colormap='summer')
plt.imshow(wordcloud.fit_words(kiva_loans['use_simplified'].value_counts().head(200)))
plt.axis('off')
plt.title('Word cloud of simplified use')
plt.show()


# In[ ]:


s = kiva_loans['tags'].value_counts().head(300)
tupples = list(zip(s.index, s))
cleaned = pd.Series(dict([i for i in tupples if '#' in i[0]]))

wordcloud = WordCloud(background_color='white', width = 1400, height = 600, random_state=42, colormap='summer')
plt.imshow(wordcloud.fit_words(cleaned.head(200)))
plt.axis('off')
plt.title('Word cloud of loan tags (just hashtags)')
plt.show()


# # Location exploration <a id="location"></a>
# ---
# 
# ## MPI Region Locations
# 
# First lets look at `kiva_mpi_region_locations.csv`. This file contains MPI data (Global Multidimensional Poverty Index) as well as location names and GPS coordinates. As we see below this file contains mostly junk data. It has almost 70% null values (1880/2772) consistent across the various features. 

# In[ ]:


#load data
kiva_mpi_locations = pd.read_csv("../input/kiva_mpi_region_locations.csv")

# explore a little
print("Total entries in kiva_mpi_locations:", len(kiva_mpi_locations))
print("Dont trust the column metrics that they give you - there are null values hiding in here:\n", kiva_mpi_locations.isnull().sum())
print("The most frequent location is (lat,lng) ", kiva_mpi_locations['geo'].value_counts()[:1])

kiva_mpi_locations = kiva_mpi_locations.dropna(axis=0, how='any')
print(" *"* 30)
print("After dropna")
print(" *"* 30)

print("Total entries in kiva_mpi_locations:", len(kiva_mpi_locations))
print("Dont trust the column metrics that they give you - there are null values hiding in here:\n", kiva_mpi_locations.isnull().sum())
print("The most frequent location is (lat,lng) ", kiva_mpi_locations['geo'].value_counts()[:1])

#kiva_mpi_locations.sample(10)


# Lets look at the MPI data for the US. Wait there is no data for the US!

# In[ ]:


kiva_mpi_locations[kiva_mpi_locations['ISO'] == 'USA']


# But if we map the GPS coordinates - we find that there are seveal coordinates that map to the US:

# In[ ]:


fig = plt.figure(figsize=(15,6))

m = Basemap(projection='cyl',lon_0=0)
m.drawcoastlines()
m.drawcountries()

x, y = m(kiva_mpi_locations['lon'].values, kiva_mpi_locations['lat'].values)

m.scatter(x, y, latlon=True,
          c=kiva_mpi_locations['MPI'].values, #s=area,
          cmap='jet', alpha=0.5)
plt.colorbar()
plt.title('MPI map - higher # means more poverty')
plt.show()


# Turns out one of the locations that map to the US is actually labeled as being in Nepal. Lets just map the Nepal coordinates.

# In[ ]:


fig = plt.figure(figsize=(15,6))

m = Basemap(projection='cyl',lon_0=0)
m.drawcoastlines()
m.drawcountries()

nepal = kiva_mpi_locations[kiva_mpi_locations['country'] == 'Nepal']
is_in_nepal = (nepal['lon'] > 80)
locs_in_nepal = is_in_nepal.sum()
locs_labeled_nepal = len(nepal)
is_in_nepal[is_in_nepal] = 'blue'
is_in_nepal[is_in_nepal==False] = 'red'

x, y = m(nepal['lon'].values, nepal['lat'].values)

m.scatter(x, y, latlon=True,
          c=is_in_nepal, #s=area,
          alpha=1)

plt.title('MPI map of Nepal labeled locations - only {} of {} are in Nepal'.format(locs_in_nepal, locs_labeled_nepal))
plt.savefig("fname0.png")


# Wow - a lot map outside of Nepal!

# In[ ]:


kiva_mpi_locations[kiva_mpi_locations['country'] == 'Nepal']


# ### MPI Locations Conclusion
# 
# Even after cleaning up the null values there is a lot of noise in this dataset. The dataset does not contain any countries labeled as USA - but when you plot the GPS coordinates on a map you will find that at least 8 of the GPS locations map to major cities in the US. Further investigation finds that one of those GPS dots is labeled as being in Nepal. Plotting all the GPS locations for Nepal I found that only 7 of the 13 locations labeled as Nepal actually map to that country. 
# 
# ** So what can you trust in this dataset?** I have been assured in the comments that while the GPS coordinates may be mixed up the rest of the information in this file is correct. However, considering the other great datasets that are being made available - it may be time to switch to a different dataset.

# # More to come - please share your votes and comments on this kernel 
