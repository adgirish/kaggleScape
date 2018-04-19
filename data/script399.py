
# coding: utf-8

# This kernel will consist of a full EDA and predictive analysis of attrition at this firm, followed by some calculations of the likely business costs under each of the various models I will use. The key insight I am pursuing here is the tradeoff between model accuracy and ROI - a more accurate model of attrition does not necessarily lead to a higher return on investment of a firm's retention budget. Understanding this and being able to talk intelligently about it is key to creating workable retention strategies that make the HR function truly strategic.
# 
# As background, there has been a great deal of talk for a couple decades now about the need for HR to be more strategic. Aside from any philosophical discussions of the true nature of 'strategy' and strategic thinking, what this normally means is that business leaders want HR to be more like their marketing departments - analytical, results-oriented, and able to directly tie their activities to the firm's desired outcomes (more money!!). Marketing firms today rely quite heavily on predictive analytics and data mining/data science. The future of HR likely lies along this same road. This kernel explores one potential problem such an analytical 'strategic' HR practicioner might reasonably expect to wrestle with.

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


data = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')
data.head()


# Time to do some EDA! First I'll use the describe() method to look at some descriptive statistics and then I'll start visualizing some of the relationships between these variables.

# In[3]:


data.describe()


# Let's do some common-sense quality checks to start with, since the appropriate information is provided. We are provided with a daily and hourly rate and the standard hours each employee is supposed to be working. We are also told whether they work overtime or not, but not how many hours they get. So first let's see if weekly hours exceeding their standard hours equates to overtime, then calculate the actual number of hours of overtime each employee is actually working.
# 
# To do this accurately I need to reflect whether these employees get paid extra for overtime. I will assume that the yes/no values in the overtime column reflect whether the employee in question is authorized overtime rather than whether or not they have actually worked it. The metric definitions provided by IBM do not specify whether this is the case or not.
# 
# If the employee is authorized overtime I'll adjust hours over 80 to reflect 150% of base pay, which is the standard rate for overtime pay under the Fair Labor Standards Act. Otherwise I'll make no adjustment.

# In[4]:


pd.options.mode.chained_assignment = None

data['RealHours'] = data['DailyRate'] / data['HourlyRate'] * 10
data['HoursDelta'] = data['RealHours'] - data['StandardHours']
print(data['HoursDelta'][:15])

data['PaidOverTime'] = data['HoursDelta'] - 80
print(data['PaidOverTime'][:15])
for row, value in enumerate(data['PaidOverTime']):
    if value < 0:
        data['PaidOverTime'][row] = 0
    if value > 0:
        data['PaidOverTime'][row] = data['PaidOverTime'][row] / 1.5
    if data['OverTime'][row] == 'No':
        data['PaidOverTime'][row] = 0
        
print(data['PaidOverTime'][:15])


# Some folks who are working more than their required hours have a 'No' in the overtime column. But how can this be when I am capturing their extra hours as a function of their Daily and Hourly rate? If they aren't getting paid for it then this should not show up here.

# In[5]:


data['OT'] = 0
data['OT'][data['OverTime'] == 'Yes'] = 1


_ = plt.scatter(data['MonthlyRate'], data['MonthlyIncome'], c=data['OT'])
_ = plt.xlabel('Monthly Rate')
_ = plt.ylabel('Monthly Income')
_ = plt.title('Monthly Rate vs. Monthly Income')
plt.show()
print(np.corrcoef(data['MonthlyRate'], data['MonthlyIncome']))

_ = plt.scatter(data['DailyRate'], data['MonthlyRate'], c=data['OT'])
_ = plt.xlabel('Daily Rate')
_ = plt.ylabel('Monthly Rate')
_ = plt.title('Daily Rate vs. Monthly Rate')
plt.show()
print(np.corrcoef(data['DailyRate'], data['MonthlyRate']))


# Well, I'm flummoxed. As you can see from the correlations above, there is basically no relationship between our various rates of pay. How can monthly rate and monthly income have NO correlation? How about daily rate and monthly rate? How can that relationship be negative, and so tiny as to be not worth considering correlated at all? One might expect imperfect correlation if there is a great deal of overtime that is skewing the results, but when I plot these variables in a scatterplot it is obvious that they are scattered pretty much entirely at random.
# 
# Now, I know that this dataset is simulated. And, unfortunately, I do not know who to ask for more information as to what these numbers are supposed to mean. If I had to make a wild guess I would say they were generated randomly by the algorithm that created this dataset. I cannot reasonably believe that there would be no relationship between the hourly and daily rate for the SAME WORKER, or that rates and income would have no correlation.
# 
# I'll make one more attempt to find a relationship here by plotting the relationships between some of these numbers. Perhaps that will shed some light on the matter.

# In[6]:


_ = plt.scatter((data['MonthlyRate'] / data['DailyRate']), data['DailyRate'])
_ = plt.xlabel('Ratio of Monthly to Daily Rate')
_ = plt.ylabel('Daily Rate')
_ = plt.title('Monthly/Daily Rate Ratio vs. Daily Rate')
plt.show()

_ = plt.scatter((data['MonthlyRate'] / data['DailyRate']), data['DailyRate'])
_ = plt.semilogx()
_ = plt.xlabel('Logarithm of Monthly/Daily Rate Ratio')
_ = plt.semilogy()
_ = plt.ylabel('Logarithm of Daily Rate')
_ = plt.title('Logarithmic Monthly/Daily Rate Ratio vs. Log. Daily Rate')
plt.show()

data['lograteratio'] = np.log(data['MonthlyRate'] / data['DailyRate'])
_ = plt.hist(data['lograteratio'], bins=50)
_ = plt.xlabel('Logarithmic Monthly/Daily Rate Ratio')
_ = plt.ylabel('Count')
_ = plt.title('Histogram of Logarithmic Monthly/Daily Rate Ratio')
plt.show()


# OK, well this is interesting. We can model this relationship a bit more profitably on a double logarithmic scatter plot. And we end up with a histogram that bears a passing (though imperfect) resemblance to a normal distribution. Curious. So there is a relationship after all, it's just difficult to see.

# In[7]:


data['left'] = 0
data['left'][data['Attrition'] == 'Yes'] = 1
x = data['left']
print('Monthly Rate:', np.corrcoef(data['MonthlyRate'], x))
print('Daily Rate', np.corrcoef(data['DailyRate'], x))
print('Hourly Rate', np.corrcoef(data['HourlyRate'], x))
print('Monthly Income', np.corrcoef(data['MonthlyIncome'], x))
print('Log Rate Ratio', np.corrcoef(data['lograteratio'] ** 35, x))


# Further, I find that if I mess around with taking exponents of these various metrics that I can change their correlation to my variable of interest (left, or attrition). This is a bad thing is some cases, good in others, indifferent in most. My biggest improvement is in raising my lograteratio to a higher power. Without alteration it correlates to attrition at 4.35%. Taken to the 15th power it gives me 8.21%. Playing around with the numbers I find that by raising that ratio to the 35th power I achieve the highest degree of correlation at 9.907%. 
# 
# I attribute this to the magnifying effect raising this variable to a higher power has on its relationship to my variable of interest. In a sense I think it is similar to the value of using the 'least squares' methodology when calculating regression lines. Exponents punish values lying further from the regression line; in this case, they exaggerate the difference between different values of my variable.
# 
# For the record, I understand that what I am doing probably seems vaguely ridiculous. Why take the logarithm of this ratio and then raise it to an exponential value? But I find that I can gain a slightly higher correlation by doing it this way (although the difference is only about two tenths of a percent).
# 
# Now that I've done all these manipulations and established that the relationship between these variables is definitely not linear, I need to delete a few of the features I created on the way to this point and clean my DataFrame back up.

# In[8]:


del data['RealHours']
del data['HoursDelta']
del data['PaidOverTime']

data['lograteratio'] = data['lograteratio'] ** 35


# Now I'll get a quick overview of some salient features. Bokeh provides a convenient way to display a ton of information in a user friendly way. It is also, incidentally, great for helping communicate important ideas to non-technical audiences.

# In[25]:


data.head()


# In[ ]:


from bokeh.plotting import figure, ColumnDataSource
from bokeh.io import output_file, show

source = ColumnDataSource(data)
p = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

p.circle(fertility, female_literacy, source=source)

# Call the output_file() function and specify the name of the file
output_file('fert_lit.html')


# I've become a fan of 3D scatter plots for visualizing HR datasets, so I'll use those to look at some key variables here. By plotting different features on each of the three axes, color, and size, I can effectively visualize five dimensions of my data simultaneously. As long as I pick the variables I want to look at well, I can learn a lot from this method.

# In[9]:


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = data['YearsAtCompany']
y = data['Age']
z = data['YearsInCurrentRole']
s = data['WorkLifeBalance']
c = data['left']
cmap = plt.get_cmap('seismic')
_ = ax.scatter(xs=x, ys=y, zs=z, c=c, cmap=cmap)
_ = ax.set_xlabel('Years at Company')
_ = ax.set_ylabel('Age')
_ = ax.set_zlabel('Years in Current Role')
_ = plt.title('')
plt.show()


# Lots of attrition among the young and inexperienced. This is no surprise. I'll try to capture some of that in a seperate feature.

# In[10]:


young = data[(data['Age'] < 30) & (data['YearsAtCompany'] <= 2) & (data['YearsInCurrentRole'] <= 1)]
data['young'] = 0
data['young'][(data['Age'] < 30) & (data['YearsAtCompany'] <= 2) & (data['YearsInCurrentRole'] <= 1)] = 1

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = young['YearsAtCompany']
y = young['Age']
z = young['YearsInCurrentRole']
s = young['WorkLifeBalance']
c = young['left']
cmap = plt.get_cmap('seismic')
_ = ax.scatter(xs=x, ys=y, zs=z, c=c, cmap=cmap)
_ = ax.set_xlabel('Years at Company')
_ = ax.set_ylabel('Age')
_ = ax.set_zlabel('Years in Current Role')
_ = plt.title('')
plt.show()

_ = sns.boxplot(young['left'], young['Age'])
plt.show()


# In[11]:


print(np.corrcoef(young['left'], young['Age']))
print(np.count_nonzero(young['left']) / len(young['left']))
percent1 = np.round(np.count_nonzero(young['left']) / len(young['left']) * 100, decimals=2)
print('{}% of workers aged under 30 leaves the firm'.format(percent1))
print(np.corrcoef(data['left'], data['Age']))
percent = np.round(np.count_nonzero(data['left']) / len(data['left']) * 100, decimals=2)
print('{}% of the total population leaves the firm.'.format(percent))
corr = np.corrcoef(data['young'], data['left'])
for item in corr[1]:
    print(np.round(item * 100, decimals=2),'%')


# I get a 24.55% correlation between my 'young' group and attrition. Not too terrible.

# In[12]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = data['YearsAtCompany']
y = data['Age']
z = data['YearsInCurrentRole']
s = data['WorkLifeBalance']
c = data['left']
cmap = plt.get_cmap('seismic')
_ = ax.scatter(xs=x, ys=y, zs=z, c=c, cmap=cmap, s=s ** 3)
_ = ax.set_xlabel('Years at Company')
_ = ax.set_ylabel('Age')
_ = ax.set_zlabel('Years in Current Role')
_ = plt.title('')
plt.show()


# I seem to observe a group in the middle of our workforce here that has pretty good work/life balance, is middle-aged, has little time at the company, and does not tend to attrit as frequently as the balance of the workforce. Let's 'zoom-in' and see if we can extract some goodness here as well.

# In[13]:


mid = data[(data['Age'] > 35) & (data['Age'] <= 40) & (data['YearsAtCompany'] <= 10) & (data['YearsAtCompany'] > 2) & (data['YearsInCurrentRole'] <= 7)]
data['mid'] = 0
data['mid'][(data['Age'] > 35) & (data['Age'] <= 40) & (data['YearsAtCompany'] <= 10) & (data['YearsAtCompany'] > 2) & (data['YearsInCurrentRole'] <= 7)] = 1

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = mid['YearsAtCompany']
y = mid['Age']
z = mid['YearsInCurrentRole']
s = mid['WorkLifeBalance']
c = mid['left']
cmap = plt.get_cmap('seismic')
_ = ax.scatter(xs=x, ys=y, zs=z, c=c, cmap=cmap)
_ = ax.set_xlabel('Years at Company')
_ = ax.set_ylabel('Age')
_ = ax.set_zlabel('Years in Current Role')
_ = plt.title('')
plt.show()

print(np.count_nonzero(mid['left']) / len(mid['left']))
percent1 = np.round(np.count_nonzero(mid['left']) / len(mid['left']) * 100, decimals=2)
print('{}% of my mid-career workers leave the firm'.format(percent1))
print('There are {} mid-career employees with exceptionally low average attrition in this firm.'.format(len(mid)))
corr = np.corrcoef(data['mid'], data['left'])
for item in corr[1]:
    print(np.round(item * 100, decimals=2),'%')


# This group seems to comprise my most stable mid-career employees. Attrition is markedly lower here than it is throughout the rest of the workforce. This group comprises about 10% of my total workforce. I could actually have gotten the attrition rate down to barely over 4% of this group if I had narrowed it down to an age range of from 36-38, but that's fewer than 100 people and seemed too restrictive to really be valuable. I'm making a subjective call here to increase the size of this group a little.
# 
# This group displays a -8.7% correlation to attrition.
# 
# I tried taking the same approach with my older workers but was not impressed with the results.

# In[14]:


_ = sns.kdeplot(data = data['Age'], data2 = data['TotalWorkingYears'])
_ = plt.scatter(data['Age'], data['TotalWorkingYears'], alpha=.5, s=20, c=data['left'])
_ = plt.xlabel('Age')
_ = plt.ylabel('Tenure (in years)')
_ = plt.title('Age vs. Tenure')
plt.show()

_ = sns.kdeplot(data=data['MonthlyIncome'], data2=data['Age'])
_ = plt.scatter(data['MonthlyIncome'], data['Age'], alpha=0.5, s=20, c=data['left'])
_ = plt.xlabel('Monthly Income')
_ = plt.ylabel('Age')
_ = plt.title('Monthly Income vs. Age')
plt.show()


# Here's something interesting. We have quite a cluster of attrition among young, low-paid employees. I've already captured this group in my 'young' column. We also have a group of highly paid individuals who rarely attrit. 

# In[15]:


data['high_income'] = 0
data['high_income'][(data['Age'] >= 25) & (data['MonthlyIncome'] > 13000)] = 1

count = np.count_nonzero(data['high_income'])
print('There are {} highly paid employees with low average attrition'.format(count))
corr = np.corrcoef(data['high_income'], data['left'])
for item in corr[0]:
    l = []
    l.append(np.round(item * 100, decimals=2))
print('Correlation between this group and attrition is {}%'.format(l[0]))


# Feature Engineering and Preparation for Machine Learning
# --------------------------------------------------------
# 
# Now that I've got a feel for my data I'll engage in a little feature engineering and preparation for classification.
# 
# NOTE: Subsequent to preparing this kernel I learned that I could use the pandas.get_dummies() method to accomplish something very similar. It accomplishes the same thing and has the advantage of being native to pandas, which I always use anyway. I won't make the change to this kernel, though.

# In[16]:


from sklearn.preprocessing import LabelEncoder as LE

data['Attrition'] = LE().fit_transform(data['Attrition'])
data['Department'] = LE().fit_transform(data['Department'])
data['EducationField'] = LE().fit_transform(data['EducationField'])
data['Gender'] = LE().fit_transform(data['Gender'])
data['JobRole'] = LE().fit_transform(data['JobRole'])
data['MaritalStatus'] = LE().fit_transform(data['MaritalStatus'])
data['Over18'] = LE().fit_transform(data['Over18'])
data['OverTime'] = LE().fit_transform(data['OverTime'])
data['BusinessTravel'] = LE().fit_transform(data['BusinessTravel'])
del data['left']
del data['OT']
del data['EmployeeNumber']
del data['EmployeeCount']


# Now let's see if we can extract some goodness with a clustering algorithm. Something I have noticed with HR datasets is that employee behavior seems to exhibit a fair amount of clustering.
# 
# For the sake of intellectual honesty I'm going to split out my train and test sets before I move forward and apply my transformations to them separately. That will reduce the accuracy of my scores in the end probably, but it more accurately reflects the experience of making actual predictions. After all, if I already know what I'm trying to predict, what's the point of predicting it?

# In[17]:


from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

X = data
y = data['Attrition']
del X['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


cluster = KMeans(n_clusters=80, random_state=42).fit_predict(X_train)
X_train['cluster'] = cluster
X_train['cluster'].plot(kind='hist', bins=80)
_ = plt.xlabel('Cluster')
_ = plt.ylabel('Count')
_ = plt.title('Histogram of Clusters')
plt.show()

_ = plt.scatter(x=X_train['Age'], y=X_train['DailyRate'], c=X_train['cluster'], cmap='Blues')
_ = sns.kdeplot(data=X_train['Age'], data2=X_train['DailyRate'])
_ = plt.xlabel('Age')
_ = plt.ylabel('Daily Rate')
_ = plt.title('Clusters within Age/Daily Rate')
plt.show()

x = np.corrcoef(X_train['cluster'], y_train)
print(x)


# It will be interesting to see if this helps. I've chosen to go with 80 clusters by trial and error. You can see in the scatter/kde plot above that most of my clusters occur among employees with a low daily rate, although the majority of my employees are in a single cluster above about 600.
# 
# Modeling
# ========
# 
# Time to get serious! First I'll run the same transformation on my test data that I just did on my train data, then I'll run some classifiers, see which one works best, and tune it for optimal performance.

# In[18]:


cluster = KMeans(n_clusters=80, random_state=42).fit_predict(X_test)
X_test['cluster'] = cluster


# Random Forest
# -------------

# In[19]:


from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score

fmodel = RFC(n_estimators=100, random_state=42, max_depth=11, max_features=11).fit(X_train, y_train)
prediction = fmodel.predict(X_test)
score = accuracy_score(y_test, prediction)
print(score)


# Support Vector Classification
# ----------------------------

# In[20]:


from sklearn.svm import SVC

model = SVC(random_state=42).fit(X_train, y_train)
prediction = model.predict(X_test)
score = accuracy_score(y_test, prediction)
print(score)


# ADA Boost Classifier
# --------------------

# In[21]:


from sklearn.ensemble import AdaBoostClassifier as ABC

model = ABC(n_estimators=100, random_state=42, learning_rate=.80).fit(X_train, y_train)
prediction = model.predict(X_test)
score = accuracy_score(y_test, prediction)
print(score)


# Bagging Classifier
# ----------------

# In[22]:


from sklearn.ensemble import BaggingClassifier as BC

model = BC(n_estimators=100, random_state=42).fit(X_train, y_train)
prediction = model.predict(X_test)
score = accuracy_score(y_test, prediction)
print(score)


# Extra Trees Classifier
# ----------------------

# In[23]:


from sklearn.ensemble import ExtraTreesClassifier as XTC

model = XTC(n_estimators=100, random_state=42, criterion='entropy', max_depth=20).fit(X_train, y_train)
prediction = model.predict(X_test)
score = accuracy_score(y_test, prediction)
print(score)


# Looks like my best models, once the parameters are tuned a little bit, are the Random Forest and the Extra Trees Classifier. All of them perform very similarly however.

# Time for some sleep! Please ***UPVOTE*** if you like what you see.
# ==================================================================
