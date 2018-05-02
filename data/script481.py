
# coding: utf-8

# <h2>KEYWORDS: imputation, missing data, K-nearest neighbor</h2>

# <h1>1. Introduction </h1>
# <p>This dataset gives an excellent opportunity to work/play with a larget dataset containing a number of missing data or "holes". Many methods for imputing missing data are available but which one to select is often a hard question to answer. It will depend on <li>the degree to which data is missing in the whole dataset,</li> <li>whether dataset has a property of continuity,</li> <li>whether the number of complete samples is sufficient to predict and test various imputation methods.</li> Let us first peek through the datset, analyze it, test various imputation methods to determine which ones to use. And then apply them to come up with a complete dataset.</p>

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing
from sklearn.preprocessing import Imputer
from sklearn.neighbors import NearestNeighbors 
import matplotlib.pyplot as plt
from IPython.display import display
import seaborn as sns
sns.set_style('whitegrid')


# <p>First, we will look into the dataset. There are 89 columns in which 84 are monthly water consumption from Jan 2009 to Dec 2015. Other columns are explained by the author <a href="https://www.kaggle.com/forums/f/1525/water-consumption-in-a-median-size-city/t/23722/captions-to-be-explained">click here</a> </p> 

# In[ ]:


aguah = pd.read_csv("../input/AguaH.csv")
aguah.head()
#aguah.info()


# For convenience, I'll rename columns in English to understand better. 

# In[ ]:


lookup = {'ENE':'01','FEB':'02','MAR':'03','ABR':'04','MAY':'05','JUN':'06','JUL':'07','AGO':'08','SEP':'09','OCT':'10','NOV':'11','DIC':'12'}
clist=[]
for col in aguah.columns[5:]:
    col = col[4:].split('_')
    clist.append('20'+col[1]+'-'+lookup[col[0]])

columns = ['LANDUSE_TYPE','USER','PIPE DIAM','VENDOR','JAN16']+clist
aguah.columns=columns
aguah.head()


# <h1>2. Analysis</h1>
# <p>We will see how many entries with null values exist in the dataset. To better visualize, I'll categorize them into groups by the number of NaN values in the rows. Bins = [0,1,10,20,0,40,50,60,70,80,90]</p>

# In[ ]:


## Countplot the number of NaN values for each entry
cons = aguah.iloc[0:, 5:]
cons['NumNull'] = cons.isnull().sum(axis=1)
print('The proportion of entries with non-NaN values is {:.2f}%'.format(len(cons[cons.NumNull==0])/len(cons)*100))
bins = [0,1,10,20,30,40,50,60,70,80,90]
cut = pd.cut(cons.NumNull, bins=bins, include_lowest=True, right=False)
fig, axis1 = plt.subplots(1,figsize=(8,4))
sns.countplot(x=cut, data=cut.to_frame(), ax=axis1)

sns.plt.show()


# <p>It has around 80% of entries without NaN values, in other words, complete entries. We'll need to look at the others (incomplete ones) and try to fill in the holes. These 80% rows of the dataset will give usthe basis of the predictions of missing data to be filled in. </p>
# <p>Now, I'll plot the dataset again to show how the null values are distributed throughout the columns. We can naturally ask whether particular columns have more NaN values than others, or is there any trend in which columns (monthly consumption throughout 7 years) contain missing values.  </p>

# In[ ]:


## Plot graphs to show how values for each entry evolves with time
NumNullwithTime = cons.drop('NumNull', axis=1).isnull().sum()

sns.set_style("darkgrid")
plt.figure(figsize=(10,4))
pbar = NumNullwithTime.plot.bar()
plt.xticks(list(range(0,len(NumNullwithTime.index),6)), list(NumNullwithTime.index[0::6]), rotation=45, ha='right')
plt.show()


# <p>Alright, the trend is very clear. entries with missing values decrease with time and shrink to about 10% (around 34,000 to 3,000) by the end of the observed period. It generally tells us that data are not available in earlier times (or water not in service but only later) but toward the year 2015 more and more data are measured (or water service in place). There are of course other factors that don't go with these assupmtions like random missing data (scattered holes for example), stop in water service, etc. We know, at least in general temrs, that data availability is increasing with time. The question to ask is now whether it is due to (unexpected) data missed during the service time, or due to the fact that water is simply not in service.</p>

# In[ ]:


## Return index of column (0-83) where the first non-NA number appears. If none, return 84
def FirstNonNull(row):
    count=0
    for col in row:
        if col==False: return count
        else: count = count+1
    return count

## Return index of column (0-83) where the last non-NA number appears, If none, return -1
def LastNonNull(row):
    count=0
    flag=-1
    for col in row:
        if col==False:
            flag=count
            count=count+1
        else: count=count+1
    return flag


# <p>I defined above two functions that return indexes of the first and last non-NaN values in rows. It tells you if water service begins later than Jan 2009 or ends before Dec 2015. </p>
# <p>With these functions above, I'm going to create another column called "NullInService" that shows the number of NaN values in rows that occur <b>during</b> each serivce period. It does not count for null values when water is not in service (in other words, before the first non-null and after the last non-null). See the result below.</p>

# In[ ]:


## I need this function for the cases of all NaN entries (NullinService value becomes 0 from -83)
def Setzero(x):
    if x<0: return 0
    else: return x

## Number of NaN values before service period
groupnull = aguah.iloc[:,0:5]
groupnull = pd.concat([groupnull, cons], axis=1)

groupnull['FirstNonNull'] = cons.copy().drop(['NumNull'], axis=1).isnull().apply(FirstNonNull,axis=1)
groupnull['LastNonNull'] = cons.copy().drop(['NumNull'],axis=1).isnull().apply(LastNonNull,axis=1)
groupnull['NullInService'] = groupnull.NumNull - groupnull.FirstNonNull - (len(cons.columns)-1-groupnull.LastNonNull) +1    
groupnull['NullInService'] = groupnull['NullInService'].apply(Setzero)
groupnull.NullInService.value_counts(sort=False).head(6) ## Print only head values


# <p>Okay the result is interesting. Above, we saw that the proportion of the entries with non-values was around 20% of the total number of data. However "NullInService" column tells us that only less than 3% of entire dataset entries have non-values <b>"during"</b> the service period.  Again, non-values during the serice period mean that data is missing abnormally (like empty scenes in video tapes, not before the start scene nor after the ending scene - video tapes... Well we used them in the 90s!)</p>
# <p>Let us now group them into 3 categories: <li>1) complete entries (without NaNs), </li><li>2) entries that start service later or end service earlier (with NaNs at both edges but not during the service period),</li><li>3) entries with NaNs within the service period (I call them "interupted")</li> </p>

# In[ ]:


contLong = groupnull[groupnull.NumNull==0]
contShort = groupnull[(groupnull.NumNull>0) & (groupnull.NullInService==0)]
interupted = groupnull[(groupnull.NullInService)>0]
print('Length of 3 groups: (Non-NA Group, Edge-NA Group, Interupted Group) = ({}, {}, {})'.format(len(contLong), len(contShort), len(interupted)))


# <h1>3. Test methods for filling up missing data </h1>
# <p>We know now that how many entries we have that are incomplete. The next step is to determine methods that we will apply to the dataset to fill in the holes. A number of imputation methods are available out there but which one to use will all depend on what the dataset looks like and what missing values one would like to fill in. Luckily we have a large group of complete entries (141,205) so we can rely on this group to come up with the most probable numbers for the missing values. And once again we can reasonably assume that numbers in rows vary in a more or less cotinuous manner (as one can naturally think how water consumption envolves with time).</p>
# <p>Before diving into the prediction, I'll test which method gives us the best prediction score when playing with the complete entry group. What I'll do is to create holes in this group (so we know the answer) and fill in the missing values using several methods. Let's get started then.</p>

# In[ ]:


## Test various imputation methods using the group of complete entries (Non-NA Group above)
rng = np.random.RandomState()
missing_rate = 0.01  ## Here, 1 for use the whole set of entries to score (0.01 only for display)

## Prepare a scoring set within Non-NA Group 
num_total = len(contLong)
num_score = int(np.floor(missing_rate*num_total))
missing_samp = np.hstack((np.zeros(num_total-num_score, dtype=np.bool), np.ones(num_score, dtype=np.bool)))
rng.shuffle(missing_samp)
cl_score = contLong.iloc[:,5:89][missing_samp.tolist()]

## Columns where holes to be made
col = rng.randint(0, 84, num_score)

## Save the answer set before making the "holes" in the scoring group (1 hole (missing data) per row)
cl_score_orig = cl_score.copy() ## save the original for KNN method (for reference)
answer = cl_score.as_matrix()[np.arange(num_score), col]
cl_score.as_matrix()[np.arange(num_score), col] = np.nan

## Function for scoring (squared mean error by answer and imputed values)
def Impute_error(imputed, answer):
    return np.sqrt(np.square(imputed-answer).sum())/len(answer)


# <p><u>Basic imputation methods</u></p>
# <p>I begin with basic imputation methods - mean, median, most frequent value, forward fill and backward fill. These are provided by the package <a href="http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html">sklearn.preprocessing.Imputer</a> and <a href="http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.fillna.html">Pandas.DataFrame.fillna API</a>. They are simple but very powerful sometimes. I'll try to fill in all NaN values with each of them and display their scores. Forward fill and backward fill need to be complemented by each other as NaN values at head or tail cannot be filled in with only one method.</p>

# In[ ]:


## Start with simple imputation methods: Mean, Median, Most frequent value, Forward fill, and Backward fill
cl_score_mean = cl_score.copy()
imp_mean = Imputer(missing_values='NaN', strategy='mean', axis=1, copy=False)
imp_mean.fit_transform(cl_score_mean)
imputed_mean = cl_score_mean.as_matrix()[np.arange(num_score), col]
                  
cl_score_median = cl_score.copy()
imp_median = Imputer(missing_values='NaN', strategy='median', axis=1, copy=False)
imp_median.fit_transform(cl_score_median)
imputed_median = cl_score_median.as_matrix()[np.arange(num_score), col]

cl_score_mfre = cl_score.copy()
imp_mfre = Imputer(missing_values='NaN', strategy='most_frequent', axis=1, copy=False)
imp_mfre.fit_transform(cl_score_mfre)
imputed_mfre = cl_score_mfre.as_matrix()[np.arange(num_score), col]

## NaN values at head can't be filled with ffill so complement with bfill
cl_score_ffill = cl_score.copy()
cl_score_ffill = cl_score_ffill.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1) 
imputed_ffill = cl_score_ffill.as_matrix()[np.arange(num_score), col]

## NaN values at tail can't be filled with bfill so complement with ffill
cl_score_bfill = cl_score.copy()
cl_score_bfill = cl_score_bfill.fillna(method='bfill', axis=1).fillna(method='ffill', axis=1)
imputed_bfill = cl_score_bfill.as_matrix()[np.arange(num_score), col]

display(Impute_error(imputed_mean, answer))
display(Impute_error(imputed_median, answer))
display(Impute_error(imputed_mfre, answer))
display(Impute_error(imputed_ffill, answer))
display(Impute_error(imputed_bfill, answer))

## Scores when missing_rate is 1 (use the entire Non-NA Group)
## 0.73146667241762797
## 0.74416261219896196
## 1.2776284010594683
## 0.43119183851023574
## 0.44032679793299145


# <p><u>Advanced imputation methods</u></p>
# <p>Interpolate API of Pandas (http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.interpolate.html#pandas.DataFrame.interpolate) provide several advanced methods of data imputation like linear, quadratic, akima, spline etc. More sophisticated, they are computationally heavier. If, however, they give much better scores, it would be worth reflecting on them.</p>

# In[ ]:


## Interpolate() in Pandas

cl_score_linear = cl_score.copy()
cl_score_linear.interpolate(method='linear', axis=1, inplace=True)
cl_score_linear = cl_score_linear.fillna(method='ffill',axis=1).fillna(method='bfill',axis=1)
imputed_linear = cl_score_linear.as_matrix()[np.arange(num_score), col] ## cl_score_linear.values works too

cl_score_akima = cl_score.copy()
cl_score_akima.columns=list(range(cl_score_akima.shape[1]))
cl_score_akima.interpolate(method='akima', axis=1, inplace=True)
cl_score_akima = cl_score_akima.fillna(method='ffill',axis=1).fillna(method='bfill',axis=1)
imputed_akima = cl_score_akima.as_matrix()[np.arange(num_score), col]

cl_score_spline = cl_score.copy()
cl_score_spline.columns=list(range(cl_score_spline.shape[1]))
cl_score_spline.interpolate(method='spline', axis=1, order=2, inplace=True)
cl_score_spline = cl_score_spline.fillna(method='ffill',axis=1).fillna(method='bfill',axis=1)
imputed_spline = cl_score_spline.as_matrix()[np.arange(num_score), col]


display(Impute_error(imputed_linear, answer))
display(Impute_error(imputed_akima, answer))
display(Impute_error(imputed_spline, answer))

## Scores when missing_rate is 1 (use the entire Non-NA Group - Warning. It may take several minutes to complete)
## 0.33370675683889633
## 0.33465191332331751
## 1.8506789840447648


# <p>Alright. The simpler the better maybe? Here the simple linear imputer makes the best score among them and so far. Akima and spline give us a reasonable score for the prediction but computational expenses vary much depending on methods of choice. </p>
# <p>The last method to test is <b>K-nearest neighbor</b>. Basically, I'll use it to find the closest complete row to the row in question and replace the empty value with the one in the complete row we found. Due to the comparison with entire NoN-NA Group rows, this is computationally much more expensive than the ones above. However let us see how good it may get and whether using this method is worthwhile. To compare with complete entries, I'll compare columns (only the columns without NaN values) using the standard K-nearest neighbor. It is reasonable to think that only the columns near the NaN values are worth comparing (you don't need to see consumption too far away in the past nor in the future to guess the consumption this month - but look around a couple of months around).  </p>

# In[ ]:


## Imputation using K-nearest neighbor (to find the k closest row(s) in the sample set to the row x in question)
def KnnImputeSimple(sample, x, k):
    ## Mask the columns with NaN value (not to compare)
    x_mask = x.notnull().tolist()
    x_mask_toggled = x.isnull().tolist()
    sample_masked = sample.iloc[:,x_mask]
    x_masked = x[x_mask]
    
    ## Extent to which column comparison is carried out. Here 6 columns (months) before and after the the column in question
    comp_size = 6
    i = x_mask_toggled.index(True)
    ## Handle when there are less than 4 columns to look at before or after the column in question
    i = max(comp_size, min(i, 84-1-comp_size))
    x_masked = x_masked[i-comp_size:i+comp_size].values
    sample_masked = sample_masked.iloc[:,i-comp_size:i+comp_size]
    
    ## I use kd_tree algorithm here.
    nbrs = NearestNeighbors(k, algorithm='kd_tree', n_jobs=-1)
    nbrs.fit(sample_masked)
    n_ones = nbrs.kneighbors([x_masked])
    
    ## Find k nearest ones and average the predicted values to return
    value = []
    for n in range(x.isnull().sum()):
        temp=[]
        for j in range(k):
            temp.append(sample.iloc[n_ones[1][0][j]][x_mask_toggled][n])
        value.append(np.sum(temp)/k)

    return value


# <p><u>K-nearest neighbor</u><p/>
# <p>Different from the other methods above, we now need sample (reference) entries in which to search the nearest ones. One way to implement this is to divide the whole set into two (<code>set1</code> and <code>set2</code>) and test one set using the other set as the sample set. For instance, we make holes in set1 and for each entry in <code>set1</code> we look at <code>set2</code> to find k-nearest entries to fill in the holes in the entry in question. Vice-versa too. It returns a list of imputed values.<p/>

# In[ ]:


## Setting for K-Nearest Neighbor method  test
div = int(num_score/2)
set1 = cl_score.copy()[0:div]
set2 = cl_score.copy()[div:2*div]

answer_set1 = answer[:div]
answer_set2 = answer[div:2*div]  
    
imputed_knn_set1 = set1.apply(lambda x: KnnImputeSimple(cl_score_orig[div:2*div], x, 2)[0], axis=1)
imputed_knn_set2 = set2.apply(lambda x: KnnImputeSimple(cl_score_orig[:div], x, 2)[0], axis=1)
imputed_knn = imputed_knn_set1.append(imputed_knn_set2)
answer = np.append(answer_set1, answer_set2)

display(Impute_error(imputed_knn, answer))

## Scores when missing_rate is 1 with various variables k and comp_size (use the entire Non-NA Group)
## Warning - It may take several hours to complete if you do it with the entire Non-NA Group i.e. missing_rate=1
## 0.33834147148949706 (k=4, comp_size=4)
## 0.30221719764528121 (k=5, comp_size=3)
## 0.29975805378644343 (k=2, comp_size=6)


# <p>KNN gives the bests estimate for the missing values. A couple of tests with two variables (<code>comp_size</code> and <code>k</code>) make somewhat different scores. The best score (with <code>com_size=6</code> and<code> k=2</code> in my test set) demonstrates a quite good score of <code>0.29975805378644343</code> (for <code>missing_rate=1</code>). However it should be noted its computational time was extremely huge (in an order of several hours - with my average laptop). We cannot therefore use this method for imputing all NaN values for a huge dataset.</p>
# <p>The second best solution was linear function and it was a fast one to calculate (in an order of seconds). And the combination of forward fill and backward fill can come in rescue when linear cannot make prediction on "Edge-NA" values. However I'll leave Edge-NA Group as it is as I consider these NaNs appear in no-service period (that is normal). Naturally, I'll use KNN for most of imputations in our dataset when the number of missing data in a row is not too big. I'll complement it with linear.</p>

# <h1>4. Impute the real missing values</h1>
# <p>Now,  it is time to impute the missing values in the real dataset. My strategy is as follows: As we tested above on the complete entries (Non-NA entries), I will use the best two methods of imputation - KNN and linear. First, I fill in the missing data in "Interupted" entries with KNN but only for those that have up to 5 missing values in a row. Second, rows with many NaN values in "Interupted" group will be imputed using linear method. "Edge-NA" entries will not be imputed as I explained above.</p>

# <p>Let me define first a function that implements imputation using <code>KnnImputeSimple</code> above. It finds the service period and imputes missing values within this period. It returns an imputed row.</p>

# In[ ]:


## Carry out imputation work using KnnImputeSimple function defined above
def KnnImputeInPlace(sample, itrp, k):
    fn = itrp['FirstNonNull']
    ln = itrp['LastNonNull']
    serv = itrp[int(fn):int(ln)+1]
    
    ## Indicate the index of NaNs in row
    index = serv[serv.isnull()].index.tolist()
    itrp_imputed = itrp
    for i in range(len(index)):
        idx = index.copy()
        ## Ignore the other NaN columns for i-th NaN calculation
        idx.pop(i)
        imp = KnnImputeSimple(sample.iloc[:,int(fn):int(ln)+1].drop(sample[idx],axis=1), serv.drop(idx), k)
        itrp_imputed.set_value(index[i],imp[0])
    
    return itrp_imputed


# <p>I'll apply <code>KnnImputeSimple</code> only to those entries with NaN values between 1 and 5 (that is still the majority of the interupted group). When missing values are numerous in one row, they tend to be clustered all together. It also makes sense to use the linear method rather than Knn.</p>

# In[ ]:


## For KNN, choose the size of sample table (same as missing_rate above)
rng = np.random.RandomState(0)
samp_rate = 0.01  ## Here, 1 for use the whole set of entries to search (0.01 only for display)

## Prepare a reference set within Non-NA Group 
num_total = len(contLong)
num_samp = int(np.floor(samp_rate*num_total))
rand_samp = np.hstack((np.zeros(num_total-num_samp, dtype=np.bool), np.ones(num_samp, dtype=np.bool)))
rng.shuffle(rand_samp)
sample = contLong.iloc[:,5:89][rand_samp.tolist()]


# In[ ]:


## KNN imputation (warning it may take a couple of minutes)
itrp = interupted.copy()[(interupted.NullInService>0) & (interupted.NullInService<6)].iloc[:,5:]
result_knn = itrp.apply(lambda x: KnnImputeInPlace(sample, x, 2), axis=1)
result_knn_head = interupted.copy()[(interupted.NullInService>0) & (interupted.NullInService<6)].iloc[:,0:5]
result_knn = pd.concat([result_knn_head, result_knn],axis=1)
result_knn.shape


# <p>Apply linear methods for the rest of Interupted Group</p>

# In[ ]:


## Linear imputation 
itrp = interupted.copy()[(interupted.NullInService>5) & (interupted.NullInService<84)].iloc[:,5:]
itrp.interpolate(method='linear', axis=1, inplace=True)
result_fill_head = interupted.copy()[(interupted.NullInService>5) & (interupted.NullInService<84)].iloc[:,0:5]
result_fill = pd.concat([result_fill_head, itrp],axis=1)
result_fill.shape


# <p>Now it's all done. Let me finalize the complete dataset that's been imputed and check its length is the same.  </p>

# In[ ]:


result = pd.concat([contLong, contShort, result_knn, result_fill])
result = result.iloc[:,:89]
result.shape


# <p>Let me plot a few graphs with imputed values highlighted in red</p>

# In[ ]:


## Plot a few imputed graphs by highlighting imputed values in red
np.random.seed(14)
num_example = 10
pltnum = np.random.randint(0,len(interupted), num_example)
before = interupted.iloc[pltnum]

idx = before.index[list(range(num_example))]
after =result.ix[idx]
before = before.iloc[:,5:89]
after = after.iloc[:,5:89]

get_ipython().run_line_magic('matplotlib', 'inline')
fig, ax = plt.subplots(5,2, figsize=(10,10))

for i in range(num_example):
    #plt.plot(before, color='r')
    after.iloc[i].plot(kind='line',color='r', alpha=0.8, ax=ax[i//2,i%2])
    #plt.subplot(5,2,i+1)
    before.iloc[i].plot(kind='line',color='b', linewidth=1.5, alpha=1, ax=ax[i//2,i%2])
    #plt.subplot(5,2,i+1)
    
plt.setp(ax, xticks=list(range(6,len(NumNullwithTime.index),24)), xticklabels=list(NumNullwithTime.index[6::24]))
plt.tight_layout()
#plt.show()


# <h1>5. Conclusion
# </h1>
# <p>It was a fun exercise to practice impute missing values. I analyzed the data set into three categorical groups namely - Non-NA, Edge-NA, Interupted. Non-NA is the groupwith complete entries, Edge-NA is the one with NaN values at both extremes (it is nothing wrong I guess as the service may start later or end earlier.). The last group is Interupted with NaN value(s) appeared during <i>service</i> period. The entire data set is relatively big however the entries that need to be imputed (Interupted) are about 3% of the dataset. As per my analysis above, the imputation method based on K-nearestneight - which was the best during the test - was used for entries with small number of missing values. Entries with large number of missing values within the service period was imputed by linear method. Some instances of the imputed dataset are presented above. It looks quite reasonable even though I used only 1% of the entire dataset sample for KNN.</p>

# <h1>End Note</h1>
# <p>I didn't impute Null values at both extremes (Edge-NAs) as I considered these as normal data. However if one needs to impute these too, then I suggest using ffill and bfill methods to do so. I didn't touch Jan2016 column as I am not convinced that this column contains the right values (too much variation from the previous month, 2015-12). One might take this as part of the dataset for analysis. </p>
# 
# <p>It became a quite long notebook but I had fun testing it. Any question or comment will be much appreciated!</p>
