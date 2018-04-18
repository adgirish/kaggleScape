
# coding: utf-8

# # What are common characteristics of employees lost in attrition compared to those who stay in IBM's fictional dataset? 
# ## We will be using point plots, box plots, kernel density diagrams, means, standard deviations, and z-tests to explore this question.

# ----------
# 
# 
# ## Set Up Dataset

# In[ ]:


from pandas import read_csv
data = read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")


# In[ ]:


target = "Attrition"


# In[ ]:


feature_by_dtype = {}
for c in data.columns:
    
    if c == target: continue
    
    data_type = str(data[c].dtype)
    
    if data_type not in feature_by_dtype.keys():
         feature_by_dtype[data_type] = [c]
    else:
        feature_by_dtype[data_type].append(c)

feature_by_dtype
feature_by_dtype.keys()


# In[ ]:


objects = feature_by_dtype["object"]


# In[ ]:


remove = ["Over18"]


# In[ ]:


categorical_features = [f for f in objects if f not in remove]


# In[ ]:


int64s = feature_by_dtype["int64"]


# In[ ]:


remove.append("StandardHours")
remove.append("EmployeeCount")


# In[ ]:


count_features = []
for i in [i for i in int64s if len(data[i].unique()) < 20 and i not in remove]:
    count_features.append(i)


# In[ ]:


count_features = count_features #+ ["TotalWorkingYears", "YearsAtCompany", "HourlyRate"]


# In[ ]:


remove.append("EmployeeNumber")


# In[ ]:


numerical_features = [i for i in int64s if i not in remove]


# ----------
# 
# 
# # Numerical Features

# In[ ]:


data[numerical_features].head()


# ----------

# # Python Source Code

# In[ ]:


def display_ttest(data, category, numeric):
    output = {}
    s1 = data[data[category] == data[category].unique()[0]][numeric]
    s2 = data[data[category] == data[category].unique()[1]][numeric]
    from scipy.stats import ttest_ind
    t, p = ttest_ind(s1,s2)
    from IPython.display import display
    from pandas import DataFrame
    display(DataFrame(data=[{"t-test statistic" : t, "p-value" : p}], columns=["t-test statistic", "p-value"], index=[category]).round(2))

def display_ztest(data, category, numeric):
    output = {}
    s1 = data[data[category] == data[category].unique()[0]][numeric]
    s2 = data[data[category] == data[category].unique()[1]][numeric]
    from statsmodels.stats.weightstats import ztest
    z, p = ztest(s1,s2)
    from IPython.display import display
    from pandas import DataFrame
    display(DataFrame(data=[{"z-test statistic" : z, "p-value" : p}], columns=["z-test statistic", "p-value"], index=[category]).round(2))
    
def display_cxn_analysis(data, category, numeric, target):
    
    from seaborn import boxplot, kdeplot, set_style, distplot, countplot
    from matplotlib.pyplot import show, figure, subplots, ylabel, xlabel, subplot, suptitle
    
    not_target = [a for a in data[category].unique() if a != target][0]
    
    pal = {target : "yellow",
          not_target : "darkgrey"}
    

    set_style("whitegrid")
    figure(figsize=(12,5))
    suptitle(numeric + " by " + category)

    # ==============================================
    
    p1 = subplot(2,2,2)
    boxplot(y=category, x=numeric, data=data, orient="h", palette = pal)
    p1.get_xaxis().set_visible(False)

    # ==============================================
    
    if(numeric in count_features):
        p2 = subplot(2,2,4)
        
        s2 = data[data[category] == not_target][numeric]
        s2 = s2.rename(not_target) 
        countplot(s2, color = pal[not_target])
        
        s1 = data[data[category] == target][numeric]
        s1 = s1.rename(target)
        ax = countplot(s1, color = pal[target])
        
        ax.set_yticklabels([ "{:.0f}%".format((tick/len(data)) * 100) for tick in ax.get_yticks()])
        
        ax.set_ylabel("Percentage")
        ax.set_xlabel(numeric)
        
    else:
        p2 = subplot(2,2,4, sharex=p1)
        s1 = data[data[category] == target][numeric]
        s1 = s1.rename(target)
        kdeplot(s1, shade=True, color = pal[target])
        #distplot(s1,kde=False,color = pal[target])

        s2 = data[data[category] == not_target][numeric]
        s2 = s2.rename(not_target)  
        kdeplot(s2, shade=True, color = pal[not_target])
        #distplot(s2,kde=False,color = pal[not_target])

        #ylabel("Density Function")
        ylabel("Distribution Plot")
        xlabel(numeric)
    
    # ==============================================
    
    p3 = subplot(1,2,1)
    from seaborn import pointplot
    from matplotlib.pyplot import rc_context

    with rc_context({'lines.linewidth': 0.8}):
        pp = pointplot(x=category, y=numeric, data=data, capsize=.1, color="black", marker="s")
        
    
    # ==============================================
    
    show()
    
    #display p value
    
    if(data[category].value_counts()[0] > 30 and data[category].value_counts()[1] > 30):
        display_ztest(data,category,numeric)
    else:
        display_ttest(data,category,numeric)
    
    #Means, Standard Deviation, Absolute Distance
    table = data[[category,numeric]]
    
    means = table.groupby(category).mean()
    stds = table.groupby(category).std()
    
    s1_mean = means.loc[data[category].unique()[0]]
    s1_std = stds.loc[data[category].unique()[0]]
    
    s2_mean = means.loc[data[category].unique()[1]]
    s2_std = means.loc[data[category].unique()[1]]
    
    print("%s Mean: %.2f (+/- %.2f)" % (category + " == " + str(data[category].unique()[0]),s1_mean, s1_std))
    print("%s Mean : %.2f (+/- %.2f)" % (category + " == " + str(data[category].unique()[1]), s2_mean, s2_std))
    print("Absolute Mean Diferrence Distance: %.2f" % abs(s1_mean - s2_mean))


# In[ ]:


def get_p_value(s1,s2):
    
    from statsmodels.stats.weightstats import ztest
    from scipy.stats import ttest_ind
    
    if(len(s1) > 30 & len(s2) > 30):
        z, p = ztest(s1,s2)
        return p
    else:
        t, p = ttest_ind(s1,s2)
        return p
    
def get_p_values(data, category, numerics):
    
    output = {}
    
    for numeric in numerics:
        s1 = data[data[category] == data[category].unique()[0]][numeric]
        s2 = data[data[category] == data[category].unique()[1]][numeric]
        row = {"p-value" : get_p_value(s1,s2)}
        output[numeric] = row
    
    from pandas import DataFrame
    
    return DataFrame(data=output).T

def get_statistically_significant_numerics(data, category, numerics):
    df = get_p_values(data, category, numerics)
    return list(df[df["p-value"] < 0.05].index)

def get_statistically_non_significant_numerics(data, category, numerics):
    df = get_p_values(data, category, numerics)
    return list(df[df["p-value"] >= 0.05].index)
    
def display_p_values(data, category, numerics):
    from IPython.display import display
    display(get_p_values(data, category, numerics).round(2).sort_values("p-value", ascending=False))


# In[ ]:


significant = get_statistically_significant_numerics(data,target,numerical_features) 
ns = get_statistically_non_significant_numerics(data,target,numerical_features)


# ----------
# 
# # Statistically Significant Numerical Features

# In[ ]:


i = iter(significant)


# ## The fictional company on average loses staff that are 3 - 4 years younger than those who stay.

# In[ ]:


display_cxn_analysis(data, target, next(i), "Yes")


# ## Employees lost in attrition tend to have lower daily rates than those who stay.
#  - Each of the group are 180 degrees flipped from each other in their kernel density diagram

# In[ ]:


display_cxn_analysis(data, target, next(i), "Yes")


# ## Employees lost in attrition tend to have longer commute distances than those who stay.

# In[ ]:


display_cxn_analysis(data, target, next(i), "Yes")


# # Employees lost in attrition are less satisfied with their work environment on average than those who stay.

# In[ ]:


display_cxn_analysis(data, target, next(i), "Yes")


# ## Employees lost in attrition are less involved with their jobs on average than those who stay.

# In[ ]:


display_cxn_analysis(data, target, next(i), "Yes")


# ## Employees lost in attrition tend to be lower in job level than those who stay.

# In[ ]:


display_cxn_analysis(data, target, next(i), "Yes")


# ## Employees who stay have more job satisfication than employees lost in attrition

# In[ ]:


display_cxn_analysis(data, target, next(i), "Yes")


# ## Employees lost in attrition tend to have lower monthly average income on average than those who stay.

# In[ ]:


display_cxn_analysis(data, target, next(i), "Yes")


# ## Employees who stay tend to have more stock options than those lost in attrition.

# In[ ]:


display_cxn_analysis(data, target, next(i), "Yes")


# ## Employees lost in attrition had less total working years than those who stay.

# In[ ]:


display_cxn_analysis(data, target, next(i), "Yes")


# ## Employees lost in attrition had less training opportunities than those who stay.

# In[ ]:


display_cxn_analysis(data, target, next(i), "Yes")


# ## Employees lost in attrition had poorer work-life balance on average than those who stay.

# In[ ]:


display_cxn_analysis(data, target, next(i), "Yes")


# ## Employees who stay had longer organization tenure than those lost in attrition by 2 years on average.

# In[ ]:


display_cxn_analysis(data, target, next(i), "Yes")


# ## Employees who stayed had 1 - 2 more years in their current role than those lost in attrition.

# In[ ]:


display_cxn_analysis(data, target, next(i), "Yes")


# ## Employees lost in attrition had less time with their current manager by 1 - 2 years on average than those who stay.

# In[ ]:


display_cxn_analysis(data, target, next(i), "Yes")


# ## Employees who stay are more satisfied with their work environment on average than those who leave.

# ----------
# # Non-Significant Features

# In[ ]:


for n in ns:
    print(n)
    
    display_cxn_analysis(data, target, n, "Yes")


# ----------
# 
# 
# ### Thank you for reading. Please upvote if you liked it or leave a critique for this report so I can improve.
# 
# ### Read more:
# - [IBM Employee Attrition Analysis by Category][1]
#   [1]: https://www.kaggle.com/slamnz/d/pavansubhasht/ibm-hr-analytics-attrition-dataset/ibm-employee-attrition-analysis-by-category/
