
# coding: utf-8

# # Why are so many employees leaving? and Who exactly are these employees?

# ** First let's do some exploratory analysis on our data! **

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")

hr_data = pd.read_csv('../input/HR_comma_sep.csv')
hr_data['dept'] = hr_data['sales']
hr_data.drop(['sales'], axis = 1, inplace = True)
hr_data.head()


# In order to make the data easier to understand I will add a column that gives the *salary* column a numerical value as follows:
# 
# Salary Description | Salary Value 
# --- | ---
# low | 30,000
# medium | 60,000
# high | 90,000

# In[ ]:


salary_dict = {'low': 30000, 'medium': 60000, 'high': 90000}
hr_data['salary_estimate'] = hr_data['salary'].map(salary_dict)


# Furthermore, I am interested in performing analysis on this data in the following groups:
# 
# - Good performers who left the company
# - Good performers who stayed with the company
# - Average/Below Average performers who left the company
# - Average/Below A performers who stayed with the company
# 
# I think it will be interesting to see the similarities and differences between these groups.
# 
# ** What is an Above Average performer?**
# 
# *I will define an above average performer as anyone who's performance evaluation was at least 1 standard deviation above the mean(Top 84th percentile). Everyone else is average or below average.*

# In[ ]:


eval_mean = hr_data['last_evaluation'].mean()
eval_std = np.std(hr_data['last_evaluation'])
hr_data['performance(standard units)'] = (hr_data['last_evaluation']- eval_mean)/eval_std

def performance_label(row):
    performance = row['performance(standard units)']
    if performance >=1:
        result = 'Above Average'
    else:
        result = 'Average or Below'
    return(result)
hr_data['performance label'] = hr_data.apply(performance_label, axis = 1)

left_dict = {1: 'left', 0: 'stayed'}

hr_data['left(as_string)'] = (hr_data['left'].map(left_dict))


# Next we'll take a look at the average values for numerical columns in our data as an initial look into the potential problem.

# In[ ]:


columns = (hr_data.columns)
num_columns = (hr_data._get_numeric_data().columns)

sep_hr_data = hr_data
sep_hr_data['Performance cluster'] = sep_hr_data['left(as_string)'] + ' : ' + sep_hr_data['performance label']

sep_hr_pivot = sep_hr_data.pivot_table(index= (['Performance cluster']), values =num_columns, aggfunc=np.mean)
sep_hr_pivot.transpose()


# The values that immediately jump out to me are the average_montly_hours, number_project, salary_estimate, and promotion_last_5years, so I will expore the more in depth.

# In[ ]:


sep_hr_data[['Performance cluster', 'average_montly_hours']].boxplot(by = 'Performance cluster')
plt.xticks(rotation = 10)
plt.show()


# Good performing employees who left the company were averaging 255 hours per month, which equates to about **12 hours per day(based on a 250 days in a year)**.
# All other groups were averaging **9 to 10 hours per day.**

# In[ ]:


sep_hr_data[['Performance cluster', 'number_project']].boxplot(by = 'Performance cluster')
plt.xticks(rotation = 10)
plt.show()


# Similarly, the Above Average performers who left, were more likely to take on projects, which most likely correlates with number of work hours.

# In[ ]:


colors = ['blue', 'green', 'red', 'purple']
for i, category in enumerate(sep_hr_data['Performance cluster'].unique()):
    hist_bin = sep_hr_data['salary_estimate'][sep_hr_data['Performance cluster'] == category]
    plt.hist(hist_bin,3, alpha=1, label=category, normed = True, linewidth=2, facecolor = 'none', edgecolor = colors[i])
plt.legend(loc='upper right')
plt.title('Salary by Perforance Group')
plt.show()


# Above, this histogram shows us that a high percentage of those who left the company fell into the low income category. Those who stayed with the company were more likely to fall into the medium to high income category.

# **Next, I want to take a look at the correlations between certain features for Above Average vs Average/Below Average performs who left the company.**

# ## Correlations for Above Average Performers

# In[ ]:


columns = ['left','average_montly_hours', 'number_project','time_spend_company']

aa_sep_hr_data = sep_hr_data[(sep_hr_data['performance label']=='Above Average')]
ab_sep_hr_data = sep_hr_data[(sep_hr_data['performance label']!='Above Average')]
average_corr = ab_sep_hr_data.corr()
above_average_corr = aa_sep_hr_data.corr().loc[columns].transpose()
below_average_corr = ab_sep_hr_data.corr().loc[columns].transpose()
above_average_corr


# ## Correlations for Below Average Performers

# In[ ]:


below_average_corr


# ### Key Points 
# 
# - For Above Average Performers:
#     - Leaving the company is highly correlated with working many hours, taking on many projects and time spent with the company
#     - Monthly hours and number of projects is highly correlated with a high level of dissatisfaction
#     - Time spent with the company also seemed to correlate with number of projects taken on
#     - At the same time, there was no correlation between number of projects and promotions or higher performance evaluations
#     
#     
# - For Average/Below Average Performers:
#     -  Leaving the company is highly correlated satisfaction level. No other metric is as correlated
#     -  For this group, as opposed to the Above Average group, Monthly hours and number of projects is highly correlated with higher performance reviews.

# ## Thoughts....
# 
# **Above Average Performers** seem to be overworked. Because they are above average performers, they may be given many difficult projects by their managers, that they may not be fully prepared for, resulting in a working overtime and high levels of stress and dissastisfaction. Also, they may be getting average performance reviews for these particular projects, resulting in little to no salary increases, or promotions. As stated earlier, time seems to be highly correlated with leaving the company. Due to lack of praise at work, and lack of pay increases, time may be the biggest factor in leaving for these valueable employees.
# 
# 
# **Average/Below Average Performers** do not seem to be overworked. These employees seem to be leaving the company because they are just very unsatisfied. It does seem that these employees are getting higher performance reviews based on the number of projects that they are taking on. I believe that these employees are given less projects, due to past underperformace, and are also given easier projects. This is resulting is higher performance reviews when these 'easy' projects are completed, but are not accompanied by promotions or salary increases.

# ### Next, I'll look into trends by department..

# In[ ]:


left_data = sep_hr_data[sep_hr_data.left == 1]
sep_hr_data.dept.value_counts().plot(kind='bar')
left_data.dept.value_counts().plot(kind='bar', color = 'red')
plt.title =('Share of Employees that Left by Department')
plt.xlabel('Departments')
plt.ylabel('Employee Count')
plt.show()
percent_left = round(left_data.dept.value_counts()/ sep_hr_data.dept.value_counts() * 100, 2)
print('Percentage of employees that left by department \n\n', percent_left.sort_values(ascending = False))


# - It also looks like HR and Accounting have employees that are most likely to leave, while RandD, Product mgt, and Marketing are amongst the least likely to leave.
# 
# Lets further explore why this may be happening.

# In[ ]:


sep_hr_data['salary_estimate_per_hour'] = sep_hr_data.salary_estimate/(sep_hr_data.average_montly_hours *12)

hr_data_by_dept = sep_hr_data.groupby(['dept']).mean()

for col in hr_data_by_dept.columns:
    hr_data_sorted = hr_data_by_dept.sort(col, ascending = False)
    hr_data_sorted[col].plot(kind = 'bar')
    plt.ylabel(col)
    plt.xticks( rotation = 30)
    plt.show()


# ## Above, I've created bar graphs which give us more insights into the working trends by department.
# ### Here is a summary of the above data:
# #### *Although HR has high turnover, I've ommitted HR from further analysis because their department is closely tied to recruitment and people managment, so the high turnover in their department may actually be due to the the high turnover in other departments.
#     
# Category |Marketing, R&D, & Prod Mgmt | Accounting, Tech & Support 
# --- | --- | ---
# Turnover|Least likely to leave | Most likely to leave
# Satisfaction|Most satified | Least satisfied
# Performance Evaluation|low performance evaluations | high performance evaluations
# Hours worked|Amongst the lowest|Accounting & Technical amongst the highest
# Promotions|Marketing & R&D are the most promoted|Amongst least promoted
# 
# ### - Accounting, Tech and support have some of the hardest working people as far as hours, and they generally have higher performance evaluations, yet they are amongst the least satisfied, and least promoted.
# 
# ### - Marketing, R&D, & Prod Mgmt generally work less as far as hours, do not do so well on evaluations, yet they are amongst the most satisfied, and most likely to get promoted.
# 
# ### Promotion to mangement results in a large boost in pay, so it is easy to see why not being promoted, would result in leaving the company.
# 
# ### While, the name of the company is undisclosed, I would guess that this company is largely dependant on Marketing and R&D, which is why those folks tend to get promoted more easily. This may be perceived as unjust by good employees of other departments, which is resulting in the high turnover by good employees.

# ## Next, I will apply machine learning techniques, in order to understand which models work best in predicting which employees will leave next.

# In[ ]:


hr_data1 = hr_data.drop(['performance label', 'left(as_string)', 'Performance cluster',                         'salary_estimate', 'satisfaction_level', 'performance(standard units)', 'Work_accident'], axis = 1)
dummies = pd.get_dummies(hr_data1[['dept', 'salary']])
hr_data1 = pd.concat([hr_data1, dummies], axis=1)
hr_data1.drop(['dept', 'salary'], axis = 1, inplace = True)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_predict, KFold

cols = hr_data1.columns
cols = cols.drop("left")
features = hr_data1[cols]
target = hr_data1["left"]

lr = LogisticRegression()
kf = KFold(features.shape[0], random_state=1)

predictions = cross_val_predict(lr, features, target, cv=kf)
predictions = pd.Series(predictions)


# In[ ]:


hr_data1.head()


# In[ ]:


def show_confusion_matrix(C,class_labels=['0','1']):
    """
    C: ndarray, shape (2,2) as given by scikit-learn confusion_matrix function
    class_labels: list of strings, default simply labels 0 and 1.

    Draws confusion matrix with associated metrics.
    """
    assert C.shape == (2,2), "Confusion matrix should be from binary classification only."
    
    # true negative, false positive, etc...
    tn = C[0,0]; fp = C[0,1]; fn = C[1,0]; tp = C[1,1];

    NP = fn+tp # Num positive examples
    NN = tn+fp # Num negative examples
    N  = NP+NN

    fig = plt.figure(figsize=(8,8))
    ax  = fig.add_subplot(111)
    ax.imshow(C, interpolation='nearest', cmap=plt.cm.gray)

    # Draw the grid boxes
    ax.set_xlim(-0.5,2.5)
    ax.set_ylim(2.5,-0.5)
    ax.plot([-0.5,2.5],[0.5,0.5], '-k', lw=2)
    ax.plot([-0.5,2.5],[1.5,1.5], '-k', lw=2)
    ax.plot([0.5,0.5],[-0.5,2.5], '-k', lw=2)
    ax.plot([1.5,1.5],[-0.5,2.5], '-k', lw=2)

    # Set xlabels
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(class_labels + [''])
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    # These coordinate might require some tinkering. Ditto for y, below.
    ax.xaxis.set_label_coords(0.34,1.06)

    # Set ylabels
    ax.set_ylabel('True Label', fontsize=16, rotation=90)
    ax.set_yticklabels(class_labels + [''],rotation=90)
    ax.set_yticks([0,1,2])
    ax.yaxis.set_label_coords(-0.09,0.65)


    # Fill in initial metrics: tp, tn, etc...
    ax.text(0,0,
            'True Neg: %d\n(Num Neg: %d)'%(tn,NN),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(0,1,
            'False Neg: %d'%fn,
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(1,0,
            'False Pos: %d'%fp,
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))


    ax.text(1,1,
            'True Pos: %d\n(Num Pos: %d)'%(tp,NP),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    # Fill in secondary metrics: accuracy, true pos rate, etc...
    ax.text(2,0,
            'False Pos Rate: %.2f'%(fp / (fp+tn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(2,1,
            'True Pos Rate: %.2f'%(tp / (tp+fn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(2,2,
            'Accuracy: %.2f'%((tp+tn+0.)/N),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(0,2,
            'Neg Pre Val: %.2f'%(1-fn/(fn+tn+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))

    ax.text(1,2,
            'Pos Pred Val: %.2f'%(tp/(tp+fp+0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w',boxstyle='round,pad=1'))


    plt.tight_layout()
    plt.show()


# ## First: Logistic Regression

# In[ ]:


C = confusion_matrix(hr_data1.left,predictions)
show_confusion_matrix(C, ['Stayed', 'Left'])


# - As we can see, this Logistic Regression model is only yielding a 70% accuracy.
# - False Postive Rate is low at just 9%, which is good.
# - True Positive Rate is also low at 3%, which is bad.
# - This model is not very good at predicting who is likely to leave.

# ## Lets try a Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(class_weight="balanced", random_state=1)
predictions1 = cross_val_predict(rf, features, target, cv=kf)
predictions1 = pd.Series(predictions1)

C1 = confusion_matrix(hr_data1.left,predictions1)
show_confusion_matrix(C1, ['Stayed', 'Left'])


# - A Random Forest Model is a better model with a 97% accuracy
# - It is able to maintain a low False Positive Rate at 2%
# - True Positive Rate is high at 93%
# - Overall, this model does a better job is prediciting whihc employees will leave next.
