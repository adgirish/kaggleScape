
# coding: utf-8

# # Simple Anomaly detection with H2O in Python
# 
# ### About dataset: 
# This data is a collection of metrics of various students a state of India. The goal was to gather as much information possible to determine if a given student would continue his/her schooling or dropout. Future dropout rates and ways to minimize this, was the ultimate goal of the data collection.
# 
# We would want Autoencoding and Anomaly detction from H2O, to differentiate the data between students who dropped out and of those who did not. 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import h2o
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler,Normalizer
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator
from pylab import rcParams
rcParams['figure.figsize']=15,10


# In[ ]:


student=pd.read_csv('../input/studentDropIndia_20161215.csv', sep=',')
student.isnull().any()


# As shown above, *total_toilets *and *establishment_year* have Null values

# In[ ]:


#student.dtypes
#student[pd.isnull(student).any(axis=1)]


# ## Basic exploratory data analysis

# ### Fill NA with 0

# In[ ]:


student=student.fillna(0)


# ### Print correlation matrix

# In[ ]:


f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(student.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


# ### Students' Marks in Math is absolutely correlated with Science 

# In[ ]:


labels = ['continue', 'drop']
sizes = [student['continue_drop'].value_counts()[0],
         student['continue_drop'].value_counts()[1]
        ]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True)
ax1.axis('equal')
plt.title('Continue vs Dropout Pie Chart', fontsize=20)
plt.show()


# 95.3% Students continued in school, whereas 4.7% dropped

# ### List column

# In[ ]:


predictors=list(range(0,15))
print(student.shape)


# ### H2O cannot use columns with character datatype. Creating Dummy variables instead

# In[ ]:


cols_to_transform = [ 'continue_drop','gender','caste','guardian','internet' ]
student = pd.get_dummies( student,columns = cols_to_transform )
student.head()


# ### Dropping student_id column

# In[ ]:


student = student.drop('student_id', 1)


# ### Ensuring all the columns are of numeric datatype

# In[ ]:


student.dtypes


# ### Standardize input data

# In[ ]:


# Copy the original dataset
scaled_features = student.copy()

# Extract column names to be standardized
col_names = ['mathematics_marks','english_marks','science_marks',
             'science_teacher','languages_teacher','school_id',
             'total_students','total_toilets','establishment_year'#,
             #'gender_F','gender_M','caste_BC','caste_OC','caste_SC',
             #'caste_ST','guardian_father','guardian_mixed','guardian_mother',
            # 'guardian_other','internet_False','internet_True'
            ]

# Standardize the columns and re-assingn to original dataframe
features = scaled_features[col_names]
scaler = RobustScaler().fit_transform(features.values)
features = pd.DataFrame(scaler, index=student.index, columns=col_names)
scaled_features [col_names] = features
scaled_features.head()


# ### Split dataset - dropped students as 'test' and continued students as 'train'

# In[ ]:


#student = student.astype(object)

train=scaled_features.loc[scaled_features['continue_drop_continue'] == 1]
test=scaled_features.loc[scaled_features['continue_drop_drop'] == 1]


# ## H2O Autoencoding and Anomaly detection

# ### Starting H2O cluster

# In[ ]:


h2o.init(nthreads=-1, enable_assertions = False)


# ### Convert panda dataframe to H2O dataframe

# In[ ]:


train.hex=h2o.H2OFrame(train)
test.hex=h2o.H2OFrame(test)


# ### Create AutoEncoder Model

# In[ ]:


model=H2OAutoEncoderEstimator(activation="Tanh",
                              hidden=[120],
                              ignore_const_cols=False,
                              epochs=100
                             )


# ### Train the model with training dataset

# In[ ]:


model.train(x=predictors,training_frame=train.hex)


# ### Print the output in JSON format

# In[ ]:


model._model_json['output']


# ### Get anomalous values

# In[ ]:


test_rec_error=model.anomaly(test.hex)
train_rec_error=model.anomaly(train.hex)


# ### Convert output to dataframe

# In[ ]:


test_rec_error_df=test_rec_error.as_data_frame()
train_rec_error_df=train_rec_error.as_data_frame()
final = pd.concat([train_rec_error_df, train_rec_error_df])


# ### Calculate top whisker value

# In[ ]:


boxplotEdges=final.quantile(.75)
iqr = np.subtract(*np.percentile(final, [75, 25]))
top_whisker=boxplotEdges[0]+(1.5*iqr)
top_whisker


# ### Add id column to dataframe 

# In[ ]:


train_rec_error_df['id']=train_rec_error_df.index
test_rec_error_df['id']=test_rec_error_df.index + 18200 #Count of train data


# ### Scatter plot with top whisker

# In[ ]:


plt.scatter(train_rec_error_df['id'],train_rec_error_df['Reconstruction.MSE'],label='Continued Students',s=1)
plt.axvline(x=18200,linewidth=1)
plt.scatter(test_rec_error_df['id'],test_rec_error_df['Reconstruction.MSE'],label='Dropped Students',s=1)
plt.axhline(y=top_whisker,linewidth=1, color='r')
plt.legend()


# ## Output:
# 
# We have trained the model to detel the students who continued in school. From the graph you can see ***all the students who dropped*** have been correctly classfifed as **Outliers**

# In[ ]:


h2o.cluster().shutdown()


# ## Reference :
# https://charleshsliao.wordpress.com/2017/06/26/denoise-with-auto-encoder-of-h2o-in-python-for-mnist/
# http://benalexkeen.com/feature-scaling-with-scikit-learn/
