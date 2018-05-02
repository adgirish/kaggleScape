
# coding: utf-8

# # Visual Analysis 
# 
# In this notebook I will try to prove some graphical analysis so get a better picture of the database, the goals of the project and how you can improve your score using interesting characteristics of the features. Fisrt lets talk shortly about the databases.
# 
# ### train dataframe
# DonorsChose.org provide us a interesting dataset with a lot of string type variables. This means that the project and the submission that we made will depend conbsiderably on how we build categorical variables and create combinations of these categories to prdict with more accuracy the probability of each application of beeing aproved.
# * **Variables:**
#     * id: ID of the application. For each ID application we need to calculate the probability of this application being pproved.
#     * **teacher_id: **ID of the teacher that is presenting the application. 
#     * **teacher_prefix: ** The title is important, it give uys information about the  gender and the title of the teacher presenting the project.
#         * ['Ms.', 'Mrs.', 'Mr.', 'Teacher', 'Dr.', nan].
#     * **school_state: ** In which state the project will be develop in case it's accepted. I'm not sure but maybe there could be some budget restrictions per state so some states may accept more projects because they have more money.
#     * **project_submitted_datetime: **  The submitted datetime of the project (No way!!) . This feature it's important for many reasons:
#         * There could be some seasonality in the events.
#         * We have to check for specific months in which we found more accpeted projects or a more accpetance rate.
#         * Maybe the budget for all the projects it's assigned in a particular month , or every X months??
#         * We should expect that during holydays and vacations there are less submitted projects (we should tottally include holyday information as external data!)
#     * **project_grade_category: ** For which school grade the project it's oriented. Maybe most of the accepted projects are for small kids because they need special prgrams to learn better? Or maybe to older students that need special materials for science projects?
#     * **project_subject_categories: **For which academic area the project was proposed? Is it math related? Music related? This is feature is interesting beacuse one application may have multiple areas. For example:
#         * -Math & Science-, -Math & Science, Applied Learning-,  -Math & Science, Warmth, Care & Hunger-, -History & Civics, Warmth, Care & Hunger- are 4 different categories that share at least one academic area so we may need to apply some string transformations to get a list of academic areas instead of a single-value string. 
#         EX:  -Math & Science, Applied Learning-, => [Math, Science, Applied Learning]
#         * We have 51 different categories single-value.
#         
#     * **project_subject_subcategories: ** This is something more specific than the category feature. Then again we may need specific words or fields of the subcategory to identify more speciffic groups. Some examples of subcategories are:
#        * 'ESL, Performing Arts', 'Gym & Fitness, Visual Arts',
#        * 'Early Development, Health & Life Science',
#        * 'Foreign Languages, Special Needs',
#        * We have 407 different subcategories.
#        
#     * **project_title: ** Name of the project.
#         * Just an interesting name:  "Wiggle While We Work" -> 149 different observations with the same name. 0.912751677852349 accpetance rate. Awesome name!! 
#     * **project_essay_1: **  When presenting the project the teachers have to make a descrption of their application in 4 paragraphs.
#         * 663.84 words average. 1st paragraph description.  
#     * **project_essay_2: **
#         * 833.552 words average  2nd paragraph description.  .
#     * **project_essay_3: ** 
#         * 19.70 words average 3rd paragraph description.  96.4% of teacher didn't include a third description paragraph.
#     * **project_essay_4: **
#         * 12.435 words average 4th paragraph description.  96.4% of teacher didn't include a third description paragraph.
#     * **project_resource_summary: ** A description of the resources required for the project.
#         * Example:   "My students need 6 Ipod Nano's to create and differentiated and engaging way to practice sight words during a literacy station." 
#     * **teacher_number_of_previously_posted_projects: ** How many applications does the teacher had presented in the past.
#         * Teachers had presented 11.23 in average.
#     * **project_is_approved: ** 
#         * 0.8476823374340949 accpetance rate. This value is really 
# 
# ### test dataframe
# Same information of the train dataset without the  **project_is_approved** variable.
# 
# ### sample_submision dataframe
# A datafarme with just two columns, id and project_is_approved. As mention in the details of the competition th order of the results doesn't matter. Also, you must predict a probability for the project_is_approved variable and the file should contain a header.
# 
# 

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import datetime
import matplotlib.patches as mpatches


# In[ ]:


# Filepath to main training dataset.
train_path = '../input/train.csv'
test_path = '../input/test.csv'

# Read data and store in DataFrame.
train_data = pd.read_csv(train_path, sep=',', date_parser="project_submitted_datetime")


# Using the train dataframe we can anwer some questions to select the variables that provide more information about the acceptance rate. For example, is there a correlation between the gender and the acceptance rate of the projects? Is there a correlation between the school state and the acceptance rate? 
# 
# In the following Chunks I will try to answer this kind of questions following the order of the column variables.

# In[ ]:


train_data.head()


# ## Acceptance rates analysis.
# 
# ### Gender and titles.

# In[ ]:


train_data.teacher_prefix.unique()


# In[ ]:


prefixAceptance = train_data[["teacher_prefix","project_is_approved"]].groupby("teacher_prefix").mean()
prefixAceptance["prefix"] = prefixAceptance.index

genderDictionary = {"Ms.": "Female", "Mrs.":"Female", "Mr.":"Male", "Teacher":"Neutral", "Dr.":"Neutral", np.nan:"Neutral"  }
train_data["gender"] = train_data.teacher_prefix.map( genderDictionary )
genderAceptance = train_data[["gender","project_is_approved"]].groupby("gender").mean()

titleDictionary = {"Ms.": "Na", "Mrs.":"Na", "Mr.":"Na", "Teacher":"Teacher", "Dr.":"Dr.", np.nan:"Na"  }
train_data["title"] = train_data.teacher_prefix.map( titleDictionary )
titleAceptance = train_data[["title","project_is_approved"]].groupby("title").mean()


# In[ ]:


fig  = plt.figure(figsize=(20,7))
ax1 = plt.subplot(1,3,1)
sns.barplot(x = prefixAceptance.index, y =prefixAceptance.project_is_approved )
ax2 = plt.subplot(1,3,2)
sns.barplot(x = genderAceptance.index, y =genderAceptance.project_is_approved )
ax3 = plt.subplot(1,3,3)
sns.barplot(x = titleAceptance.index, y =titleAceptance.project_is_approved )


# It looks like there is not a real correlation between the prefix of the title and the accpetance rate of the projects. 

# ### School Satate.

# In[ ]:


stateAceptance = train_data[["school_state","project_is_approved"]].groupby("school_state").mean()
stateAceptance["state"] = stateAceptance.index

fig = plt.figure( figsize=(20,10))
plt.title("Accpetance rate per State")
sns.barplot(x = stateAceptance.index, y =stateAceptance.project_is_approved )


# Then again it doesn't seem that the number percentaje of accepted projects change between states. If this wasn't true then we would expect bars of different sizes. Now we can reject the hipotesis that each state have a different budget and that this butget constraints the amount of projects that each state can accept. 
# 
# Just to be sure let's plot the total number projects (applications) per state.

# In[ ]:


stateAceptance = train_data[["school_state","project_is_approved"]].groupby("school_state").count()
stateAceptance["state"] = stateAceptance.index
stateAceptance = stateAceptance.sort_values( "project_is_approved", ascending=False)

fig = plt.figure( figsize=(20,10))
plt.title("Number of applications per State")
sns.barplot(x = stateAceptance.index, y =stateAceptance.project_is_approved )


# This intersting, it looks like there are some states that recieve a lot more applications than others but the acceptance rate it's almost the same for all the states. This tell us that maybe all the states have a fixed rule and they must accept at least 80% of the applications. If this was true then the competition is not about finding the good applications but finding the worst applications.

# ### Acceptance and Time.
# in this section we plot the trend of acceptance rates and the number of applications.

# In[ ]:


train_data["project_submitted_datetime"] = pd.to_datetime( train_data.project_submitted_datetime )
train_data["date"] = train_data.project_submitted_datetime.apply( lambda x: x.date() )
train_data["month"] = train_data.project_submitted_datetime.apply( lambda x: x.month )
train_data["weekday"] = train_data.project_submitted_datetime.apply( lambda x: x.weekday )
train_data["year"] = train_data.project_submitted_datetime.apply( lambda x: x.year )


# In[ ]:


dateAcceptance = train_data[["date","project_is_approved"]].groupby("date").mean()
dateAcceptanceCount = train_data[["date","project_is_approved"]].groupby("date").count() 

fig = plt.figure( figsize=(20,6))
plt.title("Acceptance rate per date and number of applications")
ax1 = plt.subplot(1,1,1)
plt.plot(dateAcceptance  )
ax2 = plt.subplot(1,1,1)
ax2 = ax1.twinx()
plt.plot(dateAcceptanceCount, "red"  )
red_patch = mpatches.Patch(color='red', label='Total number of applications')
blue_patch = mpatches.Patch(color='blue', label='Acceptance rate')
plt.legend(handles=[blue_patch, red_patch])


# In[ ]:


monthAcceptance = train_data[["month","project_is_approved"]].groupby("month").mean()
monthAcceptanceCount = train_data[["month","project_is_approved"]].groupby("month").count() 

fig = plt.figure( figsize=(20,6))

plt.title("Acceptance trend")

ax1 = plt.subplot(1,1,1)
plt.plot(monthAcceptance  )
ax2 = plt.subplot(1,1,1)
ax2 = ax1.twinx()
plt.plot(monthAcceptanceCount, "red"  )

red_patch = mpatches.Patch(color='red', label='Total number of applications')
blue_patch = mpatches.Patch(color='blue', label='Acceptance rate')
plt.legend(handles=[blue_patch, red_patch])


# In the first trend plot it's hard to notice that there is a negative correlation between the total number of applications and the acceptance rate but using a month aggregation plot allow us to see that the more applications are presented on a given month the lower the acceptance rate. Maybe this happen becauase application analist have more time to read the complete application or maybe beacause they need to accept at least X number of applications.
# 
# ### Teacher posted applications in the past.
# 
# In this section we want to test if the number of past applications hava an impact on the accpetance rate.
# 

# In[ ]:


postedAcceptance= train_data[["teacher_number_of_previously_posted_projects","project_is_approved"]].groupby("teacher_number_of_previously_posted_projects").mean()
postedAcceptanceCount = train_data[["teacher_number_of_previously_posted_projects","project_is_approved"]].groupby("teacher_number_of_previously_posted_projects").count() 
postedAcceptanceCount = postedAcceptanceCount.rename( columns= {"project_is_approved": "applications_count"})

postedAcceptance =  postedAcceptance.merge( postedAcceptanceCount, right_index=True, left_index= True)
postedAcceptance = postedAcceptance.sort_index( ascending= True)

postedAcceptance50 = postedAcceptance.head(50)


# In[ ]:




fig = plt.figure( figsize=(20,10))
fig.suptitle( "Distribution: acceptance rate and number of applications per number of past posted projects", fontsize = 20)

ax1 = plt.subplot(2,1,1)
plt.bar( postedAcceptance50.index, postedAcceptance50.project_is_approved,  color='g') 
ax2 = plt.subplot(2,1,1)
ax2 = ax1.twinx()
plt.bar( postedAcceptance50.index, postedAcceptance50.applications_count, color = 'orange' )
orange_patch = mpatches.Patch(color='orange', label='Count number of records per previously_posted_projects')
green_patch = mpatches.Patch(color='green', label='Acceptance rate')
plt.legend(handles=[green_patch, orange_patch])
postedAcceptance = postedAcceptance.sort_index( ascending= False)
postedAcceptance50 = postedAcceptance.head(50)

ax3 = plt.subplot(2,1,2)
plt.bar( postedAcceptance50.index, postedAcceptance50.project_is_approved,  color='g') 
ax4 = plt.subplot(2,1,2)
ax4 = ax3.twinx()
plt.bar( postedAcceptance50.index, postedAcceptance50.applications_count, color = 'orange' )
orange_patch = mpatches.Patch(color='orange', label='Count number of records per previously_posted_projects')
green_patch = mpatches.Patch(color='green', label='Acceptance rate')
plt.legend(handles=[green_patch, orange_patch])



# In this plots we can see that the number of previous posted projects doesn't really matter after the certain level. There is a little ascending trend on the acceptance trend when the number of previous posted projects is between 0 and 20. After this level the acceptance rate is almost the same (arround 84%) for all previous posted projects value. We also can see that there are a lot of new teacher sending their applications. The number of teachers with 0 value previous_posted_projects is 50,000 observations

# ###  School grade acceptance rate

# In[ ]:


categoryAceptance = train_data[["project_grade_category","project_is_approved"]].groupby("project_grade_category").mean()
categoryAceptance = categoryAceptance.sort_values( "project_is_approved", ascending=False)

fig = plt.figure( figsize=(20,10))
fig.suptitle( "Distribution acceptance rate and number of applications per school grade ", fontsize = 25)
plt.subplot(2,1,1)
sns.barplot(x = categoryAceptance.index, y =categoryAceptance.project_is_approved )

categoryAceptance = train_data[["project_grade_category","project_is_approved"]].groupby("project_grade_category").sum()

plt.subplot(2,1,2)
sns.barplot(x = categoryAceptance.index, y =categoryAceptance.project_is_approved )



# In[ ]:


train_data.columnsumns


# In[ ]:


categoryAceptance = train_data[["project_subject_categories","project_is_approved"]].groupby("project_subject_categories").mean()
categoryAceptance = categoryAceptance.sort_values( "project_is_approved", ascending=False)
categoryAceptance = categoryAceptance.head(15)
categoryAceptanceindex = categoryAceptance.index

fig = plt.figure( figsize=(50,20))
fig.suptitle( "Distribution acceptance rate and number of applications per category ", fontsize = 50)
ax1 = plt.subplot(2,1,1)
ax1.set_title( "Acceptance rate per categort", fontsize = 40)

sns.barplot(x = categoryAceptance.index, y =categoryAceptance.project_is_approved )

categoryAceptance = train_data[["project_subject_categories","project_is_approved"]].groupby("project_subject_categories").sum()
categoryAceptance = categoryAceptance.loc[categoryAceptanceindex]

ax2 = plt.subplot(2,1,2)
ax2.set_title( "Number of records per category", fontsize = 40)

sns.barplot(x = categoryAceptance.index, y =categoryAceptance.project_is_approved )



# In[ ]:


categoryAceptance = train_data[["project_subject_categories","project_is_approved"]].groupby("project_subject_categories").sum()
categoryAceptance = categoryAceptance.sort_values( "project_is_approved", ascending=False)
categoryAceptance = categoryAceptance.head(15)
categoryAceptanceindex = categoryAceptance.index

categoryAceptance = train_data[["project_subject_categories","project_is_approved"]].groupby("project_subject_categories").mean()
categoryAceptance = categoryAceptance.loc[categoryAceptanceindex]

fig = plt.figure( figsize=(50,20))
fig.suptitle( "Distribution acceptance rate and number of applications per category ", fontsize = 50)
plt.title("Category Accpetance")
ax1 = plt.subplot(2,1,1)
ax1.set_title( "Acceptance rate per category", fontsize = 40)

sns.barplot(x = categoryAceptance.index, y =categoryAceptance.project_is_approved )

categoryAceptance = train_data[["project_subject_categories","project_is_approved"]].groupby("project_subject_categories").sum()
categoryAceptance = categoryAceptance.loc[categoryAceptanceindex]

ax2 = plt.subplot(2,1,2)
ax2.set_title( "Number of records per category", fontsize = 40)
sns.barplot(x = categoryAceptance.index, y =categoryAceptance.project_is_approved )


