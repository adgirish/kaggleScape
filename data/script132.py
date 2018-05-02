
# coding: utf-8

# In[1]:



import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


attendance_df = pd.read_csv("../input/nba_2017_attendance.csv");attendance_df.head()


# In[ ]:


endorsement_df = pd.read_csv("../input/nba_2017_endorsements.csv");endorsement_df.head()


# In[ ]:


valuations_df = pd.read_csv("../input/nba_2017_team_valuations.csv");valuations_df.head()


# In[ ]:


salary_df = pd.read_csv("../input/nba_2017_salary.csv");salary_df.head()


# In[ ]:


pie_df = pd.read_csv("../input/nba_2017_pie.csv");pie_df.head()


# In[ ]:


plus_minus_df = pd.read_csv("../input/nba_2017_real_plus_minus.csv");plus_minus_df.head()


# In[ ]:


br_stats_df = pd.read_csv("../input/nba_2017_br.csv");br_stats_df.head()


# In[ ]:


elo_df = pd.read_csv("../input/nba_2017_elo.csv");elo_df.head()


# In[ ]:


attendance_valuation_df = attendance_df.merge(valuations_df, how="inner", on="TEAM")


# In[ ]:


attendance_valuation_df.head()


# In[ ]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"));sns.pairplot(attendance_valuation_df, hue="TEAM")


# In[ ]:


corr = attendance_valuation_df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# In[ ]:


valuations = attendance_valuation_df.pivot("TEAM", "AVG", "VALUE_MILLIONS")



# In[ ]:


plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("NBA Team AVG Attendance vs Valuation in Millions:  2016-2017 Season")
sns.heatmap(valuations,linewidths=.5, annot=True, fmt='g')


# In[ ]:


results = smf.ols('VALUE_MILLIONS ~AVG', data=attendance_valuation_df).fit()


# In[ ]:


print(results.summary())


# In[ ]:


sns.residplot(y="VALUE_MILLIONS", x="AVG", data=attendance_valuation_df)


# In[ ]:


attendance_valuation_elo_df = attendance_valuation_df.merge(elo_df, how="inner", on="TEAM")


# In[ ]:


attendance_valuation_elo_df.head()


# In[ ]:


corr_elo = attendance_valuation_elo_df.corr()
plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("NBA Team Correlation Heatmap:  2016-2017 Season (ELO, AVG Attendance, VALUATION IN MILLIONS)")
sns.heatmap(corr_elo, 
            xticklabels=corr_elo.columns.values,
            yticklabels=corr_elo.columns.values)


# In[ ]:


corr_elo


# In[ ]:



ax = sns.lmplot(x="ELO", y="AVG", data=attendance_valuation_elo_df, hue="CONF", size=12)
ax.set(xlabel='ELO Score', ylabel='Average Attendence Per Game', title="NBA Team AVG Attendance vs ELO Ranking:  2016-2017 Season")


# In[ ]:


attendance_valuation_elo_df.groupby("CONF")["ELO"].median()


# In[ ]:


attendance_valuation_elo_df.groupby("CONF")["AVG"].median()


# In[ ]:


results = smf.ols('AVG ~ELO', data=attendance_valuation_elo_df).fit()


# In[ ]:


print(results.summary())


# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


k_means = KMeans(n_clusters=3)


# In[ ]:


cluster_source = attendance_valuation_elo_df.loc[:,["AVG", "ELO", "VALUE_MILLIONS"]]


# In[ ]:


kmeans = k_means.fit(cluster_source)


# In[ ]:


attendance_valuation_elo_df['cluster'] = kmeans.labels_


# In[ ]:


ax = sns.lmplot(x="ELO", y="AVG", data=attendance_valuation_elo_df,hue="cluster", size=12, fit_reg=False)
ax.set(xlabel='ELO Score', ylabel='Average Attendence Per Game', title="NBA Team AVG Attendance vs ELO Ranking Clustered on ELO, AVG, VALUE_MILLIONS:  2016-2017 Season")


# In[ ]:


kmeans.__dict__


# In[ ]:


kmeans.cluster_centers_


# In[ ]:


cluster_1 = attendance_valuation_elo_df["cluster"] == 1


# In[ ]:


attendance_valuation_elo_df[cluster_1]

