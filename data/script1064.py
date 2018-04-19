
# coding: utf-8

# In this notebook, I leverage the power of a **Spark cluster** to explore the large (~88GB) page_views.csv dataset and analyze its relationshiop with events.csv.  
# After some hours of processing, we've got answer to questions like: 
# 
# * **How to join page_views.csv and events.csv?**
# * **Is events.csv a subset of page_views.csv?**
# * **Are there additional page views for users in events.csv?**  

# **IMPORTANT:** This Jupyter notebook was implemented in my own Spark cluster, and it was not possible to share the notebook as a Kernel without actually running it as a Kernel (which obviously would not be possible).  
# **Thus, you can see the [full notebook on my GitHub](https://github.com/gabrielspmoreira/static_resources/blob/gh-pages/Kaggle-Outbrain-PageViews_EventsAnalytics.ipynb).**

# **If this Kaggle kernel helps you, don't forget to thumbs it up! ;)**

# By [Gabriel S. P. Moreira](https://about.me/gspmoreira)
