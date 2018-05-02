
# coding: utf-8

# # <a id='xyzabc'>EDA - Fraud Detection</a>
# ---------------------------------------
# 
# In this problem, we are trying to predict the Ad clicks that lead to App download. These clicks are considered Non-Fraudulent. So I may use the terms App Downloaded and Non-Fraudulent Clicks interchangeably.
# 
# The training dataset is quite huge - 185 million
# 
# 
# # <a id='xyzabc'>List of Vizulizations</a>

# 
# - <a href='#ttd'>0. Train Vs Test Vs Test Supplement</a> 
# - <a href='#tfnfn'>1. Total Number of Fraudulent Vs Non-Fraudulent Clicks in numbers and percentage</a>
# - <a href='#codcv'>2. Count of Distinct Categorical variables</a>
# - <a href='#e'>3. Snapshot of Ips that have 100% Fraudulent Clicks</a>
# - <a href='#f'>4. Cluster Model for IPs</a>
# - <a href='#g'>5. Snapshot of OSs that have 100% Fraudulent Clicks</a>
# - <a href='#h'>6. Cluster Model for OSs</a>
# - <a href='#i'>7. Snapshot of Devices that have 100% Fradulent Clicks</a>
# - <a href='#j'>8. Cluster Model for Devices</a>
# - <a href='#k'>9. Snapshot of Apps that have 100% Fraudulent Clicks</a>
# - <a href='#l'>10. Cluster Model for Apps</a>
# - <a href='#m'>11. Snapshot of Channels that have 100% Fraudulent Clicks</a>
# - <a href='#n'>12. Cluster Model for Channels</a>
# - <a href='#o'>13. Number of Fraudulent Clicks by Device</a>
# - <a href='#p'>14. Number of Non-Fraudulent Clicks by Device</a>
# - <a href='#q'>15. Total Number of Clicks per Hour</a>
# - <a href='#r'>16. Total Number of Clicks per Day</a>
# - <a href='#s'>17. Percentage of Non-Fraudulent Clicks Day wise</a>
# - <a href='#t'>18. Percentage of Non-Fraudulent Clicks - Hour of the day</a>
# - <a href='#aa'>19. Clustering Channels</a>
# - <a href='#bb'>20. Clustering Apps</a>
# - <a href='#cc'>21. Clustering Devices</a>
# - <a href='#dd'>22. Clustering OS</a>
# - <a href='#ee'>23. Clustering IPs</a>
# - <a href='#ff'>24. Clustering Apps, Devices, Channels,OSs</a>
# - <a href='#u'>25. Scatter Plot of Fraudulent Vs Non-Fraudulent Clicks for all Channels</a>
# - <a href='#v'>26. Scatter Plot of Clicks(No Download) Vs Clicks (Download) for Hour of the Day</a>
# - <a href='#w'>27. Scatter Plot of Fraudulent Vs Non-Fraudulent Clicks for all IPs</a>
# - <a href='#x'>28. Scatter Plot of Fraudulent Vs Non-Fraudulent Clicks for all Devices</a>
# - <a href='#y'>29. Scatter Plot of Fraudulent Vs Non-Fraudulent Clicks for all Apps</a>
# - <a href='#z'>30. Scatter Plot of Fraudulent Vs Non-Fraudulent Clicks for all OSs</a>
# - <a href='#a'>31. Some Fun Visualizations</a>
# 
# 
#      
#        
#      

# # <a id='ttd'>0. Train Vs Test Vs Test Supplement</a>
# ---------------------------------------

# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/Train+Data.png)

# So the train data has partial data for 6th and 9th of November and complete data for 7th and 8th of November.  6th,7th,8th,9th are Mon,Tue, Wed, Thurs respectively
# 
# We can say that  from Monday to Thursday there is no big variation in the total number of clicks.

# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/0_2.png)

# The test data set has only data of 10th Nov which happens to be a Friday and data has been provided only for the hours - 4, 5, 9, 10, 13, 14. 

# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/Test+Supplement.png)

# This dataset was leaked unintentionally by Talkingdata during initial days of the compitition. This is only for reference. Evaluation will be done on the test set only.

# 
# # <a id='tfnfn'>1. Total Number of Fraudulent Vs Non-Fraudulent Clicks in numbers and percentage</a>
# ---------------------------------------
# 

# 
# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/1.+Total+Number+of+Fradulent+Vs+Non-Fradulent+Clicks+in+numbers.png)

# So downloading the app is a Rare Event . It is a highly unbalanced datset with only 0.25% of clicks leading to Downloads. We may have to figure out a good sampling technique.
# Now that some fraud on staggering scale.

# **IP, App, Device, OS and Channel are the categorical variables. Lets look at the distinct count of these categorical variables.**

# # <a id='codcv'>2. Count of Distinct Categorical variables</a>
# ---------------------------------------
# 

# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/3.+Count+of+Distinct+Categorical+variables.png)

# **Organised Bot Driven Fraud**

# **Lets Blackist these IPs**

# # <a id='e'>3. Snapshot of Ips that have 100% Fraudulent Clicks</a>
# ---------------------------------------

# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/4.+Snapshot+of+Ips+that+have+100%25+Fradulent+Clicks.png)

# This is just a snapshot. That's a insane nuber of clikcs coming from IPs that dont convert to a download in a period of 4 days! 
# 
# **Bold Statement : These IPs should become part of TalkingData's IP blaclist. Please comment if you guys think otherwise.**

# **Cluster Model  for IPs**

# # <a id='f'>4. Cluster Model for IPs</a>
# ---------------------------------------

# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/5.+Cluster+Model+for+IPs1.png)

# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/5.+Cluster+Model+for+IPs2.png)

# The Cluster Model gives a fair idea of the distribution of IPs that are  Fradulent and Non- Fradulent.  Majority of IPs are centered around 1% of Non - Fradulent Clicks. While the nex majority of IPs are centered around 99% of Non- fradulent Clicks.

# # <a id='g'>5. Snapshot of OSs that have 100% Fraudulent Clicks</a>
# ---------------------------------------

# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/6.+Snapshot+of+OSs+that+have+100%25+Frdulent+Clicks.png)

# **Cluster Model for OS**

# # <a id='h'>6. Cluster Model for OSs</a>
# ---------------------------------------

# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/7.+Cluster+Model+for+OSs1.png)

# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/7.+Cluster+Model+for+OSs2.png)

#  Majority of OSs are centered around 0.06% of Non - Fradulent Clicks. Only few OSs are authentic.

# # <a id='i'>7. Snapshot of Devices that have 100% Fradulent Clicks</a>
# ---------------------------------------

# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/8.+Snapshot+of+Devices+that+have+100%25+Frdulent+Clicks.png)

# **TalkingData should add Device 5, Device 182, Device 1728 to their Device Blacklist**

# # <a id='j'>8. Cluster Model for Devices</a>
# ---------------------------------------

# **Cluster Model for Devices**

# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/9.+Cluster+Model+for+Devices1.png)

# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/9.+Cluster+Model+for+Devices2.png)

# ** Majority of Devices are centered around 0.36% of Non - Fradulent Clicks. We should note that 156 devices have 100 Non-Fradulent Clicks. This is really interesting.**

# # <a id='k'>9. Snapshot of Apps that have 100% Fraudulent Clicks</a>
# ---------------------------------------

# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/10.+Snapshot+of+Apps+that+have+100%25+Frdulent+Clicks.png)

# **Cluster Model for Apps**

# # <a id='l'>10. Cluster Model for Apps</a>
# ---------------------------------------

# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/11.+Cluster+Model+for+Apps1.png)

# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/11.+Cluster+Model+for+Apps2.png)

#  Majority of Apps are centered around 0.23% of Non - Fradulent Clicks. While rest of the Apps have mixed Clicks.  Only 8 of them have 97% Non Fraduelent Clicks. Does this mean the Apps play a role in the fraud? SInce they are the benefactors of the clicks.

# # <a id='m'>11. Snapshot of Channels that have 100% Fraudulent Clicks</a>
# ---------------------------------------

# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/12.+Snapshot+of+Channels+that+have+100%25+Fradulent+Clicks.png)

# **Cluster Model for Channels**

# # <a id='n'>12. Cluster Model for Channels</a>
# ---------------------------------------

# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/13.+Cluster+Model+for+Channels1.png)

# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/13.+Cluster+Model+for+Channels2.png)

#  Majority of Channels are centered around 2% of Non - Fradulent Clicks. Wonder about the authenticity of the Channels.

# # <a id='o'>13. Number of Fraudulent Clicks by Device</a>
# ---------------------------------------

# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/Screen+Shot+2018-04-10+at+12.03.58+PM.png)

# Majority  of the Fradulent Clicks are from Device 1 and Device 2

# # <a id='p'>14. Number of Non-Fraudulent Clicks by Device</a>
# ---------------------------------------

# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/15.+Number+of+Non-Fradulent+Clicks+by+Device.png)

# Majority of Non-Fradulent Clicks are from Device 1 ans Device 0
# 

# **Effect of Click Time**

# # <a id='q'>15. Total Number of Clicks per Hour</a>
# ---------------------------------------

# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/16.+Total+Number+of+Clicks+per+Hour.png)

# # <a id='r'>16. Total Number of Clicks per Day</a>
# ---------------------------------------

# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/17.+Total+Number+of+Clicks+per+Day.png)

# 6th and 9th are partial data and 7th and 8th are full data. We can say that  from Monday to Thursday there is no big variation in the total number of clicks.

# ***Day***

# # <a id='s'>17. Percentage of Non-Fraudulent Clicks Day wise</a>
# ---------------------------------------

# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/18.+Percentage+of+Non-Fradulent+Clicks+Day+wise.png)

# ***Hour of the Day***

# # <a id='t'>18. Percentage of Non-Fraudulent Clicks - Hour of the day</a>
# ---------------------------------------

# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/19.+Percentage+of+Non-Fradulent+Clicks+-+Hour+of+the+day.png)

# Looks like we can tease out some information by extrating the Day and Hour of the Click Time. So on an average the Non- Fradulent clicks peak at 1 AM and bottoms at 3 PM 

# # <a id='u'>The following Clusters should help TalkingData to prioritize App/Device/OS/Channel/IP level investigation based on Fradulent Clicks % thresholds</a>
# ---------------------------------------

# # <a id='aa'>19. Clustering Channels</a>
# ---------------------------------------

# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/Screen+Shot+2018-04-17+at+9.00.07+PM.png)

# # <a id='bb'>20. Clustering Apps</a>
# ---------------------------------------

# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/Screen+Shot+2018-04-17+at+8.59.53+PM.png)

# # <a id='cc'>21. Clustering Devices</a>
# ---------------------------------------

# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/Screen+Shot+2018-04-17+at+8.58.29+PM.png)

# # <a id='dd'>22. Clustering OS</a>
# ---------------------------------------

# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/Screen+Shot+2018-04-17+at+8.57.58+PM.png)

# # <a id='ee'>23. Clustering IPs</a>
# ---------------------------------------

# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/Screen+Shot+2018-04-17+at+8.57.04+PM.png)

# # <a id='ff'>24. Clustering Apps, Devices, Channel, OS</a>
# ---------------------------------------

# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/Screen+Shot+2018-04-17+at+8.56.02+PM.png)

# **Scatter Plots for Fradulent Vs Non-Fradulent Clicks**

# **Channel**

# # <a id='u'>25. Scatter Plot of Fraudulent Vs Non-Fraudulent Clicks for all Channels</a>
# ---------------------------------------

# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/20.+Scatter+Plot+of+Fradulent+Vs+Non-Fradulent+Clicks+for+all+Channels.png)

# R - Aquared = 0.001
# 
# p - value =0.54
# 

# ![](http://)

# # <a id='v'>26. Scatter Plot of Clicks(No Download) Vs Clicks (Download) for Hour of the Day</a>
# ---------------------------------------

# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/21.+Scatter+Plot+of+Fradulent+Clicks+Vs+Non+Fradulent+Clicks+for+Hour+of+the+Day.png)

# R - Aquared = 0.86
# 
# p - value = < 0.0001

# # <a id='w'>27. Scatter Plot of Fraudulent Vs Non-Fraudulent Clicks for all IPs</a>
# ---------------------------------------

# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/22.+Scatter+Plot+of+Fradulent+Vs+Non-Fradulent+Clicks+for+all+IPs.png)

# R - Aquared = 0.79
# 
# p - value = < 0.0001

# # <a id='x'>28. Scatter Plot of Fraudulent Vs Non-Fraudulent Clicks for all Devices</a>
# ---------------------------------------

# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/23.+Scatter+Plot+of+Fradulent+Vs+Non-Fradulent+Clicks+for+all+Devices.png)

# R - Aquared = 0.90
# 
# p - value = < 0.0001

# # <a id='y'>29. Scatter Plot of Fraudulent Vs Non-Fraudulent Clicks for all Apps</a>
# ---------------------------------------

# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/24.+Scatter+Plot+of+Fradulent+Vs+Non-Fradulent+Clicks+for+all+Apps.png)

# R - Aquared = 0.01
# 
# p - value = < =0.00006
# 
# **Many Apps have unusually high number of Fradulent Clicks than Non-Fradulent Clicks**

# # <a id='z'>30. Scatter Plot of Fraudulent Vs Non-Fraudulent Clicks for all OSs</a>
# ---------------------------------------

# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/25.+Scatter+Plot+of+Fradulent+Vs+Non-Fradulent+Clicks+for+all+OSs.png)

# R - Aquared = 0.68
# 
# p - value = < 0.0001

# **The below one is for fun...**

# # <a id='z'>31. Some Fun Visualizations</a>
# ---------------------------------------

# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/26.+Some+Fun+Vizualizations1.png)
# ![](https://s3.us-east-2.amazonaws.com/images-kaggle/talkingdataimages/26.+Some+Fun+Vizualizations2.png)

# **To be continued ...........**
