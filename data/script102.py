
# coding: utf-8

# # Introduction to the `kagglegym` API

# Code Competitions are a new style of competition where you submit code rather than the predictions that your code creates. This allows for new types of competitions like this time-series competition hosted by Two Sigma. This notebook gives an overview of the API, `kagglegym`, which was heavily influenced by [OpenAI's Gym](https://gym.openai.com/docs) API for reinforcement learning challenges.

# ## Data Overview
# 
# Another difference with this competition is that we're using an [HDF5 file](https://support.hdfgroup.org/HDF5/) instead of a CSV file due to the size of the data. You can still easily read it and manipulate it for exploration:

# In[ ]:


# Here's an example of loading the CSV using Pandas's built-in HDF5 support:
import pandas as pd

with pd.HDFStore("../input/train.h5", "r") as train:
    # Note that the "train" dataframe is the only dataframe in the file
    df = train.get("train")


# In[ ]:


# Let's see how many rows are in full training set
len(df)


# In[ ]:


df.head()


# In[ ]:


# How many timestamps are in the full training set?
len(df["timestamp"].unique())


# **Important Note**: the raw training file is only available for exploration kernels. It will not be available when you make a competition submission. You should only use the raw training file for exploration purposes.

# ## API Overview

# The "kagglegym" API is based on OpenAI's Gym API, a toolkit for developing and comparing reinforcement learning algorithms. Read OpenAI's Gym API [documentation](https://gym.openai.com/docs) for more details. Note that ours is named "kagglegym" and not "gym" to prevent possible conflicts with OpenAI's "gym" library. This section will give an overview of the concepts to get you started on this competition.
# 
# The API is exposed through a `kagglegym` library. Let's import it to get started:

# In[ ]:


import kagglegym


# Now, we need to create an "environment". This will be our primary interface to the API. The `kagglegym` API has the concept of a default environment name for a competition, so just calling `make()` will create the appropriate one for this competition.

# In[ ]:


# Create environment
env = kagglegym.make()


# To properly initialize things, we need to "reset" the environment. This will also give us our first "observation":

# In[ ]:


# Get first observation
observation = env.reset()


# Observations are the means by which our code "observes" the world. The very first observation has a special property called "train" which is a dataframe which we can use to train our model:

# In[ ]:


# Look at first few rows of the train dataframe
observation.train.head()


# Note that this "train" is about half the size of the full training dataframe. This is because we're in an exploratory mode where we simulate the full environment by reserving the first half of timestamps for training and the second half for simulating the public leaderboard.

# In[ ]:


# Get length of the train dataframe
len(observation.train)


# In[ ]:


# Get number of unique timestamps in train
len(observation.train["timestamp"].unique())


# In[ ]:


# Note that this is half of all timestamps:
len(df["timestamp"].unique())


# In[ ]:


# Here's proof that it's the first half:
unique_times = list(observation.train["timestamp"].unique())
(min(unique_times), max(unique_times))


# Each observation also has a "features" dataframe which contains features for the timestamp you'll be asked to predict in the next "step." Note that these features are for timestamp 906 which is just passed the last training timestamp. Also, note that the "features" dataframe does *not* have the target "y" column:

# In[ ]:


# Look at the first few rows of the features dataframe
observation.features.head()


# The final part of observation is the "target" dataframe which is what we're asking you to fill in. It includes the "id"s for the timestamp next step.

# In[ ]:


# Look at the first few rows of the target dataframe
observation.target.head()


# This target is a valid submission for the step. The OpenAI Gym calls each step an "action". Each step of the environment returns four things: "observation", "reward", "done", and "info".

# In[ ]:


# Each step is an "action"
action = observation.target

# Each "step" of the environment returns four things:
observation, reward, done, info = env.step(action)


# The "done" variable tells us if we're done. In this case, we still have plenty of timestamps to go, so it returns "False".

# In[ ]:


# Print done
done


# The "info" variable is just a dictionary used for debugging. In this particular environment, we only make use of it at the end (when "done" is True).

# In[ ]:


# Print info
info


# We see that "observation" has the same properties as the one we get in "reset". However, notice that it's for the next "timestamp":

# In[ ]:


# Look at the first few rows of the observation dataframe for the next timestamp
observation.features.head()


# In[ ]:


# Note that this timestamp has more id's/rows
len(observation.features)


# Perhaps most interesting is the "reward" variable. This tells you how well you're doing. The goal in reinforcement contexts is that you want to maximize the reward. In this competition, we're using the R value that ranges from -1 to 1 (higher is better). Note that we submitted all 0's, so we got a score that's below 0. If we had correctly predicted the true mean value, we would have gotten all zeros. If we had made extreme predictions (e.g. all `-1000`'s) then our score would have been capped to -1.

# In[ ]:


# Print reward
reward


# Since we're in exploratory mode, we have access to the ground truth (obviously not available in submit mode):

# In[ ]:



perfect_action = df[df["timestamp"] == observation.features["timestamp"][0]][["id", "y"]].reset_index(drop=True)


# In[ ]:


# Look at the first few rows of perfect action
perfect_action.head()


# Let's see what happens when we submit a "perfect" action:

# In[ ]:


# Submit a perfect action
observation, reward, done, info = env.step(perfect_action)


# As expected, we get the maximum reward of 1 by submitting the perfect value:

# In[ ]:


# Print reward
reward


# ## Making a complete submission

# We've covered all of the basic components of the `kagglegym` API. You now know how to create an environment for the competition, get observations, examine features, and submit target values for a reward. But, we're still not done as there are more observations/timestamps left.

# In[ ]:


# Print done ... still more timestamps remaining
done


# Now that we've gotten the basics out of the way, we can create a basic loop until we're "done". That is, we'll make a prediction for the remaining timestamp in the data:

# In[ ]:


while True:
    target = observation.target
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))

    observation, reward, done, info = env.step(target)
    if done:        
        break


# Now we can confirm that we're done:

# In[ ]:


# Print done
done


# And since we're "done", we can take a look at at "info", our dictionary used for debugging. Recall that in this environment, we only make use of it when "done" is True.

# In[ ]:


# Print info
info


# Our score is better than 0 because we had that one submission that was perfect.

# In[ ]:


# Print "public score" from info
info["public_score"]


# This concludes our overview of the `kagglegym` API. We encourage you to ask questions in the competition forums or share public kernels for feedback on your approach. Good luck!
