
# coding: utf-8

# $$
# \huge\text{Visualizing iMaterialist Data}\\
# \large\text{March 2018}\\
# \text{Andrew Riberio @ https://github.com/Andrewnetwork}
# $$
# <img width="700" height="300" src="https://i.imgur.com/HcJJmzj.jpg" />
# In this notebook we will visualize data from the Kaggle challenge *iMaterialist Challenge (Furniture) at FGVC5*. 
# 
# **NOTE**: In order for the interactive componnents of this kernel to function, you must either fork this kernel or download it to your local machine which has the required environment and dependencies. The simplest method is to fork the kernel here on kaggle. 

# In[6]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json

# Libraries for displying the data. 
from IPython.core.display import HTML 
from ipywidgets import interact
from IPython.display import display

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Loading the data

# We use the json library to load the data into python dictionary objects. 

# In[7]:


training   = json.load(open("../input/train.json"))
test       = json.load(open("../input/test.json"))
validation = json.load(open("../input/validation.json"))


# We itterate over the json dictionaries loaded in above and produce a pandas dataframe for the training, validation, and test data. 

# In[8]:


# A function to be mapped over the json dictionary. 
def joinFn(dat):
    return [dat[0]["url"][0], dat[1]["label_id"]]

trainingDF   = pd.DataFrame(list(map(joinFn, zip(training["images"],training["annotations"]))),columns=["url","label"])
validationDF = pd.DataFrame(list(map(joinFn, zip(validation["images"],validation["annotations"]))),columns=["url","label"])
testDF       = pd.DataFrame(list(map(lambda x: x["url"],test["images"])),columns=["url"])


# In[12]:


trainingDF


# In[10]:


validationDF.head()


# In[11]:


testDF.head()


# ## Basic Visualization 

# In[13]:


print("Number of classes: {0}".format(len( trainingDF["label"].unique())))


# In[14]:


trainingDF["label"].value_counts().plot(kind='bar',figsize=(40,10),title="Number of Training Examples Versus Class").title.set_size(40)


# In the above chart we can see that we have a heavily skewed dataset in respect the the number of training examples per class. Class 20 has about 4,000 examples where class 83 has less than 500. 
# 
# We use the next funciton to view examples of each class in the training data. The overlayed number is the class label. 

# In[15]:


def displayExamples(exampleIndex=0):
    outHTML = "<div>"
    for label in range(1,129):
        img_style = "width: 180px;height:180px; margin: 0px; float: left; border: 1px solid black;"
        captionDiv = "<div style='position:absolute;right:30px;color:red;font-size:30px;background-color:grey;padding:5px;opacity:0.5'>"+str(label)+"</div>"
        outHTML += "<div style='position:relative;display:inline-block'><img style='"+img_style+"' src='"+trainingDF[trainingDF.label == label].iloc[exampleIndex][0]+"'/>"+captionDiv+"</div>"
    outHTML += "</div>"
    display(HTML(outHTML))

displayExamples()


# We can use the following function to view examples for a particular category/class. 

# In[16]:


def displayCategoryExamples(category=0,nExamples=20):
    outHTML = "<div>"
    for idx in range(0,nExamples):
        img_style = "width: 180px;height:180px; margin: 0px; float: left; border: 1px solid black;"
        outHTML += "<div style='position:relative;display:inline-block'><img style='"+img_style+"' src='"+trainingDF[trainingDF.label == category].iloc[idx][0]+"'/></div>"
    outHTML += "</div>"
    display(HTML(outHTML))
    
displayCategoryExamples(7)


# Just from looking at examples in class 7, TVs, we can note a few things: 
# * Some are natural images with the TVs included in their surroundings. 
# * Some are artifical images of cartoon TVs. 
# * Some images have digital artifacts overlayed upon the image ( like the date of the image ). 
# * There are different angles provided. 
# 
# Let's look at another category to see if this type of variablity is consistent across the classes. 
# 

# In[17]:


displayCategoryExamples(24)


# It does seem like there is a huge variablity within each class. In the next section you will be able to interactively pan around the dataset. This requires you to be running this notebook as a fork, as a static rendering will not include interactive elements. There is no preloading done here, so it may take some time for all the images to load. 

# In[18]:


def visCat(cat=1):
    displayCategoryExamples(cat,40)
    
interact(visCat, cat=(1,128))


# ## Next Steps
# For futher visualization we would need to download the images which we are unable to do in a contained Kaggle kernel. There are a few download scripts available in the kernels section. It would be nice if someone could download the images and upload a partial set of the training, validation, and test data so we can do more analysis in these notebooks. 
# 
