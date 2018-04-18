
# coding: utf-8

# # Looking For Your Files?
# 
# Your data files can be found by clicking on the **Input Files** Link above your editing area.
# 
# ![Imgur](https://i.imgur.com/TB8cntG.png)

# From there, you will find a list of files in the left sidebar.  It may look something like this.  
# ![Imgur](https://i.imgur.com/rqO331u.png)
# 
# If you have many sets of files, it could be longer.  If you don't have any data files listed, click on the button to add a data file.
# 
# To get the path you'll need to load a file, just click on the filename in this list.  That causes the file's path to show up in the main window.  It will look something like this.
# 
# ![Imgur](https://i.imgur.com/7yAY8Z1.png)
# 
# If you have a great memory, congrats. You can remember it and type in that path to load it.
# 
# But computers are finicky if you get any part of it wrong (like those darn leading dots.) . So it's usually easier to highlight the file path, then use keyboard shortcuts to copy and paste it into your code.  
# 
# Once you've copied it, click the button to the left of the **Input Files** link to get back to your editing environment.
# 
# ![Imgur](https://i.imgur.com/9RYRekU.png)
# 
# From there, you can load it with your favorite Python or R libraries.  Here's an example.

# In[ ]:


import pandas as pd

my_data = pd.read_csv('../input/train.csv')


# Now you are off to code. If you're unsure what to do with a loaded data file, Kaggle has some [great learning resources](https://www.kaggle.com/dansbecker/learning-materials-on-kaggle).
