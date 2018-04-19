
# coding: utf-8

# # Exercise Introduction
# 
# To build and test your intuition for convolutions, you will design a vertical line detector.  We'll apply it to each part of an image to create a new tensor showing where there are vertical lines.
# 
# ![Imgur](https://i.imgur.com/op9Maqr.png)
# 
# Follow these following 4 steps:
# 1. **Fork this notebook**
# 2. **Run this full notebook and scroll down to see the output**  You will see the original image, as well as an example of the image we get from applying our horizontal line detector to the image.
# 3. **Fill in the code cell for `vertical_line_conv`.**  You will have to think about what numbers in the list will create a vertical line detector.  Run this cell.
# 4. **Add `vertical_line_conv` to `conv_list`.  Run that cell.**  You will see the output of your vertical line filter underneath the horizontal line filter.  You will also see a printed hint indicating if you got this right.  
# 
# Once that's done, you are ready to learn about deep convolutional models, the key to modern computer vision breakthroughs.
# 
# ---
# 
# # Import Utility Functions
# We'll use some small utilty functions to load raw image data, visualize the results and give hints on your answert, etc.  Don't worry about these, but execute the next cell to load the utility functions.
# 

# In[ ]:


import sys
# Add directory holding utility functions to path to allow importing
sys.path.append('/kaggle/input/python-utility-code-for-deep-learning-exercises/utils')
from exercise_1 import load_my_image, apply_conv_to_image, show, print_hints


# # Example Convolution: Horizontal Line Detector
# Here is the convolution you saw in the video.  It's provided here as an example, and you shouldn't need to change it.

# In[ ]:


# Detects bright pixels over dark pixels. 
horizontal_line_conv = [[1, 1], 
                        [-1, -1]]


# # Your Turn: Vertical Line Detector
# 
# **Replace the question marks with numbers to make a vertical line detector and uncomment both lines of code in the cell below.**

# In[ ]:


#vertical_line_conv = [[?, -?], 
#                      [?, ?]]


# **Once you have created vertical_line_conv in the cell above, add it as an additional item to `conv_list` in the next cell. Then run that cell.**

# In[ ]:


conv_list = [horizontal_line_conv]

original_image = load_my_image()
print("Original image")
show(original_image)
for conv in conv_list:
    filtered_image = apply_conv_to_image(conv, original_image)
    print("Output: ")
    show(filtered_image)


# 
# **Above, you'll see the output of the horizontal line filter as well as the filter you added. If you got it right, the output of your filter will looks like this.**
# ![Imgur](https://i.imgur.com/uR2ngvK.png)
# 
# ---
# # Keep Going
# **Now you are ready to [combine convolutions into powerful models](https://www.kaggle.com/dansbecker/building-models-from-convolutions). These models are fun to work with, so keep going.**
# 
# **[Deep Learning Track Home](https://www.kaggle.com/education/deep-learning)**
