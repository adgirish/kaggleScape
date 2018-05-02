
# coding: utf-8

# ### All days of the challange:
# 
# * [Day 1: Handling missing values](https://www.kaggle.com/rtatman/data-cleaning-challenge-handling-missing-values)
# * [Day 2: Scaling and normalization](https://www.kaggle.com/rtatman/data-cleaning-challenge-scale-and-normalize-data)
# * [Day 3: Parsing dates](https://www.kaggle.com/rtatman/data-cleaning-challenge-parsing-dates/)
# * [Day 4: Character encodings](https://www.kaggle.com/rtatman/data-cleaning-challenge-character-encodings/)
# * [Day 5: Inconsistent Data Entry](https://www.kaggle.com/rtatman/data-cleaning-challenge-inconsistent-data-entry/)
# ___
# Welcome to day 4 of the 5-Day Data Challenge! Today, we're going to be working with different character encodings. To get started, click the blue "Fork Notebook" button in the upper, right hand corner. This will create a private copy of this notebook that you can edit and play with. Once you're finished with the exercises, you can choose to make your notebook public to share with others. :)
# 
# > **Your turn!** As we work through this notebook, you'll see some notebook cells (a block of either code or text) that has "Your Turn!" written in it. These are exercises for you to do to help cement your understanding of the concepts we're talking about. Once you've written the code to answer a specific question, you can run the code by clicking inside the cell (box with code in it) with the code you want to run and then hit CTRL + ENTER (CMD + ENTER on a Mac). You can also click in a cell and then click on the right "play" arrow to the left of the code. If you want to run all the code in your notebook, you can use the double, "fast forward" arrows at the bottom of the notebook editor.
# 
# Here's what we're going to do today:
# 
# * [Get our environment set up](#Get-our-environment-set-up)
# * [What are encodings?](#What-are-encodings?)
# * [Reading in files with encoding problems](#Reading-in-files-with-encoding-problems)
# * [Saving your files with UTF-8 encoding](#Saving-your-files-with-UTF-8-encoding)
# 
# Let's get started!

# # Get our environment set up
# ________
# 
# The first thing we'll need to do is load in the libraries we'll be using. Not our datasets, though: we'll get to those later!
# 
# > **Important!** Make sure you run this cell yourself or the rest of your code won't work!

# In[ ]:


# modules we'll use
import pandas as pd
import numpy as np

# helpful character encoding module
import chardet

# set seed for reproducibility
np.random.seed(0)


# Now we're ready to work with some character encodings! (If you like, you can add a code cell here and take this opportunity to take a look at some of the data.)

# # What are encodings?
# ____
# 
# Character encodings are specific sets of rules for mapping from raw binary byte strings (that look like this: 0110100001101001) to characters that make up human-readable text (like "hi"). There are many different encodings, and if you tried to read in text with a different encoding that the one it was originally written in, you ended up with scrambled text called "mojibake" (said like mo-gee-bah-kay). Here's an example of mojibake:
# 
# æ–‡å—åŒ–ã??
# 
# You might also end up with a "unknown" characters. There are what gets printed when there's no mapping between a particular byte and a character in the encoding you're using to read your byte string in and they look like this:
# 
# ����������
# 
# Character encoding mismatches are less common today than they used to be, but it's definitely still a problem. There are lots of different character encodings, but the main one you need to know is UTF-8.
# 
# > UTF-8 is **the** standard text encoding. All Python code is in UTF-8 and, ideally, all your data should be as well. It's when things aren't in UTF-8 that you run into trouble.
# 
# It was pretty hard to deal with encodings in Python 2, but thankfully in Python 3 it's a lot simpler. (Kaggle Kernels only use Python 3.) There are two main data types you'll encounter when working with text in Python 3. One is is the string, which is what text is by default.

# In[ ]:


# start with a string
before = "This is the euro symbol: €"

# check to see what datatype it is
type(before)


# The other data is the [bytes](https://docs.python.org/3.1/library/functions.html#bytes) data type, which is a sequence of integers. You can convert a string into bytes by specifying which encoding it's in:

# In[ ]:


# encode it to a different encoding, replacing characters that raise errors
after = before.encode("utf-8", errors = "replace")

# check the type
type(after)


# If you look at a bytes object, you'll see that it has a b in front of it, and then maybe some text after. That's because bytes are printed out as if they were characters encoded in ASCII. (ASCII is an older character encoding that doesn't really work for writing any language other than English.) Here you can see that our euro symbol  has been replaced with some mojibake that looks like "\xe2\x82\xac" when it's printed as if it were an ASCII string.

# In[ ]:


# take a look at what the bytes look like
after


# When we convert our bytes back to a string with the correct encoding, we can see that our text is all there correctly, which is great! :)

# In[ ]:


# convert it back to utf-8
print(after.decode("utf-8"))


# However, when we try to use a different encoding to map our bytes into a string,, we get an error. This is because the encoding we're trying to use doesn't know what to do with the bytes we're trying to pass it. You need to tell Python the encoding that the byte string is actually supposed to be in.
# 
# > You can think of different encodings as different ways of recording music. You can record the same music on a CD, cassette tape or 8-track. While the music may sound more-or-less the same, you need to use the right equipment to play the music from each recording format. The correct decoder is like a cassette player or a cd player. If you try to play a cassette in a CD player, it just won't work. 

# In[ ]:


# try to decode our bytes with the ascii encoding
print(after.decode("ascii"))


# We can also run into trouble if we try to use the wrong encoding to map from a string to bytes. Like I said earlier, strings are UTF-8 by default in Python 3, so if we try to treat them like they were in another encoding we'll create problems. 
# 
# For example, if we try to convert a string to bytes for ascii using encode(), we can ask for the bytes to be what they would be if the text was in ASCII. Since our text isn't in ASCII, though, there will be some characters it can't handle. We can automatically replace the characters that ASCII can't handle. If we do that, however, any characters not in ASCII will just be replaced with the unknown character. Then, when we convert the bytes back to a string, the character will be replaced with the unknown character. The dangerous part about this is that there's not way to tell which character it *should* have been. That means we may have just made our data unusable!

# In[ ]:


# start with a string
before = "This is the euro symbol: €"

# encode it to a different encoding, replacing characters that raise errors
after = before.encode("ascii", errors = "replace")

# convert it back to utf-8
print(after.decode("ascii"))

# We've lost the original underlying byte string! It's been 
# replaced with the underlying byte string for the unknown character :(


# This is bad and we want to avoid doing it! It's far better to convert all our text to UTF-8 as soon as we can and keep it in that encoding. The best time to convert non UTF-8 input into UTF-8  is when you read in files, which we'll talk about next.
# 
# First, however, try converting between bytes and strings with different encodings and see what happens. Notice what this does to your text. Would you want this to happen to data you were trying to analyze?

# In[ ]:


# Your turn! Try encoding and decoding different symbols to ASCII and
# see what happens. I'd recommend $, #, 你好 and नमस्ते but feel free to
# try other characters. What happens? When would this cause problems?


# # Reading in files with encoding problems
# ___
# 
# Most files you'll encounter will probably be encoded with UTF-8. This is what Python expects by default, so most of the time you won't run into problems. However, sometimes you'll get an error like this: 

# In[ ]:


# try to read in a file not in UTF-8
kickstarter_2016 = pd.read_csv("../input/kickstarter-projects/ks-projects-201612.csv")


# Notice that we get the same `UnicodeDecodeError` we got when we tried to decode UTF-8 bytes as if they were ASCII! This tells us that this file isn't actually UTF-8. We don't know what encoding it actually *is* though. One way to figure it out is to try and test a bunch of different character encodings and see if any of them work. A better way, though, is to use the chardet module to try and automatically guess what the right encoding is. It's not 100% guaranteed to be right, but it's usually faster than just trying to guess.
# 
# I'm going to just look at the first ten thousand bytes of this file. This is usually enough for a good guess about what the encoding is and is much faster than trying to look at the whole file. (Especially with a  large file this can be very slow.) Another reason to just look at the first part of the file is that  we can see by looking at the error message that the first problem is the 11th character. So we probably only need to look at the first little bit of the file to figure out what's going on.

# In[ ]:


# look at the first ten thousand bytes to guess the character encoding
with open("../input/kickstarter-projects/ks-projects-201801.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))

# check what the character encoding might be
print(result)


# So chardet is 73%  confidence that the right encoding is "Windows-1252". Let's see if that's correct:

# In[ ]:


# read in the file with the encoding detected by chardet
kickstarter_2016 = pd.read_csv("../input/kickstarter-projects/ks-projects-201612.csv", encoding='Windows-1252')

# look at the first few lines
kickstarter_2016.head()


# Yep, looks like chardet was right! The file reads in with no problem (although we do get a warning about datatypes) and when we look at the first few rows it seems to be be fine. 
# 
# > **What if the encoding chardet guesses isn't right?** Since chardet is basically just a fancy guesser, sometimes it will guess the wrong encoding. One thing you can try is looking at more or less of the file and seeing if you get a different result and then try that.

# In[ ]:


# Your Turn! Trying to read in this file gives you an error. Figure out
# what the correct encoding should be and read in the file. :)
police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv")


# # Saving your files with UTF-8 encoding
# ___
# 
# Finally, once you've gone through all the trouble of getting your file into UTF-8, you'll probably want to keep it that way. The easiest way to do that is to save your files with UTF-8 encoding. The good news is, since UTF-8 is the standard encoding in Python, when you save a file it will be saved as UTF-8 by default:

# In[ ]:


# save our file (will be saved as UTF-8 by default!)
kickstarter_2016.to_csv("ks-projects-201801-utf8.csv")


# Pretty easy, huh? :)
# 
# > If you haven't saved a file in a kernel before, you need to hit the commit & run button and wait for your notebook to finish running first before you can see or access the file you've saved out. If you don't see it at first, wait a couple minutes and it should show up. The files you save will be in the directory "../output/", and you can download them from your notebook.

# In[ ]:


# Your turn! Save out a version of the police_killings dataset with UTF-8 encoding 


# And that's it for today! We didn't do quite as much coding, but  take my word for it: if you don't have the right tools, figuring out what encoding a file is in can be a huge time sink. If you have any questions, be sure to post them in the comments below or [on the forums](https://www.kaggle.com/questions-and-answers). 
# 
# Remember that your notebook is private by default, and in order to share it with other people or ask for help with it, you'll need to make it public. First, you'll need to save a version of your notebook that shows your current work by hitting the "Commit & Run" button. (Your work is saved automatically, but versioning your work lets you go back and look at what it was like at the point you saved it. It also lets you share a nice compiled notebook instead of just the raw code.) Then, once your notebook is finished running, you can go to the Settings tab in the panel to the left (you may have to expand it by hitting the [<] button next to the "Commit & Run" button) and setting the "Visibility" dropdown to "Public".
# 
# # More practice!
# ___
# 
# Check out [this dataset of files in different character encodings](https://www.kaggle.com/rtatman/character-encoding-examples). Can you read in all the files with their original encodings and them save them out as UTF-8 files?
# 
# If you have a file that's in UTF-8 but has just a couple of weird-looking characters in it, you can try out the [ftfy module](https://ftfy.readthedocs.io/en/latest/#) and see if it helps. 
