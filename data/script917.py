
# coding: utf-8

# The Standard
# ============
# 
# [http://dicom.nema.org/standard.html][1]
# 
# 
#   [1]: http://dicom.nema.org/standard.html

# In[ ]:


import numpy as np
import pandas as pd
import dicom, glob
import multiprocessing

images = sorted(glob.glob('../input/sample_images/**/*.dcm'))
#images = glob.glob('../input/stage1/**/*.dcm')
images = pd.DataFrame([[i, i.split('/')[3], i.split('/')[4], None, []] for i in images], columns=['path','id','image', 'series_no', 'pixels'])
#train = pd.read_csv('../input/stage1_labels.csv')
#test = pd.read_csv('../input/stage1_sample_submission.csv')
#print(len(images), len(train), len(test))


# In[ ]:


#https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
from skimage import measure, morphology
import scipy.ndimage, os
import pickle

for i in images.id.unique():
    i = '0bd0e3056cbf23a1cb7f0f0b18446068' #remove
    path = '../input/sample_images/' + i
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    image = np.stack([s.pixel_array for s in slices])
    pickle.dump(image, open('' + str(i)+'.pkl', 'wb'))
    image = pickle.loads(open('' + str(i)+'.pkl','rb').read())
    print(image.shape, i)
    break


# In[ ]:


from PIL import Image, ImageDraw
from PIL import ImageFont
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

image = pickle.loads(open('0bd0e3056cbf23a1cb7f0f0b18446068.pkl','rb').read())

def set_color(mu):
    c = int((550 + mu)/100*255)
    return (c,0,c)

im = Image.new('RGBA', (int(image.shape[1] + image.shape[0]/10), int(image.shape[2] + image.shape[0]/10)))
d = ImageDraw.Draw(im)
for z in range(0,image.shape[0], 10):
    for x in range(image.shape[2]-10):
        for y in range(image.shape[1]-10):
            mu = image[z][y][x] -1024
            if -550<mu and mu<-450: #lung
                im.putpixel((int(x+z/10), int(y+z/10)), set_color(mu))
                #fill surrounding
                for o1 in range(1,10):
                    for o2 in range(1,10):
                        mu2 = image[z][y+o1][x+o2] -1024
                        if -100<mu2 and mu2<-50: #fat
                            im.putpixel((int(x+o1+z/10), int(y+o2+z/10)), (255,182,193))
                        if 30<mu2 and mu2<45: #blood
                            im.putpixel((int(x+o1+z/10), int(y+o2+z/10)), (255,0,0))
                        if -1010<mu2 and mu2<-990: #air
                            im.putpixel((int(x+o1+z/10), int(y+o2+z/10)), (135,206,235))
            if mu>500: #bone
                im.putpixel((int(x+z/10), int(y+z/10)), (35,35,35))
plt.imshow(im); plt.axis('off')

