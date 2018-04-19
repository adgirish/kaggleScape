
# coding: utf-8

# # Animating and Smoothing 3D Facial Keypoints
# **Note:** if you are impatient, please **press the "output" tab** to look at a few giff's that are the final result of this script (please wait a bit for the giff's to load, and then they will be displayed at normal speed)  
# 
# In this script I will build upon the very pretty visualizations in [DrGuillermo's 3D Animation Script](https://www.kaggle.com/drgilermo/3d-kmeans-animation) and provide a utility function to draw 3D shape animations with a surrounding 3D bounding box. In this script I also provide several additional utility functions to aid the process of working with this dataset: one function to normalize shapes (2D or 3D) and an additional one to write videos for visualization.  
# 
# After showing an animation of the movment of facial keypoints in 3D, we then continue to filter some of the noise in the shape keypoints data by utilizing the spatial correlations across the dataset and create animations of the denoised keypoints, resulting in a much smoother and nicer animations.
# 
# We then continue to apply temporal filtering on the denoised keypoint coordinates, resulting in even smoother animations.
# 
# Finally, we verify that the filtering opperations didn't ruin anything by overlaying them on the original videos and looking at the differences. We conclude that the process of spatio-temporal filtering does produce a much nicer and cleaner anontation.

# In[ ]:


import numpy as np
import pandas as pd
from sklearn import decomposition
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import plotly.offline as py
import plotly.graph_objs as go
#import imageio
import glob

py.init_notebook_mode(connected=True)


# # Load the Data

# In[ ]:


videoDF = pd.read_csv('../input/youtube_faces_with_keypoints_large.csv')

# create a dictionary that maps videoIDs to full file paths
npzFilesFullPath = glob.glob('../input/youtube_faces_*/*.npz')
videoIDs = [x.split('/')[-1].split('.')[0] for x in npzFilesFullPath]
fullPaths = {}
for videoID, fullPath in zip(videoIDs, npzFilesFullPath):
    fullPaths[videoID] = fullPath

# remove from the large csv file all videos that weren't uploaded yet
videoDF = videoDF.loc[videoDF.loc[:,'videoID'].isin(fullPaths.keys()),:].reset_index(drop=True)
print('Number of Videos uploaded so far is %d' %(videoDF.shape[0]))
print('Number of Unique Individuals so far is %d' %(len(videoDF['personName'].unique())))


# # Show Overview of Dataset Content (that has been uploaded so far)

# In[ ]:


# overview of the contents of the dataset
groupedByPerson = videoDF.groupby("personName")
numVidsPerPerson = groupedByPerson.count()['videoID']
groupedByPerson.count().sort_values('videoID', axis=0, ascending=False)

plt.close('all')
plt.figure(figsize=(25,20))
plt.subplot(2,2,1)
plt.hist(x=numVidsPerPerson,bins=0.5+np.arange(numVidsPerPerson.min()-1,numVidsPerPerson.max()+1))
plt.title('Number of Videos per Person',fontsize=30); 
plt.xlabel('Number of Videos',fontsize=25); plt.ylabel('Number of People',fontsize=25)

plt.subplot(2,2,2)
plt.hist(x=videoDF['videoDuration'],bins=20);
plt.title('Distribution of Video Duration',fontsize=30); 
plt.xlabel('duration [frames]',fontsize=25); plt.ylabel('Number of Videos',fontsize=25)
plt.xlim(videoDF['videoDuration'].min()-2,videoDF['videoDuration'].max()+2)

plt.subplot(2,2,3)
plt.scatter(x=videoDF['imageWidth'], y=videoDF['imageHeight'])
plt.title('Distribution of Image Sizes',fontsize=30)
plt.xlabel('Image Width [pixels]',fontsize=25); plt.ylabel('Image Height [pixels]',fontsize=25)
plt.xlim(0,videoDF['imageWidth'].max() +15)
plt.ylim(0,videoDF['imageHeight'].max()+15)

plt.subplot(2,2,4)
averageFaceSize_withoutNaNs = np.array(videoDF['averageFaceSize'])
averageFaceSize_withoutNaNs = averageFaceSize_withoutNaNs[np.logical_not(np.isnan(averageFaceSize_withoutNaNs))]
plt.hist(averageFaceSize_withoutNaNs, bins=20);
plt.title('Distribution of Average Face Sizes ',fontsize=30);
plt.xlabel('Average Face Size [pixels]',fontsize=25); plt.ylabel('Number of Videos',fontsize=25);


# # Define some shape normalization utility functions

# In[ ]:


#%% define shape normalization utility functions
def NormlizeShapes(shapesImCoords):
    (numPoints, numDims, _) = shapesImCoords.shape
    """shapesNomalized, scaleFactors, meanCoords  = NormlizeShapes(shapesImCoords)"""
    
    # calc mean coords and subtract from shapes    
    meanCoords = shapesImCoords.mean(axis=0)
    shapesCentered = np.zeros(shapesImCoords.shape)
    shapesCentered = shapesImCoords - np.tile(meanCoords,[numPoints,1,1])

    # calc scale factors and divide shapes
    scaleFactors = np.sqrt((shapesCentered**2).sum(axis=1)).mean(axis=0)
    shapesNormlized = np.zeros(shapesCentered.shape)
    shapesNormlized = shapesCentered / np.tile(scaleFactors, [numPoints,numDims,1])

    return shapesNormlized, scaleFactors, meanCoords


def TransformShapeBackToImageCoords(shapesNomalized, scaleFactors, meanCoords):
    """shapesImCoords_rec = TransformShapeBackToImageCoords(shapesNomalized, scaleFactors, meanCoords)"""
    (numPoints, numDims, _) = shapesNomalized.shape
    
    # move back to the correct scale
    shapesCentered = shapesNomalized * np.tile(scaleFactors, [numPoints,numDims,1])
    # move back to the correct location
    shapesImCoords = shapesCentered + np.tile(meanCoords,[numPoints,1,1])
    
    return shapesImCoords


# # Normalize the 2D and 3D Shapes
# remember that like we showed in the [Exploration Script](https://www.kaggle.com/selfishgene/exploring-youtube-faces-with-keypoints-dataset), in order to compare apples to apples (or in this case, shapes to shapes), we need first to normalize the shapes in and manually remove the things that we don't care about (in this case, we want to disregard translation and scale differences between shapes, and model only the shape's shape :-) ) 

# In[ ]:


#%% Normalize 2D and 3D shapes

# collect all 2D and 3D shapes from all frames from all videos to a single numpy array matrix
totalNumberOfFrames = videoDF['videoDuration'].sum()
landmarks2D_all = np.zeros((68,2,int(totalNumberOfFrames)))
landmarks3D_all = np.zeros((68,3,int(totalNumberOfFrames)))

shapeIndToVideoID = {} # dictionary for later useage
endInd = 0
for i, videoID in enumerate(videoDF['videoID']):
    
    # load video
    videoFile = np.load(fullPaths[videoID])
    landmarks2D = videoFile['landmarks2D']
    landmarks3D = videoFile['landmarks3D']

    startInd = endInd
    endInd   = startInd + landmarks2D.shape[2]

    # store in one big array
    landmarks2D_all[:,:,startInd:endInd] = landmarks2D
    landmarks3D_all[:,:,startInd:endInd] = landmarks3D
    
    # make sure we keep track of the mapping to the original video and frame
    for videoFrameInd, shapeInd in enumerate(range(startInd,endInd)):
        shapeIndToVideoID[shapeInd] = (videoID, videoFrameInd)

# normlize shapes
landmarks2D_normlized, _, _  = NormlizeShapes(landmarks2D_all)
landmarks3D_normlized, _, _  = NormlizeShapes(landmarks3D_all)


# # Define a utility function to Create a 3D animation
# This is essentially the same code in [DrGuillermo's 3D Animation Script](https://www.kaggle.com/drgilermo/3d-kmeans-animation), only I've wrapped it with a function and added a bounding box drawing in order to avoid plotly's automatic rescaling of the axes, thus creating a contious scene and a feeling of a face moving around that scene.

# In[ ]:


#%% define a utility function to show 3D animation
def ShowAnimation_3D(landmarks3D):
    
    landmarks3D = landmarks3D.copy()
    landmarks3D[:,0,:] = -landmarks3D[:,0,:]
    
    xMin = landmarks3D[:,0,:].min()-5
    xMax = landmarks3D[:,0,:].max()+5
    yMin = landmarks3D[:,1,:].min()-5
    yMax = landmarks3D[:,1,:].max()+5
    zMin = landmarks3D[:,2,:].min()-5
    zMax = landmarks3D[:,2,:].max()+5
    
    boxCorners = np.array([[xMin,yMin,zMin],
                           [xMin,yMin,zMax],
                           [xMin,yMax,zMin],
                           [xMin,yMax,zMax],
                           [xMax,yMin,zMin],
                           [xMax,yMin,zMax],
                           [xMax,yMax,zMin],
                           [xMax,yMax,zMax]])
    
    traversalOrder = [0,1,3,2,0,4,6,2,6,7,3,7,5,1,5,4]
    boxTraceCoords = np.zeros((len(traversalOrder),3))
    for i, corner in enumerate(traversalOrder):
        boxTraceCoords[i,:] = boxCorners[corner,:]
    
    trace1   = go.Scatter3d(name='Jawline', x=landmarks3D[:,0,1][0:17],y=landmarks3D[:,1,1][0:17],z=landmarks3D[:,2,1][0:17],
                            mode='lines+markers',marker=dict(color = 'blue',opacity=0.7,size = 5))
    
    trace2   = go.Scatter3d(name='Right Eyebrow',x=landmarks3D[:,0,1][17:22],y=landmarks3D[:,1,1][17:22],z=landmarks3D[:,2,1][17:22],
                            mode='lines+markers',marker=dict(color = 'blue',opacity=0.7,size = 5))
    
    trace3   = go.Scatter3d(name='Left Eyebrow',x=landmarks3D[:,0,1][22:27],y=landmarks3D[:,1,1][22:27],z=landmarks3D[:,2,1][22:27],
                            mode='lines+markers',marker=dict(color = 'blue',opacity=0.7,size = 5))
    
    trace4   = go.Scatter3d(name='Nose Ridge',x=landmarks3D[:,0,1][27:31],y=landmarks3D[:,1,1][27:31],z=landmarks3D[:,2,1][27:31],
                            mode='lines+markers',marker=dict(color = 'green',opacity=0.6,size = 5))
    
    trace5   = go.Scatter3d(name='Nose Base',x=landmarks3D[:,0,1][31:36],y=landmarks3D[:,1,1][31:36],z=landmarks3D[:,2,1][31:36],
                            mode='lines+markers',marker=dict(color = 'green',opacity=0.6,size = 5))
    
    trace6   = go.Scatter3d(name='Right Eye', x=landmarks3D[:,0,1][36:42],y=landmarks3D[:,1,1][36:42],z=landmarks3D[:,2,1][36:42],
                            mode='lines+markers',marker=dict(color = 'green',opacity=0.6,size = 5))
    
    trace7   = go.Scatter3d(name='Left Eye', x=landmarks3D[:,0,1][42:48],y=landmarks3D[:,1,1][42:48],z=landmarks3D[:,2,1][42:48],
                            mode='lines+markers',marker=dict(color = 'green',opacity=0.6,size = 5))
    
    trace8   = go.Scatter3d(name='Outer Mouth', x=landmarks3D[:,0,1][48:60],y=landmarks3D[:,1,1][48:60],z=landmarks3D[:,2,1][48:60],
                            mode='lines+markers',marker=dict(color = 'green',opacity=0.6,size = 5))
    
    trace9   = go.Scatter3d(name='Inner Mouth', x=landmarks3D[:,0,1][60:68],y=landmarks3D[:,1,1][60:68],z=landmarks3D[:,2,1][60:68],
                            mode='lines+markers',marker=dict(color = 'green',opacity=0.6,size = 5))
        
    boxTrace = go.Scatter3d(name='boundingBox', x=boxTraceCoords[:,0],y=boxTraceCoords[:,1],z=boxTraceCoords[:,2],
                            mode='lines+markers',marker=dict(color = 'red', opacity=1.0,size = 5))
    
    data = [trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, boxTrace]
    
    mfr = []
    for t in range(len(landmarks3D[1,1,:])):
        mfr.append({'data' :[{'type' : "scatter3d",'mode':'lines+markers',
                              'x':landmarks3D[:,0,t][ 0:17],'y':landmarks3D[:,1,t][ 0:17],'z':landmarks3D[:,2,t][ 0:17]},
                             {'type' : "scatter3d",'mode':'lines+markers',
                              'x':landmarks3D[:,0,t][17:22],'y':landmarks3D[:,1,t][17:22],'z':landmarks3D[:,2,t][17:22]},
                             {'type' : "scatter3d",'mode':'lines+markers',
                              'x':landmarks3D[:,0,t][22:27],'y':landmarks3D[:,1,t][22:27],'z':landmarks3D[:,2,t][22:27]},
                             {'type' : "scatter3d",'mode':'lines+markers',
                              'x':landmarks3D[:,0,t][27:31],'y':landmarks3D[:,1,t][27:31],'z':landmarks3D[:,2,t][27:31]},
                             {'type' : "scatter3d",'mode':'lines+markers',
                              'x':landmarks3D[:,0,t][31:36],'y':landmarks3D[:,1,t][31:36],'z':landmarks3D[:,2,t][31:36]},
                             {'type' : "scatter3d",'mode':'lines+markers',
                              'x':landmarks3D[:,0,t][36:42],'y':landmarks3D[:,1,t][36:42],'z':landmarks3D[:,2,t][36:42]},
                             {'type' : "scatter3d",'mode':'lines+markers',
                              'x':landmarks3D[:,0,t][42:48],'y':landmarks3D[:,1,t][42:48],'z':landmarks3D[:,2,t][42:48]},
                             {'type' : "scatter3d",'mode':'lines+markers',
                              'x':landmarks3D[:,0,t][48:60],'y':landmarks3D[:,1,t][48:60],'z':landmarks3D[:,2,t][48:60]},
                             {'type' : "scatter3d",'mode':'lines+markers',
                              'x':landmarks3D[:,0,t][60:68],'y':landmarks3D[:,1,t][60:68],'z':landmarks3D[:,2,t][60:68]},
                             {'type' : "scatter3d",'mode':'lines+markers',
                             'x':boxTraceCoords[:,0],'y':boxTraceCoords[:,1],'z':boxTraceCoords[:,2]}]})
                                 
    
    layout = go.Layout(width=800, height=800, title='3D Face Shape Animation',
                       scene=dict(camera=dict(up     = dict(x= 0, y=-1.0, z=0),
                                              center = dict(x= 0, y= 0.0, z=0),
                                              eye    = dict(x= 0, y= 0.7, z=2),
                                             )
                                 ),
                        updatemenus=[dict(type='buttons', showactive=False,
                                            y=1,
                                            x=1,
                                            xanchor='right',
                                            yanchor='top',
                                            pad=dict(t=0, r=10),
                                            buttons=[dict(
                                                        label='Play Animation',
                                                        method='animate',
                                                        args=[None, dict(frame       = dict(duration=0.04, redraw=True), 
                                                                         transition  = dict(duration=0),
                                                                         fromcurrent = True,
                                                                         mode = 'immediate'
                                                                        )
                                                             ]
                                                         )
                                                    ]
                                           )
                                      ]
                      )
                                                    
    fig = dict(data=data, layout=layout, frames=mfr)
    py.iplot(fig)


# # Load a Video and Present it's 3D keypoint Animation
# Press the "Play Animation" button to play a nice and fast animation, don't wait for the automatic animation to finish (it's too slow)

# In[ ]:


#%% load a 3D landmarks sequence and present it
personToUse = 'Laura_Bush_4'
videoFile = np.load(fullPaths[personToUse])
landmarks3D_curr = videoFile['landmarks3D']

ShowAnimation_3D(landmarks3D_curr)


# I don't know about you, but I think this is nice!   
# Really nice work [DrGuillermo](https://www.kaggle.com/drgilermo) did there!

# # Build a 3D Shape Model by fitting a Multivariate Gaussian (PCA)
# Remember that PCA is essentially fitting a multivariate gaussian distribution with a low rank covariance matrix

# In[ ]:


#%% build 3D shape model
numComponents = 30

normalizedShapesTable = np.reshape(landmarks3D_normlized, [68*3, landmarks3D_normlized.shape[2]]).T
shapesModel = decomposition.PCA(n_components=numComponents, whiten=True, random_state=1).fit(normalizedShapesTable)
print('Total explained percent by PCA model with %d components is %.1f%s' %(numComponents, 100*shapesModel.explained_variance_ratio_.sum(),'%'))


# # Project the original shapes onto our Shape Model and Reconstruct
# The code is very simple, you are more than welcome to unhide and check it out

# In[ ]:


#%% interpret the shapes using our shape model (project and reconstruct)

# normlize shapes (and keep the scale factors and mean coords for later reconstruction)
landmarks3D_norm, scaleFactors, meanCoords  = NormlizeShapes(landmarks3D_curr)
# convert to matrix form
landmarks3D_norm_table = np.reshape(landmarks3D_norm, [68*3, landmarks3D_norm.shape[2]]).T
# project onto shapes model and reconstruct
landmarks3D_norm_table_rec = shapesModel.inverse_transform(shapesModel.transform(landmarks3D_norm_table))
# convert back to shapes (numKeypoint, numDims, numFrames)
landmarks3D_norm_rec = np.reshape(landmarks3D_norm_table_rec.T, [68, 3, landmarks3D_norm.shape[2]])
# transform back to image coords
landmarks3D_curr_rec = TransformShapeBackToImageCoords(landmarks3D_norm_rec, scaleFactors, meanCoords)


# # Show the animation of the filtered version of the original shape

# In[ ]:


# show the new animation
ShowAnimation_3D(landmarks3D_curr_rec)


# ### Note how much smoother this shape is!
# This is just the result of applying a prior to constrain the shape at each particular frame according to the distribution of shapes across all frames of all individuals in the dataset. The power of statistics is sometimes mind blowing!
# Hard to decide what is more awsome, the plotly library that allwos us to show these nice animations, or math that allows us to create the smoothed even nicer versions of these animations

# # Plot $(x(t),y(t),z(t))$ traces for several selected keypoints

# In[ ]:


#%% plot x(t), y(t), z(t) for several keypoints before and after filtering
selectedKeypointInds    = [ 30, 33,  36, 39,  42, 45,  51, 57,  62, 66,  48, 54]
selectedKeypointColors  = ['g','g', 'r','m', 'm','r', 'b','b', 'g','g', 'y','y']
selectedKeypointStrings = ['nose tip', 'nose base',  'right eye outer','right eye inner',  'left eye inner','left eye outer',
                           'outer mouth top','outer mouth bottom',  'inner mouth top','inner mouth bottom',
                           'right mouth corner','left mouth corner']

plt.figure(figsize=(13,11)); plt.suptitle('Original Traces', fontsize=22)
for subplotInd, yLabel in enumerate(['x(t)','y(t)','z(t)']):
    plt.subplot(3,1,subplotInd+1); plt.ylabel(yLabel,fontsize=20)
    for k,c,legendLabel in zip(selectedKeypointInds,selectedKeypointColors,selectedKeypointStrings):
        plt.plot(landmarks3D_curr[k,subplotInd,:],c=c,label=legendLabel)
    if subplotInd == 0: 
        plt.legend(bbox_to_anchor=(0,1,1,0), shadow=True,
                   loc=3, ncol=4, mode="expand", borderaxespad=0, fontsize=12)
plt.xlabel('time [frame]',fontsize=20);


# # Show the "Spatially filtered" keypoint traces
# Red - Original traces  
# Blue - Spatially filtered

# In[ ]:


plt.figure(figsize=(13,11)); plt.suptitle('Original Vs. "Spatially" Filtered Traces', fontsize=22)
for subplotInd, yLabel in enumerate(['x(t)','y(t)','z(t)']):
    plt.subplot(3,1,subplotInd+1); plt.ylabel(yLabel,fontsize=25)
    plt.plot(landmarks3D_curr[selectedKeypointInds,subplotInd,:].T,c='r')
    plt.plot(landmarks3D_curr_rec[selectedKeypointInds,subplotInd,:].T,c='b')
plt.xlabel('time [frame]',fontsize=20);


# The blue traces look smoother, but not that much smoother. Maybe we should explicitly smoothen them temporally as well?

# # Temporally Smoothen the keypoint movement
# Red - Original traces  
# Blue - Spatially and Temporally filtered

# In[ ]:


#%% apply temporal filtering on the 3D points and show filtered signals
filterHalfLength = 2
temporalFilter = np.ones((1,1,2*filterHalfLength+1))
temporalFilter = temporalFilter / temporalFilter.sum()

startTileBlock = np.tile(landmarks3D_curr_rec[:,:,0][:,:,np.newaxis],[1,1,filterHalfLength])
endTileBlock = np.tile(landmarks3D_curr_rec[:,:,-1][:,:,np.newaxis],[1,1,filterHalfLength])
landmarks3D_curr_rec_padded = np.dstack((startTileBlock,landmarks3D_curr_rec,endTileBlock))
landmarks3D_curr_rec_filtered = signal.convolve(landmarks3D_curr_rec_padded, temporalFilter, mode='valid', method='fft')

plt.figure(figsize=(13,11)); plt.suptitle('Original Vs. Spatio-Temporally Filtered Traces', fontsize=22)
for subplotInd, yLabel in enumerate(['x(t)','y(t)','z(t)']):
    plt.subplot(3,1,subplotInd+1); plt.ylabel(yLabel,fontsize=20)
    plt.plot(landmarks3D_curr[selectedKeypointInds,subplotInd,:].T,c='r')
    plt.plot(landmarks3D_curr_rec_filtered[selectedKeypointInds,subplotInd,:].T,c='b')
plt.xlabel('time [frame]',fontsize=20);


# Now we can see the traces are much smoother, but we need to make sure that we didn't accidentally ruin anything.

# # Show the animation of the new filtered version of the shape

# In[ ]:


#%% show animation of the temporally filtered 3D points
ShowAnimation_3D(landmarks3D_curr_rec_filtered)


# This is much smoother looking. Even to a point that the animation appears quite slow, so we need to make a final verfication stage and view these points overlaid on the original video and see if we've messed something up or not.

# # Embed the $(x,y)$ coordinates into the video itself
# mark the the keypoints with green

# In[ ]:


#%% helper function: create videos with keypoints overlaid for each of the 3 processing stages
def CreateVideosWithMarkingsSideBySide(colorImages, landmarks3D_curr, landmarks3D_curr_rec, landmarks3D_curr_rec_filtered):
    imageWithMarkings_orig   = colorImages.copy()
    imageWithMarkings_sp     = colorImages.copy()
    imageWithMarkings_sp_tmp = colorImages.copy()
    
    # paint requested channel
    channelToMark = 1
    for frame in range(colorImages.shape[3]):
        for k in range(landmarks3D_curr.shape[0]):
            for dh in [-1,0,1]:
                for dw in [-1,0,1]:
                    locH_orig   = int(np.round(landmarks3D_curr[k,1,frame])) + dh
                    locW_orig   = int(np.round(landmarks3D_curr[k,0,frame])) + dw
                    
                    locH_sp     = int(np.round(landmarks3D_curr_rec[k,1,frame])) + dh
                    locW_sp     = int(np.round(landmarks3D_curr_rec[k,0,frame])) + dw
                    
                    locH_sp_tmp = int(np.round(landmarks3D_curr_rec_filtered[k,1,frame])) + dh
                    locW_sp_tmp = int(np.round(landmarks3D_curr_rec_filtered[k,0,frame])) + dw
                    try:
                        imageWithMarkings_orig[locH_orig,locW_orig,channelToMark,frame] = 255
                        imageWithMarkings_sp[locH_sp,locW_sp,channelToMark,frame] = 255
                        imageWithMarkings_sp_tmp[locH_sp_tmp,locW_sp_tmp,channelToMark,frame] = 255
                    except:
                        pass
            
    SideBySide = np.hstack((imageWithMarkings_orig,imageWithMarkings_sp,imageWithMarkings_sp_tmp))
    return SideBySide


# # Write a Video with all three stages side by side
# 

# In[ ]:


SideBySide = CreateVideosWithMarkingsSideBySide(videoFile['colorImages'], landmarks3D_curr, landmarks3D_curr_rec, landmarks3D_curr_rec_filtered)

#%% show video animations
def WriteColorVideo(video, filename='sample.gif', fps=20):
    writer = imageio.get_writer(filename, fps=fps)
    for frame in range(video.shape[-1]):
        writer.append_data(video[:, :, :, frame])
    writer.close()

#WriteColorVideo(SideBySide,'processingStages.gif',fps=25)


# It appears that the standard library I use for writing videos and gifs (**imageio** library) is not present in the kaggle kernels environment, so we'll need to find a **workaround** for now, but on your personal computers you can definatley use it.

# # Function that writes gif files

# In[ ]:


MSEC_PER_FRAME    = 40
MSEC_REPEAT_DELAY = 500

# Create an animated GIF file from a sequence of images
def build_gif(inImages, fname=None, show_gif=True, save_gif=True, title=''):
    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, 
                        wspace=None, hspace=None)  # removes white border
    
    imgs = [ (ax.imshow(inImages[:,:,:,frame]), 
              ax.set_title(title), 
              ax.annotate(frame,(5,5))) for frame in range(inImages.shape[3]) ] 

    img_anim = animation.ArtistAnimation(fig, imgs, interval=MSEC_PER_FRAME, 
                                         repeat_delay=MSEC_REPEAT_DELAY, blit=False)
    if save_gif:
        print('Writing:', fname)
        img_anim.save(fname, writer='imagemagick')
    if show_gif:
        plt.show();
    plt.clf() # clearing the figure when done prevents a memory leak 


# The above workaround was stolen from [this script](https://www.kaggle.com/chefele/animated-images-with-outlined-nerve-area) by [Christopher Hefele](https://www.kaggle.com/chefele)

# In[ ]:


titleStr = 'original video | spatially filtered | spatio-temporally filtered'
build_gif(SideBySide, fname='smoothing_stages_side_by_side.gif', show_gif=False, save_gif=True, title=titleStr)


# ## You should be able to view the gif file under the ***Output Tab*** of the kernel

# # Let's Repeat this process again for several additional videos

# In[ ]:


for personToUse in ['Martin_Sheen_5','Elizabeth_Berkeley_1','Tom_Hanks_3','Reese_Witherspoon_4','Kurt_Warner_1']:

    videoFile = np.load(fullPaths[personToUse])
    landmarks3D_curr = videoFile['landmarks3D']
    
    landmarks3D_norm, scaleFactors, meanCoords  = NormlizeShapes(landmarks3D_curr)
    landmarks3D_norm_table = np.reshape(landmarks3D_norm, [68*3, landmarks3D_norm.shape[2]]).T
    landmarks3D_norm_table_rec = shapesModel.inverse_transform(shapesModel.transform(landmarks3D_norm_table))
    landmarks3D_norm_rec = np.reshape(landmarks3D_norm_table_rec.T, [68, 3, landmarks3D_norm.shape[2]])
    landmarks3D_curr_rec = TransformShapeBackToImageCoords(landmarks3D_norm_rec, scaleFactors, meanCoords)
    
    startTileBlock = np.tile(landmarks3D_curr_rec[:,:,0][:,:,np.newaxis],[1,1,filterHalfLength])
    endTileBlock   = np.tile(landmarks3D_curr_rec[:,:,-1][:,:,np.newaxis],[1,1,filterHalfLength])
    landmarks3D_curr_rec_padded = np.dstack((startTileBlock,landmarks3D_curr_rec,endTileBlock))
    
    landmarks3D_curr_rec_filtered = signal.convolve(landmarks3D_curr_rec_padded, temporalFilter, mode='valid', method='fft')
    
    SideBySide = CreateVideosWithMarkingsSideBySide(videoFile['colorImages'], landmarks3D_curr, landmarks3D_curr_rec, landmarks3D_curr_rec_filtered)
    build_gif(SideBySide, fname='smoothing_stages_' + personToUse + '.gif', show_gif=False, save_gif=True, title=titleStr)

