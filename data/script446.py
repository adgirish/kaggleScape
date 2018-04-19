
# coding: utf-8

# 
# ## Feature Extractor for the EEG Data Files
# ### Cloudy with a Chance of Insight
# This is a refactored fork of [Deep's feature extractor](https://www.kaggle.com/deepcnn/melbourne-university-seizure-prediction/feature-extractor-matlab2python-translated) which was part of a winning entry in a past Kaggle EEG competition. The code below only loads the MATLAB data files and pre-processes them by extracting possibly useful features that you can use to train machine learning models on.

# ## [Melbourne University AES/MathWorks/NIH Seizure Prediction](https://www.kaggle.com/c/melbourne-university-seizure-prediction)
# 
# ### Predict seizures in long-term human intracranial EEG recordings.
# 
# __author__ = 'Tony Reina: https://www.kaggle.com/treina/'
# #### Based on the kernels from __author__ = '[Solomonk](https://www.kaggle.com/solomonk/)' and from __author__ = '[Deep](http://www.kaggle.com/deepcnn/melbourne-university-seizure-prediction/feature-extractor-matlab2python-translated)' . Thanks!

# This competition is sponsored by MathWorks, the National Institutes of Health (NINDS), the American Epilepsy Society and the University of Melbourne, and organised in partnership with the Alliance for Epilepsy Research, the University of Pennsylvania and the Mayo Clinic.
# ![alt-text](https://kaggle2.blob.core.windows.net/competitions/kaggle/5390/media/Mathwork.png) 
# ![alt-text](https://kaggle2.blob.core.windows.net/competitions/kaggle/5390/media/AES.png) 
# ![alt-text](https://kaggle2.blob.core.windows.net/competitions/kaggle/5390/media/NINDS.png) 
# ![alt-text](https://kaggle2.blob.core.windows.net/competitions/kaggle/5390/media/UMel.png)
# 
#      

# ### EEG Dataset
# 
# [Thus says the Kraggle contest website (accessed 19 SEP 2016 at 14:00PST):](https://www.kaggle.com/c/melbourne-university-seizure-prediction/data)
# 
# Human brain activity was recorded in the form of intracranial EEG (iEEG) which involves electrodes positioned on the surface of the cerebral cortex and the recording of electrical signals with an ambulatory monitoring system. iEEG was sampled from 16 electrodes at 400 Hz, and recorded voltages were referenced to the electrode group average. These are long duration recordings, spanning multiple months up to multiple years and recording large numbers of seizures in some humans.
# 
# Intracranial EEG (iEEG) data clips are organized in folders containing training and testing data for each human patient. The training data is organized into ten minute EEG clips labeled "Preictal" for pre-seizure data segments, or "Interictal" for non-seizure data segments. Training data segments are numbered sequentially, while testing data are in random order. Within folders data segments are stored in .mat files as follows:
# 
# * I_J_K.mat - the Jth training data segment corresponding to the Kth class (K=0 for interictal, K=1 for preictal) for the Ith patient (there are three patients).
# * I_J.mat - the Jth testing data segment for the Ith patient.
# 
# Each .mat file contains a data structure, dataStruct, with fields as follows:
# 
# * data: a matrix of iEEG sample values arranged row x column as time sample x electrode.
# * nSamplesSegment: total number of time samples (number of rows in the data field).
# * iEEGsamplingRate: data sampling rate, i.e. the number of data samples representing 1 second of EEG data. 
# * channelIndices: an array of the electrode indexes corresponding to the columns in the data field.
# * sequence: the index of the data segment within the one hour series of clips (see below). For example, 1_12_1.mat has a sequence number of 6, and represents the iEEG data from 50 to 60 minutes into the preictal data. This field only appears in training data.

# ### Import the necessary Python libraries

# In[ ]:


# Use inline matlib plots
get_ipython().run_line_magic('matplotlib', 'inline')

# Import python libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Get specific functions from some other python libraries
from math import floor, log
from scipy.stats import skew, kurtosis
from scipy.io import loadmat   # For loading MATLAB data (.dat) files


# ### Identify which MATLAB data file you want to analyze

# In[ ]:


'''
Opens a MATLAB file using the Qt file dialog
'''
# We could always just type the filename into this cell, but let's be slick and add a Qt dialog
# to select the file.
def openfile_dialog():
#     from PyQt4 import QtGui
#     app = QtGui.QApplication([dir])
#     fname = QtGui.QFileDialog.getOpenFileName(None, "Select a MATLAB data file...", '.', filter="MATLAB data file (*.mat);;All files (*)")
#     return str(fname)

    return '../input/train_1/1_25_1.mat'


#  ### Load the MATLAB data file
#  Here we just load it into a standard Python dictionary. In the future, I'd like to get this passed as a Pandas dataframe.

# In[ ]:


def convertMatToDictionary(path):
    
    try: 
        mat = loadmat(path)
        names = mat['dataStruct'].dtype.names
        ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
        
    except ValueError:     # Catches corrupted MAT files (e.g. train_1/1_45_1.mat)
        print('File ' + path + ' is corrupted. Will skip this file in the analysis.')
        ndata = None
    
    return ndata


# ### Calculate FFT
# <a id="cloud"></a>
# 
# The following code calculates a [Fast Fourier Transform](https://en.wikipedia.org/wiki/Fast_Fourier_transform) of the signal. It also removes the DC component and then divides the Fourier amplitudes by the sum of the amplitudes. I think the DC removal and dividing by the sum are just data normalization (remove mean and make values go from -1 to +1). This prevents features from being considered more important in the analysis solely because they are at a larger scale.
# 
# Fourier Transforms are widely used in signals analysis. Usually, signals in the real world have complex shapes and are, therefore, difficult to describe mathematically. If it can't be described with an equation, then it is hard to analyze on a computer. 
# 
# Think about **clouds in the sky**. They are-- by definition-- nebulous. In order to point them out to another person, we typically find shapes and patterns within a completely random collection of water molecules. That one looks like a duck; this one looks like a rabbit; that one is a sailboat. The clouds of course are not these objects, but by approximating them with a well-known object we have any easier time with the analysis. We can say, "Isn't the rabbit's left ear a little mishapen?" And, everyone instantly knows where in the cloud to look.
# 
# ![alt-text](https://i.ytimg.com/vi/wMvb_ZsrmMw/hqdefault.jpg "Clouds: Making shape out of form")
# 
# Similarly, any signal-- no matter how complex-- can be thought of as the addition a collection of simpler (i.e. easier to describe mathematically) signals. You can pick any ensemble of signals (sine waves, traingular waves, polynomials, wavelets) to approximate a complex signal. 
# 
# Fourier Transforms are the most commonly used because the mathematics of sine waves has been well studied and has some nice convenient properties that help process them more easily on a computer. Sine waves also give an idea of how quickly things are changing over time within a signal-- a concept gives us an intuitive insight into what information or processing is "going on" in the system being studied. We may even use a Fourier Transform to hypothesize there are separate components within the system. Filtering (i.e. separating) these components in the Fourier domain is easy and can even be done in realtime with hardware.
# 
# FFTs are important in EEG pre-processing because it is believed that different frequency bands are correlated with different observable behaviors. For example, some frequency bands seem to distinguish awake versus sleeping. Others seem to be correlated with concentration and attention. Brain disorders, such as schizophrenia, are correlated with disruptions in the FFT.

# In[ ]:


'''
Calculates the FFT of the epoch signal. Removes the DC component and normalizes the area to 1
'''
def calcNormalizedFFT(epoch, lvl, nt, fs):
    
    lseg = np.round(nt/fs*lvl).astype('int')
    D = np.absolute(np.fft.fft(epoch, n=lseg[-1], axis=0))
    D[0,:]=0                                # set the DC component to zero
    D /= D.sum()                      # Normalize each channel               

    return D


# ## Important EEG frequency bands
# An EEG measures the synaptic activity caused by post-synaptic potentials in cortical neurons. Although the EEG is sampled several hundred times per second (e.g. 400 Hz), the bands that seem most important to neurologists are all less than 30 Hz. Mainly, there a 4 bands: $$\alpha, \beta,  \Delta, \theta$$ 
# These four originated from scientists looking at sleep versus awake states. More currently, the gamma band (particulary around 40 Hz) has been hypothesized to be a sign of correlation (_aka_ synchrony) between different areas of the brain and even has been suggested to be a marker of conciousness or attention.
# ![EEG rhythms](https://www.researchgate.net/profile/Priyanka_Abhang3/publication/281801676/figure/fig4/AS:305025248186371@1449735094401/Fig-4-EEG-waves-for-different-signals.png)
# 
# The strategy is to divide the EEG frequency spectrum along these bands (using an FFT) and try to see if changes in those frequencies can be used as good features for our classifier model.

# In[ ]:


def defineEEGFreqs():
    
    '''
    EEG waveforms are divided into frequency groups. These groups seem to be related to mental activity.
    alpha waves = 8-13 Hz = Awake with eyes closed
    beta waves = 14-30 Hz = Awake and thinking, interacting, doing calculations, etc.
    gamma waves = 30-45 Hz = Might be related to conciousness and/or perception (particular 40 Hz)
    theta waves = 4-7 Hz = Light sleep
    delta waves < 3.5 Hz = Deep sleep

    There are other EEG features like sleep spindles and K-complexes, but I think for this analysis
    we are just looking to characterize the waveform based on these basic intervals.
    '''
    return (np.array([0.1, 4, 8, 14, 30, 45, 70, 180]))  # Frequency levels in Hz


# ### HELP! Can someone comment on exactly what DSpect represents?
# 
# The DSpect routine seems to do the following:
# + calculates the normalized FFT (FFT minus the 0 Hz component divided by the sum of the FFT amplitudes)
# + determines equal-sized frequency bands (lvl) to divide the FFT spectrum into
# + gets the sum of those frequency bands (multiplied by 2)
# 
# I think this is some form of a energy contained within each of these frequency bands. However, Deep disagrees. Ill defer to his expertise.
# 
# [Deep](https://www.kaggle.com/deepcnn/melbourne-university-seizure-prediction/feature-extractor-matlab2python-translated/comments) described DSpect in the following way:
# 
# > Since the D is just the spectra of the signal you can't quite say the dspect is an energy in the bands specified by lseg. But, it is fairly similar. If D wasn't a direct spectra, but a psd, then, dspect would just be the energy list with each member of the list being the energy in the frequency band specified by lseg[j+1], lseg[j]. 
# 
# I'd love to get someone else to better describe what DSpect is doing and why we need it. Is the doubling of the sum to take care of the complex part of the FFT spectrum?

# In[ ]:


def calcDSpect(epoch, lvl, nt, nc,  fs):
    
    D = calcNormalizedFFT(epoch, lvl, nt, fs)
    lseg = np.round(nt/fs*lvl).astype('int')
    
    dspect = np.zeros((len(lvl)-1,nc))
    for j in range(len(dspect)):
        dspect[j,:] = 2*np.sum(D[lseg[j]:lseg[j+1],:], axis=0)
        
    return dspect


# ### Nonstationarity
# 
# Ok. So truth time. 
# 
# FFTs can only get you so far in life because they assume that your system doesn't vary substantially with time. This is called [stationarity](https://en.wikipedia.org/wiki/Stationary_process). 
# 
# Stationarity is a statistical concept which says that if I divide a signal up into smaller windows, then each windows's mean, standard deviation, and autocorrelation are about the same. (or, more formally, that the joint probability density function does not change when shifted in time-- whatever underlying process generating the signal at time t1 is exactly the same as at time t2) So, for example, [a pure tone at 40 Hz](https://www.youtube.com/watch?v=63lFYgEGT3k) will have the same mean and standard deviation (and other nth-order statistics) regardless of when you measure the signal. 
# 
# However, suppose you turn off the signal generator at some random times. Then, there would be periods with a 40 Hz signal interspersed with periods with no signal (and periods where the signal is ramping up or down while you flick the on/off switch). So the mean and standard deviation would vary over time because whatever is generating the signal changes over time. This is *non-stationarity*.
# 
# If the on/off periods were regularly spaced, then you could say that over longer intervals there was some predictable consistency-- so maybe some *quasi-stationarity*. In fact, many approaches to adjust for non-stationarity involve defining sliding windows that are considered to be quasi-stationary.
# 
# Real world signals, such as EEG, often have some measure of nonstationarity. In fact, with EEG, we are assuming that the statistics of the signal will change between the ictal and iterictal states (hence the need for this Kaggle competition). 
# 
# However, the $20K question is really: What statistics change between these intervals and how can we best classify those differences?
# 
# To that end, the remaining functions in this feature extrator are looking at signal statistics that may change due to non-stationary effects. Again, all we are really doing at the end of the day is measuring things about the signal that we think may tell us something about how the underlying process changes. If we are lucky (or good; or both) we can find features that reliably and significantly change between the ictal and intraictal states.

# ### Shannon Entropy
# 
# Entropy is a non-linear measure quantifying the degree of complexity in a time series. It measures how well you can predict one epoch of the time series from other epochs in the series.
# 
# The formula for Shannon Entropy is:
# 
# <center>$\large H(X)=\sum_{i=1}^nP(x_{i})I(x_{i}) = -\sum_{i=1}^nP(x_{i})log_{b}P(x_{i})$</center>
# 
# References:
# + [Paper for Shannon Entropy in EEG classification](https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2014-121.pdf)
# 
# + [Another paper on Shannon Entropy in Biomedical Signals](http://logika.uwb.edu.pl/studies/download.php?volid=56&artid=56-02&format=PDF)

# In[ ]:


'''
Computes Shannon Entropy
'''
def calcShannonEntropy(epoch, lvl, nt, nc, fs):
    
    # compute Shannon's entropy, spectral edge and correlation matrix
    # segments corresponding to frequency bands
    dspect = calcDSpect(epoch, lvl, nt, nc, fs)

    # Find the shannon's entropy
    spentropy = -1*np.sum(np.multiply(dspect,np.log(dspect)), axis=0)
    
    return spentropy


# ### Spectral Edge Frequency (SEF)
# The spectral edge frequency is the frequency in which a certain percentage of the power lies below in the signal's frequency domain. So for an SEF of 43, the function would return the frequency at which 43% of the signal's power lies below. So if the SEF of 90 returned 30, then this would indicate that 90% of the signals power was less than or equal to 30 Hz.
# 
# I think this is analogous to a quantile.

# In[ ]:


'''
Compute spectral edge frequency
'''
def calcSpectralEdgeFreq(epoch, lvl, nt, nc, fs):
    
    # Find the spectral edge frequency
    sfreq = fs
    tfreq = 40
    ppow = 0.5

    topfreq = int(round(nt/sfreq*tfreq))+1
    D = calcNormalizedFFT(epoch, lvl, nt, fs)
    A = np.cumsum(D[:topfreq,:], axis=0)
    B = A - (A.max()*ppow)    
    spedge = np.min(np.abs(B), axis=0)
    spedge = (spedge - 1)/(topfreq-1)*tfreq
    
    return spedge


# ### Cross-correlation
# 
# [Cross-correlation](https://en.wikipedia.org/wiki/Cross-correlation) is the measure of similarity between two signals. It is a number between +1 and -1. A +1 means that when signal A increases, then signal B increases and when signal A decreases, signal B decreases. A -1 means that when signal A increases, signal B decreases and vice versa. A 0 means that when signal A increases, you can't say anything about what signal B will do (i.e. no correlation). 

# In[ ]:


'''
Calculate cross-correlation matrix
'''
def corr(data, type_corr):
    
    C = np.array(data.corr(type_corr))
    C[np.isnan(C)] = 0  # Replace any NaN with 0
    C[np.isinf(C)] = 0  # Replace any Infinite values with 0
    w,v = np.linalg.eig(C)
    #print(w)
    x = np.sort(w)
    x = np.real(x)
    return x


# In[ ]:


'''
Compute correlation matrix across channels
'''
def calcCorrelationMatrixChan(epoch):
    
    # Calculate correlation matrix and its eigenvalues (b/w channels)
    data = pd.DataFrame(data=epoch)
    type_corr = 'pearson'
    
    lxchannels = corr(data, type_corr)
    
    return lxchannels


# In[ ]:


'''
Calculate correlation matrix across frequencies
'''
def calcCorrelationMatrixFreq(epoch, lvl, nt, nc, fs):
    
        # Calculate correlation matrix and its eigenvalues (b/w freq)
        dspect = calcDSpect(epoch, lvl, nt, nc, fs)
        data = pd.DataFrame(data=dspect)
        
        type_corr = 'pearson'
        
        lxfreqbands = corr(data, type_corr)
        
        return lxfreqbands


# ###  Hjorth Parameters
# 
# The [Hjorth parameters](https://en.wikipedia.org/wiki/Hjorth_parameters) of **Activity**, **Mobility**, and **Complexity** are time-domain measurements commonly use to described EEGs. They have been used for many decades to reduce the data complexity in EEG studies that try to discriminate stages of sleep. Sometimes they are referred to as *normalized slope descriptors*. So in the time domain they describe the change (or slope) of the signal and in the frequency domain they describe the mean frequency.
# 
# Another way to think of the Hjorth is by considering them as a series of successive derivatives:
# + Activity is the average power in the epoch
# + Mobility is the average power of the normalized derivative in the epoch
# + Complexity is the average power of the normalized second derivative in the epoch
# 
# [Some authors](https://pub.ist.ac.at/~schloegl/publications/Vidaurre2009tdp.pdf) suggest further normalized derivatives might be useful descriptors.

# **Hjorth Activity** is the variance in the amplitude of the signal. In the frequency domain, this is the envelope of the power spectral density (~ mean power).
# 
# <center>$Activity = \large var(y(t))$</center>

# In[ ]:


def calcActivity(epoch):
    '''
    Calculate Hjorth activity over epoch
    '''
    
    # Activity
    activity = np.nanvar(epoch, axis=0)
    
    return activity


# **Hjorth Mobility** is the mean frequency or the proportion of standard deviation of the power spectrum. (RMS frequency)
# 
# <center>$Mobility = \huge\sqrt{\frac{var(\frac{\text{d}y(t)}{\text{dt}})}{var(y(t))}}$</center>

# In[ ]:


def calcMobility(epoch):
    '''
    Calculate the Hjorth mobility parameter over epoch
    '''
      
    # Mobility
    # N.B. the sqrt of the variance is the standard deviation. So let's just get std(dy/dt) / std(y)
    mobility = np.divide(
                        np.nanstd(np.diff(epoch, axis=0)), 
                        np.nanstd(epoch, axis=0))
    
    return mobility


# **Hjorth Complexity** is the ratio of the mobility of the change in signal amplitude to the mobility of the signal itself (RMS frequency spread). In the frequency domain it represents the change in frequency (or the bandwidth). A value close to one indicates that the signal is a pure sinusoid.
# 
# <center>$Complexity = \huge{\frac{Mobility(\frac{\text{d}y}{\text{d}t}y(t))}{Mobility(y(t))}}$</center>

# In[ ]:


def calcComplexity(epoch):
    '''
    Calculate Hjorth complexity over epoch
    '''
    
    # Complexity
    complexity = np.divide(
        calcMobility(np.diff(epoch, axis=0)), 
        calcMobility(epoch))
        
    return complexity  


# ### Fractal Dimensions (FD)
# >"Clouds are not spheres, mountains are not cones, coastlines are not circles, and bark is not smooth, nor does lightning travel in a straight line." ([Mandelbrot](https://en.wikipedia.org/wiki/Benoit_Mandelbrot), 1983).
# 
# So what if our "[cloud](#cloud)" approximation isn't really a good method? What if it is too simplisitic to really approximate our complex EEG patterns with simple linear methods. Well, then we have to get into some more rigorous ways of defining shapes. Go get a strong cup of coffee. The following will completely change your perception of "reality".
# 
# [Fractal dimensions](https://en.wikipedia.org/wiki/Fractal_dimension) (FD) are a more rigorous way of looking at the "dimension" of the figure. For example, why is a point considered 0 dimensional, a line 1D, a plane 2D, and a cube 3D? It turns out harder to define mathematically than you may first think. I'll try to explain it, but ([here's a good primer](http://math.bu.edu/DYSYS/chaos-game/node6.html) if you want more details.)
# 
# Fractals, or self-similarity, is one way to formally describe a dimension. With a fractal, we take the object and see if we can divide it (or multply it) into objects that have exactly the same shape as the parent object, but are just at a different scale (or magnification).
# 
# A square of length N, for instance, can be divided into $N^2$ smaller squares. Similarly, a box can be divided into $N^3$ smaller boxes. Or, another way to say it: If I were to divide a box by N, I would get exactly $N^3$ equal boxes the same shape, but N times smaller than the original box. 
# 
# _Let's take an example:_
# 
# A cube of length 4 can be divided into $4^3 (= 64)$ unit (1x1x1 cubes)-- no less, no more. I can't get 65 cubes out of it or even 63 (unless I want the cubes to be of different sizes). Or, I could divide it by a magnification of 2 and get exactly $2^3 (=8)$ equal (smaller) cubes-- no less, no more. Or, a magnification of 3 and get-- you guessed it-- $3^3 (= 9)$ similar (smaller) cubes.
# 
# So *dimension* is actually the exponent of the ratio between the parent object and how many smaller (or larger) copies I can make.
# 
# 
# <center>$\huge\frac{log (\text{number of self-similar pieces})}{log (\text{magnification factor})}$</center>
# 
# Or, for our test cube:
# 
# <center>$\huge\frac{log(64)}{log(4)} = \frac{log(8)}{log(2)} = \frac{log(9)}{log(3)} = 3$</center>
# 
# So fractal dimension is a really rigorous way of stating the "dimensionality" of an object-- something we intuitively knew but probably never knew why.
# 
# Now down the rabbit hole: Using this defintion also includes the possibility for a *"fractional"* dimension. That is, there is mathematically something in between 1D and 2D and in between 5D and 6D. So what does a dimension of 1.36 actually mean?? Well, consider the following:  a point is considered 0D and a line is considered 1D. However, a line can be thought of as being made of an infinite number of 0D points. What if the number of points was *not* infinite. Then, the line has "gaps"-- we call that a fractional dimension. The more gaps in the line, the closer it is to a 0 dimension and, conversely, the less gaps in the line, the closer it is to 1D.
# 
# Here's a more concrete example: the [Sierpinski triangle](https://en.wikipedia.org/wiki/Sierpinski_triangle) looks like this:
# 
# ![sierpinski triangle](https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Sierpinski_triangle.svg/220px-Sierpinski_triangle.svg.png "Sierpinksi triangle = Dimension of 1.58")
# 
# So imagine I double the size of the triangle (double the length of each side). The resulting triangle has 3 self-similar pieces. Or, if I quadrupule the sides, I'll get 9 self-similar Sierpinski's. The triangle is neither 1D nor 2D. Instead, it is actually 1.58D!
# 
# <center>$\huge\frac{log(3)}{log(2)} = \frac{log(9)}{log(4)} = 1.58 \text{ dimensions!}$</center>
# 
# (_I warned you to get coffee first. I couldn't make this stuff up if I tried._)
# 
# Now compare the Sierpinski to a simple triangle. It's three simple lines. If I double its size, I can get 4 copies. So that's back to 2D. Where did the 0.42D go? Well, look at the center of the Sierpinksi triangle. It might not be obvious, but that's a blank space. So the Sierpinski is drawn in a 2D space, but only takes up 1.58D of that space. Similarly, in 3D there is the [Menger Sponge](https://en.wikipedia.org/wiki/Menger_sponge) which is 2.72D due to its holes.
# 
# Fractal dimensions are used to measure complexity in the time series. You probably thought of an EEG as a collection of 2D signals of voltage versus time -- however, that is only true for a simple line (think of our simple 2D triangle). More complex lines have transients (brief spikes) and other fluctuations that disrupt the self-similarity in a time series (think of our 1.58D Sierpinski triangle). These are like might "holes" in our dimension. So our 2D EEG might actually be somewhere in between 1D and 2D. 
# 
# Why should 1.58D makes us happy? Or, more generally, how can fractal dimensions help our analysis? Well, suppose you want to measure the similarity between two very complex signals. You could do the standard cross-correlation. That will tell you if signal A increases what signal B will do. However, FD speaks to the complexity of the shapes of A and B. What if signal A was 1.34D and signal B was 1.89D. Are they alike? Different? Or, conversely, if signal A and B are both 1.734D, doesn't that gives us a little more information about how similar they really are? You can't rely on fractal dimension alone. Two completely different figures could have the same FD. However, it's unlikely that two similar signals would have different FD. So it is a necessary, but not sufficient condition. One more piece of evidence in our analysis.
# 
# Fractal dimensions have been used to analyze non-linear and non-stationary biomedical signals such as EEG and EKG. The idea is to look at the time series as if it were a geometric figure. So instead of calculating the mean or standard deviation of the signal, we are now calculating non-linear measures such as, how many times the signal changes direction and what the slope of the signal is at smaller time intervals. Of course, this makes it prone to error when there is noise in the data. The shape can be altered by random fluctuations in the signal. Many FD algorithms try to account for the noise.
# 
# There are about 7 billion methods to determine the "fractal dimension" of a time series-- all giving slightly different results. Most methods use the same basic approach-- divide the time series into different windows (and different scales) and measure the amount of similarity between windows. Nevertheless, this all depends on how you choose to measure "similarity". Remember, we are still trying to fit known shapes to unknown datasets. We're now using very complex shapes to fit our clouds. However, these complex shapes still have very precise repetitions that everyone can identify. But our real world signals are even more complex-- how much similarity between epochs of the signal indicate a true pattern? What is noise and what is a true part of the signal? So which method best calculates the Fractal Dimension is not a well solved issue. Here are a few ways people have suggested:
# 
# + [Hurst fractal dimension](https://en.wikipedia.org/wiki/Hurst_exponent)
# + Petrosian fractal dimension
# + [Katz fractal dimension](https://www.seas.upenn.edu/~littlab/Site/Publications_files/Esteller_2001.pdf)
# + [Higuchi fractal dimension](https://www.ncbi.nlm.nih.gov/pubmed/18676171)
# + [Detrended fluctuation analysis](https://en.wikipedia.org/wiki/Detrended_fluctuation_analysis) (an extension of Hurst)
# 
# All of these methods seem to be useful in classifying states of EEG ("waking", "imagining movement") etc. I've provided some references below to peruse.
# 
# References: 
# + https://www.seas.upenn.edu/~littlab/Site/Publications_files/Esteller_2001.pdf
# + https://hal.archives-ouvertes.fr/inria-00442374/document
# + http://tux.uis.edu.co/geofractales/articulosinteres/PDF/comparation.pdf - In this paper, they compared Katz, Higuchi, and Petrosian with synthetically generated signals. Higuchi was the best at estimating the true HD. Katz performs better than the other two in the presence on signficant noise. Petrosian is the fastest method, but as the time series increases aobve 4000 datapoints, all 3 methods are similarly fast.

# #### Hjorth Fractal Dimension
# 
# Hjorth FD is essentially re-sampling (or decimating) the time series and checking if we have similar curves. The similarity measure is a function of the mean value for each version of the series. 
# 
# $K_{max}$ is the scale size (or time offset). So the algorithm compares $K_{max}$ versions of the epoch for similarity. The first version skips every other datapoint, the second skips every 2 datapoints, and the $K_{max}$ skips $K_{max}$ datapoints at a time. 
# 
# Others have set this $K_{max}$ to 3. Not sure of any significance other than it is less computations.

# In[ ]:


def hjorthFD(X, Kmax=3):
    """ Compute Hjorth Fractal Dimension of a time series X, kmax
     is an HFD parameter. Kmax is basically the scale size or time offset.
     So you are going to create Kmax versions of your time series.
     The K-th series is every K-th time of the original series.
     This code was taken from pyEEG, 0.02 r1: http://pyeeg.sourceforge.net/
    """
    L = []
    x = []
    N = len(X)
    for k in range(1,Kmax):
        Lk = []
        
        for m in range(k):
            Lmk = 0
            for i in range(1,floor((N-m)/k)):
                Lmk += np.abs(X[m+i*k] - X[m+i*k-k])
                
            Lmk = Lmk*(N - 1)/floor((N - m) / k) / k
            Lk.append(Lmk)
            
        L.append(np.log(np.nanmean(Lk)))   # Using the mean value in this window to compare similarity to other windows
        x.append([np.log(float(1) / k), 1])

    (p, r1, r2, s)= np.linalg.lstsq(x, L)  # Numpy least squares solution
    
    return p[0]


# #### Petrosian fractal dimension
# 
# Petrosian's algorithm translates the series into a binary sequence (a vector of +1 and -1 values). 
# 
# <center>$\huge\frac{log_{10}(n)}{log_{10}(n) + log_{10}(\frac{n}{n + 0.4N_\Delta})}$</center>
# 
# To create the sequence, consecutive samples in the series are subtracted to get the derviative. A *+1* or *-1* is assigned for every positive or negative result respectively. So it is looking at changes in the slope at each time point in the epoch. It then counts how many times the slope changes (so **+++---++---+** would be 4  changes in slope).
# 
# $n$ is the length of the sequence and $N_\Delta$ is the number of sign changes within the sequence.

# In[ ]:


def petrosianFD(X, D=None):
    """Compute Petrosian Fractal Dimension of a time series from either two 
    cases below:
        1. X, the time series of type list (default)
        2. D, the first order differential sequence of X (if D is provided, 
           recommended to speed up)

    In case 1, D is computed by first_order_diff(X) function of pyeeg

    To speed up, it is recommended to compute D before calling this function 
    because D may also be used by other functions whereas computing it here 
    again will slow down.
    
    This code was taken from pyEEG, 0.02 r1: http://pyeeg.sourceforge.net/
    """
    
    # If D has been previously calculated, then it can be passed in here
    #  otherwise, calculate it.
    if D is None:   ## Xin Liu
        D = np.diff(X)   # Difference between one data point and the next
        
    # The old code is a little easier to follow
    N_delta= 0; #number of sign changes in derivative of the signal
    for i in range(1,len(D)):
        if D[i]*D[i-1]<0:
            N_delta += 1

    n = len(X)
    
    # This code is a little more compact. It gives the same
    # result, but I found that it was actually SLOWER than the for loop
    #N_delta = sum(np.diff(D > 0)) 
    
    return np.log10(n)/(np.log10(n)+np.log10(n/n+0.4*N_delta))


# #### Katz fractal dimension
# 
# Katz doesn't divide the time series. Instead it uses the entire epoch and measures the distance between the first data point and the data point that is furthest away. So it's the extent of the data from the first point divided by the number of data points.
# 
# <center>$\huge\frac{log(L)}{log(d)}$, where $L$ = extent of first point to furthest point, $d$ = length of epoch</center>
# 
# Despite being so simple, there is a [paper](https://www.seas.upenn.edu/~littlab/Site/Publications_files/Esteller_2001.pdf) that suggests "Katz’s algorithm is the most consistent method for **discrimination of epileptic states** from the IEEG, likely due to its exponential transformation of FD values and relative insensitivity to noise".

# In[ ]:


def katzFD(epoch):
    ''' 
    Katz fractal dimension 
    '''
    
    L = np.abs(epoch - epoch[0]).max()
    d = len(epoch)
    
    return (np.log(L)/np.log(d))


# #### Hurst Exponent
# >Taken from Markus' [StackOverflow post](http://stackoverflow.com/questions/34506130/daily-hurst-exponent). There is also a Python library called [nolds](https://pypi.python.org/pypi/nolds/0.1.1) which has several of these more advanced descriptors coded.
# 
# [Hurst](https://en.wikipedia.org/wiki/Hurst_exponent) is a measure of long-term memory in a time series that is *not* due to periodicity. It's often used as a measure how how stationary a time series is. Financial analysts use it to determine if there are truly long-term trends in data. (*Is the stock decreasing in value or is this the normal, random fluctuations in stock price?*)
# 
# The exponent is the rate at which the autocorrelation in a time series decreases as the lags increase. If the series $x(t)$ is a self-similar fractal, then $x(bt)$ is statistically equivalent to $b^Hx(t)$, where $H$ is the Hurst exponent. So if we were to take every other datapoint ($b=2$) then the mean of that would be the mean of the original signal multiplied by $2^H$. (Seems analogous to an Eigen?)
# 
# Hurst first used his exponent to describe how the Nile river's size fluctuated over long periods of time. (He was a hydrologist consulting on the building of a dam.) The width of the Nile changes with time, but Hurst tried to answer: "Are these changes (a) bouncing around a mean size or (b) steadily increasing (or decreasing)?" That's a critical point if you want the dam to be useful over decades (and probably centuries) of change.
# 
# The exponent is directly related to the *Fractal Dimension* of the time series and is basically an objective measure of whether the randomness in a time series is "mild" or "wild". In other words, how random is the randomness? It is usually one of 3 cases: mean reverting, [random walking](https://en.wikipedia.org/wiki/Random_walk), or trending.
# 
# > A *random walk* is often compared with the walk of a drunk person. With each step there is a 50-50 chance of the drunk moving foward/left or forward/right. In the short term, the walk looks "random", but over the long term the drunk stays along the mean path because of the 50-50 process.
# 
# H is a number between 0 and 1. Meanings for H exponents:
# + $H = 0.5 :$ Geometric random walk (Brownian motion), no correlation
# + $H < 0.5 :$ Mean-reverting series or antipersistent (it's a random walk with smaller than normal steps. Therefore, an increase in step size is likely followed by a decrease in step size (or vice-versa))
# + $H > 0.5 :$ Trending Series - The random walk seems to be persistently going in a direction. So a large step size is likely  followed by a large step size.

# In[ ]:


def hurstFD(epoch):

    """Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    tau = [np.sqrt(np.nanstd(np.subtract(epoch[lag:], epoch[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0


# ### Detrended Fluctuation Analysis (DFA)
# > This code is taken directly from the [nolds](https://pypi.python.org/pypi/nolds/0.1.1) Python package.
# 
# > Here's a good [primer](https://www.physionet.org/tutorials/fmnc/node5.html) on DFA
# 
# Performs a [detrended fluctuation analysis (DFA)](https://en.wikipedia.org/wiki/Detrended_fluctuation_analysis) on the given data.
# 
# Recommendations for parameter settings by Hardstone *et al.*:
# * nvals should be equally spaced on a logarithmic scale so that each window scale has the same weight
# * min(nvals) < 4 does not make much sense as fitting a polynomial (even if it is only of order 1) to 3 or less data points is very prone.
# * max(nvals) > len(data) / 10 does not make much sense as we will then have less than 10 windows to calculate the average fluctuation 
# * use overlap=True to obtain more windows and therefore better statistics (at an increased computational cost)
# 
# Explanation of DFA:
# Detrended fluctuation analysis, much like the Hurst exponent, is used to find long-term statistical dependencies in time series.
# 
# The idea behind DFA originates from the definition of self-affine processes. A process X is said to be self-affine if the standard deviation of the values within a window of length n changes with the window length factor L in a power law:
# 
# $\sigma(X,L \times n) = L^H \times \sigma(X, n)$
# 
# where $\sigma(X, k)$ is the standard deviation of the process X calculated over windows of size k. In this equation, H is called the **Hurst parameter**, which behaves indeed very similar to the **Hurst exponent**.
# 
# Like the Hurst exponent, H can be obtained from a time series by calculating $\sigma(X,n)$ for different n and fitting a straight line to the plot of $\log(\sigma(X,n))$ versus $\log(n)$.
# 
# To calculate a single $\sigma(X,n)$, the time series is split into windows of equal length n, so that the ith window of this size has the form
# 
# $W_{(n,i)} = [x_i, x_{(i+1)}, x_{(i+2)}, ... x_{(i+n-1)}]$
# 
# The value $\sigma(X,n)$ is then obtained by calculating $\sigma(W_{(n,i)})$ for each i and averaging the obtained values over i.
# 
# The aforementioned definition of self-affinity, however, assumes that the process is  non-stationary (i.e. that the standard deviation changes over time) and it is highly influenced by local and global trends of the time series.
# 
# To overcome these problems, an estimate alpha of H is calculated by using a "walk" or "signal profile" instead of the raw time series. This walk is obtained by substracting the mean and then taking the cumulative sum of the original time series. The local trends are removed for each window separately by fitting a polynomial $p_{(n,i)}$ to the window $W_{(n,i)}$ and then calculating $W'_{(n,i)} = W_{(n,i)} - p_{(n,i)}$ (element-wise substraction).
# 
# We then calculate std(X,n) as before only using the "detrended" window $W'_{(n,i)}$ instead of $W_{(n,i)}$. Instead of H we obtain the parameter $\alpha$ from the line fitting.
# 
# + $\alpha < 1/2$ : "memory" with negative correlation
# + $\alpha \cong 0.5$ : we have no correlation or "memory", white noise
# + $1/2 < \alpha < 1$ : "memory" with positive correlation
# + $\alpha \cong 1$ : $1/f$ noise, pink noise  
# + $\alpha > 1$ : the underlying process is non-stationary and is unbounded
# + $\alpha \cong 3/2$ : Brownian noise
# 
# References:
# 1. C.-K. Peng, S. V. Buldyrev, S. Havlin, M. Simons, H. E. Stanley, and 
#     A. L. Goldberger, “Mosaic organization of DNA nucleotides,” Physical 
#     Review E, vol. 49, no. 2, 1994.
# 2. R. Hardstone, S.-S. Poil, G. Schiavone, R. Jansen, V. V. Nikulin, 
#     H. D. Mansvelder, and K. Linkenkaer-Hansen, “Detrended fluctuation 
#     analysis: A scale-free view on neuronal oscillations,” Frontiers in 
#     Physiology, vol. 30, 2012.
# 
# Reference code:
# a. Peter Jurica, "Introduction to MDFA in Python",
#       url: http://bsp.brain.riken.jp/~juricap/mdfa/mdfaintro.html
# b. JE Mietus, "dfa",
#       url: https://www.physionet.org/physiotools/dfa/dfa-1.htm
# c. "DFA" function in R package "fractal"
# 
# **Args:**
# data (array of float): time series
# 
# **Kwargs:**
# 
# | Name   | Type     | Description |
# | ---------- | -----    |----------- |
# | nvals      | integer  | subseries sizes at which to calculate fluctuation (default: logarithmic_n(4, 0.1xlen(data), 1.2)) |
# | overlap    | boolean  | if True, the windows $W_{(n,i)}$ will have a 50% overlap, otherwise non-overlapping windows will be used |
# | order | integer       | (polynomial) order of trend to remove |
# | debug_plot | boolean  | if True, a simple plot of the final line-fitting step will be shown |
# | plot_file | string    | if debug_plot is True and plot_file is not None, the plot will be saved under the given file name instead of directly showing it through plt.show() |
#                  
# **Returns:**
# float: the estimate alpha for the Hurst parameter
# + $\alpha$ < 1: stationary process similar to fractional Gaussian noise with $H = \alpha$ 
# + $\alpha$ > 1: non-stationary process similar to fractional brownian motion with $H = \alpha - 1$

# In[ ]:


def logarithmic_n(min_n, max_n, factor):
    """
    Creates a list of values by successively multiplying a minimum value min_n by
    a factor > 1 until a maximum value max_n is reached.

    Non-integer results are rounded down.

    Args:
    min_n (float): minimum value (must be < max_n)
    max_n (float): maximum value (must be > min_n)
    factor (float): factor used to increase min_n (must be > 1)

    Returns:
    list of integers: min_n, min_n * factor, min_n * factor^2, ... min_n * factor^i < max_n
                      without duplicates
    """
    assert max_n > min_n
    assert factor > 1
    
    # stop condition: min * f^x = max
    # => f^x = max/min
    # => x = log(max/min) / log(f)
    
    max_i = int(np.floor(np.log(1.0 * max_n / min_n) / np.log(factor)))
    ns = [min_n]
    
    for i in range(max_i+1):
        n = int(np.floor(min_n * (factor ** i)))
        if n > ns[-1]:
            ns.append(n)
            
    return ns


# In[ ]:


def dfa(data, nvals= None, overlap=True, order=1, debug_plot=False, plot_file=None):

    total_N = len(data)
    if nvals is None:
        nvals = logarithmic_n(4, 0.1*total_N, 1.2)
        
    # create the signal profile (cumulative sum of deviations from the mean => "walk")
    walk = np.nancumsum(data - np.nanmean(data))
    fluctuations = []
    
    for n in nvals:
        # subdivide data into chunks of size n
        if overlap:
            # step size n/2 instead of n
            d = np.array([walk[i:i+n] for i in range(0,len(walk)-n,n//2)])
        else:
            # non-overlapping windows => we can simply do a reshape
            d = walk[:total_N-(total_N % n)]
            d = d.reshape((total_N//n, n))
            
        # calculate local trends as polynomes
        x = np.arange(n)
        tpoly = np.array([np.polyfit(x, d[i], order) for i in range(len(d))])
        trend = np.array([np.polyval(tpoly[i], x) for i in range(len(d))])
        
        # calculate standard deviation ("fluctuation") of walks in d around trend
        flucs = np.sqrt(np.nansum((d - trend) ** 2, axis=1) / n)
        
        # calculate mean fluctuation over all subsequences
        f_n = np.nansum(flucs) / len(flucs)
        fluctuations.append(f_n)
        
        
    fluctuations = np.array(fluctuations)
    # filter zeros from fluctuations
    nonzero = np.where(fluctuations != 0)
    nvals = np.array(nvals)[nonzero]
    fluctuations = fluctuations[nonzero]
    if len(fluctuations) == 0:
        # all fluctuations are zero => we cannot fit a line
        poly = [np.nan, np.nan]
    else:
        poly = np.polyfit(np.log(nvals), np.log(fluctuations), 1)
    if debug_plot:
        plot_reg(np.log(nvals), np.log(fluctuations), poly, "log(n)", "std(X,n)", fname=plot_file)
        
    return poly[0]


# ### [Higuchi Fractal Dimension](http://tswww.ism.ac.jp/higuchi/index_e/papers/PhysicaD-1988.pdf)
# 
# Higuchi is similar to Hjorth in that we resample the time series $x(t)$ into $k$ different versions. 
# So using $x(t)$ an ensemble of time series versions $x_m^k$ are defined as:
# 
# <center>$\large x_m^k = \left \{ x(m), x(m+k), ..., x(m + \left \lfloor \frac{N-m}{k} \right \rfloor k \right \}$</center>
# 
# <center>for $m=1,2,...,k$</center>
# 
# So, for example, let's consider a 100-point time series $x(t)$. For $k=4$ and ($N=100$), we get four time series to compare:
# 
# >  $x_4^1 : \left \{ x(1), x(5), x(9),   ..., x(97) \right \}$
# 
# >  $x_4^2 : \left \{ x(2), x(6), x(10), ..., x(98) \right \}$
# 
# >  $x_4^3 : \left \{ x(3), x(7), x(11), ..., x(99) \right \}$
# 
# >  $x_4^4 : \left \{ x(4), x(8), x(12), ..., x(100) \right \}$
# 
# Based on these ensembles, we subtract successive values and then sum over the ensemble:
# 
# <center>$\large L_m(k) = \frac{1}{k} \left ( {\sum_{i=1}^{\left \lfloor \frac{N-m}{k} \right \rfloor} \left | x(m+ik) - x(m+(i-1)k)) \right | } \right ) \left ( {\frac{N-1}{\left \lfloor \frac{N-m}{k} \right \rfloor k}} \right )$</center>
# 
# where:
# + $N$ is the length of the time series $x(t)$
# + $\left \lfloor \frac{N-m}{k} \right \rfloor$ is the floor of the ratio (so `floor((N-m)/k)`)
# 
# **QUESTION: Isn't subtracting successive values just the derivative? Then summing will be the integral of the derivative. So the dervative and then integral is the same as just subtracting the mean from the time series?? Or am I missing something? Why not do that instead?**
# 
# After getting $L_m(k)$ we compute the average $L_m$:
# 
# <center>$\large \langle L(k) \rangle = \frac{1}{k} \sum_{m=1}^k L_m(k)$</center>
# 
# The procedure is repeated for all k from 1 to $K_{max}$. 
# 
# Higuchi showed that this was the time-domain equivalent of the power spectrum ([PSD](https://en.wikipedia.org/wiki/Spectral_density)). For self-similar processes, the power spectrum $P(f)$ has an interesting property:
# 
# <center>$P(f) \propto f^{-\alpha}$</center>
# 
# where $f$ is the frequency and $\alpha$ is the fractal dimension (FD). Therefore, if we plot the power spectrum versus the frequency on a log-log axis, there should be a line with slope of $-\alpha$. If the exponent is 0, then the power spectrum is independent of frequency (white noise). if the exponent is 1, then there is moderate correlation. If the exponent is 2, then there's a strong [(Brownian) correlation](https://en.wikipedia.org/wiki/Distance_correlation).
# 
# Higuchi said that in the time domain, if we plot $\langle L(k) \rangle$ versus $\frac{1}{k}$ on a log-log plot, then the slope is also $\alpha$. So $\alpha$ is Higuchi's exponent (or FD).
# 
# <center>$ \langle L(k) \rangle \propto {(\frac{1}{k})}^{FD}$</center>
# 
# The [Higuchi exponent](http://iopscience.iop.org/article/10.1088/1742-6596/475/1/012002/pdf) is considered to be a measure of the *irregularity* of a time series.

# In[ ]:


def higuchiFD(epoch, Kmax = 8):
    '''
    Ported from https://www.mathworks.com/matlabcentral/fileexchange/30119-complete-higuchi-fractal-dimension-algorithm/content/hfd.m
    by Salai Selvam V
    '''
    
    N = len(epoch)
    
    Lmk = np.zeros((Kmax,Kmax))
    
    #TODO: I think we can use the Katz code to refactor resampling the series
    for k in range(1, Kmax+1):
        
        for m in range(1, k+1):
               
            Lmki = 0
            
            maxI = floor((N-m)/k)
            
            for i in range(1,maxI+1):
                Lmki = Lmki + np.abs(epoch[m+i*k-1]-epoch[m+(i-1)*k-1])
             
            normFactor = (N-1)/(maxI*k)
            Lmk[m-1,k-1] = normFactor * Lmki
    
    Lk = np.zeros((Kmax, 1))
    
    #TODO: This is just a mean. Let's use np.mean instead?
    for k in range(1, Kmax+1):
        Lk[k-1,0] = np.nansum(Lmk[range(k),k-1])/k/k

    lnLk = np.log(Lk) 
    lnk = np.log(np.divide(1., range(1, Kmax+1)))
    
    fit = np.polyfit(lnk,lnLk,1)  # Fit a line to the curve
     
    return fit[0]   # Grab the slope. It is the Higuchi FD


# In[ ]:


def calcFractalDimension(epoch):
    
    '''
    Calculate fractal dimension
    '''
    
    # Fractal dimensions
    [nt, no_channels] = epoch.shape
    fd = []
     
    for j in range(no_channels):
        
        fd.append( [petrosianFD(epoch[:,j])      # Petrosan fractal dimension
                    , hjorthFD(epoch[:,j],3)     # Hjorth exponent
                    , hurstFD(epoch[:,j])        # Hurst fractal dimension
                    , katzFD(epoch[:,j])         # Katz fractal dimension
                    , higuchiFD(epoch[:,j])      # Higuchi fractal dimension
                   #, dfa(epoch[:,j])    # Detrended Fluctuation Analysis - This takes a long time!
                   ] )
    
    return pd.DataFrame(fd, columns=['Petrosian FD', 'Hjorth FD', 'Hurst FD', 'Katz FD', 'Higuichi FD'])
    #return pd.DataFrame(fd, columns=['Petrosian FD', 'Hjorth FD', 'Hurst FD', 'Katz FD', 'Higuichi FD', 'DFA'])


# In[ ]:


def calcPetrosianFD(epoch):
    
    '''
    Calculate Petrosian fractal dimension
    '''
    
    # Fractal dimensions
    [nt, no_channels] = epoch.shape
    fd = []
     
    for j in range(no_channels):
        
        fd.append(petrosianFD(epoch[:,j]))    # Petrosan fractal dimension
                   
    
    return fd


# In[ ]:


def calcHjorthFD(epoch):
    
    '''
    Calculate Hjorth fractal dimension
    '''
    
    # Fractal dimensions
    [nt, no_channels] = epoch.shape
    fd = []
     
    for j in range(no_channels):
        
        fd.append(hjorthFD(epoch[:,j],3))     # Hjorth exponent
                   
    
    return fd


# In[ ]:


def calcHurstFD(epoch):
    
    '''
    Calculate Hurst fractal dimension
    '''
    
    # Fractal dimensions
    [nt, no_channels] = epoch.shape
    fd = []
     
    for j in range(no_channels):
        
        fd.append(hurstFD(epoch[:,j]))       # Hurst fractal dimension
                   
    
    return fd


# In[ ]:


def calcHiguchiFD(epoch):
    
    '''
    Calculate Higuchi fractal dimension
    '''
    
    # Fractal dimensions
    [nt, no_channels] = epoch.shape
    fd = []
     
    for j in range(no_channels):
        
        fd.append(higuchiFD(epoch[:,j]))      # Higuchi fractal dimension
                   
    
    return fd


# In[ ]:


def calcKatzFD(epoch):
    
    '''
    Calculate Katz fractal dimension
    '''
    
    # Fractal dimensions
    [nt, no_channels] = epoch.shape
    fd = []
     
    for j in range(no_channels):
        
        fd.append(katzFD(epoch[:,j]))      # Katz fractal dimension
                   
    
    return fd


# In[ ]:


def calcDFA(epoch):
    
    '''
    Calculate Detrended Fluctuation Analysis
    '''
    
    # Fractal dimensions
    [nt, no_channels] = epoch.shape
    fd = []
     
    for j in range(no_channels):
        
        fd.append(dfa(epoch[:,j]))      # DFA
                   
    
    return fd


# ### Other geometric parameters of signal shape

# #### Skewness
# 
# [Skewness](https://en.wikipedia.org/wiki/Kurtosis) measures how asymmetric a probability density function is around its mean value.
# 
# ![skewness](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Negative_and_positive_skew_diagrams_%28English%29.svg/446px-Negative_and_positive_skew_diagrams_%28English%29.svg.png "Skewness: Left or right shift around mean")

# In[ ]:


def calcSkewness(epoch):
    '''
    Calculate skewness
    '''
    # Statistical properties
    # Skewness
    sk = skew(epoch)
        
    return sk


# #### Kurtosis
# [Kurtosis](https://en.wikipedia.org/wiki/Kurtosis) is how long the tail is of a probability density function. So longer tails mean more kurtosis. Short, "tight" distrubtions centered around the mean have little kurtosis.
# 
# ![kurtosis](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Standard_symmetric_pdfs.png/300px-Standard_symmetric_pdfs.png "Kurtosis: Tails get longer")

# In[ ]:


def calcKurtosis(epoch):
    
    '''
    Calculate kurtosis
    '''
    # Kurtosis
    kurt = kurtosis(epoch)
    
    return kurt


# ### Dyadic analysis
# 
# (Thanks to [Jason McNeill](https://www.kaggle.com/txtrouble) for helping with this section)
# 
# We're also calculating the FFT using dyadic frequency bands.  I think this is similar to a [Wavelet](http://www.wavelet.org/tutorial/wbasic.htm) approach whereby you are looking for frequency bands of successive scales (in reality they are "scales" of the mother wavelet). So Fourier is approximating the signal using a combination of sine waves whereas Wavelet is approximating the signal using a combination of non-stationary basis functions. With FT you get a 2D plot (frequency versus amplitude). With Wavelets you are going for a 3D plot (time versus scale versus amplitude).
# 
# ![sine verus wavelet](http://www.wavelet.org/tutorial/gifs/sine.gif "Sine wave versus Wavelet")
# 
# Remember, the problem with the Fourier Transform is that we lose temporal information when we go to the frequency domain. So it only works well if the signal is stationary. Wavelets (or more generally time-frequency domain methods) can localize to both frequency and time. They give a more robust snapshot of how the signal component frequencies evolve over time. For example, in the figures below, the x axis is time, the y axis is scale (of the wavelet), and the greyscale value is the amplitude.
# 
# ![wavelet versus dyad](http://www.wavelet.org/tutorial/gifs/dwtcwt.gif "Discrete versus Continuous Wavelet Transform of Chirp")
# 
# Essentially, dyads are a "poor man's wavelet analysis" (compare how coarse the DWT is compared to the CWT in the figure above.) I think typically they are defined as scales of 2 (hence dyad). So the second scale is twice the time length of the first and the third is quadruple the first. This lends nicely to computer analysis (which is base-2 math) and even works well with acoustical analysis (which uses octaves for musical notes-- i.e. base-8). Since the power of 2 scale is the smallest regular scale possible, it should be our best hope at tracking changes over time.

# In[ ]:


'''
Computes Shannon Entropy for the Dyads
'''

def calcShannonEntropyDyad(epoch, lvl, nt, nc, fs):
    
    dspect = calcDSpectDyad(epoch, lvl, nt, nc, fs)
                           
    # Find the Shannon's entropy
    spentropyDyd = -1*np.sum(np.multiply(dspect,np.log(dspect)), axis=0)
        
    return spentropyDyd


# In[ ]:


def calcDSpectDyad(epoch, lvl, nt, nc, fs):
    
    # Spectral entropy for dyadic bands
    # Find number of dyadic levels
    ldat = int(floor(nt/2.0))
    no_levels = int(floor(log(ldat,2.0)))
    seg = floor(ldat/pow(2.0, no_levels-1))

    D = calcNormalizedFFT(epoch, lvl, nt, fs)
    
    # Find the power spectrum at each dyadic level
    dspect = np.zeros((no_levels,nc))
    for j in range(no_levels-1,-1,-1):
        dspect[j,:] = 2*np.sum(D[int(floor(ldat/2.0))+1:ldat,:], axis=0)
        ldat = int(floor(ldat/2.0))

    return dspect


# In[ ]:


def calcXCorrChannelsDyad(epoch, lvl, nt, nc, fs):
    
    dspect = calcDSpectDyad(epoch, lvl, nt, nc, fs)
    
    # Find correlation between channels
    data = pd.DataFrame(data=dspect)
    type_corr = 'pearson'
    lxchannelsDyd = corr(data, type_corr)
    
    return lxchannelsDyd


# In[ ]:


def removeDropoutsFromEpoch(epoch):
    
    '''
    Return only the non-zero values for the epoch.
    It's a big assumption, but in general 0 should be a very unlikely value for the EEG.
    '''
    return epoch[np.nonzero(epoch)]


# ### Putting it all together to get the Feature Space
# 
# So now we use all of the above functions to transform our EEG signals into a matrix of numbers that represent the various measurements of the EEG signal for the epoch provided and hope that some of these have some predictive power on the question at hand.
# 
# #### What's an Epoch?
# 
# The dataset contains 1 hour sequences of 10 minute intervals. So each data run is 10 minutes long. We've chosen to further divide the 10 minute data run

# In[ ]:


def calculate_features(file_name):
    
    f = convertMatToDictionary(file_name)
    if (f == None):
        return
    
    fs = f['iEEGsamplingRate'][0,0]
    eegData = f['data']
    [nt, nc] = eegData.shape
    print('EEG shape = ({} timepoints, {} channels)'.format(nt, nc))
    
    lvl = defineEEGFreqs()
    
    subsampLen = floor(fs * 60)  # Grabbing 60-second epochs from within the time series
    numSamps = int(floor(nt / subsampLen));      # Num of 1-min samples
    sampIdx = range(0,(numSamps+1)*subsampLen,subsampLen)
     
    functions = { 'shannon entropy': 'calcShannonEntropy(epoch, lvl, nt, nc, fs)'
                 , 'spectral edge frequency': 'calcSpectralEdgeFreq(epoch, lvl, nt, nc, fs)'
                 , 'correlation matrix (channel)' : 'calcCorrelationMatrixChan(epoch)'
                 , 'correlation matrix (frequency)' : 'calcCorrelationMatrixFreq(epoch, lvl, nt, nc, fs)'
                 , 'shannon entropy (dyad)' : 'calcShannonEntropyDyad(epoch, lvl, nt, nc, fs)'
                 , 'crosscorrelation (dyad)' : 'calcXCorrChannelsDyad(epoch, lvl, nt, nc, fs)'
                 , 'hjorth activity' : 'calcActivity(epoch)'
                 , 'hjorth mobility' : 'calcMobility(epoch)'
                 , 'hjorth complexity' : 'calcComplexity(epoch)'
                 , 'skewness' : 'calcSkewness(epoch)'
                 , 'kurtosis' : 'calcKurtosis(epoch)'
                 , 'Petrosian FD' : 'calcPetrosianFD(epoch)'
                 , 'Hjorth FD' : 'calcHjorthFD(epoch)'
                 , 'Katz FD' : 'calcKatzFD(epoch)'
                 , 'Higuchi FD' : 'calcHiguchiFD(epoch)'
                # , 'Detrended Fluctuation Analysis' : 'calcDFA(epoch)'  # DFA takes a long time!
                 }
    
    # Initialize a dictionary of pandas dataframes with the features as keys
    feat = {key[0]: pd.DataFrame() for key in functions.items()}  

    for i in range(1, numSamps+1):
    
        print('processing file {} epoch {}'.format(file_name,i))
        epoch = eegData[sampIdx[i-1]:sampIdx[i], :]  
   
        for key in functions.items():
            feat[key[0]] = feat[key[0]].append(pd.DataFrame(eval(key[1])).T)
             
    # This is kludge but it gets the correct time to seizure to the rows
    for key in functions.items():
        # The sequence is 1,2,3,4,5,6
        # Sequence represents the ten minute intervals in the hour before the seizure
        # (Technically it is from 65 minutes to 5 minutes before the seizure)
        # We've subdivided each of those six 10-minute intervals into ten 1-minute intervals
        # So we'd like to have the 
        feat[key[0]]['Minutes to Seizure'] = np.subtract(range(numSamps), 70-10*f['sequence'][0][0] + 5)
        feat[key[0]] = feat[key[0]].set_index('Minutes to Seizure')
        #feat[key[0]]['Epoch #'] = range(numSamps)
        #feat[key[0]] = feat[key[0]].set_index('Epoch #')
    
    return feat


# ### Preprocessing the features
# 
# Here we do things like identifying bad data and normalizing the data.

# In[ ]:


def replaceZeroRuns(df):
    '''
    Replace runs of 0s within the pandas dataframe with the NaN and then
    replace the NaN with the mean value
    '''
    return (df.replace(0, np.nan).fillna())


# ### Normalize the features
# 
# Features are always normalized (typically, subtract the mean and divide by the standard deviation or subtract the minimum and divide by the range). Otherwise, some features will dominate the algorithms simply because their scale is so large. You should pass a dataframe with the features and epochs for a given channel.
# 
# `df = normalizePanel(featPanel.ix[:, :, 0]) # Passing channel 0`

# In[ ]:


from sklearn import preprocessing

def normalizeFeatures(df):
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(df)
    df_normalized = pd.DataFrame(x_scaled, columns=df.columns)
    
    return df_normalized

def normalizePanel(pf):
    
    pf2 = {}
    for i in range(pf.shape[2]):
        pf2[i] = normalizeFeatures(pf.ix[:,:,i])
        
    return pd.Panel(pf2)


# In[ ]:


DATA_FILE = openfile_dialog()
feat = calculate_features(DATA_FILE)


# ### How did you do?
# 
# The point of this exercise is to reduce the vast amount of EEG data we have into a smaller dataset that we think more succinctly describes the data. So originally we had lots of datapoints (look at 'EEG shape' in the cell above-- it's 240,000 x 16. That's 16 electrode channels at 400 samples per second over 10 minutes (or 600 seconds) each. So our .MAT files contain easily over 3 million data points each. It will take a long time to process that much raw data.
# 
# With the help of our feature extractor, we've reduced it to the number of datapoints below. The hope is that-- if we've chosen the features wisely-- that we have a smaller amount of data that accurately describes the original data. Therefore, we have--once again-- tried to turn a cloud into something more easier to describe.

# ### Normalize all of the Pandas data panel
# 
# Our features should be "normalized" before we use them in our modeling. This means we scale the data so that it goes between 0 and 1. There are several ways to normalize. We'll use [sklearn's MinMax Scaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) which follows this formula:
# 
# <center>${\displaystyle x'={\frac {x-{\text{min}}(x)}{{\text{max}}(x)-{\text{min}}(x)}}}$</center>
# 
# Normalization is also called [feature scaling](https://en.wikipedia.org/wiki/Feature_scaling). Scaling is important because many of our data modeling algorithms use Euclidean distance (i.e. the vector magnitude or `norm`) in their calculations. If one feature is much larger than the others, then the Euclidean distance will be distorted in favor of that classifier. So scaling ensures that all features are equally-weighted in our model.

# In[ ]:


featPanel = normalizePanel(pd.Panel(feat))
print('Total # of datapoints reduced from {:,} to {:,}'.format(10*60*400*16, # mins * sec/min * samples/sec * channels
                                                           featPanel.shape[0]*featPanel.shape[1]*featPanel.shape[2]))

featPanel


# ### Pandas Panels
# 
# Pandas (aka *pan(el)-da(ta)-s*) was originally created to use [Panels](http://pandas.pydata.org/pandas-docs/stable/dsintro.html#panel) of economic data for Wall Street data scientists. Panels are 3D data structures-- a series of 2D dataframes. 
# 
# For the features, we've organized the data into a Panel of three dimensions:
# 1. Channel # (0 to 11)
# 2. Epoch of time in 10 minute interval (0 to 9)
# 3. Feature (Higuchi FD to spectral edge frequency)
# 
# This just seems like a convenient container to mangle our data. We can now slice our data with Pandas in any way we want. Conveniently, libraries like matplotlib, numpy, and sklearn know how to handle our sliced dataframes. 

# In[ ]:


chanNum = 6   # Electrode channel to select
featPanel.ix[chanNum,:,:].plot(subplots=True, figsize=(14,40), sharex=True, style='.-');
plt.suptitle('Normalized features for channel #{}'.format(chanNum+1));
plt.xlabel('Epoch #')
plt.subplots_adjust(top=0.96);


# ### Let's see what the average channel looks like feature-wise
# 
# So we'll just calculate the mean over all 16 channels and plot the mean for each feature as a function on epoch. The error bars represent one standard deviation.

# In[ ]:


featPanel.mean(axis=0).plot(subplots=True, figsize=(14,40), sharex=True, style='.-', yerr=featPanel.std(axis=0));
plt.suptitle('Normalized features for average channel\n(with standard deviation errors)');
plt.xlabel('Epoch #')
plt.subplots_adjust(top=0.96);


# ### Loading the datafiles
# 
# According to the [Kaggle website](https://www.kaggle.com/c/melbourne-university-seizure-prediction/data):
# > Within folders data segments are stored in .mat files as follows:
# 
# > + I_J_K.mat - the Jth training data segment corresponding to the Kth class (K=0 for interictal, K=1 for preictal) for the Ith patient (there are three patients).
# > + I_J.mat - the Jth testing data segment for the Ith patient.
# 
# So the file ../input/train_2_103_1.mat contains the 103rd data segment recording when patient 2 was within 10 minutes of having a seizure.

# In[ ]:


from os import listdir

def ieegGetFilePaths(directory, extension='.mat'):
    filenames = sorted(listdir(directory))
    files_with_extension = [directory + '/' + f for f in filenames if f.endswith(extension) and not f.startswith('.')]
    return files_with_extension


# In[ ]:


# Get preictal (K=1) files
preIctalFiles = ieegGetFilePaths('../input/train_1', extension='1.mat')


# In[ ]:


preIctalFiles

