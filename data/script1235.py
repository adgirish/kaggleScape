
# coding: utf-8

# This notebook will show you how to load and manipulate audio data in Python. 
# 
# The Heartbeat Sounds dataset is primarily audio-based: all of the heartbeat sounds are stored as WAV files that record either normal or abnormal heartbeats. So let's learn how to load and play with WAVs in Python.

# In general, uncompressed audio is stored as a sequence of numbers that indicate the amplitude of the recorded sound pressure at each time point. In the WAV standard, these numbers are packed into a bytestring. The interpretation of this byestring depends primarily on two factors: first, the sampling rate, usually given in Hertz, which indicates how many number samples comprise a second's worth of data; and second, the bit depth (or sample width), which indicates how many bits comprise a single number.
# 
# These parameters, along with other parameters like the number of channels (e.g., is the audio mono or stereo) are stored in the header of the WAV file.
# 
# The `wave` library handles the parsing of WAV file headers, which include the parameters mentioned above. Let's load the `wave` library and use it to open a sound file.

# In[ ]:


import wave

FNAME = '../input/set_a/normal__201101070538.wav'

f = wave.open(FNAME)

# frames will hold the bytestring representing all the audio frames
frames = f.readframes(-1)
print(frames[:20])


# So `frames` now holds the entire bytestring representing all the audio samples in the sound file. We need to unpack this bytestring into an array of numbers that we can actually work with.
# 
# The first question is: how many bytes represent a single observation? In my experience in voice recording, 16-bit and 24-bit are the most common sample widths, but you can find a whole collection [on Wikipedia](https://en.wikipedia.org/wiki/Audio_bit_depth).
# 
# Powers of 2 tend to be the easiest to work with, and luckily for us the heartbeat audio seems to be 16-bit. We can check this by using the getsamplewidth() method on the wave file:

# In[ ]:


print(f.getsampwidth())


# The result of getsamplewidth() is in bytes, so multiply it by 8 to get the bit depth. Since the result from the call is 2, that means we're looking at a 16-bit file.
# 
# We'll unpack the bytestring by using the `struct` library in Python. `struct` requires a format string based on C format characters, which you can take a look at [on the documentation page for Python's struct library](https://docs.python.org/2/library/struct.html).
# 
# We're in luck with the 16-bit depth, since the `struct` library prefers powers of 2. 16 bits corresponds to 2 bytes, so we'll use the signed format that corresponds to 2 bytes; according to [the C format characters](https://docs.python.org/2/library/struct.html#format-characters), we should use the format character 'h'.
# 
# A slight trick in the `struct` library is that it wants its format string to exactly match the expected size, so we have to multiply the format character 'h' by the number of frames in the bytestring:

# In[ ]:


import struct
samples = struct.unpack('h'*f.getnframes(), frames)
print(samples[:10])


# To get the timing, we'll grab the sampling rate from the wave object.

# In[ ]:


framerate = f.getframerate()
t = [float(i)/framerate for i in range(len(samples))]
print(t[:10])


# Now we can take a look at the waveform.

# In[ ]:


from pylab import *
plot(t, samples)

