#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.cluster import KMeans
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import cv2


# In[2]:


# load the image and grab its width and height
image = cv2.imread("leaf.jpg")
(h, w) = image.shape[:2]
cv2.imshow('image', image)
plt.show()
print((h, w))
# load the image and convert it to a floating point data type
image = img_as_float(image)
for numSegments in (200, 300, 500, 800):
	# apply SLIC and extract (approximately) the supplied number of segments
	segments = slic(image, n_segments = numSegments, sigma = 5)
    
	# show the output of SLIC
	fig = plt.figure("Superpixels -- %d segments" % (numSegments))
	ax = fig.add_subplot(1, 1, 1)
	ax.imshow(mark_boundaries(image, segments))
	plt.axis("off")
    
# show the plots
plt.show()


# In[3]:


# load the image and grab its width and height
image = cv2.imread("lena.jpg")
(h, w) = image.shape[:2]

plt.imshow(image)
plt.show()

print((h, w))

# load the image and convert it to a floating point data type
image = img_as_float(image)

# loop over the number of segments
for numSegments in (200, 300, 500, 800):
	# apply SLIC and extract (approximately) the supplied number of segments
	segments = slic(image, n_segments = numSegments, sigma = 5)
    
	# show the output of SLIC
	fig = plt.figure("Superpixels -- %d segments" % (numSegments))
	ax = fig.add_subplot(1, 1, 1)
	ax.imshow(mark_boundaries(image, segments))
	plt.axis("off")
    
# show the plots
plt.show()


# In[4]:


# load the image and grab its width and height
image = cv2.imread("imgproc.jpg")
(h, w) = image.shape[:2]

plt.imshow(image)
plt.show()

print((h, w))

# load the image and convert it to a floating point data type
image = img_as_float(image)

# loop over the number of segments
for numSegments in (200, 300, 500, 800):
	# apply SLIC and extract (approximately) the supplied number of segments
	segments = slic(image, n_segments = numSegments, sigma = 5)
    
	# show the output of SLIC
	fig = plt.figure("Superpixels -- %d segments" % (numSegments))
	ax = fig.add_subplot(1, 1, 1)
	ax.imshow(mark_boundaries(image, segments))
	plt.axis("off")
    
# show the plots
plt.show()

