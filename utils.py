# Utilities for PoolVision

import skimage
import skimage.io
import skimage.transform
import numpy as np
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt

def truncate(f, n):
  # Truncates/pads a float f to n decimal places without rounding
  s = '{}'.format(f)
  if 'e' in s or 'E' in s:
    return '{0:.{1}f}'.format(f, n)
  i, p, d = s.partition('.')
  return '.'.join([i, (d+'0'*n)[:n]])

def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

r = 2 # radius of squisher
squisher = np.array([[.51,.51,.51,.51,.51],[.51,.1,.1,.1,.51],[.51,.1,.01,.1,.51],[.51,.1,.1,.1,.51],[.51,.51,.51,.51,.51]])
assert(squisher.shape[0] == 2 * r + 1)

def personalspace(heatmap,thresh):
    count = 0
    balls = []
    bustmap = np.zeros((heatmap.shape[0]+2*r,heatmap.shape[1]+2*r))
    bustmap[r:-r,r:-r] = heatmap
    maxcoord = np.argmax(bustmap)
    maxcoord = (maxcoord // bustmap.shape[1], maxcoord % bustmap.shape[1]) # converts into 2d
    maxprob = bustmap[maxcoord[0],maxcoord[1]]
    while(maxprob > thresh):
        balls.append(maxcoord)
        bustmap[(maxcoord[0]-r):(maxcoord[0]+r+1),(maxcoord[1]-r):(maxcoord[1]+r+1)] *= squisher
        maxcoord = np.argmax(bustmap)
        maxcoord = (maxcoord // bustmap.shape[1], maxcoord % bustmap.shape[1])
        maxprob = bustmap[maxcoord[0],maxcoord[1]]
    balls = list(map(lambda ball: (ball[1]-r,ball[0]-r), balls)) # flipped?
    return balls

