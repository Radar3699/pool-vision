import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import os
import csv
import glob
import cv2
import time

import classifier
import priority
import utils

### Tensorflow Setup
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

### Global Variables
smallwindow_size = 48
smallwindow_step = 23

### Main Classification loop. (folder of JPEGS) -> (set of txt files with ball types and locations)
folders = glob.glob('games_folder/*') # change to suit your needs
folders.sort()

# Each folder coresponds to a set of JPEG images of pool table game states
for folder in folders:
  images = glob.glob(folder+'/*.jpg')
  images.sort()

  # Each image corresponds to a pool game state
  for imagename in images:
    print(imagename)
    image = cv2.imread(imagename)
    new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    (winW, winH) = (smallwindow_size, smallwindow_size)

    if new_image.shape != (790,1580):
      new_image = new_image[60:850,40:1620]

    # Get probability heatmap with priority network
    # This step utilizes a 5 layer "priority network" to quickly scan
    # through the image to find areas of the pool table in which something
    # is probably there, to eventually feed to the much deeper classifier network
    heatmap = np.zeros((67,33,3))
    count = 0
    start = time.time()
    for (x, y, window) in utils.sliding_window(new_image, stepSize=smallwindow_step, windowSize=(winW, winH)):
      if window.shape[0] != winH or window.shape[1] != winW:
        continue
      window = utils.skimage.transform.resize(window, (24, 24))
      predictions = priority.isball(window)
      heatmap[int(x/smallwindow_step), int(y/smallwindow_step),0] = utils.truncate(predictions[0],3)
      heatmap[int(x/smallwindow_step), int(y/smallwindow_step),1] = utils.truncate(predictions[1],3)
      count += 1

    # DEBUG: Print runtime for priority network to slide over image
    end = time.time()
    print("RUNTIME", end - start)
    heatmap = heatmap.transpose(1,0,2)
    heatmap = heatmap[:,:,1]

    # Get precise ball positions from local max finder
    balls = utils.personalspace(heatmap,0.49)

    # DEBUG: Print results
    # plt.imshow(heatmap)
    # for each in balls:
      # plt.plot(int(each[0]), int(each[1]), 'ro')
    # plt.show()

    # Feed small areas of image with high priority to classifier network
    interesting_count = 0
    names = []
    for ball in balls:
      xcoord = int(ball[0] * smallwindow_step)
      ycoord = int(ball[1] * smallwindow_step)
      small_image = new_image[max(ycoord, 0): min(ycoord + smallwindow_size, 790), max(xcoord, 0): min(xcoord + smallwindow_size, 1580)]
      predictions = classifier.isball(small_image)
      small_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)
      labels = ['eight_ball', 'cue', 'neither', 'solids', 'stripes']
      maxnum = predictions[0][0]
      index = 0
      for i in range(1, len(labels)):
          if maxnum < predictions[0][i]:
            maxnum = predictions[0][i]
            index = i
      names.append(labels[index])
      interesting_count += 1

    # Output txt file with ball types and locations
    csvname = imagename.split('.jpg')[0]
    with open(csvname, 'w') as csvfile:
      writer = csv.writer(csvfile) #, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
      for i in range(len(balls)):
        writer.writerow([names[i],balls[i][0],balls[i][1]])
