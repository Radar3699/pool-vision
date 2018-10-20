from keras.models import load_model
import cv2
import argparse
import time
import numpy as np
import utils
import matplotlib.pyplot as plt

# load model
model = load_model('priority_weights/model4.h5')

def isball(image):
  image = image.reshape((1,24,24,3))
  predictions = model.predict(image)
  return predictions[0]












