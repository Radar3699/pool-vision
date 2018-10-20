import tensorflow as tf
import numpy as np
import sys
import os

### TENSORFLOW SETUP
# todo: disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("classifier_weights/trained_labels.txt")]

with tf.gfile.FastGFile("classifier_weights/trained_graph.pb", 'rb') as f:
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(f.read())
  g2 = tf.import_graph_def(graph_def, name='g2')

def isball(image_data):
  with tf.Session(graph=g2) as sess2:

    softmax_tensor = sess2.graph.get_tensor_by_name('g2/final_result:0')
    predictions = sess2.run(softmax_tensor, {'g2/DecodeJpeg:0': image_data})

    # DEBUG: Print results
    # print(predictions[0])
    # top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    # for node_id in top_k:
    #   human_string = label_lines[node_id]
    #   score = predictions[0][node_id]
    #   print('%s (score = %.5f)' % (human_string, score))

    return predictions





















