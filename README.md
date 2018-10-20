# Pool Vision

Pool Vision is a project for quickly converting top-down images of pool games to a list of ball types (solids, stripes, cue, eight) and positions. Pool Vision leverages a 2 network hybrid approach to combine the speed of classical computer vision with the accuracy of deep learning methods, all without any specialized hardware. 

## Motivation 

This project was created because existing fast computer vision systems for analyzing pool games were highly innacurate and non-generalizable (not robust to changes in lighting, cue sticks or people in the image, etc.). Additionally accurate deep learning solutions were very slow and only practical with GPU computation.

## Description

![alt text](https://github.com/Radar3699/PoolVision/blob/master/example_images/L0.jpg)

As mentioned above, Pool Vision leverages two neural networks, a lightweight 'priority' network and a heavyweight Inception-based 'classifier' network to analyze an image from a pool game. The lightweight 'priority network' scans through the image with a sliding window in traditional computer vision style. It is trained to discriminate game-relevant objects (different types of pool balls in different conditions and orientations) from non-game relevant objects (hands, arms, people, cue sticks, shadows, empty table, chalk, etc.) and can be fed-forward in about a milisecond on a CPU. At this point we have a heatmap of probability that a game-relevant object is in that area. (Note the human hand has high probability, this is a false-positive which will eventually be weeded out by the classifier network)

![alt text](https://github.com/Radar3699/PoolVision/blob/master/example_images/L4.jpg)


We then run a local-maximum algorithm on the heatmap to obtain a list of coordinates in which something game-revelant is probably there. 

![alt text](https://github.com/Radar3699/PoolVision/blob/master/example_images/L1.jpg)

 The small images around these coordinates are then fed to the heavyweight 'classifier' network. It is an Inception V3 very deep convoltional neural network base modified in a transfer learning process to discriminate between game objects (solid ball, striped ball, cue ball, eight ball, none-of-the-above) and takes about 1 second to feed forward on a CPU. (Note the hand is labelled 'none' because this network can weed out false-positives from the priority network)
 
![alt text](https://github.com/Radar3699/PoolVision/blob/master/example_images/L2.jpg)
 
By leveraging this dual-network approach we are able to obtain the speed of hard-coded approaches, but the accuracy and robustness of deep learning solutions, without specialized hardware. 

## Usage

### Installing 

Pool Vision uses TensorFlow and Keras for deep learning, as well as cv2 for image loading as well as numpy and matpotlib for numerical computation and visualization. All requirements can be installed by running 

**CPU only:**
```
pip install requirements.txt
```

**CPU + GPU**
```
pip install requirements_gpu.txt
```


### Multiple games with multiple images

With each game in a different folder in games_folder, and a sequence of game images within each game folder, run:

```
python classify_game_images.py 
```
And the output will be a txt file for each image in each game. By default they are in the same location as the images but this can be changed in the script.

### Individual Networks

The function which takes an image and feeds it forward through the priority network can be found in ```priority.py``` and likewise for the classifier network can be found in ```classifier.py```.

## Built With

* [Tensorflow](https://www.tensorflow.org/) - The framework used for the classifier network
* [Keras](https://keras.io/) - The framework used for the priority network
* [Google Inception](https://github.com/google/inception) - The base network transfer learned from for the classifier network