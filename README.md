# Person Detection (Follow-Me)

## Overview

This project was built as part of Udacity's 'Robotics Nanodegree Term 1' curriculum. It involves training a deep neural network using various mechanisms - such as Fully Convolutional Networks, Skip Connections, 1x1 Convolutions etc - to detect a person-of-interest from images captured by a Follow-Me drone, the purpose eventually being to be able to train a drone to follow-along with that person as they go jogging, walking etc.

## Components

An ipython notebook contains a majority of the relevant code for the network, with various support utilities implemented in python files.

The folder hierarchy is as follows:
```
  root/
    README.md
    code/
      model_training.ipynb      <-- IPython notebook
      utils/
        ...
      ...
    data/
      masks/
      weights/
      runs/
      inferences/
    docs/
    logs/                     <-- Tensorboard logs      
```

## Network Architecture

The network consists of 
