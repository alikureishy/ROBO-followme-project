# Person Detection (Follow-Me)

## Overview

This project was built as part of Udacity's 'Robotics Nanodegree Term 1' curriculum. It involves training a deep neural network using various mechanisms - such as Fully Convolutional Networks, Skip Connections, 1x1 Convolutions etc - to detect a person-of-interest from images captured by a Follow-Me drone, the purpose eventually being to be able to train a drone to follow-along with that person as they go jogging, walking etc.

## Components

There is an ipython notebook that contains a majority of the relevant code for the network, with various support utilities implemented in python files. Aside from this there are a data folder, a weights folder and an inferences folder.

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
      inferences/
    docs/
    logs/                     <-- Tensorboard logs      
```

## Network Architecture

This is a Deep Neural Network consisting, at a high level, of an Encoder, followed by a Decoder, with a 1x1 convolution in between.

Here is a diagram of the architecture:

### Encoder Section

### 1x1 Convolution

### Decoder Section

## Performance

IoU Achieved: 0.416

## Assessments

## Future Improvements

## References
