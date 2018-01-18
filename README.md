# Image Segmentation - Person Detection (Follow-Me Project)

## Overview

This is an Image Segmentation project built as part of Udacity's 'Robotics Nanodegree Term 1' curriculum. It involves training a deep neural network using various mechanisms - such as Fully Convolutional Networks, Skip Connections, 1x1 Convolutions etc - to detect a person-of-interest from images captured by a Follow-Me drone, the purpose eventually being to be able to train a drone to follow-along with that person as they go jogging, walking etc. It was evaluated [https://review.udacity.com/#!/rubrics/1155/view] based on its IoU (Intersection-over-Union) performance on a provided test set.

The folder hierarchy is as follows:
```
  root/
    README.md
    code/
      model_training.ipynb      <-- IPython notebook (network architecture implementation)
      utils/
        ...
      ...
    data/
      masks/                    <-- Trainable data, including validation and test sets
        train/
          images/               <-- Raw images provided as inputs
          masks/                <-- Expected image segmentation output
        validation/
          images/
          masks/
        test/
          images/
          masks/
      weights/
      inferences/
    docs/
    logs/                     <-- Tensorboard logs      
```

## Data

### Inputs

The network would be fed images (scenes) of a simulated world, as seen from the point of view of a drone that would, hypothetically speaking, continuously detect a designated person-of-interest (aka 'hero') within that scene and follow behind her. An example of such a scene captured from the drone's camera is illustrated at the top of this document.

There are three types of images included as training data:
- _Images containing the hero nearby_: For incremental adjustments of the drone's guidance system
- _Images containing the hero far away_: For helping a drone get back-on-track in case it falls behind
- _Images containing no hero at all_: Presumably to have the drone remain stationary and patrol its surroundings until one of the above two images is encountered

## Expected Output

The objective was to train the network to output a segmented image of the same size as the original, containing 3 classes of pixels:
- _Hero: (Blue pixels)_ These represent the POI (Person-Of-Interest, or "hero"): Identified with blue pixels
- _Generic person (Green pixels)_: These are all other people in the scene who are *not* the 'hero'
- _Surroundings (Red pixels)_: A catch-all category that includes everything else such as grass, road surfaces, buildings, sky etc.

Such a segmentation would presumably provide the drone the coordinates of the object that it is to follow behind. Ultimately this could be used in an actual drone implementation, but would of course require various other components to fully implement, and is as such outside the scope of this project.

## Simulator

A simulator was provided with the project to merely collect 'real-world' data in the event that the data included with the project was not sufficient. This was not used in this project however, as the provided data proved sufficient to satisfy the rubric [https://review.udacity.com/#!/rubrics/1155/view].

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

Areas of future improvement:
- I have achieved the requisite rubric [https://review.udacity.com/#!/rubrics/1155/view] for this project by using the provided data itself, without any additional data generated from the simulator. That, however, would likely be necessary to improve upon the performance of the DNN (Deep Neural Network).

## References
