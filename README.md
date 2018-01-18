![Overview](https://github.com/safdark/ROBO-followme-project/blob/master/docs/images/image-2.png)

<!-- Side-by-side images:
        ![alt-text-1](image1.png "title-1") ![alt-text-2](image2.png "title-2")
-->

# Image Segmentation - Person Detection (Follow-Me Project)

## Table of Contents

- [Overview](#overview)
- [Components](#components)
	- [Simulation Environment](#simulation-environment)
	- [Training Pipeline](#training-pipeline)
		- [Sample Collection](#sample-collection)
		- [Feature Extraction](#feature-extraction)
		- [Training](#training)
	- [Perception Pipeline](#perception-pipeline)
		- [RGBD Camera View](#rgbd-camera-view)
		- [Downsampling](#downsampling)
		- [Cleaning](#cleaning)
		- [Passthrough Filter](#passthrough-filter)
		- [Segmentation](#segmentation)
		- [Clustering](#clustering)
		- [Classification](#classification)
		- [Labeling](#labeling)
- [Debugging](#debugging)
- [Results](#results)
	- [World 1](#world-1)
	- [World 2](#world-2)
	- [World 3](#world-3)
- [Conclusions](#conclusions)

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

### Expected Output

The objective was to train the network to output a segmented image of the same size as the original, containing 3 classes of pixels:
- _Hero: (Blue pixels)_ These represent the POI (Person-Of-Interest, or "hero"): Identified with blue pixels
- _Generic person (Green pixels)_: These are all other people in the scene who are *not* the 'hero'
- _Surroundings (Red pixels)_: A catch-all category that includes everything else such as grass, road surfaces, buildings, sky etc.

Such a segmentation would presumably provide the drone the coordinates of the object that it is to follow behind. Ultimately this could be used in an actual drone implementation, but would of course require various other components to fully implement, and is as such outside the scope of this project.

### Simulator

A simulator was provided with the project to merely collect 'real-world' data in the event that the data included with the project was not sufficient. This was not used in this project however, as the provided data proved sufficient to satisfy the rubric [https://review.udacity.com/#!/rubrics/1155/view].

## Network Architecture

This is a Deep Neural Network consisting, at a high level, of an Encoder, followed by a Decoder, with a 1x1 convolution in between.

Here is a diagram of the architecture:

### Encoder Section


### 1x1 Convolution

### Decoder Section

### Training Hooks (Callbacks)

#### Preexisting
Here are some callbacks I used to simplify the training bookkeeping:
- _keras.callbacks.ModelCheckpoint_: To checkpoint the model after an epoch only if val_loss improves.
- _keras.callbacks.EarlyStopping_: To stop the training if a certain number of epochs have passed without any improvement in val_loss

#### Custom
Here is a custom callback that was implemented for special handling at the end of an epoch
- _plotting_tools.LoggerPlotter_: At the end of every epoch, this plots a graph of the val_loss history before that epoch. It also saves that plot in a folder created for that particular training run.


## Training

### AWS

Training was done on a p2.xlarge (GPU-Compute) EC2 instance on AWS, which performed brilliantly, achieving almost a 10x improvement in training time. The _Udacity Robotics Deep Learning Laboratory (ami-e4fd199e) AMI_ was used, as below:

![AWS AMI](https://github.com/safdark/ROBO-followme-project/blob/master/docs/images/aws-ami.png)

During training, I was able to max out the GPU usage on this machine as well, as suggested by this output from the command-line ```nvidia-smi'' command:

![GPU utilization](https://github.com/safdark/ROBO-followme-project/blob/master/docs/images/gpu-utilization.png)

### Jupyter Notebook Server

This goes without saying. Nevertheless, on an EC2 instance (as mentioned above), the command to launch the server is:
```
The jupyter notebook server was launched on the EC2 instance using this command:
    jupyter notebook --ip='*' --port=8888 --no-browser
```

## Performance

### Take # 1

This was achieved with a network with 5 encoder layers and 5 decoder layers, and a 1x1 convolution between them. Filter depths varied from 32 to 512, depending on the layer, both for the encoder and decoder sections. Below is a diagram showing the evaluation of this network.

Here's the graph of the val_loss as calculated over ~60 epochs for this training run:
![Take1 - Validation Loss History](https://github.com/safdark/ROBO-followme-project/blob/master/docs/images/val-loss-history-plot-take1.png)


### Take # 2



## Assessments

## Future Improvements

To improve upon this network further, here are some additional tasks to be attempted.

### Collecting more data

I have achieved the requisite rubric [https://review.udacity.com/#!/rubrics/1155/view] for this project by using the provided data itself, without any additional data generated from the simulator. However, collecting more data would likely be necessary to push the IoU higher for this project.

### Larger kernels with dimensionality reduction

I have mostly focused on a kernel of size 3 in this project. A kernel of 5 not only deteriorated accuracy but also performance of the network during both training and inference. One way to optimize convolution with such a kernel is to insert a 1x1 convolution before it that reduces the dimensionality of the prior layer. This reduction in dimensionality dramatically reduces the number of trainable parameters when convolving the larger sized kernels.

### Converting to Inception layers

It is conceivable that using different sized kernels together might help find more relevant features for the segmentation task. The inception network targets just that scenario. If each encoder layer were an inception network, combining kernels of sizes 3, 5 and 7, in addition to a 1x1 convolution itself, as well as a maxpooling layer, we could achieve better accuracy. The degradation in performance from adding such large kernels can be worked around by inserting a 1x1 convolution to reduce the dimensionality of the prior layer before it is convolved with the associated kernel.



## References
