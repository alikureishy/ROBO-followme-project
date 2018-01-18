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

<img src="https://github.com/safdark/ROBO-followme-project/blob/master/docs/images/gpu-utilization.png" width="900" height="250">

<!--![GPU utilization](https://github.com/safdark/ROBO-followme-project/blob/master/docs/images/gpu-utilization.png)-->

### Jupyter Notebook Server

This goes without saying. Nevertheless, on an EC2 instance (as mentioned above), the command to launch the Jupyter server is:
```
    jupyter notebook --ip='*' --port=8888 --no-browser
```

## Performance

Thanks to AWS, I was able to pretty quickly train the network over ~60 epochs, which allowed me to experiment with different network topologies. Below are two different topologies that I experimented with.

*Common Hyperparameters*
These hyperparameters were common to both topologies that were explored.
- Learning Rate = 0.001
- Train Batch Size = 32
- Batches Per Epoch: 130
- Validation Batch Size = 32
- Batches per Validation: 25

### Take # 1

This was achieved using a network with 4 encoder layers and 4 decoder layers, and a 1x1 convolution between them. Filter depths varied from 32 to 256, depending on the layer, both for the encoder and decoder sections. The encoding and decoding layers were essentiallly mirror images of each other. The encoding layers included just one separable convolution. Each decoding layer included an upsampling layer (doubling the image size in both x and y dimensions), followed by a concatenation of a skip connection input from its corresponding encoding layer, followed then by 2 separable convolution layers. I ran the training for ~60 epochs, though the network had almost fully saturated near ~30 epochs, as you can see in the validation loss graph below. Nevertheless, it appears there was still some marginal improvement going up to 60 epochs, which helped push the IoU score above 0.40.

![Take1 - Validation Loss History](https://github.com/safdark/ROBO-followme-project/blob/master/docs/images/take1-val-loss-history-plot.png)

#### Network Diagram
![Take1 - Network Diagram](https://github.com/safdark/ROBO-followme-project/blob/master/docs/images/take1-network.png)

#### Network Evaluation
![Take1 - IoU Evaluation](https://github.com/safdark/ROBO-followme-project/blob/master/docs/images/take1-evaluation.png)

#### Segmentation Outputs

*Hero close by*
![Take1 - Hero Close By](https://github.com/safdark/ROBO-followme-project/blob/master/docs/images/take1-hero-close.png)

*No hero*
![Take1 - No Hero](https://github.com/safdark/ROBO-followme-project/blob/master/docs/images/take1-no-hero.png)

*Hero far away*
![Take1 - Hero Far Away](https://github.com/safdark/ROBO-followme-project/blob/master/docs/images/take1-hero-far.png)

### Take # 2

This was a deeper network (5 encoding layers and 5 decoding layers) than in take # 1, and consequently its filter depths varied from 32 to 512, depending on the layer, both for the encoder and decoder sections. The encoding and decoding layers were mirror images of each other. The encoding layers included just one separable convolution. Each decoding layer included an upsampling layer (doubling the image size in both x and y dimensions), followed by a concatenation of a skip connection input from its corresponding encoding layer, followed then by 3 separable convolution layers. I ran the training for ~60 epochs, though the network had almost fully saturated near ~30 epochs, , as you can see in the validation loss graph below. Nevertheless, it appears there was still some marginal improvement going up to 60 epochs, which helped push the IoU metric above 0.40.

![Take2 - Validation Loss History](https://github.com/safdark/ROBO-followme-project/blob/master/docs/images/take2-val-loss-history-plot.png)

#### Network Diagram
![Take2 - Network Diagram](https://github.com/safdark/ROBO-followme-project/blob/master/docs/images/take2-network.png)

#### Network Evaluation
![Take2 - IoU Evaluation](https://github.com/safdark/ROBO-followme-project/blob/master/docs/images/take2-evaluation.png)

#### Segmentation Outputs

*Hero close by*
![Take2 - Hero Close By](https://github.com/safdark/ROBO-followme-project/blob/master/docs/images/take2-hero-close.png)

*No hero*
![Take2 - No Hero](https://github.com/safdark/ROBO-followme-project/blob/master/docs/images/take2-no-hero.png)

*Hero far away*
![Take2 - Hero Far Away](https://github.com/safdark/ROBO-followme-project/blob/master/docs/images/take2-hero-far.png)

## Future Improvements

To improve upon this network further, here are some additional tasks to be attempted.

### Collecting more data

I have achieved the requisite rubric [https://review.udacity.com/#!/rubrics/1155/view] for this project by using the provided data itself, without any additional data generated from the simulator. However, collecting more data would likely be necessary to push the IoU higher for this project.

### Larger kernels with dimensionality reduction

I have mostly focused on a kernel of size 3 in this project. A kernel of 5 not only deteriorated accuracy but also performance of the network during both training and inference. One way to optimize convolution with such a kernel is to insert a 1x1 convolution before it that reduces the dimensionality of the prior layer. This reduction in dimensionality dramatically reduces the number of trainable parameters when convolving the larger sized kernels.

### Converting to Inception layers

It is conceivable that using different sized kernels together might help find more relevant features for the segmentation task. The inception network targets just that scenario. If each encoder layer were an inception network, combining kernels of sizes 3, 5 and 7, in addition to a 1x1 convolution itself, as well as a maxpooling layer, we could achieve better accuracy. The degradation in performance from adding such large kernels can be worked around by inserting a 1x1 convolution to reduce the dimensionality of the prior layer before it is convolved with the associated kernel.



## References
