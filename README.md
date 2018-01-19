![Overview](https://github.com/safdark/ROBO-followme-project/blob/master/docs/images/image-2.png)

<!-- Side-by-side images:
        ![alt-text-1](image1.png "title-1") ![alt-text-2](image2.png "title-2")
-->

# Image Segmentation - Person Detection (Follow-Me Project)

## Table of Contents

- [Overview](#overview)
- [Data](#data)
	- [Inputs](#inputs)
		- [Provided Data](#provided-data)
	- [Expected Output](#expected-output)
	- [Simulator](#simulator)
- [Network Architecture](#network-architecture)
	- [Encoder](#encoder)
	- [1x1 Convolution](#1x1-convolution)
	- [Decoder](#decoder)
- [Training](#training)
	- [Hyperparameter Tuning](#hyperparameter-tuning)
		- [Learning Rate](#learning-rate)
		- [Batch Size](#batch-size)
		- [Optimizer](#optimizer)
		- [Network Depth](#network-depth)
		- [Number of Epochs](#number-of-epochs)
		- [Data Augmentation](#data-augmentation)
		- [Data Filteration](#data-filteration)
		- [Batch Size](#batch-size)
	- [AWS](#aws)
	- [Jupyter Notebook Server](#jupyter-notebok-server)
	- [Training Hooks](#training-hooks-(callbacks))
		- [Preexisting](#preexisting)
		- [Custom](#custom)
- [Performance](#performance)
- [Other Use Cases](#other-use-cases)
- [Future Improvements](#future-improvements)
- [References](#references)

## Overview
This is an Image Segmentation project built as part of Udacity's 'Robotics Nanodegree Term 1' curriculum. It involves training a deep neural network using a Fully Convolutional Network, as well as various other mechanisms - such as Skip Connections, 1x1 Convolutions etc - to detect a person-of-interest from images captured by a Follow-Me drone, the purpose eventually being to be able to train a drone to follow-along with that person as they go jogging, walking etc. It was evaluated [https://review.udacity.com/#!/rubrics/1155/view] based on its IoU (Intersection-over-Union) performance on a provided test set.

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

The network would be fed 160x160x3-pixel images (scenes) of a simulated world, as seen from the point of view of a drone that would, hypothetically speaking, continuously detect a designated person-of-interest (aka 'hero') within that scene and follow behind her. An example of such a scene captured from the drone's camera is illustrated at the top of this document.

There are three types of images included as training data:
- _Images containing the hero nearby_: For incremental adjustments of the drone's guidance system
- _Images containing the hero far away_: For helping a drone get back-on-track in case it falls behind
- _Images containing no hero at all_: Presumably to have the drone remain stationary and patrol its surroundings until one of the above two images is encountered

#### Provided Data

The data provided was not balanced across the three types of scenarios above. Here is a breakdown of the counts:
```
- Hero nearby: 413 (Images with masks containing > 400 blue pixels)
- Hero far away: 1142 (Images with masks containing 1 - 400 blue pixels)
	- Hero far = 621 (Images with masks containing between 40-400 blue pixels)
	- Hero very far = 521 (Images with masks containing btween 1-40 blue pixels)
- No hero at all: 2576 (Images with masks containing 0 blue pixels)
```
Notice that only 37% of the images actually had the hero at all (first two categories) in them. The remaining 63% were of the 3rd category (with no hero at all).

### Expected Output

The objective was to train the network to output a segmented image of the same dimensions as the input image, containing 3 classes of pixels:
- _Hero: (Blue pixels)_ These represent the POI (Person-Of-Interest, or "hero"): Identified with blue pixels
- _Generic person (Green pixels)_: These are all other people in the scene who are *not* the 'hero'
- _Surroundings (Red pixels)_: A catch-all category that includes everything else such as grass, road surfaces, buildings, sky etc.

Therefore the output of the image would be 160x160x3 images. It so happens that the 3 classes above, through a softmax activation layer, could be trivially translated to the 3 RGB channels, which is why the encodings above were given equivalence to those 3 specific colors. Had there been more classes of images, or a different color mapping, a separate conversion would have been needed to convert the category into the appropriate 3-channel RGB pixel of the segmented image.

The drone would use the CoG of the blue pixels in the segmented image to follow behind the hero. Ultimately this DNN could therefore be used in an actual follow-me-drone implementation, but would of course require numerous other components that are outside the scope of this project.

### Simulator

A simulator was provided with the project to merely collect 'real-world' data in the event that the data included with the project was not sufficient. This was not used in this project however, as the provided data proved sufficient to satisfy the rubric [https://review.udacity.com/#!/rubrics/1155/view].

## Network Architecture

Fully Convolutional networks are well suited for segmentation tasks.

...
...
...

Here is a diagram of the architecture I finally settled on:

### Encoder


### 1x1 Convolution


### Decoder


## Training

The immediate goal of this project was to satisfy the 40% IoU metric required, which is the yardstick I used when evaluating various hyper parameters during training. Therefore, it is worth noting that I did not explore optimizations to the network beyond what was needed to reach the requisite IoU score. Possible optimizations are discussed at the end of this document. Furthermore, I tuned the hyperparameters manually, though I could have used tools like TensorBoard to compare various permutations of hyperparameters.

### Hyperparameter Tuning

In the following sections, I discuss the hyperparameters I tweaked, before I settled on the 'winning' network architecture and accompanying paramters.

#### Learning Rate

I started with a learning rate of 0.01, which was too high because the training and validation losses fluctuated a lot during training. I reduced that to 0.001, which proved to be sufficient to produce a smooth asymptotic training loss curve, and a reasonably smooth validation loss curve too, before stopping the training.

#### Batch Size
On my local machine, where I started the training attempts, a batch size of 32 seemed more appropriate given the memory constraints of the system. I could theoretically try increasing the batch size when I moved over to AWS, but I did not attempt that.

#### Optimizer

I started with an Adam optimizer but later settled on _Nadam_ based on input from fellow students on Slack. I did not however notice a substantial enough difference between the two.

#### Network Depth (# of encoding/decoding layers)

Even with 5 layers and filters ranging in depth from 32 to 512, the network did not seem to overfit to the training set. This was likely because of batch normalization being applied after every convolution layer in the network.

#### Number of Epochs
Initially, I kept the training to only ~20-25 epochs, which consistently fell short of the target IoU metric of 40%. I tried increasing the depth of the network (adding additional encoding and decoding layers) but that did not help.

#### Data Augmentation

Through the data_iterator.BatchIterator() class, I augmented the data as follows:
- Random (0.5) horizontal flipping of input images

#### Data Filteration

Filteration was also necessary for this data, in order to balance the training data across the 3 types of images (hero close by, hero far away, and no hero). The provided data had only 37% images with the hero in it, and 63% without. To balance this, I randomly filtered out 30% of all non-hero images, to bring down the count to ~44%. Note that this however is not optimal. A more optimal filtering approach is discussed under the Improvements section below.

### AWS

Training was done on a p2.xlarge (GPU-Compute) EC2 instance on AWS, which performed brilliantly, achieving almost a 10x improvement in training time. The _Udacity Robotics Deep Learning Laboratory (ami-e4fd199e) AMI_ was used, as below:

![AWS AMI](https://github.com/safdark/ROBO-followme-project/blob/master/docs/images/aws-ami.png)

During training, I was able to max out the GPU usage on this machine as well, as suggested by this output from the command-line ```nvidia-smi'' command:

<img src="https://github.com/safdark/ROBO-followme-project/blob/master/docs/images/gpu-utilization.png" width="900" height="250">

### Jupyter Notebook Server

This goes without saying. Nevertheless, on an EC2 instance (as mentioned above), the command to launch the Jupyter server is:
```
    jupyter notebook --ip='*' --port=8888 --no-browser
```

### Training Hooks (Callbacks)

#### Preexisting
Here are some callbacks I used to simplify the training bookkeeping:
- _keras.callbacks.ModelCheckpoint_: To checkpoint the model after an epoch only if val_loss improves.
- _keras.callbacks.EarlyStopping_: To stop the training if a certain number of epochs have passed without any improvement in val_loss

#### Custom
Here is a custom callback that was implemented for special handling at the end of an epoch
- _plotting_tools.LoggerPlotter_: At the end of every epoch, this plots a graph of the val_loss history before that epoch. It also saves that plot in a folder created for that particular training run.

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

## Other Use Cases

This model, generally, would work with any kind of image segmentation task provided sufficient training data relevant to the problem being addressed.

### Different types of objects
The model in this case was trained with specific data pertaining to human beings - a hero and others - walking in a city-like environment (with roads, bridges, buildings, lawns etc). In other words, the model weights have been specifically tuned to recognize features pertaining to the categories that it was trained to recognize using the training data provided. As examples, the model probably uses the color of the clothing to distinguish between the hero and other people. Therefore, this trained model, as it stands, will not work for segmenting other classes of objects (like cats, dogs etc), or different environments. In fact, it will also fail if the test data were to present the hero in a different set of clothes.

The model has learned features specific to the domain for which it was trained. Therefore, if provided sufficient training data for other domains, with 3 classes of objects in the segmentation, this model is likely to segment those images well too. However, there are some limitations there too ...

### Different number of object categories
One such limitation is that the model has been architected to address 3 classes of objects in this segmentation problem. If provided sufficient data, the model might still be trainable on a minor increase or decrease in the number of cateogories it is to classify pixels into. But, the model may not be able to train well for a vastly larger count of categories in the segmentation task.

### More complex environments
I also expect that, depending on the environments being processed, the network might need to be deeper. In other words, it should be possible to come up with an environment wherein the objects being classified require the network to learn more features than it is presently capable of learning. In such situations too, this network will not be trainable.

## Future Improvements

To improve upon this network further, here are some additional tasks to be attempted.

### Training with an appropriately balanced categorical distribution of data

I have achieved the requisite rubric [https://review.udacity.com/#!/rubrics/1155/view] for this project by using the provided data itself, without any additional data generated from the simulator. However, collecting more data would likely be necessary to push the IoU higher for this project.

More specifically, the training data was imbalanced, with the following percentages:
- Hero close: ~9.5%
- Hero far: ~27.5%
- No hero: ~63%

As a result, the network in general was rather good at segmenting images that had no hero, or where the hero was very close. Images where the hero was far away had the worst accuracy as compared to the other two categories, even though 27.5% of the input data was of those types of images, arguably because that segmentation problem is more complicated, requiring a very small cluster of pixels in the image to be recognized as the hero. The case where the hero is very close is arguably an easier segmentation problem, because of the number of pixels that stand out as belonging to the hero.

Therefore, in my view, an approximate distribution that would achieve better results in all 3 categories of images is as follows:
- Hero close: ~20%
- Hero far: ~50%
- No hero: ~30%

I have obviously not tested this hypothesis, but it remains a future goal.

### Larger kernels with dimensionality reduction

I have mostly focused on a kernel of size 3 in this project. A kernel of 5 not only deteriorated accuracy but also performance of the network during both training and inference. One way to optimize convolution with such a kernel is to insert a 1x1 convolution before it that reduces the dimensionality of the prior layer. This reduction in dimensionality dramatically reduces the number of trainable parameters when convolving the larger sized kernels.

### Using Inception layers in the encoder

It is conceivable that using different sized kernels together might help find more relevant features for the segmentation task. The inception network targets just that scenario. If each encoder layer were an inception network, combining kernels of sizes 3, 5 and 7, in addition to a 1x1 convolution itself, as well as a maxpooling layer, we could achieve better accuracy. The degradation in performance from adding such large kernels can be worked around by inserting a 1x1 convolution to reduce the dimensionality of the prior layer before it is convolved with the associated kernel.

### Using Transposed Convolutions in addition to (or instead of) Biliniear Upsampling

In bilinear upsampling, the upsampled image is based purely on the input image. When using Transposed Convolutions, the relation between the input and upsampled image pixels can be learned by the network, thereby not only introducing additional non-linearities in the upsampling but also a more nuanced transformation relation between the pixels in the input and output. But for this same reason, this approach will also impact performance, both during training and also with inference. However, that can potentially be resolved by removing the 'same' padded convolution layers at each decoding layer, after the transposed convolution step. It is not clear to me which approach will produce better accuracy, but it is perhaps worth exploring.

### Better Hyperparemter Tuning

I could have used TensorBoard to compare various hyperparameter permutations, but did not on account of time constraints with getting familiar with all the TensorBoard tools and quirks.

I settled on a batch size of 32 since that was appropriate to the memory constraints of my home use laptop. This could potentially have been increased when training was done on the GPU instances on AWS.






## References
