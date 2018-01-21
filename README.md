# Image Segmentation - Person Detection (Follow-Me Project)
![Overview](https://github.com/safdark/ROBO-followme-project/blob/master/docs/images/follow-me.png)

<!-- Side-by-side images:
        ![alt-text-1](image1.png "title-1") ![alt-text-2](image2.png "title-2")
-->

## Table of Contents

- [Overview](#overview)
- [Environment & Setup](#environment-and-setup)
	- [AWS](#aws)
	- [Jupyter Notebook Server](#jupyter-notebook-server)
	- [Local Machine Setup](#local-machine-setup)
- [Data](#data)
	- [Inputs](#inputs)
		- [Provided Data](#provided-data)
	- [Expected Output](#expected-output)
	- [Simulator](#simulator)
- [Training](#training)
	- [Hyperparameter Tuning](#hyperparameter-tuning)
		- [Learning Rate](#learning-rate)
		- [Batch Size](#batch-size)
		- [Batches Per Epoch](#batches-per-epoch)
		- [Number of Epochs](#number-of-epochs)
	- [Training Hooks](#training-hooks-callbacks)
		- [Preexisting](#preexisting)
		- [Custom](#custom)
	- [Regularization](#regularization)
		- [Data Augmentation](#data-augmentation)
		- [Data Filteration](#data-filteration)
		- [Batch Normalization](#batch-normalization)
- [Network Architecture](#network-architecture)
	- [Techniques Used](#techniques-used)
		- [Fully Convolutional Networks](#fully-convolutional-networks)
		- [Skip Connections](#skip-connections)
	- [Components](#components)
		- [Encoder](#encoder)
		- [1x1 Convolution](#1x1-convolution)
		- [Decoder](#decoder)
		- [Output Layer](#output-layer)
	- [Network Depth](#network-depth)
	- [Final Architeture](#final-architecture)
	- [Loss Function](#loss-function)
	- [Optimizer](#optimizer)
- [Results!](#-results)
	- [Validation Loss Graph](#validation-loss-graph)
	- [Network Evaluation](#network-evaluation)
	- [Segmentation Outputs](#segmentation-outputs)
	- [Deploying To The Simulator](#deploying-to-the-simulator)
- [Other Use Cases](#other-use-cases)
	- [Different Classes](#different-classes)
	- [Different Number of Classes](#different-classes)
	- [More Complex Environments](#more-complex-environments)
- [Future Improvements](#future-improvements)
	- [Appropriately Balancing Training Data](#appropriately-balancing-training-data)
	- [Using Inception Layers](#using-inception-layers)
	- [Transposed Convolutions Instead of Biliniear Upsampling](#transposed-convolutions-instead-of-bilinear-upsampling)
	- [TensorBoard for Exhaustive Hyperparemter Tuning](#tensorboard-for-exhaustive-hyperparameter-tuning)
- [References](#references)

## Overview
This is an Image Segmentation project built as part of Udacity's 'Robotics Nanodegree Term 1' curriculum. It involves training a deep neural network using a Fully Convolutional Network, as well as various other mechanisms - such as Skip Connections, 1x1 Convolutions etc - to detect a person-of-interest from images captured by a Follow-Me drone, the purpose eventually being to be able to guide a drone to follow-along with that person as they go jogging, walking etc. It was [evaluated](https://review.udacity.com/#!/rubrics/1155/view) based on its IoU (Intersection-over-Union) performance on a provided test set, but can be deployed to the simulated drone to validate its effectiveness as well.

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

## Environment and Setup

### AWS

Training was done on a p2.xlarge (GPU-Compute) EC2 instance on AWS, which performed brilliantly, achieving a 10x-15x reduction in training time. The _Udacity Robotics Deep Learning Laboratory (ami-e4fd199e) AMI_ was used, as shown below:

![AWS AMI](https://github.com/safdark/ROBO-followme-project/blob/master/docs/images/aws-ami.png)

During training, I was able to max out the GPU usage on this machine as well, as suggested by this output from the command-line ```nvidia-smi``` command:

<img src="https://github.com/safdark/ROBO-followme-project/blob/master/docs/images/gpu-utilization.png" width="900" height="250">

One issue I faced was with getting the plotting of the keras model to work. [This youtube video](https://www.youtube.com/watch?v=8f2lOxsCDHM) helped explain how to resolve that problem.

### Jupyter Notebook Server

On an EC2 instance (as mentioned above), the command to launch the Jupyter server was:
```
    jupyter notebook --ip='*' --port=8888 --no-browser
```

No conda environment needs to be activated to launch jupyter. Just cd to the repository and launch this command. The AMI above has all the necessary libraries pre-installed without needing conda.

### Local Machine Setup

The local machine setup requires following [these instructions](https://github.com/udacity/RoboND-Python-StarterKit/blob/master/doc/configure_via_anaconda.md).

After finishing those setup steps, invoke the following on the command line:
```
source activate RoboND
pip install tensorflow==1.2.1
pip install socketIO-client
pip install transforms3d
pip install PyQt5
pip install pyqtgraph
```

The environment should now be ready to launch the jupyter notebook locally, for which just launch:
```
jupyter notebook
```

See the [deployment](#deploying-to-the-simulator) section for information on running the simulator.

## Data

### Inputs

The network would be fed 160x160x3-pixel images (scenes) of a simulated world, as seen from the point of view of a drone that would, hypothetically speaking, continuously detect a designated person-of-interest (aka 'hero') within that scene and follow behind her. An example of such a scene captured from the drone's camera is illustrated at the top of this document.

There are three types of images included as training data:
- _Images containing the hero nearby_: For incremental adjustments of the drone's guidance system
- _Images containing the hero far away_: For helping the drone get back-on-track in case it falls behind
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

The drone then uses the CoG of the blue pixels in the segmented output image to follow behind the hero.

### Simulator

A simulator was provided with the project to merely collect 'real-world' data in the event that the data included with the project was not sufficient. This was not used in this project however, as the provided data proved sufficient to satisfy the [rubric](https://review.udacity.com/#!/rubrics/1155/view).

## Training

The immediate goal of this project was to satisfy the 40% IoU metric required, which is the yardstick I used when evaluating various hyper parameters during training. Therefore, it is worth noting that I did not explore optimizations to the network beyond what was needed to reach the requisite IoU score. Possible optimizations are discussed at the end of this document. Furthermore, I tuned the hyperparameters manually, though I could have used tools like TensorBoard to compare various permutations of hyperparameters.

### Hyperparameter Tuning

Thanks to the [p2.xlarge instance on AWS](#aws), I was able to pretty quickly explore various hyperparameters and network architectures, and settled on these common hyperparameters, prior to exploring different network architectures:
```
- Learning Rate = 0.001
- Train Batch Size = 32
- Batches Per Epoch: 130
- Validation Batch Size = 32
- Batches per Validation: 25
```

In the following sections, I discuss the reasoning for the selection of the various 'winning' hyperparameters.

#### Learning Rate

I started with a learning rate of 0.01, which was too high because the training and validation losses fluctuated a lot during training. I reduced that to 0.001, which proved to be sufficient to produce a smooth asymptotic training loss curve, and a reasonably smooth validation loss curve too, before stopping the training.

Here is a side-by-side illustration of the validation loss for the two learning rates used (all else being equal). Left is with 0.01, right with 0.001. Notice the sawtooth-like curve with the 0.01 learning rate.

<div>
	<img src="https://github.com/safdark/ROBO-followme-project/blob/master/docs/images/lr_0_01_val_loss_history.png" width="400" height="200">
	<img src="https://github.com/safdark/ROBO-followme-project/blob/master/docs/images/take2-val-loss-history-plot.png" width="400" height="200">
</div>

#### Batch Size
On my local machine, where I started the training attempts, a batch size of 32 seemed more appropriate given the memory constraints of the system. I could theoretically try increasing the batch size when I moved over to AWS, but I did not attempt that since the performance was satisfactory with the chosen batchsize.

#### Batches Per Epoch

With a batch size of 32, and a total training set size of 4131, there would need to be 160 batches per epoch. I chose one epoch to be the process of going through the entire training set once through.

#### Number of Epochs

Training using a learning rate of 0.001, and the chosen [network depth](#network-depth) would appear to largely saturate after around ~15 epochs, but consistently fell short of the target IoU metric of 40%. However, if left to train for ~50-60 epochs, there was still sufficient improvement (though infinitesimally small) with each additional epoch for the network to satisfy the >40% IoU metric. Running the [network](#network-depth) for ~60 epochs yielded an IoU of 46.65!

### Training Hooks (Callbacks)

#### Preexisting
Here are some callbacks I used to simplify the training bookkeeping:
- _keras.callbacks.ModelCheckpoint_: To checkpoint the model after an epoch only if val_loss improves.
- _keras.callbacks.EarlyStopping_: To stop the training if a certain number of epochs have passed without any improvement in val_loss

#### Custom
Here is a custom callback that was implemented for special handling at the end of an epoch
- _plotting_tools.LoggerPlotter_: At the end of every epoch, this plots a graph of the val_loss history before that epoch. It also saves that plot, and the model weights, in a folder created for that particular training run, which helps with post-mortem analysis of different runs.

### Regularization

As a result of the following regularizations used, even the deepest network - with 5 layers and filter depths ranging from 32 to 512 - did not seem to overfit to the training set, and achieved . This was likely because of batch normalization being applied after every convolution step in the network. I also added regularization by [augmenting](#data-augmentation) the data.

#### Data Augmentation

Through the data_iterator.BatchIterator() class, I augmented the data as follows:
- Random (0.5) horizontal flipping of input images, as a form or regularization

#### Data Filteration

Filteration was also necessary for this data, in order to balance the training data across the 3 types of images (hero close by, hero far away, and no hero). The provided data had only 37% images with the hero in it, and 63% without. To balance this, I implemented a randomness-based filter that, given a number between 0 and 1, would filter out that percentage of all non-hero images, bringing down the count of those images (to ~44% of the total training set). Note that this however is not optimal. A more optimal filtering approach is discussed [here](#appropriately-balancing-training-data).

#### Batch Normalization

Batch Normalization was applied to the output of every convolution step in the network (whether it was a regular convolution or a separable convolution) by normalizing the outputs of that convolution across each mini-batch.

## Network Architecture

### Techniques Used

#### Fully Convolutional Networks

This segmentation network is a _Fully Convolutional_ network. Fully Convolutional networks are well suited for segmentation tasks because in such networks, spatial information is processed through the network, and is essential for the generation of the new (segmented) image that gets output by the network. The network goes through an _encoding_ phase, wherein finer grained features are extracted from the input image as it gets progressively 'squeezed'. Then, when all the relevant information is held as fine grained features, the _decoding_ phase generates the image step by step in the same fashion as it was deconstructed in the encoding phase. In the case of merely an object classification network, the deep set of features available at the end of the encoding phase can be fed into a _Fully Connected_ network which spits out categorical information that is devoid of any spatial information at that point. Contrary to this, in the decoding phase of a Fully Convolutional Network, that spatial information is _retained_ through the rest of the network, through a progressive sequence of Convolutional layers rather than Fully Connected layers.

#### Skip Connections

One disadvantage of the decoding phase in a fully convolutional network is that we _do lose some_ of the coarser grained features (the 'big picture'), since we are progressively trading image size for finer grained features. This drawback is resolved by using a technique called _Skip Connections_. Skip connections allow us to retain the coarser grained features from prior layers and use them in conjunction with the finer grained features while recreating an image from subsequent layers.

### Components

We now dive deeper into the specific components of the network.

#### Encoder

The encoder layers gradually reduce the size of each feature tensor passing through the network, while increasing the number of features it extracts at each layer.

#### 1x1 Convolution

1x1 convolutions are useful in the following scenarios:
- A 1x1 convolution in the middle adds a layer of non-linearity to the network before the decoder starts. It can also serve to increase or decrease the number of features extracted from the last layer of the encoder.
- One would use a 1x1 convolution to reduce the number of trainable parameters, for convolutions with large sized-kernels, for example, by reducing the dimensionality of input tensor before it is fed into the regular convolution. This dimensionality reduction, in practice, does not impact the accuracy or trainability of the resulting convolution step.

In this project, the 1x1 convolution used between the encoder and decoder serves only the purpose of adding a layer of non-linearity to the network to improve trainability. However, it is not being used for either reducing or increasing dimensionality at this point, as should be apparent from the [network model](#final-architecture).

#### Decoder

Understandably, the decoding layer does the opposite of the encoder -- converts smaller feature tensors into larger ones, while reducing the feature count at the same time. The decoder layer then outputs a softmax activation for each pixel across the number of classes being segmented out of the original image, which essentially ends up being an image of the same X and Y dimensions, and possibly a different set of channels.

To produce larger feature tensors, each decoding layer includes a _Bilinear Upsampling layer_ (doubling the image size in both x and y dimensions), followed then by a concatenation of _Skip Connections_ from its corresponding encoding layer with the same feature tensor shape, followed then by 2 _Separable Convolution Layers_ to add non-linearity and more nuanced feature-of-feature extraction.

#### Output Layer

In the case of the network in question, the output of the image is a 160x160x3 tensor retrieved through a _convolution layer_ followed by a softmax activation. It so happens that the 3 classes above could be trivially paired with the 3 RGB channels, which is why the 3 types of scenes (hero close, hero far, and no hero) were given exact equivalence to the RGB color channels -- i.e, RGB channel values of [0,0,255], [0,255,0] or [255,0,0]. Had there been more classes of objects to be detected (say 'n'), a separate conversion step (for example, classes 1 and 3 with RGB channel mappings of [0,128, 255] and [15, 180, 150] respectvively) would have been needed at the output layer, to convert the 'n' softmax activations (across n-classes) into an appropriate distribution over the 3 RGB channels so as to produce a discernable segmentation output.

### Network Depth

Depth made a significant impact to the IoU scores achieved by the network. A shallower network didn't seem to have sufficient variance to be able to closely approximate the data generating process, unless the filter depth of each layer was significantly increased. However, doing that immediately put a strain on the resources available (even on an AWS p2.xlarge instance). This was because of the explosion in the number of parameters being learned, which not only put a strain on memory but also on performance. I could have worked around this by using [1x1 convolutions for dimensionality reduction](#using-inception-layers) at each of the two layers, but I chose instead to keep it simple for now, and increase the network depth to 4, and subsequently to 5 layers.

Depths explored:
- Depth-of-4:
	- 4 encoding and 4 decoding layers
	- 2 separable convolutions after each bilinear upsampling
	- filter depths ranging between 32 and 256, depending on the layer (later layers had deeper filters)
	- IoU Achieved: 41.6%
- Depth-of-5:
	- 5 encoding and 5 decoding layers
	- 3 separable convolutions after each bilinear upsampling
	- filter depths ranging between 32 and 512, depending on the layer (later layers had deeper filters)
	- IoU Achieved: 46.65%

### Final Architecture
The final architecture, with Depth-of-5, is illustrated here:
![Final Architecture](https://github.com/safdark/ROBO-followme-project/blob/master/docs/images/take3-model.png)

### Loss Function

The loss function was the Cross-Entropy Loss over the output Softmax Activation layer. Since the expected y-value of a 3-class segmentation for a given pixel would be one of [0,0,1], [0,1,0] or [1,0,0], for this problem space, the closer the softmax activation would be to these values, obviously the lower the cross entropy loss. The resulting loss calculated for the entire segmented image would be the average cross-entropy loss of the softmax activation vector over all of the pixels in that image. Typical cross-entropy loss values for the validation set by a trained network were seen to be in the 0.01 - 0.04 range. Values above 0.05 would almost certainly not achieve the desired IoU score on the test set.

### Optimizer

I started with an _Nadam_ optimizer based on input from fellow students on Slack (particularly as a form of regularization), but later settled on an _Adam_ optimizer after witnessing significantly better performance of the latter compared to the former, on the Depth-of-5 architecture.

## Results

The [network](network-depth) hit optimal performance on the test set around epoch # 64, with an IoU of:
```
          46.65%!
```

Here are the optimally trained and loadable [weights file](https://github.com/safdark/ROBO-followme-project/blob/master/data/weights/weights.hd5) and [architecture file](https://github.com/safdark/ROBO-followme-project/blob/master/data/weights/architecture.json) files for this network. Please use the load_architecture() API in the [model_tools.py file](https://github.com/safdark/ROBO-followme-project/blob/master/code/utils/model_tools.py) to load the architecture file. The weights file can be loaded using the model.load_weights() API once you've constructed the model from the load_architecture() API.

### Validation Loss Graph
Here is the graph of the validation loss seen at the end of every epoch, until epoch #64.

<img src="https://github.com/safdark/ROBO-followme-project/blob/master/docs/images/take3-val-loss-history-plot.png" width="850" height="300">

### Network Evaluation
![Take3 - IoU Evaluation](https://github.com/safdark/ROBO-followme-project/blob/master/docs/images/take3-evaluation.png)

### Segmentation Outputs

Here are the segmentation outputs of this network. From left to right - hero close by, no hero (patrol mode), and hero very far:

<div>
	<img src="https://github.com/safdark/ROBO-followme-project/blob/master/docs/images/take3-hero-close.png" width="290" height="290">
	<img src="https://github.com/safdark/ROBO-followme-project/blob/master/docs/images/take3-no-hero.png" width="290" height="290">
	<img src="https://github.com/safdark/ROBO-followme-project/blob/master/docs/images/take3-hero-far.png" width="290" height="290">
</div>

### Deploying To The Simulator

Please follow these steps to deploy the trained [model](https://github.com/safdark/ROBO-followme-project/blob/master/data/weights/architecture.json) and [weights](https://github.com/safdark/ROBO-followme-project/blob/master/data/weights/weights.hd5) files to the simulator:

```
1. Launch the Quad Simulator
2. Check the "Spawn people" checkbox and then click on "Follow Me"
3. From the command line, activate the RoboND conda [environment](#local-machine-setup)
4. Navigate to the root of this repository, then cd to the code folder.
5. Launch:
   	python follower.py ../data/weights/architecture.json ../data/weights/weights.hd5 --pred_viz
6. Switch back to the Quad Simulator and watch the drone eventually find the hero and start following behind her.
```

I will post a video of the simulated drone on youtube soon.

## Other Use Cases

This model, generally, would work with any kind of image segmentation task provided sufficient training data relevant to the problem being addressed.

### Different Classes
The model in this case was trained with specific data pertaining to human beings - a hero and others - walking in a city-like environment (with roads, bridges, buildings, lawns etc). In other words, the model weights have been specifically tuned to recognize features pertaining to the categories that it was trained to recognize using the training data provided. As examples, the model probably uses the color of the clothing to distinguish between the hero and other people. Therefore, this trained model, as it stands, will not work for segmenting other classes of objects (like cats, dogs etc), or different environments. In fact, it will also fail if the test data were to present the hero in a different set of clothes.

The model has learned features specific to the domain for which it was trained. Therefore, if provided sufficient training data for other domains, with 3 classes of objects in the segmentation, this model is likely to segment those images well too. However, there are some limitations there too ...

### Different Number of Classes
One such limitation is that the model has been architected to address 3 classes of objects in this segmentation problem. If provided sufficient data, the model might still be trainable on a minor increase or decrease in the number of categories it is to classify pixels into. But, the model may not be able to train well for a vastly larger count of categories in the segmentation task, possibly because of higher than requisite bias in those cases.

### More Complex Environments
I also expect that the network might need to be deeper, or use other optimizations/tweaks to handle more complex environments. In other words, it could be possible to generate an input image where the objects being classified require the network to learn more discernable features than it is presently capable of learning. In such situations, this network will not be optimally trainable.

## Future Improvements

To improve upon this network further, here are some additional tasks to be attempted.

### Appropriately Balancing Training Data

I have achieved the requisite [rubric](https://review.udacity.com/#!/rubrics/1155/view) for this project by using the provided data itself, without any additional data generated from the simulator. However, collecting more data would likely be necessary to push the IoU higher for this project.

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

### Using Inception Layers

I have mostly focused on a kernel of size 3 in this project, since a kernel of size 5 was not very effective for this problem. However, as a generic model, it would make more sense to use a mix of different kernel sizes in each of the convolution steps because that might help the network find a wider range of features and not only improve accuracy for the given network but more importantly for a wider range of classes.

The degradation in performance from adding larger kernels can also be worked around by inserting a 1x1 convolution before each kernel, to reduce the dimensionality prior to convolution with that kernel thereby reducing the number of trainable parameters. The inception network does precisely that. If each encoder layer were an inception layer, combining kernels of sizes 3, 5 and 7, in addition to a 1x1 convolution and a maxpooling step, we could achieve better accuracy while also minimizing the associated performance penalty.

### Transposed Convolutions Instead of Biliniear Upsampling

In bilinear upsampling, the upsampled image is based purely on the input image. When using Transposed Convolutions, the relation between the input and upsampled image pixels can be learned by the network, thereby not only introducing additional non-linearities in the upsampling but also a more nuanced transformation relation between the pixels in the input and output. But for this same reason, this approach will also impact performance, both during training and also with inference. However, that can potentially be resolved by removing the 'same' padded convolution layers at each decoding layer, after the transposed convolution step. It is not clear to me which approach will produce better accuracy, but it is perhaps worth exploring.

### TensorBoard for Exhaustive Hyperparemter Tuning

I could have used TensorBoard to compare various hyperparameter permutations, but did not on account of time constraints with getting familiar with all the TensorBoard tools and quirks.

I settled on a batch size of 32 since that was appropriate to the memory constraints of my home use laptop. This could potentially have been increased when training was done on the GPU instances on AWS.

## References

No references to call out at the moment.
