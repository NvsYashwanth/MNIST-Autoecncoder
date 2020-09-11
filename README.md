# MNIST Image reconstruction using Autoencoders
[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://github.com/NvsYashwanth)

![](https://badgen.net/badge/Code/Python/blue?icon=https://simpleicons.org/icons/python.svg&labelColor=cyan&label)        ![](https://badgen.net/badge/Library/Pytorch/blue?icon=https://simpleicons.org/icons/pytorch.svg&labelColor=cyan&label)       ![](https://badgen.net/badge/Tools/pandas/blue?icon=https://simpleicons.org/icons/pandas.svg&labelColor=cyan&label)       ![](https://badgen.net/badge/Tools/numpy/blue?icon=https://upload.wikimedia.org/wikipedia/commons/1/1a/NumPy_logo.svg&labelColor=cyan&label)        ![](https://badgen.net/badge/Tools/matplotlib/blue?icon=https://upload.wikimedia.org/wikipedia/en/5/56/Matplotlib_logo.svg&labelColor=cyan&label)
## Autoencoders
![](https://github.com/NvsYashwanth/MNIST-Autoecncoder/blob/master/assets/autoencoder.png)
* With autoencoders, we pass input data through an encoder that makes a compressed representation of the input. Then, this representation is passed through a decoder to reconstruct the input data. Generally the encoder and decoder will be built with neural networks, then trained on example data.
* A compressed representation can be great for saving and sharing any kind of data in a way that is more efficient than storing raw data. In practice, the compressed representation often holds key information about an input image and we can use it for denoising images or oher kinds of reconstruction and transformation!

## MNIST dataset

The `MNIST` database is available at http://yann.lecun.com/exdb/mnist/

The `MNIST` database is a dataset of handwritten digits. It has 60,000 training
samples, and 10,000 test samples. Each image is represented by 28x28 pixels, each
containing a value 0 - 255 with its grayscale value.

<p align='left'>
<img src ='https://github.com/NvsYashwanth/MNIST-Handwritten-Digits-Recognition/blob/master/images/samples.png'>
</p>

It is a subset of a larger set available from NIST.
The digits have been size-normalized and centered in a fixed-size image.

Thanks to Yann LeCun, Corinna Cortes, Christopher J.C. Burges.


## Results
***`A validation dataset of size 12,000 was deduced from the Training dataset with its size being changed to 48,000. We train the following models for 20 epochs.`***

### Prarameters Initialization for FFNN
* Feed Forward Neural Netword model has been initialized with random weights sampled from a normal distribution and bias with 0.
* These parameters have been intialized only for the Linear layers present in both of the models.
* If `n` represents number of nodes in a Linear Layer, then weights are given as a sample of normal distribution in the range `(0,y)`. Here `y` represents standard deviation calculated as `y=1.0/sqrt(n)`
* Normal distribution is chosen since the probability of choosing a set of weights closer to zero in the distribution is more than that of the higher values. Unlike in Uniform distribution where probability of choosing any value is equal.

### Model -1: FFNN Autoencoder
<p align='center'>
<img src ='https://github.com/NvsYashwanth/MNIST-Autoecncoder/blob/master/assets/simple_autoencoder.png'>
</p>

* We'll train an autoencoder with MNIST images by flattening them into 784 length vectors. The images from this dataset are already normalized such that the values are between 0 and 1. Let's start by building a simple autoencoder. The encoder and decoder should be made of one linear layer. The units that connect the encoder and decoder will be the compressed representation.
* Since the images are normalized between 0 and 1, we need to use a sigmoid activation on the output layer to get values that match this input value range.

### Model -2: Transpose CNN Autoencoder
* The decoder needs to convert from a narrow representation to a wide, reconstructed image. For example, the representation could be a 7x7x4 max-pool layer. This is the output of the encoder, but also the input to the decoder. We want to get a 28x28x1 image out from the decoder so we need to work our way back up from the compressed representation. A schematic of the network is shown below.

<p align='center'>
<img src ='https://github.com/NvsYashwanth/MNIST-Autoecncoder/blob/master/assets/tran_conv.png'>
</p>

* Here our final encoder layer has size 7x7x4 = 196. The original images have size 28x28 = 784, so the encoded vector is 25% the size of the original image. These are just suggested sizes for each of the layers.

* This decoder uses transposed convolutional layers to increase the width and height of the input layers. They work almost exactly the same as convolutional layers, but in reverse. A stride in the input layer results in a larger stride in the transposed convolution layer. For example, if you have a 3x3 kernel, a 3x3 patch in the input layer will be reduced to one unit in a convolutional layer. Comparatively, one unit in the input layer will be expanded to a 3x3 path in a transposed convolution layer. PyTorch provides us with an easy way to create the layers, nn.ConvTranspose2d.
* It is important to note that transpose convolution layers can lead to artifacts in the final images, such as checkerboard patterns. This is due to overlap in the kernels which can be avoided by setting the stride and kernel size equal. In this Distill article from Augustus Odena, et al, the authors show that these checkerboard artifacts can be avoided by resizing the layers using nearest neighbor or bilinear interpolation (upsampling) followed by a convolutional layer.

### Model -3: Upsampled CNN Autoencoder
<p align='center'>
<img src ='https://github.com/NvsYashwanth/MNIST-Autoecncoder/blob/master/assets/up_conv.png'>
</p>

* This decoder uses a combination of nearest-neighbor upsampling and normal convolutional layers to increase the width and height of the input layers.
* It is important to note that transpose convolution layers can lead to artifacts in the final images, such as checkerboard patterns. This is due to overlap in the kernels which can be avoided by setting the stride and kernel size equal. In this Distill article from Augustus Odena, et al, the authors show that these checkerboard artifacts can be avoided by resizing the layers using nearest neighbor or bilinear interpolation (upsampling) followed by a convolutional layer. This is the approach we take, here.

### Why MSE?
* We're comparing pixel values in input and output images, it will be best to use a loss that is meant for a regression task. Regression is all about comparing quantities rather than probabilistic values. So, in this case, I'll use MSELoss. And compare output images and input images as follows:

```loss = criterion(outputs, images)```
