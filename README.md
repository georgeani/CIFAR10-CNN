# CIFAR10-CNN

This is a coursework undertaken for the module Advanced Artificial Intelligence.

It is written in python and utilizes the TensorFlow and Keras libraries.

The goal was to learn and understand on how to build Convolutional Neural Networks (CNNs) using the Keras library.
Focus was given in the accuracy of the network but also to the loss function. Different aspects of the CNN was researched in order to reach a more comprehensive idea on what influences the learning of the network.

The observations can be seen later on in this file.

The base architecture used was the LeNet5 implementation.

The code can be found [here](/Code/CIFAR10-CNN.ipynb).


# Tools and modifications applied

## Tools used

For this coursework I chose to use Keras and Tensorflow as my main libraries. The choice
stems from the easiness of implementing a neural network but also because they are supported
by Google Colab which is the primary tool that I used.

Tensorflow is a Machine Learning library that allows for an easy construction and training
of Neural Networks. It allows for easily training and evaluating the models while not relying
on the underlying hardware, which can be a cloud implementation or a local computer.
Keras is an API that is part of Tensorflow and allows abstraction in the network construction
by providing an automated way to build the layers and pass configuration parameters to them
as Keras acts as a wrapper to the underlying low level libraries.

Google Colab is a free Jupyter Notebook environment that runs on the the cloud providing
computation in order to train and evaluate networks. The environment does not need any set
up to be performed as it is up to date already with the latest libraries.

## Base Network

In terms of datasets used, the CIFAR10 dataset was used. It was downloaded using the
dataset package provided from tensorflow.keras. The dataset is already split into training and
testing datasets. We performed pixel normalization to ensure that no value dominates the
network. The next step was to standardize our images with mean of 1 and variance of 0 and to
resize depending on the input of the input layer. The buffer size was set to equal the size of the
dataset in order to ensure completely random shuffling of the dataset. Batch size was tested for
both 32 and 28 in order to determine the importance of it.

Finally, the model was based on a LeNet5 implementation, the main modification was the
change of the loss function used Sparse Categorical Cross-Entropy instead of Categorical Cross-
Entropy which is used for binary classification and the addition of the BatchNormalisation layer
after the MaxPooling layers. Dropout has also been added between the Densely connected layers
to aid the avoidance of overfitting and improve performance.

## Network Modifications

Another important aspect of a network is the input size. The LeNet5 architecture uses
an input of 28x28x3 which requires down-sampling of the images obtained from the dataset
CIFAR10 which has images of 32x32x3. As such it was an interesting test to see how different
inputs can affect the parameters and complexity of the network and its performance and
accuracy.

Strides are the shift of the image over the input matrix. The default value in the 1 which
denotes that the shift of the image over the matrix is 1. Yet, what would the change be if we
changed the stride to 2 in the input layer. Increase in the stride will result in down-sampling. In
our case, this down-sampling would be by 2 as we increase the stride to 2. It would be important
to see how this affects the performance of the model. The model used in this experiment has
input 28x28x3.

Another important feature of the network is to see how the increase of the filter sizes affects
the parameters of the network and how can this impact the performance of the network. Larger
filters results in larger number of activation maps that increase the size of the output. It
is considered that more filters can extract more information from the network and increase
accuracy.

Dropout was included as a modification over the original architecture as it is considered as
a good tool to counter overfitting. Dropout works by randomly disconnecting neurons from the
network. The neuron is then not dependent to other neurons and has to learn more robust
features, thus removing the susceptibility of the model to noise. Up until this point all experiments
have been conducted using dropout. We will be running the experiment using the initial
model, with input 28x28x3 as it is faster and seeing the accuracy of both models.

## Second Architecture
The second architecture will mainly be used to test modifications that would be harder to
be tested in the main architectures. Activation functions will be tested and their combination
for each layer of the network.

Finally, callbacks will be used for the final evaluation of the baseline model with the simplified
architecture. Reduce on LRonPlane will be used in order to reduce the learning rate when
the validation accuracy does not improve more. An early stop callback will be used to avoid
overfitting as it may result in bad results. The callback will allow up to 5 epochs for the
improvement in the error rate before it intervenes and stop the learning.

# Experiments and results

Focusing on the importance of the input area and its effects on the network performance.
This information is important in order to understand the effects of different input sizes in the
network. For this reason we built 2 models one with input shape of 32x32x3 and the other with
28x28x3(baseline model). Initially, the baseline model has 660,008 parameters in total whereas
the modified model has 850,326 parameters indicating that it can adapt closer to the aspects
of what we want it to be. This is an indication that changing the input shape we increase
the complexity of the network. Testing has indicated that the algorithm performs better with
a larger input, but extra time is needed for training. The change in input has resulted in an
accuracy of 0.7572 compared to 0.7561 after 8 epochs. The performance increase is miniscule
and does not justify the increase in time needed. Using smaller images allows the network to
have more generalised information of the image thus leading to an increase in performance.
Additionally the increase in parameters may result in overfitting of the network. Yet, the larger
inputs seems to indicate that the algorithm learns faster although the smaller input catches up
on the larger input and has similar performance. Epochs start from 0 instead of 1.

![Accuracy of using different inputs](/ImagesUsedForREADME/DifferentInputs.png "a title")