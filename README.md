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

![Accuracy of using different inputs](/ImagesUsedForREADME/DifferentInputs.png "Accuracy of using different inputs")

To check the stride parameter in the input convolutional layer. The input layer was modified
to have a stride of 2. This implementation resulted in a network with less parameters than the original one, the original one had 666,006 parameters while the new one had 341,862 parameters.
The performance would be less optimal due to the parameters number, which is the case in this
situation as the highest accuracy recorded was 0.7143 vs 0.7561 recorded in the model without
a stride. This can be attributed to down-sampling, as some of the images’ pixels are skipped.
Thus missing important information.

![Accuracy of using stride vs without](/ImagesUsedForREADME/Stride.png "Accuracy of using stride vs without")

Furthermore, interest has been given to the filters in the convolutional layers and how
they can affect the model. The LeNet5 implementation seen up until this point has a input
convolutional layer with filter 32, and second layer with filter 48. To check whether this had
any impact on the performance a filter of 48 will be used followed by the next layer having a
filter of 64 replacing the previous filters, the input used was 28x28x3. The total parameters of
the new model are 906,470 compared to 850,326 of the previous implementation. The results of
the training indicate a small increase of accuracy, new accuracy is 0.7572 compared to 0.7561.
The new filters seem to indicate a better start in the training but at the end the original filters
got similar results which does not justify the increase in time and computational power needed.

![Accuracy of using old and new filters](/ImagesUsedForREADME/DifferentFilters.png "Accuracy of using old and new filters")

As mentioned in the previous section dropout plays an important role in helping avoiding
overfitting and learning on robust features. It would be interesting to see how it affects the
learning and the overall performance of the network. There is no change in the parameters of
the network but the maximum accuracy achieved is 0.7175 compared to 0.7561 which is a clear difference in performance. Additionally, the time needed to run the network without dropout is larger than running it with dropout making dropout a good choice. Yet, without dropout the
network seems to have better initial performance.

![Accuracy of using dropout vs with no dropout](/ImagesUsedForREADME/NoDropout.png "Accuracy of using dropout vs with no dropout")

Working on the second architecture. Initially, evaluation of the best activation function for
the input layer and for the output layer. The activation functions that will be examined are
ReLu, Sigmoid, Tanh and Softmax.

When it comes to the activation function of the convolutionary layer, the dense layer uses
Softmax. The best performing is ReLu which has the steepest and it also has the least computational
overhead from all the other functions. Second in performance is the Tanh which
initially has less performance than ReLu but eventually catches on it and surpasses it yet, its
seems to overfit quite easily.Softmax is also performing good as well as Sigmoid which was not
performing that good in initial iterations.

For the application of the activation functions in the dense layer the activation function is
ReLu. The best performing activation function seems to be Sigmoid function followed by the
Softmax. Sigmoid achieves the highest performance of 0.6495 followed by Softmax with highest
performance of 0.6461. The other functions have a performance that hovers around 0.1 and as
such they should not be used. Strictly using their performance as an evaluation would result in
selecting the sigmoid, yet due to the sigmoid’s saturation problem, as values reach close either
to 0 or 1, the gradient’s vanish which could hinder the performance of the network. This is not
present in the SoftMax function and as such it is preferred.

Finally, the performance of the baseline network will be evaluated with input 28x28x3 and
with input 32x32x3. This will also include the simplified architecture to see the effects of more
layers. It is worth noting that the simplified architecture has 65,290 parameters which are
significantly less than the other architectures. To ensure the best performance the accuracy of
the validation dataset will be used as well as the loss. Finally, callbacks have been implemented
to detect possible overfitting or plateau of the learning parameter. Once this is detected the
training stops or the learning parameter changes.

Having run the experiment it is observed that the majority of the training did not reach end
of the training period of 25 epochs. The more complex models seem to reach the 13 and 12 epoch
mark before the callbacks stop the training. It is worth noting the the 32x32x3 input model is
performing better than the 28x28x3 model although the performance is within the parameters
that had seen before. More importantly though would be to focus on the val loss graph. This
graph shows the change of the loss value that is used to indicate whether a model is overfitting
or not. From the graph, the best performing is again the model with input 32x32x3 followed
by 28x28x3. What is interesting is that the 32x32x3 model improves significantly faster than
than 28x28x3 although both seem to follow a similar curve, although 32x32x3 seems to further
ahead than 28x28x3. Yet, both terminate with similar values of the best epoch, for 28x28x3
val loss 0.7870 with accuracy 0.7405 and for 32x32x3 val loss 0.7415 and accuracy 0.7559 thus
there is not much of a real difference.

The simplified architecture is observed under-performing significantly compared to the two
previous architectures. Thus showing that it is not ideal to be implemented. It also shows the
importance of the extra layers and how much more performance is improved by adding them.
It is worth noting that the simpler architecture takes longer to reach accuracy and it seems it
struggles to reach accuracy over 0.675. Also, it has a high val loss value meaning that it does
not perform well.

In conclusion, as it has been observed the best performing architecture in terms of performance
is the 32x32x3 although if time is to be considered then 28x28x3 is the best. Increased
filter size does not guarantee increased performance and the use of dropout layers increase the
model’s performance significantly. Finally, in terms of performance the best activation functions
is ReLu and then Softmax at the last layer.

![Metrics of the use of the activation functions](/ImagesUsedForREADME/ActivationConv.png "Metrics of the use of the activation functions")

![Metrics of the use of the activation functions](/ImagesUsedForREADME/ActivationDense.png "Metrics of the use of the activation functions")

![Metrics of the performance of the evaluated networks](/ImagesUsedForREADME/FinalEvaluation.png "Metrics of the performance of the evaluated networks")

![Metrics of the performance of the evaluated networks](/ImagesUsedForREADME/FinalEvaluationLoss.png "Metrics of the performance of the evaluated networks")