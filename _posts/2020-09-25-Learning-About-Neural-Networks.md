---
layout: default
title: Learning how to implement a Neural Network from Scratch
---

## Learning to code a neural network with base python and numpy

When I first started learning about neural networks what really stood out was how much I really didn’t understand. That is pretty much how I feel whenever I embark on something new, but it was even more apparent for this topic because a lot of the details had been abstracted away. Keras is an open-source neural-network library. Tensorflow is another library released by Google which is often used alongside Keras, as Keras has the ability to work on top of Tensorflow. Using these tools, a very simple neural network can be implemented within about five or six lines of code, but there is a lot going on behind the scenes that I want to have a better understanding of.

For this project, I was tasked with choosing an algorithm to implement. When using Python, we have many powerful machine learning algorithms that can be imported and used as part of a library. The task was to implement an algorithm using only Python and a math library such as numpy or scipy in order to understand how it works at a lower level. Because I would like to have a better understanding of how the different components within a neural network interact with each other, I chose to implement a very simple one using base Python and numpy.
For the data, I am using the sklearn.datasets library to generate scatter plots. This was very quick and convenient because you can easily change the number of classes to predict. For some datasets, such as 'make moons', you can adjust the amount of noise and make the classes more difficult to differentiate. In the future I would like to use a more realistic dataset to make predictions, such as image classification.

Before getting started I need to cite some of my sources. [Sentdex](https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3) has a great series on how to build neural networks from scratch, and this is where I started. I've also read quite a bit of his e-book which contains a lot of detail about how each function within an ANN works.

The architecture for this neural network is very basic. It is a multi layer perceptron containing an input layer, two dense layers, and an output layer. The data starts at the input layer and is passed forward to the first dense, or hidden layer. Then the data is passed forward to the second hidden layer, and finally the output layer. Once the information has reached the output layer you have a prediction based on which output has the highest value. The layers within a neural network are made up of neurons, which are interconnected units. When the inputs are transmitted between the neurons, weights and bias are then applied to the input (weight * input + bias). Every input has a weight, and every neuron has a bias. Later in the training process the weights and bias are adjusted using backpropagation.

So one of the first functions that has an impact on the weights and biases is the activation function. The activation determines whether or not a neuron will turn on or off. Once the calculation for weights times input plus bias has been made, that value is fed to the activation function which determines the output.The output from the activation function is the input into the next layer. The first activation function I looked at for this project was rectified linear, or ReLU. For ReLU, if x is less than zero the output is zero, however if x is greater than or equal to zero it is equal to the value of x. This condition is very simple and easy to implement. The reason ReLU is very popular is because it deals with the issue of vanishing gradient. This occurs when the gradient is vanishingly small and prevents the weights from changing value. ReLU is often chosen over other activation functions, such as Sigmoid, because it is not differentiable at zero.

![Different Activation Functions](https://miro.medium.com/max/1200/1*ZafDv3VUm60Eh10OeJu1vw.png)

Example of how to implement ReLU:
~~~
output = []

def ReLU(input):
   for i in inputs:
       if i > 0:
           output.append(i)
       else:
           output.append(0)
~~~
For this implementation np.maximum was used, which appends either 0 or the input value:
~~~
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
~~~

The next activation function I looked at was softmax, which is a little more complicated and useful if your output needs to be a probability distribution. In fact softmax is often used as the last activation function in a neural network when predicting multiple classes. For this project I could have stuck with one activation function and chosen a binary classification problem. However, I do want to continue to learn how all of the different activation functions work and adapt this project to a more realistic dataset in the future. Because a Softmax function outputs a probability distribution which will add up to 1, the values are exponentiated. This will return only non-negative values. The values are then normalized by taking a given value and dividing it by the sum of all the values within the distribution.

Example of how to implement Softmax:
~~~
#take exponent and get unnormalized probabilities
values = np.exp(inputs - np.max(inputs))
        
#Normalize the probabilities
probabilities = exp_values / np.sum(exp_values)
~~~

We haven't even gone backwards yet! After each forward pass through the neural network, backpropagation is used to perform a backward pass and adjust the weights and biases. An optimizer is used to minimize loss. I used one of the most common optimizers, stochastic gradient descent. There are many different optimizer functions that can be used in a neural network, and many of them are a variation of stochastic gradient descent. SGD uses a learning rate to adjust the weights and biases over multiple iterations in order to minimize error. The reason Stochastic Gradient Descent is often used over gradient descent is because SGD uses a subset of training samples to update the parameters, as opposed to gradient descent which uses *all* of the sample data, making it less efficient for larger amounts of data. The model is shown one training instance at a time, a prediction is made, and then the error is calclated and the model is updated in order to reduce error for the next prediction.  

A simple implementation of stochastic gradient descent:
~~~
coefficient = coefficient - learning_rate * error * input_value
~~~

The learning rate is the amount that the weights are updated during training. It is an important hyperparameter that needs to be tuned for a neural networks and an important part of stochastic gradient descent. A smaller learning rate will require more training epochs which will make the model adapt more slowly. A larger learning rate will require less training epochs, however this could cuase the model to converge too quickly. Learning rate decay is a technique which begins with a larger learning rate which will decay over multiple iterations. The initially larger learning rate helps to accelerate training and escape local minima.

~~~
class Optimizer_SGD:
    #larger default learning rate of 1
    def __init(self, learning_rate=1.0, decay=1e-3):
        self.learning_rate = learning_rate
        self.decay = decay
        self.iter = 0
       
    def decay(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * 
            ((1./(1. + self.decay*self.iter))
    
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases
~~~

The loss function is an algorithm that calculates the model's error, or how wrong the model is. Loss functions are not only used in neural networks, but in machine learning algorithms in general. For example, mean squared error loss is commonly used for linear regression. In a multi-class classification task, categorical crossentropy is used. It calculates the difference between two probability distributions. Because we are using a softmax activation function, it also makes sense to use a categorical crossentropy function for loss.

![Categorical crossentropy diagram found on peltarion](https://peltarion.com/static/categorical_crossentropy_setup.svg)

~~~
An example of categorical crossentropy:
softmax_output = [.5, .3, .2]
target = [0, 1, 0]

loss = -(np.log(softmax_output[0]) * target[0] +
         np.log(softmax_output[1]) * target[1] +
         np.log(softmax_output[2]) * target[2])
~~~



## What I'd like to learn more about

In the future, I plan to continue to learn how the different algorithms within a NN work. I'm looking forward to learning how the different parameters work, and how to adapt them to different problems. I would also like to be able to compare how these different parameters work within Tensorflow and Keras to what I have learned so far.
