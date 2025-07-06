#training non-linear data. Linear data is coded in way easier ways, so lets do this! The non linear data data sets are created using nnfs library which randmly generates data sets which we can use to train models 

#start with 'pip install nnfs'

from nnfs.datasets import spiral_data #this is the data we will be training our model on. Plots on a 2D plane points that look like a spiral galaxy. The spiral dataset lets us generate as many classes and samples as want

import numpy as np
import nnfs
#nnfs.init() should always be the starting. This sets ranodm seed to 0, overrides original dot product from Numpy
nnfs.init()

#create dataset
x,y = spiral_data(samples=100,classes=3)

#lets create the main automation class
class dense_Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons)) #return an array of this shape, filled with zeros. Starting point.
    def forward(self, inputs):
        self.Layer_output = np.dot(inputs, self.weights) + self.biases

#lets create a dense layer
dense1 = dense_Layer(2,3)

dense1.forward(x) #X here is the data we got from the data set
print(dense1.Layer_output)