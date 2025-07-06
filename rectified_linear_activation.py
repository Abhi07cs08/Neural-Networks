from nnfs.datasets import spiral_data #this is the data we will be training our model on. Plots on a 2D plane points that look like a spiral galaxy. The spiral dataset lets us generate as many classes and samples as want

import numpy as np
import nnfs
nnfs.init()

#create dataseet
x,y = spiral_data(samples=100,classes=3)

#lets create the main automation class
class dense_Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons)) #return an array of this shape, filled with zeros. Starting point.
    def forward(self, inputs):
        self.Layer_output = np.dot(inputs, self.weights) + self.biases

class ReLU_activation:
    def __init__(self) -> None: #formality
        pass
    def forward(self, inputs):
        self.output = np.maximum(0,inputs) #ReLU: will replace <0 values with 0

#lets create a dense layer
dense1 = dense_Layer(2,3)

dense1.forward(x) #X here is the data we got from the data set

Activation_result = ReLU_activation() #create activation 

Activation_result.forward(dense1.Layer_output)
print(Activation_result.output)