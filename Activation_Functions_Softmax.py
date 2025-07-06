import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

#creating Dense layer 
class Dense_Layer:
    def __init__(self, n_inputs, n_neurons):
        self.n_weights = 0.01* np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))
    
    #forward layer pass
    def forward(self, inputs):
        #calsulate output from inpurs, weights and biases
        self.Layer_output = np.dot(inputs, self.n_weights) + self.biases

#ReLU Activation
class ReLu:
    def forward(self, input):
        self.output = np.maximum(0, input) #will return 0 for negative outputs. 

#Softmax activation
class Softmax_activation:
    def forward(self, inputs):
        #get unnormalised probablitlies
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        #normalize them for each sample
        probabilities = exp_values/np.sum(exp_values, axis = 1, keepdims=True)
        self.output = probabilities

#create dataset

x,y = spiral_data(samples=100, classes=3)

#creating dense layere with 2 inputs x,y features and 3 output calues 

dense1 = Dense_Layer(2,3)

#creating a RelU activation to be used with Dense1

activation1 = ReLu()

#creating a second dense layer with 3 inputs this time as we take the output from the first layer as input

dense2 = Dense_Layer(3,3)

#create the softmax activation layer to be used as Dense2 is the last hidden layer

softmax = Softmax_activation()
#lets pass samples data though layer 1
dense1.forward(x) #print(dense1.Layer_output)

#make forward pass thorugh activation function. taking output of first dense layer 
activation1.forward(dense1.Layer_output)
print(activation1.output) #with 0 for negative values now!

#input this value into our second layer
dense2.forward(activation1.output)

#firwarding output from layer 2 to the softmax function 
softmax.forward(dense2.Layer_output)
print(softmax.output)