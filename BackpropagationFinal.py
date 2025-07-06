# this is the backpropagation based code that will cal culate the gradients of the outputs wuith respenct to the weights, inputs, and biases. 

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

#lets make the dense layer
class dense_layer:

    #layer init
    def __init__(self, n_inputs, n_neurons):
        #init weights and biases
        self.weights = 0.01*np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))

    #forward pass
    def forward(self, inputs):
        self.inputs = inputs

        self.output = np.dot(inputs, self.weights)+self.biases

    #backward pass
    def backward(self, dvalues):
        #the gradients of the parameters:
        self.dweights = np.dot(self.inputs.T, dvalues) #because dL/dW = (dL/dY) * (dY/dW)
        self.dbiases = np.sum(dvalues, axis = 0, keepdims=True) #because same  bias for each neuron for that sample
        #gradinet of values
        self.dinputs = np.dot(dvalues, self.weights.T) #very simple math begind these


#ReLU Activation
class ReLU:
    #forward pass
    def forward(self, inputs):
        #remeber input values
        self.inputs = inputs

        #calculate output
        self.output = np.maximum(0, inputs)

    #backward pass
    def backward(self, dvalues):
        #since we have to modify original dinputs, lets make a copy of dvalues first
        self.dinputs = dvalues.copy()

        #zero grad where negative values
        self.dinputs[self.inputs<=0] = 0


#softmax activation 
class softmax_activation:

    #forward pass
    def forward(self, inputs):
        #initialize inputs
        self.inputs = inputs

        #get unnormalized probabilities 
        exp_values = np.exp(inputs-np.max(inputs, axis=1, keepdims=True))

        #normalize then for each sample
        probabilities = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    #backward pass
    def backward(self, dvalues):
        #initialize empty arrays
        self.dinputs = np.empty_like(dvalues)

        #enumerate each output and its gradient
        for index, (single_output, single_dvalue) in enumerate(zip(self.output, dvalues)):
            #jacobian matrix time
            #flatten output array
            single_output.reshape(-1,1)

            #now lets caluclate the jacobian matrix
            jacobian_matrix = np.diagflat(single_output)  - np.dot(single_output, single_output.T)

            #calculate the gradients sample wise and then add them to dinputs
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalue) #final output of the backprop layer. 

#common loss class
class Loss:
    #this caluclates the losses given predicted and true values
    def calculate_Loss(self, output, y):
        #calculate sample Losses
        sample_losses = self.forward(output, y)

        #calualate the mean loss and then return it
        data_loss = np.mean(sample_losses)

        return data_loss

class Loss_categoricalCrossEntropy(Loss):

    #forward pass
    def forward(self, y_pred, y_true):
        #number of sampes in a batch
        samples = len(y_pred)

        #clip data samples to prevent log 0 and from both sides to prevent problems with the mean
        clipped_data = np.clip(y_pred, 1e-7, 1-1e-7)
        
        #probabilities for target values only if categorical
        if (len(y_true.shape) == 1):
            correct_confidences = clipped_data[
                range(samples), y_true #runs a for loop using just commas.
            ]

        #mask values only for one-hot encoded
        elif (len(y_true.shape)==2):
        #     y_true = np.argmax(y_true*y_pred, axis=1) cannot do this as the max class will not always be the correct class
            correct_confidences = np.sum(y_pred * y_true, axis = 1)
            
        #losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
        
    #backward pass
    def backward(self, dvalues, y_true):
        #number of samples 
        samples = len(dvalues)
        #number of labels in every sample: We'll be using thefirst sample to count them
        lables = len(samples[0])
        
        #if y_true values are sparse, turn them into one hot vectors
        if (y_true.shape==1):
            y_true = np.eye(lables)[y_true] #makes a list with columns(lables) and with values with values of 1 at the index of y_pred
        
        #calculate gradients 
        self.dinputs = -dvalues/y_true #differential of log obtained from loss gradient wrt softmax output
        #bomalize the gradient
        self.dinputs = self.dinputs/samples

#LETS MAKE THE ULTIMATE FINAL STEP: a softmax classifier: a combined softmax axtivation and cross entropy loss fir faster backward step
class Activation_Softmax_Loss_Categorical_Cross_Entropy():
    
    #init creates activation and loss function objects
    def __init__(self):
        self.activation = softmax_activation()
        self.Loss = Loss_categoricalCrossEntropy()

    #forward pass
    def forward(self, inputs, y_true):
        #output layers activation function
        self.activation.forward(inputs)
        #set the output
        self.output = self.activation.output
        #calculate and reutn loss value
        return self.Loss.calculate_Loss(self.output, y_true)
    
    #backward pass
    def backward(self, dvalues, y_true):

        #number of samples
        samples = len(dvalues)

        #here, if one hot, turn to discrete
        if (y_true.shape==2):
            y_true = np.argmax(y_true, axis = 1)

        #make a cope for safe modification
        self.dinputs = dvalues.copy()
        #calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        #norma;ize gradient
        self.dinputs = self.dinputs/samples

#done with classes!
#lets create a dataset

x,y = spiral_data(samples=100,classes=3)

#create dense layer with 2 input features and 3 output values
dense1 = dense_layer(2,3)
#create ReLU to be used with dense layer
activation1 = ReLU()

#secibd dense layer with 3 inputs feturs and 3 output values
dense2 = dense_layer(3,3)

#create combined softmax loss 
loss_activation = Activation_Softmax_Loss_Categorical_Cross_Entropy()

#perform a forward pass of our data thoruhg this layer
dense1.forward(x)
#perform forward pass though activation
activation1.forward(dense1.output) #takes computed output from dense1

#oerform forward pass thorugh our second dense layer and takes outputs from activatio1 as inpits
dense2.forward(activation1.output) #(3,3) because the dense1 output a=has 3 colums because 3 classes

#perform a pass tghough our combined layer taking the output of the second dense layer to return the loss
loss = loss_activation.forward(dense2.output, y)

#lets see the output of the first few samples hereL
print(f"Sample Loss: {loss_activation.output[:5]}")

#print loss value
print("loss: ", loss)

#calculate the predictions
predictions = np.argmax(loss_activation.output, axis = 1)

if len(y.shape)==2: #y is the output class need to have it discrete
    y = np.argmax(y, axis=1)

accuracy = np.mean(predictions==y) #again, its a lot like a mini for loop

print(f"accuracy: {accuracy}")

#forward passes done 

#backward passes:

#initialize the backward passes
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

#print the gradients 
print(dense1.dweights)
print(dense1.dbiases)

print(dense2.dweights)
print(dense2.dbiases)