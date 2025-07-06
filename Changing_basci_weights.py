#same code as LossFuncAndAccuracy. 
import numpy as np
import nnfs
from nnfs.datasets import vertical_data # a less complicated data sample than spiral data

nnfs.init()

#dense layer
class Dense_Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights  = 0.01*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights)+self.biases

class ReLu_Activation:
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)


class Softmax_Activation: 
    def forward(self, inputs):
        #getting unnormalized probabilities
        self.exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        
        #normalize them 
        self.probabilities = self.exp_values/np.sum(self.exp_values, axis=1, keepdims=True)
        self.output = self.probabilities

class Loss:
    #Calculates the data and regularization losses given model outup and ground truth values
    def calculate(self, output, Real_y):
        #calculate sample losses
        sample_losses = self.forward(output, Real_y) #this method is described in the next class

        data_loss = np.mean(sample_losses) #will calculate means of negative logs
        return data_loss

class Categorical_Cross_Entropy_Loss(Loss):
    def forward(self, output, Real_y):
        #number of samples in batch
        samples = len(output)

        #clip data to prevent the possibility log(0) etc and clip both sides to not drag mean at all
        output_clipped = np.clip(output, 1e-7, 1-1e-7) #1e-7 is the minimum allowed value and the other is a max all
        #print(output_clipped)
        #probabilities for target values
        #for sparse
        if len(Real_y.shape)==1:
            correct_confidences = output_clipped[range(samples), Real_y] #this row, this term
            #No need for range(len(sample)) because samepls is already in int format!

        #for hot-encoded 
        if len(Real_y.shape)==2: 
            correct_confidences = np.sum(output_clipped*Real_y,axis=1) #the sum of all terms row-wise

        #lets get the losses now
        negative_logs = -np.log(correct_confidences)
        return negative_logs

#create dataset

x,y = vertical_data(samples=100, classes=3)

#creating dense layere with 2 inputs x,y features and 3 output calues 

dense1 = Dense_Layer(2,3)

#creating a RelU activation to be used with Dense1

activation1 = ReLu_Activation()

#creating a second dense layer with 3 inputs this time as we take the output from the first layer as input

dense2 = Dense_Layer(3,3)

#create the softmax activation layer to be used as Dense2 is the last hidden layer
softmax = Softmax_Activation()

#lets make a few variables to track the best loss we encountered so far
lowest_loss = 999999#random large initial value
best_dense1_weights = dense1.weights.copy() #created a copy of the weights
best_dense1_biases = dense1.biases.copy() #created a copy of the weights
best_dense2_weights = dense2.weights.copy() #created a copy of the weights
best_dense2_biases = dense2.biases.copy() #created a copy of the weights

#lets do 10000 iterations.
for iteration in range(100000):
    #generate a new set of weights for each iteration
    dense1.weights = 0.05*np.random.randn(2,3)
    dense1.biases = 0.05*np.random.randn(1,3)
    dense2.weights = 0.05*np.random.randn(3,3)
    dense2.biases = 0.05*np.random.randn(1,3)

    #lets pass samples data though layer 1
    dense1.forward(x) #print(dense1.Layer_output)

    #make forward pass thorugh activation function. taking output of first dense layer 
    activation1.forward(dense1.output)
    #print(activation1.output) #with 0 for negative values now!

    #input this value into our second layer
    dense2.forward(activation1.output)

    #forwarding output from layer 2 to the softmax function 
    softmax.forward(dense2.output)
        
    #perfrom forward pass thorugh loss function takes output of seconddense(softmax) and returns loss

    loss_function = Categorical_Cross_Entropy_Loss()
    loss = loss_function.calculate(softmax.output, y)

    #print loss

    #lets add an accuracy function 
    predictions = np.argmax(softmax.output, axis=1) #picks the prediced target per row

    #for one_hot_encoded
    if len(y.shape) == 2: #y are the true values
        y = np.argmax(y,axis=1, keepdims=True) #the 1 among the zeros to be choses 

    accuracy = np.mean(predictions==y)
    #if loss is smaller, print and save weights and biases aside. 

    if loss<lowest_loss:
        print(f"new set of weights found, iteration: {iteration}, loss: {loss}, accuracy: {accuracy}")
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss