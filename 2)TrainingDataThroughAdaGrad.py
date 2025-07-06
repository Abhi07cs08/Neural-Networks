# AdaGrad, instead of a global LR, lets each parametr have its own learning rate. Important for unupdated weights and neurons to stay activated

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






#NEW OPTIMIZER CLASS!!
class Optimizer_Adagrad():
    #initialize optimizer by setting a primary rate, which defaults to one
    # a constant learning rate is not ideal. hence, let's use constant decay function to lower the learning rate
    def __init__(self, learning_rate = 1.0, decay = 0, epsilon = 1e-7):
        self.learning_rate = learning_rate
        self.current_Learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
   
    #lets create a pre param update method so that if we have a non zero decay, our current learning rate will update
    def pre_update_param(self):
        #only activate if decay is activated
        if self.decay: #if decau is not set to zero basically if self.decay == True or 1
          self.current_Learning_rate = self.learning_rate * (1/(1+(self.decay*self.iterations)))

    #update parameters 
    def Params_update(self, layer):
        
        if not hasattr(layer, 'weights_cache'):
            layer.weights_cache = np.zeros_like(layer.weights)
            layer.biases_cache = np.zeros_like(layer.biases)

        #update cache with current squared gradients
        layer.weights_cache += layer.dweights**2
        layer.biases_cache += layer.dbiases**2

        layer.weights += -self.current_Learning_rate * layer.dweights / (np.sqrt(layer.weights_cache)+self.epsilon)
        layer.biases += -self.current_Learning_rate * layer.dbiases / (np.sqrt(layer.biases_cache)+self.epsilon)
       
        #the correct class, though jacobian matrix calc is given a negative gradient, because of which thar wieght increases while the others fall in quantity
    
    
    #call once after any paramteter was updated
    def Post_param_update(self):
        self.iterations+=1




#done with classes!
#lets create a dataset

x,y = spiral_data(samples=100,classes=3)

#create dense layer with 2 input features(x,y) and 64 output values, making a 1x64 hidden layer
dense1 = dense_layer(2,64)
#create ReLU to be used with dense layer
activation1 = ReLU()

#secibd dense layer with 64 inputs(as we take in outputs from the previous layers) and feturs and 3 output values(classes)
dense2 = dense_layer(64,3)

#create combined softmax loss 
loss_activation = Activation_Softmax_Loss_Categorical_Cross_Entropy()




#CREATE A NEW OPTIMIZER CLASS
optimizer = Optimizer_Adagrad(decay=1e-4)


#now, intead of doing the loop once, lets train our model in a loop

for epoch in range(10001):
    #perform a forward pass of our data thoruhg this layer
    dense1.forward(x)

    #perform forward pass though activation
    activation1.forward(dense1.output) #takes computed output from dense1

    #oerform forward pass thorugh our second dense layer and takes outputs from activatio1 as inpits
    dense2.forward(activation1.output) #(3,3) because the dense1 output a=has 3 colums because 3 classes

    # #lets see the output of the first few samples hereL
    # print(f"Sample Loss: {loss_activation.output[:5]}")

    loss = loss_activation.forward(dense2.output, y)

    #calculate the predictions
    predictions = np.argmax(loss_activation.output, axis = 1)

    if len(y.shape)==2: #y is the output class need to have it discrete
        y = np.argmax(y, axis=1)

    accuracy = np.mean(predictions==y) #again, its a lot like a mini for loop

    #forward passes done 

    #backward passes:

    #initialize the backward passes
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)


        
    #Now We need to Update the paramters!!
    optimizer.pre_update_param()
    optimizer.Params_update(dense1)
    optimizer.Params_update(dense2)
    optimizer.Post_param_update()


    if not epoch % 100: 
        #epoch % 100 would return 0 when epoch is divisible by 10 
        #the not conditional would make that zero a one and all non zero values a 0 so would return true when any other value

        print(f'epoch: {epoch} ')
        print(f"accuracy: {accuracy} ")
            #perform a pass tghough our combined layer taking the output of the second dense layer to return the loss
            #print loss value
        print(f"loss: {loss}")
        print(f"LR: {optimizer.current_Learning_rate}\n")
            
        #print the gradients 
        # print(dense1.dweights)
        # print(dense1.dbiases)

        # print(dense2.dweights)
        # print(dense2.dbiases)