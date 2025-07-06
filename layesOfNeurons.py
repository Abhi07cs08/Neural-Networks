#lets create a layer of 3 neurons, which is recieving 4 inputs. The differentiating factor is that they have different weights and biases

inputs = [1,2,3,2.5] # the inputs are the same for all of them 

weights1 = [0.2,0.8,-0.5,1]
weights2 = [0.5,-0.91,0.26,-0.5]
weights3 = [-0.26,-0.27,0.17,0.87]
weightterms = [weights1,weights2,weights3]

bias1 = 2
bias2 = 3
bias3 = 0.5
biases = [bias1,bias2,bias3]

#output of the current layer
outputs = []

#for each neuron 
for neuron_Weights, neuron_bias in zip(weightterms,biases): #iterate over multiple iterables in one go
    #zeroing output of the current neuron
    print(neuron_Weights)
    neuron_Output = 0
    #for each input and weight to the neuron
    for n_input, weight in zip(inputs, neuron_Weights): #so they were able to break down the list's list
        print(weight)
        neuron_Output += n_input*weight
    #add bias
    neuron_Output +=  neuron_bias
    #putting output into the output list
    outputs.append(neuron_Output)
print(outputs)