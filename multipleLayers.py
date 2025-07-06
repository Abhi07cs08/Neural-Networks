import numpy as np

inputs = [
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12]
]

weights = [
    [0.2,0.1,0.4,0.5],
    [0.9,0.9,0.4,0.1],
    [0.5,0.2,0.3,0.8]
]

biases = [2,3,4]

output_layer1= np.dot(inputs, np.array(weights).T) + biases

#lets program the second layer
weights_2 = [
    [0.4,0.3,0.4],
    [0.9,0.7,0.8],
    [0.5,0.6,0.3]
]

biases_2 = [3,6,34]

output_layer2 = np.dot(output_layer1, np.array(weights_2).T) + biases_2

print(output_layer2) 