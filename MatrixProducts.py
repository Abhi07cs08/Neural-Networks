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

# check notebook but: the number of rows in inputs(index 1) should be equal tothe columns in weights(index 0) so we need to transpose the weights. The measure of the matric would be the index 0 of the inputs and the index 1 of the transposed weights. in this case 3

output = np.dot(inputs, np.array(weights).T) + biases
print(output)