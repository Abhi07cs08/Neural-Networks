import numpy

inputs = [1, 2, 3, 2.5] 
weights = [
    [0.2 ,0.8, -0.5, 1],
    [0.5,-0.91,0.26,-0.5],
    [-0.26,-0.27,0.17,0.87]
    ] #this made a matrix (4,3)

biases = [2,3,0.5]

output = numpy.dot(weights, inputs)+biases
#what exactly this does is shown b`elow
#numpy.dot(weights, inputs)=numpy.dot(weights[0],inputs),numpy.dot(weights[1],inputs),numpy.dot(weights[2],inputs)

print(output)

