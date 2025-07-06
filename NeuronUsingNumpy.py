import numpy
#compare to our firstNeuron code
inputs = [1,2,3]
weights = [0.2,0.8,-0.5]
bias = 2.0

outputs = numpy.dot(weights,inputs)+bias #dot product of the 2 vectors. multiplying values of the same index
print(outputs)