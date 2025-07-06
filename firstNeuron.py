#a neuron takes inputs, with associated wieghts, accepts the bias value, and gives the output using the formula Ill write

inputs = [1,2,3]
weights = [0.2,0.8,-0.5]
bias = 2.0

output = bias
for term in range(len(inputs)):
    output += inputs[term]*weights[term] #+bias
print(output) #this is the output of the neuron


