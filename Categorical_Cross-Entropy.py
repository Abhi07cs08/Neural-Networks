import math
#this loss control function would work with the output payer so lets create a sample output layer 

softmax_output = [0.7,0.1,0.2]

#lets make the ground truth: the values that we want to be true
target_output = [1,0,0] #this is how we ideally wanted the probabilities to be 

loss = -(math.log(softmax_output[0])*target_output[0]+
math.log(softmax_output[1])*target_output[1]+
    math.log(softmax_output[2])*target_output[2]) # = value + 0 + 0

print(loss)

#remeber IN PYTHON NUMPY THE LOG WITHOUT BASE REFERS TO NATURAL LOG.
