# #how to calculate the loss of batches 

# softmax_outputs = [[0.7,0.1,0.2],
#                    [0.1,0.5,0.4],
#                    [0.02,0.9,0.08]]
 
# class_targets = [0,1,1]

# for target_class, distribution in zip(class_targets, softmax_outputs):
#     print(distribution[target_class]) #will help us return the highest confidence value we determined from the class targets. 

#now what if we make an array?
import numpy as np

softmax_outputs = np.array([[0.7,0.1,0.2],
                   [0.1,0.5,0.4],
                   [0.02,0.9,0.08]])

class_targets = [0,1,1]
print(softmax_outputs[[0,1,2], class_targets]) #the class target'th term in the [list number here in the array]

#even more efficient:
print(softmax_outputs[range(len(softmax_outputs)), class_targets])

#now let us apply neagtive log to these key values

Neg_log = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets]) #individual losses

#now we need to calculate average loss of the process Using Numpy's avg method

avg_loss = np.mean(Neg_log)

print(avg_loss)