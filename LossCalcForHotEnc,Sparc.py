import numpy as np

softmax_outputs = np.array([[0.7,0.1,0.2],
                            [0.1,0.5,0.4],
                            [0.02,0.9,0.08]])

class_targets = np.array([[1,0,0],
                          [0,1,0],
                          [0,1,0]])

#probablities fot target values 

#for sparse 
# #SHAPE FUNTION: import numpy as np

# arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

# print(arr.shape) returns (2,4) 2 columns and 4 rows 
if (len(class_targets.shape) == 1):
    corect_confidence = softmax_outputs[
        range(len(softmax_outputs)), class_targets
    ]
  #  print(corect_confidence)

#for onehot
elif (len(class_targets.shape)==2): #meaning two dimensional
    corect_confidence = np.sum(
        softmax_outputs * class_targets, axis=1 #axis 1 means do this for all similar indexies row-wise 
    )
 #   print(corect_confidence)

neg_log = -np.log(corect_confidence)

avg_loss = np.mean(neg_log)
print(avg_loss)