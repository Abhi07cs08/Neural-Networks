import numpy as np
layer_outputs = np.array([[2.23,4.34,1],
                 [2.4,-4.2,3.3],
                 [1.41,0.53,4.55]]) #now we can perfomr array specific functions such as axis and dims on it

print("sum without axis function")
print(np.sum(layer_outputs))
print("thiss will be identical to above since AXIS is zero")
print(np.sum(layer_outputs, axis=None))

#axis:0 is columns and axis 1 is rows axis 1 would sume row-wise and 0 would sum column wise 

print(np.sum(layer_outputs, axis=1, keepdims=False))
print(np.sum(layer_outputs, axis=1, keepdims=True))
