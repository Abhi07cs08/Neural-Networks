softmax_output = [[0.6,0.1,0.3]]

import numpy as np

softmax_output = np.array(softmax_output).reshape(1,-1) 
#output: [[0.6 0.1 0.3 2.  3.  4. ]]

softmax_output = np.array(softmax_output).reshape(-1,1)
#  #output:[[0.6]
#  [0.1]
#  [0.3]
#  [2. ]
#  [3. ]
#  [4. ]]

#print(softmax_output)
#print(softmax_output*np.eye(softmax_output.shape[0]))

#use np.diagflat to create an array outputs that outputs the input vector as a diagnoal 
print(np.diagflat(softmax_output))
