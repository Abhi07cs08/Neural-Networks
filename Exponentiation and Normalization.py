#no numpy first

import math
output_layer = [4.9,2,21,3,435]

math.e # e

exp_Values = []

for value in output_layer:
    exp_Values.append(math.e**value)

print(exp_Values)

# to normalize, the list will basically be one exponentiated value divided by the sum of the exponentiated values, so that their sum would be 1

norm_base = sum(exp_Values)

norm_values = []
for value in exp_Values:
    norm_values.append(value/norm_base)
print(norm_values)