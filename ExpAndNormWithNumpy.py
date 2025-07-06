import numpy

output_layer = [4.9,2,21,3,435]

exp_Values = numpy.exp(output_layer)

print(exp_Values)

# to normalize, the list will basically be one exponentiated value divided by the sum of the exponentiated values, so that their sum would be 1

norm_values = exp_Values/numpy.sum(exp_Values)
print(norm_values)


