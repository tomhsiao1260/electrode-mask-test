import math

# EPSILON for computing floating point error during build
# https://en.wikipedia.org/wiki/Machine_epsilon#Values_for_standard_hardware_floating_point_arithmetics
FLOAT32_EPSILON = math.pow(2, -24)