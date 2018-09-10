import sys
import numpy as np


class Initializer:
    def from_shape(self, shape):
        print("Cannot initialize with Initializer baseclass")
        sys.exit(1)
        pass

class Zeros(Initializer):
    def from_shape(self, shape):
        return np.zeros(shape, dtype=float)

class Xavier(Initializer):
    def from_shape(self, shape):
        return np.random.normal(0, np.sqrt(2 / (sum(shape))), shape)
