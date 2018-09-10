import numpy as np

from rnn import RNN

class CharRNN(RNN):
    def __init__(self, in_out_size, state_size,
                  init_W=None,
                  init_U=None,
                  init_V=None,
                  init_b=None,
                  init_c=None):

        super().__init__(in_out_size, in_out_size, state_size,
                         init_W, init_U, init_V,
                         init_b, init_c)

    def generate(self, x, prev_state, timesteps):
        res = np.empty((timesteps, self.out_size), dtype=np.int)
        for t in range(timesteps):
            res[t], prev_state = self.predict_class(x, prev_state)
            x = res[t]
        return res, prev_state
