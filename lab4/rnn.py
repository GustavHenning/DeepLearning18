import numpy as np

from inits import Xavier, Zeros

class RNN:
    def __init__(self, in_size, out_size, state_size,
                  init_W=None,
                  init_U=None,
                  init_V=None,
                  init_b=None,
                  init_c=None):
        init_W = init_W if init_W is not None else Xavier()
        init_U = init_U if init_U is not None else Xavier()
        init_V = init_V if init_V is not None else Xavier()
        init_b = init_b if init_b is not None else Zeros()
        init_c = init_c if init_c is not None else Zeros()
        # For input checking
        self.in_size = in_size
        self.state_size = state_size
        self.out_size = out_size

        # Trainable parameters
        self.W = init_W.from_shape((state_size, state_size))
        self.U = init_U.from_shape((state_size, in_size))
        self.b = init_b.from_shape(state_size)
        self.V = init_V.from_shape((out_size, state_size))
        self.c = init_c.from_shape(out_size)

        # Gradients
        self.g_W = np.empty_like(self.W)
        self.g_U = np.empty_like(self.U)
        self.g_b = np.empty_like(self.b)
        self.g_V = np.empty_like(self.V)
        self.g_c = np.empty_like(self.c)

        self.timesteps = 1
        self.sequence = np.empty((self.timesteps, in_size))
        self.prev_states = np.empty((self.timesteps + 1, state_size))
        self.probs = np.empty((self.timesteps, out_size))

    def weights_gradients_pairs(self):
        yield (self.W, self.g_W, 'W')
        yield (self.U, self.g_U, 'U')
        yield (self.b, self.g_b, 'b')
        yield (self.V, self.g_V, 'V')
        yield (self.c, self.g_c, 'c')

    def forward(self, sequence, prev_state):
        # Check input size
        assert sequence.shape[1] == self.in_size
        assert prev_state.size == self.state_size
        self.timesteps = sequence.shape[0]

        # Bookkeeping for backpropagation
        self.sequence = sequence
        self.prev_states = np.empty((self.timesteps + 1, self.state_size))
        self.probs = np.empty((self.timesteps, self.out_size))

        for t in range(self.timesteps):
            self.prev_states[t] = prev_state
            self.probs[t], prev_state = self.predict_prob(sequence[t],
                                                          prev_state)
        self.prev_states[-1] = prev_state

        return self.probs, self.prev_states

    def predict_prob(self, x, prev_state):
        assert x.size == self.in_size
        assert prev_state.size == self.state_size
        a = self.W @ prev_state + self.U @ x + self.b
        h = np.tanh(a)
        o = self.V @ h + self.c
        p = self._softmax(o)
        return p, h

    def predict_class(self, x, prev_state):
        probs, prev_state = self.predict_prob(x, prev_state)
        one_hot = np.zeros(self.out_size)
        one_hot[np.random.choice(self.out_size, p=probs)] = 1
        return one_hot, prev_state

    def _softmax(self, o):
        try:
            e = np.exp(o)
            res = e / e.sum()
        except FloatingPointError:
            res = np.full_like(o, fill_value=np.finfo(float).eps)
            res[np.argmax(o)] = 1 - \
                                (self.out_size - 1) * np.finfo(float).eps
        return res

    def backward(self, targets):
        """
        Note: the network will use the intermediary results of
        the previous run to propagate the gradients back
        :param targets: the target sequence to compare the output against,
                        one timestep per row
        """
        assert self.probs.shape == targets.shape
        # dL/do
        dL_do = self.probs - targets

        # dL/dc
        self.g_c = dL_do.sum(axis=0)

        # dL/dV
        self.g_V = np.zeros_like(self.V)
        for t in range(self.timesteps):
            self.g_V += np.outer(dL_do[t], self.prev_states[t+1])

        # dL/dW, dL/dU, dL/db computed iteratively going back in time
        self.g_W = np.zeros_like(self.W)
        self.g_U = np.zeros_like(self.U)
        self.g_b = np.zeros_like(self.b)
        dL_da = np.zeros(self.state_size)
        for t in range(self.timesteps - 1, 0 - 1, -1):
            dL_dh = dL_do[t] @ self.V + dL_da @ self.W
            dL_da = dL_dh * (1 - self.prev_states[t + 1] ** 2)

            self.g_W += np.outer(dL_da, self.prev_states[t])
            self.g_U += np.outer(dL_da, self.sequence[t])
            self.g_b += dL_da

        """
        dL_dh = dL_do[self.timesteps-1] @ self.V
        for t in range(self.timesteps-1, 0-1, -1):
            dL_da = dL_dh * (1 - self.prev_states[t+1]**2)
            dL_dh = dL_do[t] @ self.V + dL_da @ self.W
        """

    def cost(self, targets):
        assert self.probs.shape == targets.shape
        log_arg = (self.probs * targets).sum(axis=1)
        log_arg[log_arg == 0] = np.finfo(float).eps
        return - np.log(log_arg).sum()

    def __str__(self):
        return 'RNN {} -> {} -> {}'.format(
            self.in_size, self.state_size, self.out_size)
