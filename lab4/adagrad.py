import numpy as np

class Adagrad():
    def __init__(self, rnn, init_learning_rate,
                 decay_factor=1.0, stateful=False, clip=None):
        self.rnn = rnn
        self.smooth_costs = []
        self.stateful = stateful
        self.steps = 0
        self.learning_rate = init_learning_rate
        self.decay_factor = decay_factor
        self.clip = clip
        self.learning_rates = []
        # store gradient history
        self.hist = {name: np.zeros_like(weights)
                     for weights, _, name in self.rnn.weights_gradients_pairs()}

    def train(self, sequence_pairs, epochs=1,
              callback=None, callback_every=0,
              epoch_callback=None):
        for _ in range(epochs):
            prev_state = np.zeros(self.rnn.state_size)
            for i, sp in enumerate(sequence_pairs):
                probs, states = self.rnn.forward(sp.input, prev_state)
                self.rnn.backward(sp.output)
                self.execute_update()
                cost = self.rnn.cost(sp.output)
                self.update_metrics(cost)
                if self.stateful:
                    prev_state = states[-1]
                else:
                    prev_state = np.zeros(self.rnn.state_size)
                if callback is not None and i % callback_every == 0:
                    callback(self, probs[-1], states[-1])
                self.learning_rate *= self.decay_factor
            if epoch_callback:
                epoch_callback(self)

    def update_metrics(self, cost):
        self.learning_rates.append(self.learning_rate)
        self.steps += 1
        if len(self.smooth_costs) > 0:
            cost = .999 * self.smooth_costs[-1] + .001 * cost
        self.smooth_costs.append(cost)

    def execute_update(self):
        for weights, grad, name in self.rnn.weights_gradients_pairs():
            if self.clip is not None:
                grad = np.clip(grad, -self.clip, +self.clip)
            self.hist[name] += grad ** 2
            update = self.learning_rate * grad / \
                     np.sqrt(self.hist[name] + np.finfo(float).eps)
            # if __debug__:
            #    self.learning_rate_warning(weights, update)
            weights -= update
