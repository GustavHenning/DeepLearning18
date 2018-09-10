import unittest
from tempfile import TemporaryFile

import numpy as np

from inits import Xavier, Zeros
from crnn import CharRNN
from test.test_rnn import TestRNN
from util import save_crnn, load_crnn


class TestCharRNN(TestRNN):
    @classmethod
    def setUpClass(cls):
        cls.rnn = None
        cls.in_size = 2
        cls.out_size = 2
        cls.state_size = 5
        cls.rnn = CharRNN(
            in_out_size=cls.in_size,
            state_size=cls.state_size
        )

    def test_save_restore(self):
        outfile = TemporaryFile()
        save_crnn(self.rnn, outfile)
        outfile.seek(0)
        restored = load_crnn(outfile)
        for x, y in zip(self.rnn.weights_gradients_pairs(),
                        restored.weights_gradients_pairs()):
            self.assertTrue(np.allclose(x[0], y[0]))

    def test_generate(self):
        seq, last_state = self.rnn.generate(
            x=np.array([1, 0]),
            prev_state=np.random.random(size=self.state_size),
            timesteps=10
        )
        for t in range(10):
            self.is_valid_class(seq[t])
        self.is_valid_state(last_state)

    def test_forward(self):
        sequence = np.array([
            [0, 1],
            [1, 0],
            [0, 1],
            [0, 1],
            [1, 0],
            [0, 1],
        ])
        probs, states = self.rnn.forward(
            sequence=sequence,
            prev_state=np.random.random(size=self.state_size)
        )
        for t in range(5):
            self.is_valid_output(probs[t])
            self.is_valid_state(states[t])

        targets = np.array([
            [0, 1],
            [1, 0],
            [0, 1],
            [1, 0],
            [1, 0],
            [0, 1],
        ])
        cost = self.rnn.cost(targets)

    def test_backprop(self):
        sequence = np.array([
            [0, 1],
            [1, 0],
            [0, 1],
            [0, 1],
            [1, 0],
            [0, 1],
        ])
        probs, states = self.rnn.forward(
            sequence=sequence,
            prev_state=np.random.random(size=self.state_size)
        )

        targets = np.array([
            [0, 1],
            [1, 0],
            [0, 1],
            [1, 0],
            [1, 0],
            [0, 1],
        ])
        self.rnn.backward(targets)


if __name__ == '__main__':
    unittest.main()
