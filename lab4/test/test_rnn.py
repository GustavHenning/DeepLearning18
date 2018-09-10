import unittest

import numpy as np

from inits import Xavier, Zeros
from rnn import RNN

class TestRNN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.in_size = 2
        cls.out_size = 3
        cls.state_size = 5
        cls.rnn = RNN(
            in_size=cls.in_size,
            out_size=cls.out_size,
            state_size=cls.state_size
        )

    def test_predict_prob(self):
        p, h = self.rnn.predict_prob(
            x=np.array([0, 1]),
            prev_state=np.random.random(size=self.state_size)
        )
        self.is_valid_output(p)
        self.is_valid_state(h)

    def test_predict_class(self):
        c, h = self.rnn.predict_class(
            x=np.array([1, 0]),
            prev_state=np.random.random(size=self.state_size)
        )
        self.is_valid_class(c)
        self.is_valid_state(h)

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
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
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
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
        ])

        self.rnn.backward(targets)

    def is_valid_class(self, c):
        self.is_valid_output(c)
        self.assertTrue((c == 1).sum(), 1)

    def is_valid_state(self, h):
        self.assertEqual(h.size, self.state_size)

    def is_valid_output(self, p):
        self.assertEqual(p.size, self.out_size)
        self.assertAlmostEqual(p.sum(), 1)


if __name__ == '__main__':
    unittest.main()
