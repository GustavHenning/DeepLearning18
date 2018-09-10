import unittest

import numpy as np

from data import TextData
from inits import Xavier, Zeros
from crnn import CharRNN
from util import generate_text


class TestTextGeneration(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.goblet = TextData('data/goblet_book.txt')
        cls.rnn = CharRNN(
            in_out_size=cls.goblet.num_classes,
            state_size=100
        )

    def test_text_generation(self):
        seq, last_state = generate_text(self.rnn, self.goblet)
        print(seq)
        self.assertEqual(last_state.size, self.rnn.state_size)

        seq, last_state = generate_text(
            self.rnn, self.goblet, self.goblet.encode('A'),
            np.random.random(self.rnn.state_size), 100)
        print(seq)
        self.assertEqual(last_state.size, self.rnn.state_size)


if __name__ == '__main__':
    unittest.main()
