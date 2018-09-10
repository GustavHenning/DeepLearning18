import itertools

import numpy as np

from inits import Zeros
from crnn import CharRNN


def generate_text(rnn, data, first_char=None, initial_state=None, length=200):
    if first_char is None:
        first_char = np.zeros(rnn.in_size)
        first_char[np.random.randint(0, rnn.in_size)] = 1
    if initial_state is None:
        initial_state = np.zeros(rnn.state_size)
    seq, last_state = rnn.generate(first_char, initial_state, length)
    return data.decode_to_strings(seq), last_state

def save_crnn(rnn, filename):
    np.savez_compressed(
        filename,
        **{name: weight for weight, _, name in rnn.weights_gradients_pairs()},
        in_out_size=rnn.in_size,
        state_size=rnn.state_size
    )


def load_crnn(filename):
    loaded = np.load(filename)
    rnn = CharRNN(
        in_out_size=loaded['in_out_size'],
        state_size=loaded['state_size'],
        init_W=Zeros(),
        init_U=Zeros(),
        init_V=Zeros()
    )
    for weight, _, name in rnn.weights_gradients_pairs():
        weight += loaded[name]
    return rnn


def id_gen(prefix=''):
    for i in itertools.count():
        yield '{}_{}'.format(prefix, i)
