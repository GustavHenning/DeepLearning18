from collections import namedtuple

import numpy as np
from sklearn import preprocessing

InputTargetSequence = namedtuple('InputTargetSequence', 'input output')


class TextData:
    def __init__(self, filename):
        self.char_seq = list(self.load_text(filename))
        self.label_encoder = preprocessing.LabelBinarizer()
        self.enc_text = self.label_encoder.fit_transform(
            self.char_seq)
        self.tot_chars, self.num_classes = self.enc_text.shape

    def encode(self, *vals):
        if len(vals) == 1:
            return np.squeeze(self.label_encoder.transform(list(vals[0])))
        else:
            return [self.encode(s) for s in vals]

    def decode_to_strings(self, *seqs):
        if len(seqs) == 1:
            return ''.join(self.label_encoder.inverse_transform(seqs[0]))
        else:
            return [self.decode_to_strings(s) for s in seqs]

    def get_seqs(self, length=25):
        for i in range(0, self.tot_chars - length, length):
            yield InputTargetSequence(
                input=self.enc_text[i:i + length],
                output=self.enc_text[i + 1:i + length + 1]
            )

    @staticmethod
    def load_text(filename):
        with open(filename, 'r') as f:
            return f.read()
