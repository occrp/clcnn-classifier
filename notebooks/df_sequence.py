from keras.utils import Sequence
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

class DfSequence(Sequence):

    def __init__(self, df, batch_size, max_length, alphabet_size, num_classes):
        self.df = df
        self.batch_size = batch_size
        self.max_length = max_length
        self.alphabet_size = alphabet_size
        self.num_classes = num_classes

    def __len__(self):
        return int(np.ceil(self.df.shape[0] / float(self.batch_size)))

    def __getitem__(self, idx):
        
        actual_batch_size = 0
        if self.df.shape[0]-self.batch_size*idx > self.batch_size:
            actual_batch_size = self.batch_size
        else:
            actual_batch_size = self.df.shape[0]-self.batch_size*idx
        
        xy = self.df.iloc[idx*self.batch_size:idx*self.batch_size + actual_batch_size]
        x_cat = pad_sequences(xy.iloc[:, 0], maxlen=self.max_length, padding='post')
        x_cat = to_categorical(x_cat, num_classes=self.alphabet_size)
        y_cat = to_categorical(xy.iloc[:, 1], num_classes=self.num_classes)

        return (x_cat, y_cat)