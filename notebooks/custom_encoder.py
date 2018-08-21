import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from normality.transliteration import ascii_text

class Encoder:
    def __init__(self, alphabet):
        """Initialize encoder object with alphabet string."""
        self.alphabet = alphabet
        self.tokenizer = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
        char_dict = {}
        for i, char in enumerate(alphabet):
            char_dict[char] = i + 1
        self.tokenizer.word_index = char_dict 
        # Add 'UNK' to the vocabulary 
        self.tokenizer.word_index[self.tokenizer.oov_token] = max(char_dict.values()) + 1
        
    def tokenize(self, names, latin=True):
        """Transliterate characters when latin-True and encode them as numbers"""
        names_lower = list(map(lambda x : x.lower(), names))
        if latin:
            names_lower = list(map(ascii_text, names_lower))
        names_lower = np.array(names_lower)
        names_tokenized = self.tokenizer.texts_to_sequences(names_lower)
        return names_tokenized
    
    def hot_encode(self, names, max_length):
        """Turn numbers into 01 lists of equal length"""
        sequences = self.tokenize(names)
        sequences_padded = pad_sequences(sequences, maxlen=max_length, padding='post')
        categorical = np.array(to_categorical(sequences_padded, num_classes=len(self.alphabet)+2))
        return categorical