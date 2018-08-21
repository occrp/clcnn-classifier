from keras.models import load_model
import constants as const
from custom_encoder import Encoder
from IPython.display import display, clear_output

class CharLevelClassifier:
    """Class for character level classification using pretrained convolutional neural network"""
    
    def __init__(self, model_path, alphabet):
        """Initialize object with pretrained model in h5 format and alphabet used for this model training"""
        self.model = load_model(model_path)
        self.encoder = Encoder(alphabet)
    
    def predict(self, names, batch_size = 10, verbose = False):
        """Predict result based on provided list of strings.
        
        Return list of lists, each of length equal to number of classes.
        """
        input_length = self.model.layers[0].input_shape[1]
        result = []
        for batch in self._create_batches(names, batch_size):            
            batch_result = self.model.predict(self.encoder.hot_encode(batch, input_length), batch_size = batch_size)
            batch_result = batch_result.tolist()
            result.extend(batch_result)
            if verbose:
                clear_output(wait=True)
                display(str(len(result)) + '/' + str(len(names)))
        return result
    
    def _create_batches(self, sequence, n=1):
        """Split sequences into sequences to save memory when one-hot encoding"""
        l = len(sequence)
        for ndx in range(0, l, n):
            yield sequence[ndx:min(ndx + n, l)]