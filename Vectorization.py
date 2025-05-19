import tensorflow
import keras

from tensorflow.keras.layers import TextVectorization

# Create the vectorizer
def vector(vocab_size, max_len):
    vectorizer = TextVectorization(
        max_tokens = vocab_size,
        output_mode = 'int',
        output_sequence_length = max_len
    return vectorizer
    )