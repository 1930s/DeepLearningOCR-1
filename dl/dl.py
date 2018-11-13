import json
import numpy as np
import os.path as path
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras

# Number of input nodes
n_input = 27
book_name = ""
label_encoder = LabelEncoder()
loaded_model = ""


# Function that the C-program calls, return the predicted character
def ocrValue(tuple_in, max_length, model_json):
    loaded_model = model_json[0]
    onehot_encoded = model_json[1]
    label_encoder = model_json[2]
    # For some reason, the list comes in as length 54 but with only 27 elements
    # Extract the elements from indices 27 through 53
    temp = []
    for dimension in range(max_length, len(tuple_in)):
        temp.append(tuple_in[dimension])
    # print "Length:", len(tuple), "\t data:", tuple
    tuple_out = tf.constant(temp, shape=[1, n_input])
    ret_char = label_encoder.inverse_transform([np.argmax(onehot_encoded[np.argmax(loaded_model.predict(tuple_out, steps=1))])])[0]
    # print(ret_char)
    return ret_char


# Import pre-trained font data
def load_font(filename):
    trained_font = []
    expected = []
    with open(filename) as f:
        for line in f.readlines():
            splits = line.strip().split(" ")
            character_data = [float(num) for num in splits[:-1]]
            trained_font.append(character_data)
            expected.append(splits[-1])

    # data = tf.convert_to_tensor(trained_font)
    # outputs = tf.convert_to_tensor(expected)
    return trained_font, expected
