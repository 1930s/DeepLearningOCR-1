import json
import numpy as np
import os.path as path
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow import keras
from sklearn.externals import joblib

import time # only use for demo video

# Number of input nodes
n_input = 27

def load_saved_model(font_name):
    base_name = path.splitext(path.basename(font_name))[0]
    # load HDF5 file into a model
    loaded_model = keras.models.load_model(base_name + "_dl_model.h5")

    onehot_encoded = joblib.load(base_name + r'_dl_onehot.pkl')
    label_encoder = joblib.load(base_name + r'_dl_label_data.pkl')

    '''
    with open(base_name + "_onehot.pkl", "rb") as onehot_encoded:
        pickle.load(onehot_encoded)

    with open(base_name + "_label_data.pkl", "rb") as label_encoder:
        pickle.load(label_encoder)


    dataset = np.genfromtxt(font_name, dtype='str', delimiter=" ", encoding="utf8")
    expecteds = dataset[:, -1]
    # Integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(expecteds)
    #integer_encoded.sort()
    # Binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    '''
    return loaded_model, onehot_encoded, label_encoder

# Function that the C-program calls, return the predicted character
def ocrValueDL(tuple_in, max_length, args):
    loaded_model = args[0]
    onehot_encoded = args[1]
    label_encoder = args[2]

    # For some reason, the list comes in as length 54 but with only 27 elements
    # Extract the elements from indices 27 through 53
    temp = []
    for dimension in range(-1 * max_length, 0):
        if (dimension >= -2):
            temp.append(float(tuple_in[dimension]) / 3.0)
        else:
            temp.append(float(tuple_in[dimension]))

    tuple_out = tf.constant(temp, shape=[1, n_input])
    hot_index = np.argmax(loaded_model.predict(tuple_out, steps=1))
    ret_char = str(label_encoder.inverse_transform([hot_index])[0]).encode('utf-8')

    # print("\nPrediction: ", ret_char.decode("utf-8"), "Hot Index: ", hot_index, "\n", np.array(temp))
    return ret_char
