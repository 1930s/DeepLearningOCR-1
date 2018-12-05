import json
import numpy as np
import os.path as path
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
import pickle
import tensorflow as tf
from tensorflow import keras

import time # only use for demo video

# Function that the C-program calls, return the predicted character
def ocrValueSVM(tuple_in, max_length, book_name):
    base_name = path.splitext(path.basename(book_name))[0]
    # load the classifier
    with open(base_name + '_svm_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

    # For some reason, the list comes in as length 54 but with only 27 elements
    # Extract the elements from indices 27 through 53
    temp = []
    for dimension in range(max_length, len(tuple_in)):
        if (dimension >= len(tuple_in) - 2):
            temp.append(tuple_in[dimension] / 3)
        else:
            temp.append(tuple_in[dimension])

    # this bit of code does the actual predicting and makes sure that the
    # arrays and characters are properly formatted and encoded for return
    # to the C program.
    char_array = np.array(temp)
    char_array = np.reshape(char_array, (1, 27))
    ret_charray = loaded_model.predict(char_array)
    ret_char = (ret_charray[0])
    ret_char = str(ret_char[0]).encode('utf-8')

    return ret_char
