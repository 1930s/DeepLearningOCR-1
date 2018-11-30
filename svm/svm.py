import json
import numpy as np
import os.path as path
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
import pickle
import tensorflow as tf
from tensorflow import keras

import time # only use for demo video

# Number of input nodes
n_input = 27
book_name = ""
label_encoder = LabelEncoder()
loaded_model = ""
onehot_encoded = ""


# Function that the C-program calls, return the predicted character
def ocrValue(tuple_in, max_length):

    # load the classifier
    with open('my_dumped_classifier.pkl', 'rb') as g:
        loaded_model = pickle.load(g)

    #print("yea hey cool")
    #ret_char = 'a'
    #print(tuple_in)
    '''
    loaded_model = model_json[0]
    onehot_encoded = model_json[1]
    label_encoder = model_json[2]
    '''
    # For some reason, the list comes in as length 54 but with only 27 elements
    # Extract the elements from indices 27 through 53
    temp = []
    # print("reeeee", range(max_length, len(tuple_in)))
    for dimension in range(max_length, len(tuple_in)):
        if (dimension >= len(tuple_in) - 2):
            temp.append(tuple_in[dimension] / 3)
        else:
            temp.append(tuple_in[dimension])
    '''
    # print("Length:", len(temp), "\t data:", temp)
    tuple_out = tf.constant(temp, shape=[1, n_input])
    ret_char = str(label_encoder.inverse_transform([np.argmax(onehot_encoded[np.argmax(loaded_model.predict(tuple_out, steps=1))])])[0]).encode('utf-8')
    # print("Python returned:", ret_char, type(ret_char))
    # print(ret_char.encode('utf-8'))
    '''
    #temp2 = [[]]
    char_array = np.array(temp)
    #print("char_array", char_array)
    char_array = np.reshape(char_array, (1, 27))
    #temp2.append(char_array)
    #char_array2 = temp2)
    #print("char_array2", temp2)
    ret_charray = loaded_model.predict(char_array)
    ret_char = (ret_charray[0])
    ret_char = str(ret_char[0]).encode('utf-8')
    #print(ret_char)
    #print("hey yea also cool")
    return ret_char

'''
book = "english"
tuples = np.genfromtxt("../fontData/" + book + ".data", dtype='str', delimiter=" ", encoding="utf8")
max = 27
with open("model.json", 'r') as f:
    mod = f.read()
    correct_count = 0
    for tup in tuples:
        # print(tup)
        expected_char = tup[-1]
        predicted = ocrValue(tup[0:-1], max, load_model(mod, book))
        print("\n  Expected:", expected_char, "\nPrediction:", predicted)
        if (expected_char == predicted):
            correct_count += 1
        # time.sleep(0.25)
    print("Num correct:", correct_count, "out of", len(tuples))
'''
