import json
import numpy as np
import os.path as path
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras

import time # only use for demo video

# Number of input nodes
n_input = 27
book_name = ""
label_encoder = LabelEncoder()
loaded_model = ""
onehot_encoded = ""


def load_model(model_json, book_name):
    # load json and create model
    loaded_model = keras.models.model_from_json(model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    # print("Loaded model from disk")

    # Compile the model and test on it
    loaded_model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
    # test_loss, test_acc = loaded_model.evaluate(test_images, test_labels)

    # print("Test accuracy:", test_acc)

    onehot_json_file = open("onehot.json", 'r')
    onehot_json = onehot_json_file.read()
    onehot_json_file.close()

    global onehot_encoded
    onehot_encoded = json.loads(onehot_json)
    # print(onehot_encoded)

    dataset = np.genfromtxt("../fontData/" + book_name + ".data", dtype='str', delimiter=" ", encoding="utf8")
    expecteds = dataset[:, -1]
    label_encoder.fit_transform(expecteds)

    return loaded_model, onehot_encoded, label_encoder


# Function that the C-program calls, return the predicted character
def ocrValue(tuple_in, max_length, model_json):
    loaded_model = model_json[0]
    onehot_encoded = model_json[1]
    label_encoder = model_json[2]
    # For some reason, the list comes in as length 54 but with only 27 elements
    # Extract the elements from indices 27 through 53
    temp = []
    # print("reeeee", range(max_length, len(tuple_in)))
    for dimension in range(max_length, len(tuple_in)):
        if (dimension >= len(tuple_in) - 2):
            temp.append(tuple_in[dimension] / 3)
        else:
            temp.append(tuple_in[dimension])
    # print("Length:", len(temp), "\t data:", temp)
    tuple_out = tf.constant(temp, shape=[1, n_input])
    ret_char = str(label_encoder.inverse_transform([np.argmax(onehot_encoded[np.argmax(loaded_model.predict(tuple_out, steps=1))])])[0]).encode('utf-8')
    # print("Python returned:", ret_char, type(ret_char))
    # print(ret_char.encode('utf-8'))
    return ret_char

'''
book = "english"
tuples = np.genfromtxt("../fontData/" + book + ".data", dtype='str', delimiter=" ", encoding="utf8")
max = 27
with open("model.json", 'r') as f:
    mod = f.read()
    model = load_model(mod, book)
    correct_count = 0
    for tup in tuples:
        # print(tup)
        expected_char = tup[-1]
        predicted = ocrValue(tup[0:-1], max, model)
        print("\n  Expected:", expected_char, "\nPrediction:", predicted)
        if (expected_char == predicted):
            correct_count += 1
        # time.sleep(0.25)
    print("Num correct:", correct_count, "out of", len(tuples))
'''
