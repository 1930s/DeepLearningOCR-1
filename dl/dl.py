import json
import numpy as np
import os.path as path
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras

import time # only use for demo video

# Number of input nodes
n_input = 27

def load_saved_model(book_name):
    # load json and create model
    # loaded_model = keras.models.model_from_json(model_json)
    # load weights into new model
    # loaded_model.load_weights("model.h5")
    # print("Loaded model from disk")

    loaded_model = keras.models.load_model("model.h5")

    # Compile the model and test on it
    # loaded_model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
    # test_loss, test_acc = loaded_model.evaluate(test_images, test_labels)

    # print("Test accuracy:", test_acc)

    with open("onehot.json", 'r') as onehot_json:
        onehot_encoded = json.loads(onehot_json.read())
    # print(onehot_encoded)

    dataset = np.genfromtxt("../fontData/" + book_name + ".data", dtype='str', delimiter=" ", encoding="utf8")
    expecteds = dataset[:, -1]
    label_encoder = LabelEncoder()
    label_encoder.fit_transform(expecteds)

    return loaded_model, onehot_encoded, label_encoder

# Function that the C-program calls, return the predicted character
def ocrValue(tuple_in, max_length, args):
    # Start a timer to report the length of time
    # different parts of this code take to run
    start_time = time.time()

    # loaded_model = model_json[0]
    # loaded_model = keras.models.load_model("model.h5")
    loaded_model = args[0]
    onehot_encoded = args[1]
    label_encoder = args[2]
    load_model_time = time.time()

    # onehot_encoded = model_json[1]
    '''with open("onehot.json", 'r') as onehot_json:
        onehot_encoded = json.loads(onehot_json.read())'''

    load_onehot_time = time.time()

    '''label_encoder = LabelEncoder()
    dataset = np.genfromtxt("../fontData/" + "english" + ".data", dtype='str', delimiter=" ", encoding="utf8")
    expecteds = dataset[:, -1]
    label_encoder.fit_transform(expecteds)'''

    # For some reason, the list comes in as length 54 but with only 27 elements
    # Extract the elements from indices 27 through 53
    temp = []
    # print("reeeee", range(max_length, len(tuple_in)))
    for dimension in range(max_length, len(tuple_in)):
        if (dimension >= len(tuple_in) - 2):
            temp.append(tuple_in[dimension] / 3)
        else:
            temp.append(tuple_in[dimension])

    fix_tuple_time = time.time()

    # print("Length:", len(temp), "\t data:", temp)
    tuple_out = tf.constant(temp, shape=[1, n_input])

    make_tensorflow_var_time = time.time()

    hot_index = np.argmax(loaded_model.predict(tuple_out, steps=1))
    ret_char = str(label_encoder.inverse_transform([np.argmax(onehot_encoded[np.argmax(loaded_model.predict(tuple_out, steps=1))])])[0]).encode('utf-8')
    # print("Python returned:", ret_char, type(ret_char))
    # print(ret_char.encode('utf-8'))
    # print(ret_char)
    end_time = time.time()

    '''
    print("Load Model: %3f   Load OneHot: %3f   Fix Tuple: %3f   Make TF Var: %3f   Rev Transform: %3f"
            % (load_model_time - start_time,
            load_onehot_time - load_model_time,
            fix_tuple_time - load_onehot_time,
            make_tensorflow_var_time - fix_tuple_time,
            end_time - make_tensorflow_var_time))
    '''

    print("\nPrediction: ", ret_char.decode("utf8"), "Hot Index: ", hot_index, "\n", np.array(temp))
    return ret_char

'''
book = "english"
tuples = np.genfromtxt("../fontData/" + book + ".data", dtype='str', delimiter=" ", encoding="utf8")
max = 27
correct_count = 0
model = load_saved_model(book)
for tup in tuples:
    expected_char = tup[-1]
    predicted = ocrValue(tup[0:-1], max, model).decode("utf8")
    print("\n  Expected:", expected_char, "\nPrediction:", predicted)
    print(tup)
    if (expected_char == predicted):
        correct_count += 1
    # time.sleep(0.25)
print("Num correct:", correct_count, "out of", len(tuples))
'''
