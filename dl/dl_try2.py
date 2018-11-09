# https://www.tensorflow.org/guide/keras
# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# https://www.pyimagesearch.com/2018/09/10/keras-tutorial-how-to-get-started-with-keras-deep-learning-and-python/
# https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
# https://machinelearningmastery.com/save-load-keras-deep-learning-models/

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os.path as path
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Don't truncate printing numpy lists (i.e. don't use "...")
# np.set_printoptions(threshold=np.nan)

# TODO: Pass filename into this file
# TODO: Output onehot_encoded into a file, load it into C, then pass it to this script
# TODO: Separate training and predicting code into two separate files
#       - Add entry in Makefile "make train" to reference training
#       - Call prediction through C-Python interface

# Function that the C-program calls, return the predicted character
def ocrValue(tuple_in, max_length):
    # For some reason, the list comes is as length 54 but with only 27 elements
    # Extract the elements from indices 27 through 53
    temp = []
    for dimension in range(max_length, len(tuple_in)):
        temp.append(tuple_in[dimension])
    # print "Length:", len(tuple), "\t data:", tuple
    tuple = tf.constant(temp, shape=[1, n_input])
    # return neural_network_model(tuple)
    return model.predict(tuple)


# Import pre-trained font data
def load_font(filename):
    trained_font = []
    expected = []
    # count = 0
    with open(filename, encoding="utf8") as f:
        for line in f.readlines():
            # count += 1
            splits = line.strip().split(" ")
            # print(count, splits)
            character_data = [float(num) for num in splits[:-1]]
            trained_font.append(character_data)
            expected.append(splits[-1])

    # data = tf.convert_to_tensor(trained_font)
    # outputs = tf.convert_to_tensor(expected)
    return trained_font, expected


'''
    Start training code
'''

# Pre-trained English font data and expected characters with matching array indices
font, expected = load_font(path.abspath(path.join(__file__, "../../fontData/complete.data")))
uniques = list(set(expected))
print("Number of unique characters:", len(uniques))


n_input = 27
n_neurons_in_h1 = 512
n_neurons_in_h2 = 256
n_neurons_in_h3 = 128
n_classes = len(uniques)

learning_rate = 0.01
num_epochs = 6
steps_per_epoch = 100

dataset = np.genfromtxt("../fontData/complete.data", dtype='str', delimiter=" ", encoding="utf8")

X = dataset[:,0:-1].astype(dtype='float')
Y = dataset[:,-1]

# Integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(Y)
print("int encoded:", len(integer_encoded))
# print(integer_encoded)
# Binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
# print(onehot_encoded)
# Invert first example
print("encoded:", len(onehot_encoded))
inverted = label_encoder.inverse_transform([np.argmax(onehot_encoded[500, :])])
print(inverted)

# TODO: output the onehot_encoded list and load it into C
# Pass it as an arg to this python file, then call the LabelEncoder to inverse it

# onehot_encoded is 2D array of either ones or zeroes, where only one 1 is in each line
# its length is the same as the length of the fontData file (bad?)

Y = onehot_encoded

# print("X:", X)
# print("Y:", Y)

'''
    THIS CAN ALL GO INTO SEPARATE TRAINING FILE
'''

train_images = np.array(X)
train_labels = np.array(Y)

test_images = np.array(X[np.random.choice(len(Y), size=1000, replace=False)])
test_labels = np.array(Y[np.random.choice(len(Y), size=1000, replace=False)])

model = keras.Sequential([
    keras.layers.Dense(units=n_neurons_in_h1, input_shape=(n_input,), activation="sigmoid"),
    keras.layers.Dense(units=n_neurons_in_h2, activation="sigmoid"),
    keras.layers.Dense(units=n_neurons_in_h3, activation="sigmoid"),
    keras.layers.Dense(units=n_classes, activation="softmax")
])

model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# print(train_images[0])
# print(train_labels[0])

model.fit(train_images, train_labels, epochs=num_epochs, steps_per_epoch=steps_per_epoch)

# test_loss, test_acc = model.evaluate(test_images, test_labels)

# print("Test accuracy:", test_acc)

predictions = model.predict(test_images)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
# print("Saved model to disk")

'''
    END TRAINING-ONLY STUFF
'''

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = keras.models.model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
# print("Loaded model from disk")

# Compile the model and test on it
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
test_loss, test_acc = loaded_model.evaluate(test_images, test_labels)

print("Test accuracy:", test_acc)
