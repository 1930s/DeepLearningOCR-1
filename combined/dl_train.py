# https://www.tensorflow.org/guide/keras
# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# https://www.pyimagesearch.com/2018/09/10/keras-tutorial-how-to-get-started-with-keras-deep-learning-and-python/
# https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
# https://machinelearningmastery.com/save-load-keras-deep-learning-models/

import codecs
import json
import numpy as np
import os.path as path
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import sys
import tensorflow as tf
from tensorflow import keras

import time # only use for demo video

# Don't shorten printing numpy lists (i.e. don't use "...")
# np.set_printoptions(threshold=np.nan)


# Train the Deep Neural Network using a given filename
# Then output the resulting network model to a file to be loaded later
def main():
    if len(sys.argv) < 2:
        print("Specify fontData file name")
        return 1
    elif not path.isfile(path.abspath(path.join(__file__, "../" + sys.argv[1]))):
        print("Given filename for fontData doesn't exist")
        return 1
    else:
        # Pre-trained fontData and expected characters with matching array indices
        dataset = np.genfromtxt(sys.argv[1], dtype='str', delimiter=" ", encoding="utf8")

        X = dataset[:,0:-1].astype(dtype='float')
        Y = dataset[:,-1]
        uniques = list(set(Y))

        n_input = 27
        n_neurons_in_h1 = 10
        n_neurons_in_h2 = 50
        n_neurons_in_h3 = 25
        n_classes = len(uniques)

        learning_rate = 0.001
        num_epochs = 20
        steps_per_epoch = 100

        # Integer encode
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(Y)
        # Binary encode
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        # onehot_encoded is 2D array of either ones or zeroes, where only one 1 is in each line
        # its length is the same as the length of the fontData file (bad?)

        # Get lists to train the Neural Network on
        train_images = np.array(X)
        train_labels = np.array(onehot_encoded)

        # Make testing samples
        test_sample_size = int(0.4 * len(X))
        choices = np.random.choice(len(X), size=test_sample_size, replace=False)
        test_images = np.array(X[choices])
        test_labels = np.array(onehot_encoded[choices])

        # Create a Neural Network with 3 hidden layers and 1 output layer
        model = keras.Sequential([
            keras.layers.Dense(units=n_neurons_in_h1, input_shape=(n_input,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(units=n_neurons_in_h2),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(units=n_neurons_in_h3),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(units=n_classes, activation="sigmoid")
        ])

        optimizer = keras.optimizers.Adam(lr=learning_rate)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])

        # Train the model to the given fontData
        model.fit(train_images, train_labels, epochs=num_epochs, steps_per_epoch=steps_per_epoch)

        print("Performing tests")
        test_loss, test_acc = model.evaluate(test_images, test_labels)

        print("Test accuracy:", test_acc * 100, "%")

        # serialize weights to HDF5
        model.save("dl_model.h5")

        return 0


if __name__ == "__main__":
    main()
