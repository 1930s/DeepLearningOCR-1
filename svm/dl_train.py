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
    elif not path.isfile(path.abspath(path.join(__file__, "../../fontData/" + sys.argv[1]))):
        print("Given filename for fontData doesn't exist")
        return 1
    else:
        # Pre-trained fontData and expected characters with matching array indices
        dataset = np.genfromtxt("../fontData/" + sys.argv[1], dtype='str', delimiter=" ", encoding="utf8")

        X = dataset[:,0:-1].astype(dtype='float')
        Y = dataset[:,-1]
        uniques = list(set(Y))

        n_input = 27
        n_neurons_in_h1 = 100
        n_neurons_in_h2 = 50
        n_neurons_in_h3 = 25
        n_classes = len(uniques)

        learning_rate = 0.001
        num_epochs = 10
        steps_per_epoch = 100

        # print("X", X)
        # print("Y", Y)

        # Integer encode
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(Y)
        # print("int encoded:", len(integer_encoded))
        # print(integer_encoded)
        # Binary encode
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        # print(onehot_encoded)
        # Invert to get character back
        # print("encoded:", len(onehot_encoded))
        # inverted = label_encoder.inverse_transform([np.argmax(onehot_encoded[0, :])])
        # print(inverted)

        onehot_list = onehot_encoded.tolist()
        onehot_json = "onehot.json"
        json.dump(onehot_list, codecs.open(onehot_json, 'w', encoding='utf-8'), sort_keys=True, indent=4)

        # onehot_encoded is 2D array of either ones or zeroes, where only one 1 is in each line
        # its length is the same as the length of the fontData file (bad?)

        # Y = onehot_encoded

        # print("X:", X)
        # print("Y:", Y)

        # Get lists to train the Neural Network on
        train_images = np.array(X)
        train_labels = np.array(onehot_encoded)

        # Make testing samples
        test_sample_size = int(0.4 * len(X))
        choices = np.random.choice(len(X), size=test_sample_size, replace=False)
        test_images = np.array(X[choices])
        test_labels = np.array(onehot_encoded[choices])

        # print("test", test_images[0])
        # print("test", test_labels[0])

        # print(label_encoder.inverse_transform([np.argmax(test_labels[0, :])]))

        # Create a Neural Network with 3 hidden layers and 1 output layer
        model = keras.Sequential([
            keras.layers.Dense(units=n_neurons_in_h1, input_shape=(n_input,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(units=n_neurons_in_h2),
            keras.layers.Dense(units=n_neurons_in_h3, kernel_initializer="normal"),
            keras.layers.Dense(units=n_classes, activation="softmax")
        ])

        model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Train the model to the given fontData
        model.fit(train_images, train_labels, epochs=num_epochs, steps_per_epoch=steps_per_epoch)

        print("Performing tests")
        test_loss, test_acc = model.evaluate(test_images, test_labels)

        print("Test accuracy:", test_acc * 100, "%")

        # Test the model on the fontData, comparing expected and predicted output
        '''
        print("Testing on fontData file...")
        correct_count = 0
        predictions = [model.predict(tf.constant(X[index], shape=[1, n_input]), steps=1) for index in range(len(X))]
        for index in range(len(predictions)):
            # print(predictions[index])
            character = label_encoder.inverse_transform([np.argmax(predictions[index])])[0]
            expected_char = label_encoder.inverse_transform([np.argmax(onehot_encoded[index])])[0]
            print("\n  Expected:", expected_char)
            print("Prediction:", character)
            if (expected_char == character):
                correct_count += 1
            # time.sleep(0.25)
        print("Num correct:", correct_count, "out of", len(predictions))
        '''

        # serialize model to JSON
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model.h5")
        # print("Saved model to disk")

        return 0


if __name__ == "__main__":
    main()
