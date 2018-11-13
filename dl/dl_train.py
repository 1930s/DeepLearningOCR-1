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

# Don't shorten printing numpy lists (i.e. don't use "...")
# np.set_printoptions(threshold=np.nan)

# TODO: Output onehot_encoded into a file, load it into C, then pass it to this script
# TODO: Separate training and predicting code into two separate files
#       - Add entry in Makefile "make train" to reference training
#       - Call prediction through C-Python interface


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
        font, expected = load_font(path.abspath(path.join(__file__, "../../fontData/" + sys.argv[1])))
        uniques = list(set(expected))
        # print("Number of unique characters:", len(uniques))

        n_input = 27
        n_neurons_in_h1 = 512
        n_neurons_in_h2 = 256
        n_neurons_in_h3 = 128
        n_classes = len(uniques)

        learning_rate = 0.01
        num_epochs = 10
        steps_per_epoch = 100

        dataset = np.genfromtxt("../fontData/" + sys.argv[1], dtype='str', delimiter=" ", encoding="utf8")

        X = dataset[:,0:-1].astype(dtype='float')
        Y = dataset[:,-1]

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

        # TODO: output the onehot_encoded list and load it into C
        # Pass it as an arg to this python file, then call the LabelEncoder to inverse it

        # onehot_encoded is 2D array of either ones or zeroes, where only one 1 is in each line
        # its length is the same as the length of the fontData file (bad?)

        Y = onehot_encoded

        # print("X:", X)
        # print("Y:", Y)

        # Get lists to train the Neural Network on
        train_images = np.array(X)
        train_labels = np.array(Y)

        # Make testing samples
        test_sample_size = int(0.2 * len(Y))
        test_images = np.array(X[np.random.choice(len(Y), size=test_sample_size, replace=False)])
        test_labels = np.array(Y[np.random.choice(len(Y), size=test_sample_size, replace=False)])

        # Create a Neural Network with 3 hidden layers and 1 output layer
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

        # Train the model to the given fontData
        model.fit(train_images, train_labels, epochs=num_epochs, steps_per_epoch=steps_per_epoch)

        print("Performing tests")
        test_loss, test_acc = model.evaluate(test_images, test_labels)

        print("Test accuracy:", test_acc)

        predictions = model.predict(test_images)

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
