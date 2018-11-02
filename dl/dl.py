import tensorflow as tf
import numpy as np
import os.path as path
from tensorflow.examples.tutorials.mnist import input_data

'''mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
print mnist
n_features = 27
n_classes = 60

training_epochs = 500
n_neurons_in_h1 = 100
n_neurons_in_h2 = 100
n_neurons_in_h3 = 100
learning_rate = 0.01

# Length of input
x = tf.placeholder('float', [None, 27])
y = tf.placeholder('float')

def neural_network_model(data):
    # (input_data * weights) + biases
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([27, n_neurons_in_h1])),
                      'biases': tf.Variable(tf.random_normal(n_neurons_in_h1))}
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_neurons_in_h1, n_neurons_in_h2])),
                      'biases': tf.Variable(tf.random_normal(n_neurons_in_h2))}
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_neurons_in_h2, n_neurons_in_h3])),
                      'biases': tf.Variable(tf.random_normal(n_neurons_in_h3))}
    output_layer = {'weights': tf.Variable(tf.random_normal([n_neurons_in_h3, n_classes])),
                    'biases': tf.Variable(tf.random_normal(n_classes))}

    l1 = tf.add(tf.matmul(data,))'''

# Import pre-trained font data
def load_font(filename):
    trained_font = []
    expected = []
    with open(filename) as f:
        for line in f.readlines():
            splits = line.strip().split(" ")
            character_data = splits[:-1]
            trained_font.append(character_data)
            expected.append(splits[-1])

    data = tf.convert_to_tensor(trained_font)
    return data, expected


# Pre-trained English font data and expected characters with matching array indices
font, expected = load_font(path.abspath(path.join(__file__, "../../fontData/english.data")))
# print font

def ocrValue(tuple_in, max_length):
    # For some reason, the list comes is as length 54 but with only 27 elements
    # Extract the elements from indices 27 through 53
    temp = []
    for dimension in range(max_length, len(tuple_in)):
        temp.append(tuple_in[dimension])
    # print "Length:", len(tuple), "\t data:", tuple
    tuple = tf.convert_to_tensor(temp)
    return "a"
