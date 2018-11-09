import tensorflow as tf
import numpy as np
import os.path as path
#from tensorflow.examples.tutorials.mnist import input_data

'''
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
print mnist
'''

#begin tensorflow stuff
#ocr = input_data.read_data_sets(tuple, one_hot=True)
n_features = 27
# n_classes = 60
batch_size = 10

n_neurons_in_h1 = 100
n_neurons_in_h2 = 100
n_neurons_in_h3 = 100
learning_rate = 0.01


# Given Python list of the 27 data points describing a glyph,
# Return what character that glyph represents, based on the network
def neural_network_model(data):
    # (input_data * weights) + biases
    # data = tf.constant(data, shape=[1, 27])

    global l_1, a_1, l_2, a_2, l_3, a_3, a_4
    l_1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    a_1 = tf.nn.sigmoid(l_1)

    l_2 = tf.add(tf.matmul(a_1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    a_2 = tf.nn.sigmoid(l_2)

    l_3 = tf.add(tf.matmul(a_2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    a_3 = tf.nn.sigmoid(l_3)

    output = tf.matmul(a_3, output_layer['weights']) + output_layer['biases']
    a_4 = tf.nn.sigmoid(output)

    sess = tf.Session()
    result = sess.run(a_4).tolist()[0]
    print(uniques[result.index(max(result))])
    return uniques[result.index(max(result))]


# Return the derivative of the sigmoid function
def sigmoid_prime(data):
    return tf.multiply(tf.sigmoid(data), tf.subtract(tf.constant(1.0), tf.sigmoid(data)))


# Using a 2-D 1-hot matrix, compare how different the output matrix is
# def my_loss_function(output, expected_index):


# Back propagation
def back_prop():
    global diff
    diff = tf.subtract(a_4, y)
    # Start from end, work to beginning
    d_z_3 = tf.multiply(diff, sigmoid_prime(l_3))
    d_bias_3 = d_z_3
    d_weight_3 = tf.matmul(tf.transpose(a_2), d_z_3)

    d_a_2 = tf.matmul(d_z_3, tf.transpose(hidden_3_layer['weights']))
    d_z_2 = tf.multiply(d_a_2, sigmoid_prime(l_2))
    d_bias_2 = d_z_2
    d_weight_2 = tf.matmul(tf.transpose(a_1), d_z_2)

    d_a_1 = tf.matmul(d_z_2, tf.transpose(hidden_2_layer['weights']))
    d_z_1 = tf.multiply(d_a_1, sigmoid_prime(l_1))
    d_bias_1 = d_z_1
    d_weight_1 = tf.matmul(tf.transpose(a_0), d_z_1)

    # Begin updating the network
    eta = tf.constant(0.5)

    # Deltas that the weights and biases will change by
    step = [
        tf.assign(hidden_1_layer['weights'],
                  tf.subtract(hidden_1_layer['weights'], tf.multiply(eta, d_weight_1))),
        tf.assign(hidden_1_layer['biases'],
                  tf.subtract(hidden_1_layer['biases'], tf.multiply(eta, tf.reduce_mean(d_bias_1, axis=[0])))),

        tf.assign(hidden_2_layer['weights'],
                  tf.subtract(hidden_2_layer['weights'], tf.multiply(eta, d_weight_2))),
        tf.assign(hidden_2_layer['biases'],
                  tf.subtract(hidden_2_layer['biases'], tf.multiply(eta, tf.reduce_mean(d_bias_2, axis=[0])))),

        tf.assign(hidden_3_layer['weights'],
                  tf.subtract(hidden_3_layer['weights'], tf.multiply(eta, d_weight_3))),
        tf.assign(hidden_3_layer['biases'],
                  tf.subtract(hidden_3_layer['biases'], tf.multiply(eta, tf.reduce_mean(d_bias_3, axis=[0])))),
    ]

    return step


# Given the size of each batch, return a random selection from the training data
def next_batch(batch_size):
    indices = [np.random.random_integers(0, 56) for count in range(batch_size)]
    ret_batch = [font[index] for index in indices]
    ret_expected = [expected[index] for index in indices]
    return ret_batch, ret_expected


# Train the neural network based on existing trained font data
def train_neural_network(font_data, expeted_chars):
    print("train start")

    training_epochs = 10

    # acct_mat = tf.equal(tf.argmax(a_3, 1), tf.argmax(y, 1))
    # acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))

    print("before with")
    with tf.Session() as sess:
        print("before run")
        # sess.run(tf.initialize_all_variables())
        sess.run(tf.global_variables_initializer())

        for epoch in range(training_epochs):
            for _ in range(int(len(expected) / batch_size)):
                epoch_x, epoch_y = next_batch(batch_size)

                # prediction = neural_network_model(x)
                # print("prediction worked")

                step = back_prop()
                sess.run(step, feed_dict={a_0: epoch_x, y: epoch_y})

        correct = tf.equal(tf.argmax(a_0, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy', accuracy.eval({a_0: font, y: expected}))

    print("after with")


# Import pre-trained font data
def load_font(filename):
    trained_font = []
    expected = []
    with open(filename) as f:
        for line in f.readlines():
            splits = line.strip().split(" ")
            character_data = [float(num) for num in splits[:-1]]
            trained_font.append(character_data)
            expected.append(splits[-1])

    # data = tf.convert_to_tensor(trained_font)
    # outputs = tf.convert_to_tensor(expected)
    return trained_font, expected


# Function that the C-program calls, return the predicted character
def ocrValue(tuple_in, max_length):
    # For some reason, the list comes is as length 54 but with only 27 elements
    # Extract the elements from indices 27 through 53
    temp = []
    for dimension in range(max_length, len(tuple_in)):
        temp.append(tuple_in[dimension])
    # print "Length:", len(tuple), "\t data:", tuple
    tuple = tf.constant(temp, shape=[1, n_features])
    # return neural_network_model(tuple)
    return neural_network_model(tuple)


# Pre-trained English font data and expected characters with matching array indices
font, expected = load_font(path.abspath(path.join(__file__, "../../fontData/english.data")))
uniques = list(set(expected))
# print(len(uniques))
n_classes = len(uniques)

diff = tf.placeholder('float', [None, n_classes])

# Length of input
a_0 = tf.placeholder('float', [None, 27])
y = tf.placeholder('float', [None, n_classes])

hidden_1_layer = {'weights': tf.Variable(tf.random_normal([27, n_neurons_in_h1])).initialized_value(),
                  'biases': tf.Variable(tf.random_normal([n_neurons_in_h1])).initialized_value()}

hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_neurons_in_h1, n_neurons_in_h2])).initialized_value(),
                  'biases': tf.Variable(tf.random_normal([n_neurons_in_h2])).initialized_value()}

hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_neurons_in_h2, n_neurons_in_h3])).initialized_value(),
                  'biases': tf.Variable(tf.random_normal([n_neurons_in_h3])).initialized_value()}

output_layer = {'weights': tf.Variable(tf.random_normal([n_neurons_in_h3, n_classes])).initialized_value(),
                'biases': tf.Variable(tf.random_normal([n_classes])).initialized_value()}

# Before training
print(neural_network_model(tf.constant(font[0], shape=[1, 27])))

train_neural_network(font, expected)
# end tensorflow stuff

# After training
print(neural_network_model(tf.constant(font[0], shape=[1, 27])))
